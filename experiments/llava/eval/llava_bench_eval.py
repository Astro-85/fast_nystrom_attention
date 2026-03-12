"""
LLaVA-Bench (In-the-Wild) evaluation entry-point for LLaVA-NeXT + Fast Nyström Attention (FNA).

This script mirrors the COCO VQA eval runner, but targets LLaVA-Bench style JSONL prompts.

Inputs (typical LLaVA-Bench-in-the-Wild format):
  questions.jsonl lines like:
    {"image": "001.jpg", "text": "...", "category": "conv|detail|complex", "question_id": 0}
  Images live under --images-root (e.g., llava-bench-in-the-wild/images)

Outputs:
  * predictions.jsonl  - detailed per-question generations + timing/token stats (for profiling).
  * answers.jsonl      - LLaVA-bench-compatible answer file:
        {"question_id": 0, "prompt": "...", "answer_id": "...", "model_id": "...", "metadata": {...}, "text": "..."}

You can feed answers.jsonl into whatever judge/eval pipeline you use (GPT-4 judge, Mixtral judge, etc.).
See common answer-file format described in NVIDIA NeMo NeVA docs. (Not required by this script.)

Example:
  python experiments/llava/eval/llava_bench_eval.py \
    --model-id llava-hf/llava-v1.6-vicuna-7b-hf \
    --questions-jsonl /path/to/llava-bench-in-the-wild/questions.jsonl \
    --images-root /path/to/llava-bench-in-the-wild/images \
    --output-dir outputs/llava_fna_llavabench \
    --fna-layer-range 18:32 --fna-num-sample 256 \
    --max-new-tokens 256 --temperature 0.0

Notes:
  - Long-answer benchmarks: don't use short-answer guidance by default.
  - Decodes ONLY generated tokens (prompt is excluded).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple
import sys
import uuid

import torch
from PIL import Image
from tqdm import tqdm

# Ensure the project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fast_nystrom_attention import LlavaNextForConditionalGenerationFNA
from transformers import LlavaNextProcessor


# ---------------------------------------------------------------------------
# Optional: shortuuid for LLaVA-Bench answer_id (fallback to uuid4 if missing)
# ---------------------------------------------------------------------------

try:
    import shortuuid  # type: ignore
except Exception:  # pragma: no cover
    shortuuid = None


def make_answer_id() -> str:
    if shortuuid is not None:
        return shortuuid.uuid()
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BenchExample:
    question_id: int
    image: str
    text: str
    category: Optional[str] = None

    @staticmethod
    def from_json(obj: Dict[str, object]) -> "BenchExample":
        return BenchExample(
            question_id=int(obj["question_id"]),
            image=str(obj["image"]),
            text=str(obj["text"]),
            category=str(obj.get("category")) if obj.get("category") is not None else None,
        )


@dataclass
class GenerationRecord:
    question_id: int
    image: str
    category: Optional[str]
    prompt: str
    answer: str
    raw_completion: str
    latency_s: float
    num_starting_tokens: int
    num_generated_tokens: int
    generation_latency_s: Optional[float] = None

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    num_questions: int
    num_predicted: int
    avg_latency_s: Optional[float]
    median_latency_s: Optional[float]
    avg_generated_tokens: Optional[float]
    avg_tokens_per_s: Optional[float] = None
    avg_generation_latency_s: Optional[float] = None
    avg_generation_tokens_per_s: Optional[float] = None
    # breakdown by category (conv/detail/complex), if present
    by_category: Optional[Dict[str, Dict[str, float]]] = None

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core helpers (mostly shared with your COCO VQA runner)
# ---------------------------------------------------------------------------

def dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {sorted(mapping)}")
    return mapping[name]


def parse_layer_selection(layer_range: str, explicit: Optional[Sequence[int]], disabled: bool) -> List[int]:
    if disabled:
        return []
    if explicit:
        return sorted(set(int(layer) for layer in explicit))
    if not layer_range:
        return []
    try:
        start_str, end_str = layer_range.split(":", maxsplit=1)
        start, end = int(start_str), int(end_str)
    except ValueError as exc:  # pragma: no cover
        raise ValueError("--fna-layer-range must be formatted as start:end") from exc
    if end <= start:
        raise ValueError("--fna-layer-range end must be > start")
    return list(range(start, end))


def move_batch_to_device(
    batch: MutableMapping[str, torch.Tensor],
    device: Optional[str],
    dtype: torch.dtype,
) -> MutableMapping[str, torch.Tensor]:
    if device is None:
        return batch
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device=device)
            if value.dtype in {torch.float32, torch.float16, torch.bfloat16}:
                batch[key] = batch[key].to(dtype)
    return batch


def make_cuda_sync_fn(device: Optional[str]) -> Callable[[], None]:
    should_sync = bool(device) and str(device).startswith("cuda") and torch.cuda.is_available()
    if not should_sync:
        return lambda: None

    def _sync_fn() -> None:
        torch.cuda.synchronize()

    return _sync_fn


def _classify_generation_phase(args: Tuple[object, ...], kwargs: Dict[str, object]) -> str:
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        candidate = args[0]
        if torch.is_tensor(candidate):
            input_ids = candidate
    if torch.is_tensor(input_ids) and input_ids.ndim >= 2:
        seq_len = input_ids.shape[1]
        if seq_len <= 1:
            return "decode"
    return "prefill"


class _ForwardTimingTracker:
    def __init__(self, model: "LlavaNextForConditionalGenerationFNA", sync_fn: Callable[[], None]):
        self.model = model
        self.sync_fn = sync_fn
        self.timings = {"prefill": 0.0, "decode": 0.0}
        self._start_time: Optional[float] = None
        self._phase: str = "prefill"
        self._pre_handle = model.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        self._post_handle = model.register_forward_hook(self._post_hook, with_kwargs=True)

    def remove(self) -> None:
        if self._pre_handle is not None:
            self._pre_handle.remove()
            self._pre_handle = None
        if self._post_handle is not None:
            self._post_handle.remove()
            self._post_handle = None

    def _pre_hook(self, _module, args, kwargs):
        self.sync_fn()
        self._phase = _classify_generation_phase(args, kwargs)
        self._start_time = time.perf_counter()

    def _post_hook(self, _module, args, kwargs, _output):
        if self._start_time is None:
            return
        self.sync_fn()
        elapsed = time.perf_counter() - self._start_time
        self.timings[self._phase] += elapsed
        self._start_time = None


@contextmanager
def track_generation_timings(
    model: "LlavaNextForConditionalGenerationFNA",
    sync_fn: Callable[[], None],
):
    tracker = _ForwardTimingTracker(model, sync_fn)
    try:
        yield tracker.timings
    finally:
        tracker.remove()


def prepare_prompt(
    processor: LlavaNextProcessor,
    user_text: str,
    system_prompt: Optional[str],
) -> str:
    """
    LLaVA-NeXT chat template: [{system?}, {user: [image, text]}] -> add_generation_prompt.
    """
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    conversation.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    )
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def postprocess_answer(answer: str) -> str:
    """
    Keep long-form answers (LLaVA-Bench is judge-based), but remove common role prefixes.
    """
    text = answer.strip()
    text = re.sub(r"^\s*(assistant|assistant:)\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def load_model_and_processor(
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
) -> Tuple[LlavaNextForConditionalGenerationFNA, LlavaNextProcessor]:
    processor_id = args.processor_id or args.model_id
    processor = LlavaNextProcessor.from_pretrained(processor_id, use_fast=False)

    fna_layers = parse_layer_selection(args.fna_layer_range, args.fna_layers, args.disable_fna)
    fna_config = {
        "fna_layers": fna_layers,
        "num_sample": args.fna_num_sample,
        "resample_every_layer": args.fna_resample_every_layer,
        "sampling_strategy": args.fna_sampling_strategy,
        "sampling_features": args.fna_sampling_features,
    }

    model_source = args.checkpoint_path or args.model_id
    logging.info("Loading LLaVA checkpoint from %s", model_source)
    model = LlavaNextForConditionalGenerationFNA.from_pretrained(
        model_source,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None if args.device_map in {None, "none"} else args.device_map,
        fna_config=fna_config,
        fna_cache={},
    )
    model.eval()
    if args.device and (args.device_map in {None, "none"}):
        logging.info("Moving model to %s", args.device)
        model.to(args.device)
    return model, processor


# ---------------------------------------------------------------------------
# LLaVA-Bench dataset IO
# ---------------------------------------------------------------------------

def load_bench_questions(path: Path, limit: Optional[int] = None) -> List[BenchExample]:
    """
    Accepts JSONL (recommended) or JSON list/dict containing the same fields.
    HF dataset is JSONL with keys: image, text, category, question_id.
    """
    raw = path.read_text()
    examples: List[BenchExample] = []
    try:
        payload = json.loads(raw)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("questions"), list):
            rows = payload["questions"]  # type: ignore[index]
        else:
            raise ValueError("Not a JSON list/dict; try JSONL")
        for row in rows:
            if not isinstance(row, dict):
                continue
            examples.append(BenchExample.from_json(row))
    except Exception:
        # JSONL fallback
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                examples.append(BenchExample.from_json(obj))

    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"No questions found in {path}")
    return examples


def resolve_image_path(images_root: Path, image_rel: str) -> Path:
    """
    LLaVA-Bench-in-the-Wild stores 'image' like '001.jpg'. Typically images_root points
    directly to the folder containing those files.
    """
    candidate = images_root / image_rel
    if candidate.exists():
        return candidate

    # defensive: maybe images are under a subfolder like images_root/images
    alt = images_root / "images" / image_rel
    if alt.exists():
        return alt

    raise FileNotFoundError(f"Image not found: tried {candidate} and {alt}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(
    model: LlavaNextForConditionalGenerationFNA,
    processor: LlavaNextProcessor,
    example: BenchExample,
    images_root: Path,
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
) -> GenerationRecord:
    img_path = resolve_image_path(images_root, example.image)
    image = Image.open(img_path).convert("RGB")

    user_text = example.text
    if args.answer_guidance:
        user_text = f"{user_text}\n\n{args.answer_guidance.strip()}"

    prompt = prepare_prompt(processor, user_text, args.system_prompt)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = move_batch_to_device(inputs, args.device, torch_dtype)

    # Prompt length in tokens (used to slice generated tokens)
    prompt_len = int(inputs["input_ids"].shape[-1])

    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": pad_token_id,
        # Often helps end cleanly:
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )

    sync_fn = make_cuda_sync_fn(args.device)
    with torch.inference_mode():
        sync_fn()
        start = time.perf_counter()
        with track_generation_timings(model, sync_fn) as timings:
            output_ids = model.generate(**inputs, **generation_kwargs)
        sync_fn()
        latency = time.perf_counter() - start
    generation_latency = timings.get("decode", None)

    # Decode ONLY generated tokens (exclude the prompt/prefill tokens)
    gen_ids = output_ids[:, prompt_len:]
    if gen_ids.numel() == 0:
        decoded_gen = ""
    else:
        decoded_gen = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    answer = postprocess_answer(decoded_gen)

    num_generated = int(gen_ids.shape[-1]) if gen_ids.ndim == 2 else 0

    if args.save_full_completion:
        decoded_full = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        raw_completion = decoded_full
    else:
        raw_completion = decoded_gen

    return GenerationRecord(
        question_id=example.question_id,
        image=example.image,
        category=example.category,
        prompt=example.text,
        answer=answer,
        raw_completion=raw_completion,
        latency_s=latency,
        generation_latency_s=generation_latency,
        num_starting_tokens=prompt_len,
        num_generated_tokens=num_generated,
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def read_existing_predictions(path: Path) -> List[GenerationRecord]:
    if not path.exists():
        return []
    records: List[GenerationRecord] = []
    with path.open("r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(GenerationRecord(**json.loads(line)))
    logging.info("Resuming from %d existing predictions", len(records))
    return records


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a") as fp:
        fp.write(json.dumps(payload) + "\n")


def write_prediction(predictions_path: Path, record: GenerationRecord) -> None:
    append_jsonl(predictions_path, record.to_json())


def write_llava_bench_answer(
    answers_path: Path,
    record: GenerationRecord,
    model_id_for_file: str,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    """
    LLaVA-bench-in-the-wild answer JSONL format used by common judge scripts:
      {"question_id": 0, "prompt": "...", "answer_id": "...", "model_id": "...", "metadata": {}, "text": "..."}
    """
    payload: Dict[str, object] = {
        "question_id": record.question_id,
        "prompt": record.prompt,
        "answer_id": make_answer_id(),
        "model_id": model_id_for_file,
        "metadata": metadata or {},
        "text": record.answer,
    }
    append_jsonl(answers_path, payload)


def dump_metrics(metrics_path: Path, summary: EvaluationSummary) -> None:
    with metrics_path.open("w") as fp:
        json.dump(summary.to_json(), fp, indent=2)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=level)


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LLaVA-Bench-in-the-Wild inference on LLaVA-NeXT + FNA.")
    # model / hf
    p.add_argument("--model-id", default="llava-hf/llava-v1.6-vicuna-7b-hf",
                   help="Hugging Face model id or local path")
    p.add_argument("--processor-id", default=None,
                   help="Optional processor id (defaults to --model-id)")
    p.add_argument("--checkpoint-path", default=None,
                   help="Optional local checkpoint directory overriding --model-id")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None,
                   help="Pass through to transformers.from_pretrained (e.g., 'auto')")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "float64"])

    # dataset
    p.add_argument("--questions-jsonl", type=Path, required=True,
                   help="Path to LLaVA-Bench questions file (JSONL recommended)")
    p.add_argument("--images-root", type=Path, required=True,
                   help="Path to images folder for LLaVA-Bench (contains 001.jpg, ...)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="Debug: limit number of examples")

    # generation
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)

    # prompts
    p.add_argument("--system-prompt",
                   default="You are a helpful assistant.",
                   help="System prompt for chat template")
    p.add_argument("--answer-guidance", default="",
                   help="Optional extra instruction appended after each prompt (keep empty for natural long answers)")

    # FNA config (mirrors your COCO eval script)
    p.add_argument("--fna-layer-range", default="12:32", help="Inclusive:exclusive layer range using FNA")
    p.add_argument("--fna-layers", type=int, nargs="*", default=None, help="Explicit list of layers using FNA")
    p.add_argument("--fna-num-sample", type=int, default=256)
    p.add_argument("--fna-resample-every-layer", action="store_true", help="Resample landmarks before each FNA layer")
    p.add_argument("--fna-sampling-strategy", default="fps", choices=["fps", "random"])
    p.add_argument("--fna-sampling-features", default="q", choices=["input", "q", "k", "v"])
    p.add_argument("--disable-fna", action="store_true")

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-full-completion", action="store_true",
                   help="If set, raw_completion stores full decoded prompt+answer (bigger files).")
    p.add_argument("--model-id-for-answers", default=None,
                   help="Optional override for 'model_id' field in answers.jsonl (defaults to --model-id)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    set_random_seed(args.seed)

    torch_dtype = dtype_from_string(args.dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    answers_path = args.output_dir / "answers.jsonl"
    metrics_path = args.output_dir / "metrics.json"

    examples = load_bench_questions(args.questions_jsonl, args.limit)

    # Resume logic: avoid regenerating existing question_ids
    existing = read_existing_predictions(predictions_path)
    known_ids = {r.question_id for r in existing}

    model, processor = load_model_and_processor(args, torch_dtype)

    model_id_for_file = args.model_id_for_answers or args.model_id
    # store some run metadata into each answer record
    run_metadata: Dict[str, object] = {
        "dtype": args.dtype,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "fna_layer_range": args.fna_layer_range,
        "fna_layers": args.fna_layers,
        "fna_num_sample": args.fna_num_sample,
        "fna_resample_every_layer": bool(args.fna_resample_every_layer),
        "fna_sampling_strategy": args.fna_sampling_strategy,
        "fna_sampling_features": args.fna_sampling_features,
        "disable_fna": bool(args.disable_fna),
    }

    predictions: List[GenerationRecord] = list(existing)

    progress = tqdm(examples, desc="LLaVA-Bench", unit="q")
    for ex in progress:
        if ex.question_id in known_ids:
            continue

        rec = generate_answer(model, processor, ex, args.images_root, args, torch_dtype)
        write_prediction(predictions_path, rec)
        write_llava_bench_answer(answers_path, rec, model_id_for_file=model_id_for_file, metadata=run_metadata)

        predictions.append(rec)
        known_ids.add(ex.question_id)
        maybe_empty_cuda_cache()

    # Aggregate metrics
    latencies = [r.latency_s for r in predictions]
    tokens = [r.num_generated_tokens for r in predictions]
    generation_latencies = [r.generation_latency_s for r in predictions if r.generation_latency_s is not None]
    generation_tokens = [r.num_generated_tokens for r in predictions if r.generation_latency_s is not None]

    avg_latency = float(sum(latencies) / len(latencies)) if latencies else None
    avg_generated_tokens = float(sum(tokens) / len(tokens)) if tokens else None

    avg_tokens_per_s = None
    if (avg_latency is not None) and (avg_latency > 0) and (avg_generated_tokens is not None):
        avg_tokens_per_s = avg_generated_tokens / avg_latency

    avg_generation_latency = float(sum(generation_latencies) / len(generation_latencies)) if generation_latencies else None
    avg_generation_tokens_per_s = None
    if generation_latencies:
        total_gen_latency = float(sum(generation_latencies))
        if total_gen_latency > 0:
            avg_generation_tokens_per_s = float(sum(generation_tokens) / total_gen_latency)

    # category breakdown (if present)
    by_cat: Dict[str, List[GenerationRecord]] = defaultdict(list)
    for r in predictions:
        if r.category:
            by_cat[str(r.category)].append(r)

    by_category_summary: Optional[Dict[str, Dict[str, float]]] = None
    if by_cat:
        by_category_summary = {}
        for cat, rs in by_cat.items():
            lats = [x.latency_s for x in rs]
            toks = [x.num_generated_tokens for x in rs]
            by_category_summary[cat] = {
                "n": float(len(rs)),
                "avg_latency_s": float(sum(lats) / len(lats)) if lats else 0.0,
                "avg_generated_tokens": float(sum(toks) / len(toks)) if toks else 0.0,
                "avg_tokens_per_s": float((sum(toks) / sum(lats)) if sum(lats) > 0 else 0.0),
            }

    summary = EvaluationSummary(
        num_questions=len(examples),
        num_predicted=len(predictions),
        avg_latency_s=avg_latency,
        median_latency_s=float(statistics.median(latencies)) if latencies else None,
        avg_generated_tokens=avg_generated_tokens,
        avg_tokens_per_s=avg_tokens_per_s,
        avg_generation_latency_s=avg_generation_latency,
        avg_generation_tokens_per_s=avg_generation_tokens_per_s,
        by_category=by_category_summary,
    )
    dump_metrics(metrics_path, summary)

    logging.info("Done. Wrote:\n  %s\n  %s\n  %s", predictions_path, answers_path, metrics_path)
    logging.info("Metrics: %s", json.dumps(summary.to_json(), indent=2))


if __name__ == "__main__":
    main()
