from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import string
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import LlavaNextProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fast_nystrom_attention import LlavaNextForConditionalGenerationFNA


@dataclass
class GenerationRecord:
    question_id: str
    question: str
    answer_choices: List[str]
    ground_truth_answer: str
    predicted_answer: str
    full_generation: str
    generation_latency_s: Optional[float] = None
    prefill_time_s: Optional[float] = None
    decode_time_s: Optional[float] = None

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ScienceQAMetrics:
    total_questions: int
    correct_answers: int
    accuracy: float
    average_latency_s: Optional[float] = None
    median_latency_s: Optional[float] = None
    average_prefill_time_s: Optional[float] = None
    median_prefill_time_s: Optional[float] = None
    average_decode_time_s: Optional[float] = None
    median_decode_time_s: Optional[float] = None

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def choice_in_text(norm_choice: str, norm_text: str) -> bool:
    if len(norm_choice.split()) == 1 and len(norm_choice) <= 3:
        return re.search(rf"\b{re.escape(norm_choice)}\b", norm_text) is not None
    return norm_choice in norm_text


def letter_to_choice(candidate: str, answer_choices: List[str]) -> Optional[str]:
    m = re.search(r"\b([A-E])\b", candidate.strip(), flags=re.IGNORECASE)
    if not m:
        return None

    idx = ord(m.group(1).upper()) - ord("A")
    if 0 <= idx < len(answer_choices):
        return answer_choices[idx].strip()
    return None


def match_candidate_to_choice(candidate: str, answer_choices: List[str]) -> Optional[str]:
    candidate = candidate.strip()
    norm_candidate = normalize_text(candidate)

    letter_choice = letter_to_choice(candidate, answer_choices)
    if letter_choice:
        return letter_choice

    for choice in answer_choices:
        if normalize_text(choice) == norm_candidate:
            return choice.strip()

    for choice in answer_choices:
        norm_choice = normalize_text(choice)
        if norm_choice and norm_candidate.startswith(norm_choice):
            return choice.strip()

    for choice in answer_choices:
        norm_choice = normalize_text(choice)
        if norm_candidate and norm_choice.startswith(norm_candidate):
            return choice.strip()

    return None


def extract_predicted_choice_text(decoded: str, answer_choices: List[str]) -> str:
    cleaned = decoded.strip()

    matches = re.findall(r"Answer:\s*(.+)", cleaned, flags=re.IGNORECASE)
    if matches:
        parsed = match_candidate_to_choice(matches[-1].splitlines()[0], answer_choices)
        if parsed:
            return parsed

    matches = re.findall(r"(?:the\s+)?answer\s+is:?\s*(.+)", cleaned, flags=re.IGNORECASE)
    if matches:
        parsed = match_candidate_to_choice(matches[-1].splitlines()[0], answer_choices)
        if parsed:
            return parsed

    matches = re.findall(r"(?:the\s+)?letter\s+is:?\s*([A-E])", cleaned, flags=re.IGNORECASE)
    if matches:
        parsed = match_candidate_to_choice(matches[-1], answer_choices)
        if parsed:
            return parsed

    first_line = cleaned.splitlines()[0] if cleaned else ""
    norm_first = normalize_text(first_line)

    for choice in answer_choices:
        norm_choice = normalize_text(choice)
        if norm_choice and choice_in_text(norm_choice, norm_first):
            return choice.strip()

    norm_output = normalize_text(cleaned)

    for choice in answer_choices:
        if normalize_text(choice) == norm_output:
            return choice.strip()

    for choice in answer_choices:
        norm_choice = normalize_text(choice)
        if norm_choice and choice_in_text(norm_choice, norm_output):
            return choice.strip()

    output_words = set(norm_output.split())
    best_choice = ""
    best_score = -1

    for choice in answer_choices:
        choice_words = set(normalize_text(choice).split())
        score = len(output_words & choice_words)
        if score > best_score:
            best_score = score
            best_choice = choice.strip()

    return best_choice if best_score > 0 else cleaned


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ScienceQA Evaluation with LLaVA + FNA")

    parser.add_argument("--model-id", default="llava-hf/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--checkpoint-path", default=None)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fna-layer-range", default="12:32")
    parser.add_argument("--fna-layers", type=int, nargs="*", default=None)
    parser.add_argument("--fna-num-sample", type=int, default=256)
    parser.add_argument("--fna-resample-every-layer", action="store_true")
    parser.add_argument("--fna-sampling-strategy", default="random", choices=["fps", "random"])
    parser.add_argument("--disable-fna", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--scienceqa-hf-dataset", default="derek-thomas/ScienceQA")
    parser.add_argument("--scienceqa-split", default="validation")
    parser.add_argument("--scienceqa-cache-dir", type=Path, default=None)

    parser.add_argument("--CLIP-feature-layer", type=int, default=-1, help="Which CLIP layer to take features from for FNA (counting from the end, -1 is the final layer)")

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "float64"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def parse_layer_selection(layer_range: str, explicit: Optional[Sequence[int]], disabled: bool) -> List[int]:
    if disabled:
        return []
    if explicit:
        return sorted(set(int(layer) for layer in explicit))
    if not layer_range:
        return []

    start_str, end_str = layer_range.split(":", maxsplit=1)
    start, end = int(start_str), int(end_str)

    if end <= start:
        raise ValueError("--fna-layer-range end must be > start")

    return list(range(start, end))


def dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping[name]


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


def load_model_and_processor(
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
) -> Tuple[torch.nn.Module, LlavaNextProcessor]:
    processor_id = args.processor_id or args.model_id
    processor = LlavaNextProcessor.from_pretrained(processor_id, use_fast=False)

    fna_layers = parse_layer_selection(args.fna_layer_range, args.fna_layers, args.disable_fna)
    fna_config = {
        "fna_layers": fna_layers,
        "num_sample": args.fna_num_sample,
        "resample_every_layer": args.fna_resample_every_layer,
        "sampling_strategy": args.fna_sampling_strategy,
    }

    base_model = LlavaNextForConditionalGenerationFNA.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None if args.device_map in {None, "none"} else args.device_map,
        fna_config=fna_config,
        fna_cache={},
    )

    if args.checkpoint_path is not None:
        logging.info("Loading LoRA adapter from %s", args.checkpoint_path)
        model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    else:
        model = base_model

    model.eval()

    if args.device and (args.device_map in {None, "none"}):
        logging.info("Moving model to %s", args.device)
        model.to(args.device)

    return model, processor


def load_scienceqa_dataset(args: argparse.Namespace) -> Dataset:
    df = load_dataset(
        args.scienceqa_hf_dataset,
        split=args.scienceqa_split,
        cache_dir=str(args.scienceqa_cache_dir) if args.scienceqa_cache_dir else None,
    )

    if args.limit is not None:
        df = df.select(range(min(args.limit, len(df))))

    return df


def prepare_answer_choices(raw_choices: List[str]) -> List[str]:
    return [choice.strip() for choice in raw_choices if choice.strip()]


def prepare_prompt(
    processor: LlavaNextProcessor,
    question: str,
    hint: str,
    answer_choices: List[str],
    has_image: bool,
) -> str:
    system_prompt = """
You are taking a multiple-choice exam.

Your task is to choose the correct option and explain your reasoning.

Rules:
- Use the image if one is provided.
- Use the hint if one is provided.
- Reason step by step using the available information.
- Then provide the final correct answer choice text.

Output format:
Reasoning: <your reasoning>
Answer: <exact answer choice text>
"""

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        }
    ]

    content = []

    if has_image:
        content.append({"type": "image"})

    prepared_choices = prepare_answer_choices(answer_choices)
    choices_text = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(prepared_choices)])

    content.append({"type": "text", "text": question})

    if hint:
        content.append({"type": "text", "text": f"Hint: {hint}"})

    content.append({"type": "text", "text": "Choices:\n" + choices_text})
    content.append({"type": "text", "text": "Reasoning:"})

    conversation.append({"role": "user", "content": content})

    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def hf_generate(model, inputs, tokenizer, max_new_tokens, device):
    sync_fn = make_cuda_sync_fn(device)

    sync_fn()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    sync_fn()
    total_time = time.perf_counter() - t0

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]

    return generated_ids, total_time


def generate_answer_scienceqa(
    model: torch.nn.Module,
    processor: LlavaNextProcessor,
    sample: Dict[str, object],
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
    sample_index: int,
) -> GenerationRecord:
    question_text = str(sample["question"])
    answer_choices = list(sample["choices"])
    hint = str(sample.get("hint") or "").strip()

    gt_index = sample.get("answer", None)
    if isinstance(gt_index, int) and 0 <= gt_index < len(answer_choices):
        ground_truth_answer = answer_choices[gt_index].strip()
    else:
        ground_truth_answer = ""

    question_id = str(sample.get("question_id", sample_index))

    image_obj = sample.get("image", None)
    has_image = image_obj is not None

    prompt = prepare_prompt(processor, question_text, hint, answer_choices, has_image)

    if has_image:
        image = image_obj.convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    else:
        inputs = processor(text=prompt, return_tensors="pt")

    inputs = move_batch_to_device(inputs, args.device, torch_dtype)

    generated_ids, total_time = hf_generate(
        model,
        inputs,
        processor.tokenizer,
        args.max_new_tokens,
        args.device,
    )

    decoded = processor.tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()

    predicted_answer = extract_predicted_choice_text(decoded, answer_choices)

    return GenerationRecord(
        question_id=question_id,
        question=question_text,
        answer_choices=answer_choices,
        ground_truth_answer=ground_truth_answer,
        predicted_answer=predicted_answer,
        full_generation=decoded,
        generation_latency_s=total_time,
        prefill_time_s=None,
        decode_time_s=None,
    )


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=level,
    )


def maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_record(path: Path, record: GenerationRecord) -> None:
    with path.open("a") as fp:
        fp.write(json.dumps(record.to_json()) + "\n")


def dump_metrics(metrics_path: Path, metrics: ScienceQAMetrics) -> None:
    with metrics_path.open("w") as fp:
        json.dump(metrics.to_json(), fp, indent=2)


def read_existing_predictions(path: Path) -> List[GenerationRecord]:
    if not path.exists():
        return []

    records: List[GenerationRecord] = []

    with path.open("r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(GenerationRecord(**payload))

    logging.info("Resuming from %d existing predictions", len(records))
    return records


def run_scienceqa_eval(args: argparse.Namespace) -> None:
    dtype = dtype_from_string(args.dtype)
    set_random_seed(args.seed)
    configure_logging(args.verbose)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    predictions_path = out_dir / "predictions.jsonl"

    model, processor = load_model_and_processor(args, dtype)
    CLIP_LAYER = args.CLIP_feature_layer
    if hasattr(model.config, "vision_feature_layer"):
        model.config.vision_feature_layer = CLIP_LAYER
        logging.info("Set CLIP feature layer to %d", CLIP_LAYER)
    if hasattr(model.config, "mm_vision_select_layer"):
        model.config.mm_vision_select_layer = CLIP_LAYER
        logging.info("Set CLIP feature layer to %d", CLIP_LAYER)
    df = load_scienceqa_dataset(args)
    
    df = load_scienceqa_dataset(args)

    predictions: List[GenerationRecord] = read_existing_predictions(predictions_path)
    known_ids = {row.question_id for row in predictions}

    progress = tqdm(enumerate(df), total=len(df), desc="Evaluating ScienceQA", unit="sample")

    for i, row in progress:
        idx = str(row.get("question_id", i))

        if idx in known_ids:
            continue

        record = generate_answer_scienceqa(
            model=model,
            processor=processor,
            sample=row,
            args=args,
            torch_dtype=dtype,
            sample_index=i,
        )

        predictions.append(record)
        write_record(predictions_path, record)
        known_ids.add(idx)
        maybe_empty_cuda_cache()

    latencies = [rec.generation_latency_s for rec in predictions if rec.generation_latency_s is not None]
    avg_latency = float(sum(latencies) / len(latencies)) if latencies else None
    median_latency = float(statistics.median(latencies)) if latencies else None

    prefill_times = [rec.prefill_time_s for rec in predictions if rec.prefill_time_s is not None]
    decode_times = [rec.decode_time_s for rec in predictions if rec.decode_time_s is not None]

    avg_prefill_time = float(sum(prefill_times) / len(prefill_times)) if prefill_times else None
    median_prefill_time = float(statistics.median(prefill_times)) if prefill_times else None
    avg_decode_time = float(sum(decode_times) / len(decode_times)) if decode_times else None
    median_decode_time = float(statistics.median(decode_times)) if decode_times else None

    num_correct = sum(
        1
        for rec in predictions
        if normalize_text(rec.predicted_answer) == normalize_text(rec.ground_truth_answer)
    )

    accuracy = float(num_correct / len(predictions)) if predictions else 0.0

    metrics = ScienceQAMetrics(
        total_questions=len(predictions),
        correct_answers=int(num_correct),
        accuracy=accuracy,
        average_latency_s=avg_latency,
        median_latency_s=median_latency,
        average_prefill_time_s=avg_prefill_time,
        median_prefill_time_s=median_prefill_time,
        average_decode_time_s=avg_decode_time,
        median_decode_time_s=median_decode_time,
    )

    dump_metrics(metrics_path, metrics)
    logging.info("Evaluation complete: %s", json.dumps(metrics.to_json(), indent=2))


def main() -> None:
    args = parse_args()
    run_scienceqa_eval(args)

if __name__ == "__main__":
    main()