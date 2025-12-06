"""Utility to obtain ChatGPT answers for VQA-style questions.

Example usage
-------------

    python -m experiments.llava.chatgpt_vqa \
    --questions-file /path/to/v2_OpenEnded_mscoco_val2014_questions_shortened_100.json \
    --output-json outputs/chatgpt_answers.json \
    --image-root /datasets/coco/val2014 \
    --send-images \
    --model gpt-4o-mini \
    --limit 128 --resume

The script expects the OpenAI API key to be provided via the
``OPENAI_API_KEY`` environment variable and stores the responses as a JSON
list (one entry per question). It supports COCO-style ``questions.json``
files, plain lists, or JSONL files that contain at least ``question_id`` and
``question``/``text`` fields. When ``--send-images`` is enabled, the script reads
the referenced image files, converts them to base64 data URLs, and attaches them
to each ChatGPT request.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

OPENAI_ERROR_CLASSES: Tuple[Type[BaseException], ...]

try:  # pragma: no cover - optional dependency may be missing until runtime
    from openai import APIConnectionError, APIError, OpenAI, RateLimitError
    from openai import BadRequestError, Timeout

    def _sanitize_openai_errors(*candidates: object) -> Tuple[Type[BaseException], ...]:
        valid: List[Type[BaseException]] = []
        for candidate in candidates:
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                valid.append(candidate)
        if not valid:
            valid.append(Exception)
        return tuple(valid)

    OPENAI_ERROR_CLASSES = _sanitize_openai_errors(
        RateLimitError,
        APIError,
        APIConnectionError,
        Timeout,
        BadRequestError,
    )
except Exception as exc:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    OPENAI_ERROR_CLASSES = (Exception,)
    _OPENAI_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _OPENAI_IMPORT_ERROR = None

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("Please install 'tqdm' to use this script (pip install tqdm).") from exc

LOGGER = logging.getLogger("chatgpt_vqa")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for visual question answering."
)
DEFAULT_PROMPT_TEMPLATE = (
    "Question ID: $question_id\n"
    "Image ID: $image_id\n"
    "Question: $question\n"
)


@dataclass
class VqaQuestion:
    """Normalized VQA question entry."""

    question_id: int
    question: str
    image_id: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], fallback_id: Optional[int] = None) -> "VqaQuestion":
        question_text = payload.get("question") or payload.get("text")
        if not question_text:
            raise ValueError("Question payload is missing a 'question' or 'text' field")
        question_id = payload.get("question_id") or payload.get("id")
        if question_id is None:
            if fallback_id is None:
                raise ValueError("No question_id present and no fallback supplied")
            question_id = fallback_id
        image_id = payload.get("image_id") or payload.get("image") or payload.get("imageId")
        raw_copy = dict(payload)
        return cls(question_id=int(question_id), question=str(question_text).strip(), image_id=image_id, raw=raw_copy)


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query OpenAI ChatGPT for VQA answers and store them as JSON.")
    parser.add_argument("--questions-file", type=Path, required=True, help="Path to questions JSON/JSONL file.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to store the ChatGPT responses (JSON list).")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model to use (default: gpt-4o-mini).")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Custom system prompt for ChatGPT.")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Template for the user prompt. Use $question, $question_id, $image_id placeholders.",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for ChatGPT (default: 0.2).")
    parser.add_argument("--max-output-tokens", type=int, default=256, help="Max tokens to generate per question (default: 256).")
    parser.add_argument("--limit", type=int, help="Optional cap on how many questions to send.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSON instead of overwriting.")
    parser.add_argument(
        "--send-images",
        action="store_true",
        help="Attach image pixels to each ChatGPT request (requires --image-root).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Directory containing the VQA images. Joined with per-question filenames.",
    )
    parser.add_argument(
        "--image-field",
        default=None,
        help="Question JSON key that already stores the image filename (relative to --image-root).",
    )
    parser.add_argument(
        "--image-template",
        default="COCO_val2014_{image_id:012d}.jpg",
        help="Python format string used to derive the image filename when --image-field is absent.",
    )
    parser.add_argument(
        "--image-mime",
        default=None,
        help="Override MIME type for encoded images (guessed from file extension by default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and emit placeholder answers. Useful for smoke testing.",
    )
    parser.add_argument("--max-retries", type=int, default=6, help="Maximum API retry attempts per question (default: 6).")
    parser.add_argument(
        "--initial-retry-wait",
        type=float,
        default=1.0,
        help="Base wait seconds between retries (exponential backoff).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=90.0,
        help="Per-request timeout in seconds passed to the OpenAI client.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity (can repeat).")
    return parser.parse_args(argv)


def load_questions(path: Path) -> List[VqaQuestion]:
    if not path.exists():
        raise FileNotFoundError(path)
    payloads: List[Dict[str, Any]] = []
    suffix = path.suffix.lower()
    LOGGER.info("Loading questions from %s", path)
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fp:
            for idx, line in enumerate(fp):
                line = line.strip()
                if not line:
                    continue
                payloads.append(json.loads(line))
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict) and "questions" in data:
            payloads.extend(list(data["questions"]))
        elif isinstance(data, list):
            payloads.extend(data)
        else:
            raise ValueError(
                "Unsupported JSON structure. Provide either a list or a dict containing a 'questions' list."
            )
    else:
        raise ValueError(f"Unsupported file extension '{suffix}'. Use .json or .jsonl")

    questions: List[VqaQuestion] = []
    for idx, payload in enumerate(payloads):
        try:
            questions.append(VqaQuestion.from_payload(payload, fallback_id=idx))
        except Exception as exc:
            raise ValueError(f"Failed to parse question at index {idx}: {exc}") from exc
    if not questions:
        raise ValueError(f"No questions were parsed from {path}")
    return questions


def ensure_api_key_present() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it before running the script: 'export OPENAI_API_KEY=sk-...'"
        )


def create_openai_client() -> Any:
    if OpenAI is None:
        raise RuntimeError(
            "Unable to import the 'openai' package. Install it via 'pip install openai'."
        ) from _OPENAI_IMPORT_ERROR
    return OpenAI()


def make_user_prompt(question: VqaQuestion, template: str) -> str:
    image_id = question.image_id if question.image_id is not None else "unknown"
    return Template(template).safe_substitute(
        question=question.question,
        question_id=question.question_id,
        image_id=image_id,
    ).strip()


def resolve_image_path(question: VqaQuestion, args: argparse.Namespace) -> Optional[Path]:
    if not args.send_images:
        return None
    if args.image_root is None:
        raise ValueError("--image-root must be provided when --send-images is enabled")

    root = args.image_root
    candidate: Optional[Path] = None
    if args.image_field:
        raw_value = question.raw.get(args.image_field)
        if raw_value:
            candidate = Path(str(raw_value))
    if candidate is None and question.image_id is not None and args.image_template:
        context: Dict[str, Any] = dict(question.raw)
        context.setdefault("image_id", question.image_id)
        context.setdefault("question_id", question.question_id)
        try:
            rendered = args.image_template.format(**context)
        except KeyError as exc:
            raise ValueError(f"Image template is missing key {exc} for question {question.question_id}") from exc
        candidate = Path(rendered)

    if candidate is None:
        raise ValueError(
            f"Unable to resolve image path for question {question.question_id}. Provide --image-field or --image-template capable of producing a path."
        )

    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def encode_image_to_data_url(image_path: Path, mime_override: Optional[str] = None) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime_type = mime_override or mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    with image_path.open("rb") as fp:
        payload = base64.b64encode(fp.read()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def build_user_content(user_prompt: str, image_data_url: Optional[str]) -> Any:
    if image_data_url:
        return [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    return user_prompt


def extract_usage(response: Any) -> Optional[Dict[str, Any]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    for attr in ("model_dump", "dict", "to_dict"):
        method = getattr(usage, attr, None)
        if callable(method):  # pragma: no branch - deterministic order
            return method()
    try:
        return json.loads(usage.json())  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - best effort fallback
        return None


def call_chatgpt(
    client: Any,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    initial_wait: float,
) -> Dict[str, Any]:
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            choice = response.choices[0]
            answer_text = (choice.message.content or "").strip()
            return {
                "answer": answer_text,
                "finish_reason": choice.finish_reason,
                "usage": extract_usage(response),
                "raw_response": response.model_dump() if hasattr(response, "model_dump") else None,
            }
        except OPENAI_ERROR_CLASSES as exc:
            if attempt == max_retries:
                raise
            wait_time = initial_wait * (2 ** (attempt - 1))
            LOGGER.warning("API call failed (%s). Retrying in %.1f s (%d/%d)...", exc.__class__.__name__, wait_time, attempt, max_retries)
            time.sleep(wait_time)
    raise RuntimeError("Exceeded maximum retry attempts")


def load_existing_results(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Existing output JSON must contain a list")
    mapping: Dict[int, Dict[str, Any]] = {}
    for entry in data:
        qid = entry.get("question_id")
        if qid is None:
            continue
        mapping[int(qid)] = entry
    return mapping


def persist_results(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if args.output_json.exists() and not args.resume:
        raise FileExistsError(
            f"{args.output_json} already exists. Pass --resume to append or delete the file manually."
        )

    questions = load_questions(args.questions_file)
    if args.limit is not None:
        questions = questions[: args.limit]
    LOGGER.info("Loaded %d questions", len(questions))

    existing_records = load_existing_results(args.output_json) if args.resume else {}
    ordered_results: List[Dict[str, Any]] = []

    if args.resume and existing_records:
        LOGGER.info("Resuming with %d pre-existing answers", len(existing_records))

    client: Optional[Any] = None
    if args.dry_run:
        LOGGER.info("Running in dry-run mode – API calls will be skipped.")
    else:
        ensure_api_key_present()
        client = create_openai_client()

    unanswered = 0
    progress = tqdm(questions, desc="ChatGPT VQA", unit="q")
    for question in progress:
        if args.resume and question.question_id in existing_records:
            ordered_results.append(existing_records[question.question_id])
            progress.set_postfix(skipped="yes")
            continue

        image_path: Optional[Path] = None
        image_data_url: Optional[str] = None
        if args.send_images:
            try:
                image_path = resolve_image_path(question, args)
                if not args.dry_run:
                    image_data_url = encode_image_to_data_url(image_path, args.image_mime)
            except Exception as exc:
                unanswered += 1
                LOGGER.error("Failed to prepare image for question %s: %s", question.question_id, exc)
                continue

        if args.dry_run:
            answer_payload = {
                "answer": f"[DRY-RUN] Placeholder answer for question {question.question_id}",
                "finish_reason": "dry_run",
                "usage": None,
                "raw_response": None,
            }
        else:
            user_prompt = make_user_prompt(question, args.prompt_template)
            user_content = build_user_content(user_prompt, image_data_url)
            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": user_content},
            ]
            if client is None:  # pragma: no cover - defensive guard
                raise RuntimeError("OpenAI client is not initialized.")
            try:
                answer_payload = call_chatgpt(
                    client=client,
                    messages=messages,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_output_tokens,
                    timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    initial_wait=args.initial_retry_wait,
                )
            except Exception as exc:
                unanswered += 1
                LOGGER.error("Failed to obtain answer for question %s: %s", question.question_id, exc)
                continue

        record = {
            "question_id": question.question_id,
            "image_id": question.image_id,
            "question": question.question,
            "model": args.model,
            "system_prompt": args.system_prompt,
            "image_path": str(image_path) if image_path else None,
            "answer": answer_payload["answer"],
            "finish_reason": answer_payload["finish_reason"],
            "usage": answer_payload["usage"],
        }
        ordered_results.append(record)
        existing_records[question.question_id] = record

        persist_results(args.output_json, ordered_results)
        progress.set_postfix(skipped="no")

    LOGGER.info("Completed. %d/%d questions lacked answers due to errors.", unanswered, len(questions))


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
