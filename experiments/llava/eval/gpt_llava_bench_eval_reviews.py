"""
GPT-based evaluation for LLaVA-Bench (In-the-Wild) style outputs.

This script compares two answer files (Assistant 1 vs Assistant 2) question-by-question,
and asks an OpenAI evaluator model to score both answers 1-10.

Expected inputs:
  - questions.jsonl: each line like:
      {"image":"001.jpg","text":"...","category":"conv","question_id":0}
  - answers_gpt4.jsonl: reference answers, each line like:
      {"question_id":0,"text":"...","answer_id":"...","model_id":"gpt-4"}
  - answers.jsonl: your model answers, same schema (must include question_id, text)

Optional but recommended (classic LLaVA-Bench protocol):
  - context.jsonl: each line like:
      {"image":"001.jpg","caption":[...]} or {"image":"001.jpg","caption":"..."}

Outputs:
  - reviews.jsonl: each line includes the judge response + parsed (score1, score2)

Default evaluator model: gpt-4o-mini (cost-efficient). Uses the Responses API by default.

Example (text-only classic):
  export OPENAI_API_KEY="..."
  python experiments/llava/eval/gpt_llava_bench_eval_reviews.py \
    --question /path/to/questions.jsonl \
    --context  /path/to/context.jsonl \
    --answer-list /path/to/answers_gpt4.jsonl /path/to/answers.jsonl \
    --output /path/to/reviews.jsonl \
    --eval-model gpt-4o-mini

Example (image-based judging, no context captions):
  python experiments/llava/eval/gpt_llava_bench_eval_reviews.py \
    --question /path/to/questions.jsonl \
    --answer-list /path/to/answers_gpt4.jsonl /path/to/answers.jsonl \
    --output /path/to/reviews.jsonl \
    --eval-model gpt-4o-mini \
    --use-images --images-root /path/to/images
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Embedded LLaVA-Bench rubrics (equivalent to llava_bench_{conv,detail,complex})
# -----------------------------------------------------------------------------

ROLE = "Assistant"

LLAVA_BENCH_RUBRIC = {
    "conv": (
        "We would like to request your feedback on the performance of two AI assistants in response "
        "to the user question displayed above. The user asks the question on observing an image. "
        "For your reference, the visual content in the image is represented with a few sentences "
        "describing the image.\n"
        "Please rate the helpfulness, relevance, accuracy, level of details of their responses.\n"
        "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates "
        "better overall performance.\n"
        "Please first output a single line containing only two values indicating the scores for "
        f"{ROLE} 1 and 2, respectively. The two scores are separated by a space.\n"
        "In the subsequent line, please provide a comprehensive explanation of your evaluation, "
        "avoiding any potential bias and ensuring that the order in which the responses were presented "
        "does not affect your judgment."
    ),
    "detail": (
        "We would like to request your feedback on the performance of two AI assistants in response "
        "to the user question displayed above. The user asks the question on observing an image. "
        "For your reference, the visual content in the image is represented with a few sentences "
        "describing the image.\n"
        "Please rate the helpfulness, relevance, accuracy, level of details of their responses.\n"
        "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates "
        "better overall performance.\n"
        "Please first output a single line containing only two values indicating the scores for "
        f"{ROLE} 1 and 2, respectively. The two scores are separated by a space.\n"
        "In the subsequent line, please provide a comprehensive explanation of your evaluation, "
        "avoiding any potential bias and ensuring that the order in which the responses were presented "
        "does not affect your judgment."
    ),
    "complex": (
        "We would like to request your feedback on the performance of two AI assistants in response "
        "to the user question displayed above. The user asks the question on observing an image. "
        "For your reference, the visual content in the image is represented with a few sentences "
        "describing the image.\n"
        "Please rate the helpfulness, relevance, accuracy, level of details of their responses.\n"
        "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates "
        "better overall performance.\n"
        "Please first output a single line containing only two values indicating the scores for "
        f"{ROLE} 1 and 2, respectively. The two scores are separated by a space.\n"
        "In the subsequent line, please provide a comprehensive explanation of your evaluation, "
        "avoiding any potential bias and ensuring that the order in which the responses were presented "
        "does not affect your judgment."
    ),
}

DEFAULT_RUBRIC = (
    "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\n"
    "Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, "
    "where a higher score indicates better overall performance.\n"
    f"Please first output a single line containing only two values indicating the scores for {ROLE} 1 and 2, respectively. "
    "The two scores are separated by a space.\n"
    "In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that "
    "the order in which the responses were presented does not affect your judgment."
)

SYSTEM_INSTRUCTIONS = "You are a helpful and precise assistant for checking the quality of the answer."


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_questions(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def load_answers_by_qid(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = read_jsonl(path)
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        if "question_id" not in r:
            continue
        try:
            qid = int(r["question_id"])
        except Exception:
            continue
        out[qid] = r
    return out


def load_context_by_image(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    out: Dict[str, str] = {}
    for r in rows:
        img = r.get("image")
        cap = r.get("caption")
        if not isinstance(img, str):
            continue
        if isinstance(cap, list):
            cap_str = "\n".join(str(x) for x in cap)
        elif isinstance(cap, str):
            cap_str = cap
        else:
            cap_str = ""
        out[img] = cap_str
    return out


# -----------------------------------------------------------------------------
# OpenAI call helpers (Responses API preferred; fallback to chat.completions)
# -----------------------------------------------------------------------------

def _init_openai_client():
    try:
        # new SDK
        from openai import OpenAI  # type: ignore
        return OpenAI()
    except Exception:
        return None


def _responses_create_text(client, model: str, content: str, max_output_tokens: int, temperature: float) -> str:
    # New SDK (Responses API)
    resp = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=content,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    return getattr(resp, "output_text", None) or resp.output_text  # type: ignore[attr-defined]


def _responses_create_with_image(
    client,
    model: str,
    content_text: str,
    image_data_url: str,
    max_output_tokens: int,
    temperature: float,
    image_detail: str,
) -> str:
    # New SDK (Responses API), multimodal message
    resp = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": content_text},
                    {"type": "input_image", "image_url": image_data_url, "detail": image_detail},
                ],
            }
        ],
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    return getattr(resp, "output_text", None) or resp.output_text  # type: ignore[attr-defined]


def _chat_completions_fallback(model: str, content: str, max_tokens: int, temperature: float) -> str:
    # Fallback: openai<1.0 style or openai>=1.0 chat.completions style
    import openai  # type: ignore

    # Try new style first
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    # Old style
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp["choices"][0]["message"]["content"]


def get_eval(
    client,
    model: str,
    content: str,
    max_output_tokens: int,
    temperature: float,
    sleep_s: float,
    max_retries: int,
    use_images: bool,
    image_data_url: Optional[str],
    image_detail: str,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            if client is not None:
                if use_images:
                    if not image_data_url:
                        raise ValueError("use_images=True but image_data_url is None")
                    return _responses_create_with_image(
                        client=client,
                        model=model,
                        content_text=content,
                        image_data_url=image_data_url,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        image_detail=image_detail,
                    )
                return _responses_create_text(
                    client=client,
                    model=model,
                    content=content,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
            # fallback path
            return _chat_completions_fallback(model=model, content=content, max_tokens=max_output_tokens, temperature=temperature)
        except Exception as e:
            last_err = e
            # jittered exponential-ish backoff
            time.sleep(sleep_s * (1.0 + 0.2 * random.random()) * min(10, (attempt + 1)))
            continue
    raise RuntimeError(f"OpenAI call failed after {max_retries} retries. Last error: {last_err}") from last_err


# -----------------------------------------------------------------------------
# Scoring parse + prompt assembly
# -----------------------------------------------------------------------------

_SCORE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def parse_score(review: str) -> Tuple[float, float]:
    """
    The judge is instructed to output: "<s1> <s2>" on the first line.
    We parse the first two numbers found on the first line.
    """
    try:
        first_line = review.splitlines()[0].strip()
        nums = _SCORE_RE.findall(first_line.replace(",", " "))
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    except Exception:
        pass
    return -1.0, -1.0


def build_eval_content(
    question: Dict[str, Any],
    ans1_text: str,
    ans2_text: str,
    cap_str: str,
) -> Tuple[str, str]:
    """
    Returns: (category_key, content_string)
    """
    q_text = str(question.get("text", ""))
    img = str(question.get("image", ""))
    cat = str(question.get("category", "")).strip().lower()

    rubric = LLAVA_BENCH_RUBRIC.get(cat, DEFAULT_RUBRIC)

    parts: List[str] = []
    if cap_str:
        parts.append(f"[Context]\n{cap_str}\n")
    else:
        # Still include a Context section for consistency; keeps judge prompt structure stable
        parts.append("[Context]\n\n")

    parts.append(f"[Question]\n{q_text}\n")
    parts.append(f"[{ROLE} 1]\n{ans1_text}\n\n[End of {ROLE} 1]\n")
    parts.append(f"[{ROLE} 2]\n{ans2_text}\n\n[End of {ROLE} 2]\n")
    parts.append(f"[System]\n{rubric}\n")

    content = "\n".join(parts)
    category_key = f"llava_bench_{cat}" if cat else "llava_bench_unknown"
    return category_key, content


def encode_image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix not in {"jpg", "jpeg", "png", "webp"}:
        # default to jpeg
        mime = "image/jpeg"
    else:
        mime = f"image/{'jpeg' if suffix in {'jpg','jpeg'} else suffix}"
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-based evaluation for LLaVA-Bench style QA.")
    p.add_argument("-q", "--question", type=Path, required=True, help="questions.jsonl")
    p.add_argument("-c", "--context", type=Path, default=None, help="context.jsonl (optional but recommended)")
    p.add_argument("-a", "--answer-list", type=Path, nargs=2, required=True, help="Two answer jsonl files: ans1 ans2")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output reviews jsonl")

    p.add_argument("--eval-model", default="gpt-4o-mini", help="Evaluator model id")
    p.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens for evaluator")
    p.add_argument("--temperature", type=float, default=0.2, help="Evaluator temperature")
    p.add_argument("--sleep", type=float, default=0.5, help="Base sleep seconds between retries")
    p.add_argument("--max-retries", type=int, default=50, help="Max retries per judge call")

    # Optional image-based judging (instead of, or in addition to, context captions)
    p.add_argument("--use-images", action="store_true", help="Send the image to the evaluator model")
    p.add_argument("--images-root", type=Path, default=None, help="Folder containing images referenced by questions.jsonl")
    p.add_argument("--image-detail", default="low", choices=["low", "auto", "high"], help="Image detail level for evaluator")

    # Resume behavior
    p.add_argument("--skip-existing", action="store_true", help="Skip question_ids already present in output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    questions = load_questions(args.question)
    ans1_by_qid = load_answers_by_qid(args.answer_list[0])
    ans2_by_qid = load_answers_by_qid(args.answer_list[1])
    context_by_img = load_context_by_image(args.context)

    print("[info] starting LLaVA-Bench evaluation")
    print(f"[info] questions={len(questions)} ans1={len(ans1_by_qid)} ans2={len(ans2_by_qid)}")
    print(f"[info] eval_model={args.eval_model} max_tokens={args.max_tokens} temperature={args.temperature}")
    print(f"[info] use_images={bool(args.use_images)} image_detail={args.image_detail}")
    if args.use_images:
        print(f"[info] images_root={args.images_root}")
    print(f"[info] output={args.output} skip_existing={args.skip_existing}")

    # Resume: collect already reviewed qids
    reviewed_qids = set()
    if args.output.exists() and args.skip_existing:
        for r in read_jsonl(args.output):
            if "question_id" in r:
                try:
                    reviewed_qids.add(int(r["question_id"]))
                except Exception:
                    pass
        print(f"[info] resuming: {len(reviewed_qids)} already reviewed")

    # Init OpenAI client (Responses API)
    client = _init_openai_client()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.output.open("a", encoding="utf-8")

    idx = 0
    total = len(questions)
    start_time = time.time()
    for q in questions:
        idx += 1
        try:
            qid = int(q["question_id"])
        except Exception:
            continue

        print(f"[progress] {idx}/{total} qid={qid}")

        if args.skip_existing and qid in reviewed_qids:
            print(f"[skip] question_id={qid} already in {args.output}")
            continue

        ans1 = ans1_by_qid.get(qid)
        ans2 = ans2_by_qid.get(qid)
        if ans1 is None or ans2 is None:
            print(f"[warn] missing answers for question_id={qid} (ans1={ans1 is not None}, ans2={ans2 is not None})")
            continue

        ans1_text = str(ans1.get("text", "")).strip()
        ans2_text = str(ans2.get("text", "")).strip()

        img_name = str(q.get("image", ""))
        cap_str = context_by_img.get(img_name, "")
        if cap_str:
            print(f"[info] context chars={len(cap_str)} image={img_name}")
        else:
            print(f"[info] no context image={img_name}")

        category_key, content = build_eval_content(q, ans1_text, ans2_text, cap_str)

        image_data_url: Optional[str] = None
        if args.use_images:
            if args.images_root is None:
                raise ValueError("--use-images requires --images-root")
            image_path = args.images_root / img_name
            if not image_path.exists():
                # common layout: images_root/images/xxx.jpg
                alt = args.images_root / "images" / img_name
                if alt.exists():
                    image_path = alt
                else:
                    raise FileNotFoundError(f"Could not find image: {image_path} (or {alt})")
            print(f"[info] using image: {image_path}")
            image_data_url = encode_image_to_data_url(image_path)

        eval_start = time.time()
        print(f"[eval] requesting judge (qid={qid}, cat={category_key})")

        review = get_eval(
            client=client,
            model=args.eval_model,
            content=content,
            max_output_tokens=args.max_tokens,
            temperature=args.temperature,
            sleep_s=args.sleep,
            max_retries=args.max_retries,
            use_images=bool(args.use_images),
            image_data_url=image_data_url,
            image_detail=args.image_detail,
        )

        eval_elapsed = time.time() - eval_start

        s1, s2 = parse_score(review)

        cur_js: Dict[str, Any] = {
            "id": idx,
            "question_id": qid,
            "answer1_id": ans1.get("answer_id", ans1.get("question_id", qid)),
            "answer2_id": ans2.get("answer_id", ans2.get("question_id", qid)),
            "category": category_key,
            "review_model": args.eval_model,
            "content": review,
            "tuple": [s1, s2],
        }

        out_f.write(json.dumps(cur_js) + "\n")
        out_f.flush()

        total_elapsed = time.time() - start_time
        avg_per = total_elapsed / max(1, idx)
        remaining = max(0, total - idx)
        eta_s = int(avg_per * remaining)

        print(
            f"[{idx:04d}] qid={qid} cat={category_key} scores=({s1},{s2}) "
            f"eval={eval_elapsed:.2f}s avg={avg_per:.2f}s eta~{eta_s}s"
        )

    out_f.close()
    print("[info] done")


if __name__ == "__main__":
    main()
