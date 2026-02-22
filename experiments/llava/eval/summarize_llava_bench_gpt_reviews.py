"""
Summarize GPT review JSONL from eval_gpt_review_bench.py (LLaVA-Bench style).

Input format (one JSON per line), example:
  {
    "question_id": 0,
    "category": "llava_bench_conv",
    "tuple": [7.0, 6.0],
    "review_model": "gpt-4o-mini",
    ...
  }

This script reports:
  - mean score for Assistant 1 and Assistant 2
  - relative score (%) = mean(A2) / mean(A1) * 100
  - category breakdown (conv/detail/complex if present)

Usage:
  python summarize_llava_bench_gpt_reviews.py -f /path/to/reviews.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def safe_mean(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def rel_percent(a2: float, a1: float) -> float:
    if not (math.isfinite(a1) and a1 > 0):
        return float("nan")
    return 100.0 * (a2 / a1)


def normalize_category(cat: str) -> str:
    """
    Expect categories like:
      - llava_bench_conv / llava_bench_detail / llava_bench_complex
      - conv / detail / complex
    Returns one of: conv, detail, complex, other
    """
    if not cat:
        return "other"
    c = cat.strip().lower()
    if "conv" in c:
        return "conv"
    if "detail" in c:
        return "detail"
    if "complex" in c:
        return "complex"
    return "other"


def extract_scores(row: Dict[str, Any]) -> Tuple[float, float]:
    """
    Prefer 'tuple': [s1, s2]. Fallback: parse first 2 numbers from 'content' first line.
    """
    t = row.get("tuple")
    if isinstance(t, list) and len(t) >= 2:
        try:
            return float(t[0]), float(t[1])
        except Exception:
            pass

    content = row.get("content", "")
    if isinstance(content, str) and content.strip():
        first = content.splitlines()[0].strip()
        # crude numeric extraction
        nums: List[float] = []
        cur = ""
        for ch in first:
            if ch.isdigit() or ch in ".-":
                cur += ch
            else:
                if cur:
                    try:
                        nums.append(float(cur))
                    except Exception:
                        pass
                    cur = ""
        if cur:
            try:
                nums.append(float(cur))
            except Exception:
                pass
        if len(nums) >= 2:
            return nums[0], nums[1]

    return float("nan"), float("nan")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns a structured summary dict:
      {
        "overall": {...},
        "by_category": {"conv": {...}, ...},
        "n_total": ...,
        "n_valid": ...,
      }
    """
    a1_all: List[float] = []
    a2_all: List[float] = []
    by_cat = defaultdict(lambda: {"a1": [], "a2": []})

    n_total = 0
    n_valid = 0

    for r in rows:
        n_total += 1
        s1, s2 = extract_scores(r)
        if not (math.isfinite(s1) and math.isfinite(s2) and s1 >= 0 and s2 >= 0):
            continue
        n_valid += 1
        a1_all.append(s1)
        a2_all.append(s2)
        cat = normalize_category(str(r.get("category", "")))
        by_cat[cat]["a1"].append(s1)
        by_cat[cat]["a2"].append(s2)

    def pack(a1: List[float], a2: List[float]) -> Dict[str, Any]:
        m1 = safe_mean(a1)
        m2 = safe_mean(a2)
        return {
            "n": len(a1),
            "mean_assistant1": m1,
            "mean_assistant2": m2,
            "relative_percent": rel_percent(m2, m1),
        }

    out = {
        "n_total": n_total,
        "n_valid": n_valid,
        "overall": pack(a1_all, a2_all),
        "by_category": {cat: pack(v["a1"], v["a2"]) for cat, v in sorted(by_cat.items())},
    }
    return out


def print_human(summary: Dict[str, Any]) -> None:
    print(f"n_total={summary['n_total']}  n_valid={summary['n_valid']}")
    print("")
    ov = summary["overall"]
    print("Overall")
    print(f"  n                 : {ov['n']}")
    print(f"  mean Assistant 1  : {ov['mean_assistant1']:.4f}")
    print(f"  mean Assistant 2  : {ov['mean_assistant2']:.4f}")
    print(f"  relative (%)      : {ov['relative_percent']:.2f}")
    print("")
    print("By category")
    for cat, s in summary["by_category"].items():
        print(f"  {cat}")
        print(f"    n                : {s['n']}")
        print(f"    mean Assistant 1 : {s['mean_assistant1']:.4f}")
        print(f"    mean Assistant 2 : {s['mean_assistant2']:.4f}")
        print(f"    relative (%)     : {s['relative_percent']:.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize GPT judge reviews (LLaVA-Bench style).")
    p.add_argument("-f", "--file", type=Path, required=True, help="reviews.jsonl produced by eval_gpt_review_bench.py")
    p.add_argument("--json-out", type=Path, default=None, help="Optional path to write summary JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.file)
    summary = summarize(rows)
    print_human(summary)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
