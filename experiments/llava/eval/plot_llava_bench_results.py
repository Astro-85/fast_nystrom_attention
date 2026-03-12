#!/usr/bin/env python3
"""Plot LLaVA-Bench GPT-judge results vs timing metrics.

This script expects the directory structure produced by
`experiments/llava/scripts/run_llava_bench_sweep.sh` and GPT review outputs
created by `experiments/llava/eval/gpt_llava_bench_eval_reviews.py`:

```
experiments/llava/eval/outputs_q_fpsample/llava_bench/
  ├─ llava_bench_baseline/
  │    ├─ metrics.json
  │    └─ gpt_reviews.jsonl
  ├─ llava_bench_fna-layers_12_32-samples_64-q/
  │    ├─ metrics.json
  │    └─ gpt_reviews.jsonl
  └─ ...
```

Each point uses:
    - color: FNA layer range
    - size : number of landmarks (sample size)

The plots show GPT judge score (Assistant 2) vs:
    - generation tokens/s (avg_generation_tokens_per_s)
    - total/actual tokens/s (avg_tokens_per_s)
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

RESULT_DIR_PATTERN = re.compile(
    r"llava_bench_fna-layers_(?P<start>\d+)_(?P<end>\d+)-samples_(?P<samples>\d+)(?:-(?P<features>[A-Za-z0-9_]+))?"
)
DEFAULT_RESULTS_ROOT = (
    Path(__file__).resolve().parents[0] / "outputs_q_fpsample" / "llava_bench"
)


@dataclass
class ReviewSummary:
    mean_assistant1: float
    mean_assistant2: float
    relative_percent: float
    n_valid: int


@dataclass
class ResultPoint:
    label: str
    layer_range: Optional[Tuple[int, int]]
    sample_size: Optional[int]
    avg_tokens_per_s: float
    avg_generation_tokens_per_s: float
    score_assistant2: float
    score_assistant1: float
    score_relative_percent: float
    n_reviews: int
    is_baseline: bool = False

    def layer_label(self) -> str:
        return _format_layer_label(self.layer_range)


class MetricsLoadError(RuntimeError):
    pass


class ReviewLoadError(RuntimeError):
    pass


def _format_layer_label(layer_range: Optional[Tuple[int, int]]) -> str:
    if layer_range is None:
        return "Baseline (FNA disabled)"
    lo, hi = layer_range
    return f"layers[{lo}:{hi}]"


def _load_metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise MetricsLoadError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _extract_scores(row: Dict[str, object]) -> Tuple[float, float]:
    t = row.get("tuple")
    if isinstance(t, list) and len(t) >= 2:
        try:
            return float(t[0]), float(t[1])
        except Exception:
            pass

    content = row.get("content", "")
    if isinstance(content, str) and content.strip():
        first = content.splitlines()[0].strip()
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


def _safe_mean(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _rel_percent(a2: float, a1: float) -> float:
    if not (math.isfinite(a1) and a1 > 0):
        return float("nan")
    return 100.0 * (a2 / a1)


def _load_review_summary(run_dir: Path) -> ReviewSummary:
    summary_candidates = [
        run_dir / "gpt_reviews_summary.json",
        run_dir / "gpt_review_summary.json",
    ]
    for candidate in summary_candidates:
        if candidate.exists():
            data = json.loads(candidate.read_text())
            overall = data.get("overall", {})
            return ReviewSummary(
                mean_assistant1=float(overall.get("mean_assistant1", float("nan"))),
                mean_assistant2=float(overall.get("mean_assistant2", float("nan"))),
                relative_percent=float(overall.get("relative_percent", float("nan"))),
                n_valid=int(overall.get("n", 0)),
            )

    reviews_path = run_dir / "gpt_reviews.jsonl"
    if not reviews_path.exists():
        raise ReviewLoadError(f"Missing GPT review file: {reviews_path}")
    rows = _read_jsonl(reviews_path)
    a1: List[float] = []
    a2: List[float] = []
    for r in rows:
        s1, s2 = _extract_scores(r)
        if not (math.isfinite(s1) and math.isfinite(s2) and s1 >= 0 and s2 >= 0):
            continue
        a1.append(s1)
        a2.append(s2)
    return ReviewSummary(
        mean_assistant1=_safe_mean(a1),
        mean_assistant2=_safe_mean(a2),
        relative_percent=_rel_percent(_safe_mean(a2), _safe_mean(a1)),
        n_valid=len(a2),
    )


def _extract_point(dir_path: Path, min_reviews: int) -> ResultPoint:
    metrics = _load_metrics(dir_path / "metrics.json")
    avg_tokens_per_s = metrics.get("avg_tokens_per_s")
    avg_gen_tokens_per_s = metrics.get("avg_generation_tokens_per_s")
    if avg_tokens_per_s is None:
        raise MetricsLoadError(
            f"Missing avg_tokens_per_s in {dir_path / 'metrics.json'}"
        )
    if avg_gen_tokens_per_s is None:
        raise MetricsLoadError(
            f"Missing avg_generation_tokens_per_s in {dir_path / 'metrics.json'}"
        )

    reviews = _load_review_summary(dir_path)
    if reviews.n_valid < min_reviews:
        raise ReviewLoadError(
            f"Insufficient GPT reviews in {dir_path} (n_valid={reviews.n_valid}, min={min_reviews})"
        )

    if dir_path.name == "llava_bench_baseline":
        return ResultPoint(
            label="llava_bench_baseline",
            layer_range=None,
            sample_size=None,
            avg_tokens_per_s=float(avg_tokens_per_s),
            avg_generation_tokens_per_s=float(avg_gen_tokens_per_s),
            score_assistant2=float(reviews.mean_assistant2),
            score_assistant1=float(reviews.mean_assistant1),
            score_relative_percent=float(reviews.relative_percent),
            n_reviews=reviews.n_valid,
            is_baseline=True,
        )

    match = RESULT_DIR_PATTERN.fullmatch(dir_path.name)
    if not match:
        raise MetricsLoadError(
            "Could not infer layer/sample configuration from directory "
            f"name: {dir_path.name}"
        )

    layer_start = int(match.group("start"))
    layer_end = int(match.group("end"))
    num_samples = int(match.group("samples"))

    return ResultPoint(
        label=dir_path.name,
        layer_range=(layer_start, layer_end),
        sample_size=num_samples,
        avg_tokens_per_s=float(avg_tokens_per_s),
        avg_generation_tokens_per_s=float(avg_gen_tokens_per_s),
        score_assistant2=float(reviews.mean_assistant2),
        score_assistant1=float(reviews.mean_assistant1),
        score_relative_percent=float(reviews.relative_percent),
        n_reviews=reviews.n_valid,
    )


def gather_points(results_root: Path, min_reviews: int) -> List[ResultPoint]:
    points: List[ResultPoint] = []
    if not results_root.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_root}. "
            "Have you run run_llava_bench_sweep.sh?"
        )

    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            points.append(_extract_point(child, min_reviews=min_reviews))
        except (MetricsLoadError, ReviewLoadError) as exc:
            print(f"[warning] {exc}")
            continue

    if not points:
        raise RuntimeError(
            f"No valid results found under {results_root}. "
            "Ensure each run has metrics.json and gpt_reviews.jsonl."
        )
    return points


def _build_layer_color_map(layer_ranges: Sequence[Optional[Tuple[int, int]]]) -> Dict[
    Optional[Tuple[int, int]], Tuple[float, float, float]
]:
    palette = np.array(
        [
            [127, 113, 240],
            [247, 214, 124],
            [76, 186, 182],
            [245, 154, 110],
            [217, 17, 17],
            [240, 127, 189],
            [127, 242, 107],
            [237, 92, 208],
        ],
        dtype=float,
    ) / 255.0

    unique_layers = [layer for layer in dict.fromkeys(layer_ranges) if layer is not None]
    color_map: Dict[Optional[Tuple[int, int]], Tuple[float, float, float]] = {
        None: (0.0, 0.0, 0.0)
    }
    for idx, layer in enumerate(unique_layers):
        color_map[layer] = tuple(palette[idx % len(palette)])
    return color_map


def _build_sample_size_map(points: Iterable[ResultPoint]) -> Dict[Optional[int], float]:
    sample_sizes = sorted({p.sample_size for p in points if p.sample_size})
    if not sample_sizes:
        return {None: 160.0}
    min_size = sample_sizes[0]
    base = 60.0
    size_map: Dict[Optional[int], float] = {None: 220.0}
    for sample in sample_sizes:
        relative = sample / min_size if min_size else 1.0
        size_map[sample] = base * (relative ** 0.9)
    return size_map


def plot_points(
    points: Sequence[ResultPoint],
    output_path: Optional[Path],
    title: str,
    show: bool,
    layer_color_map: Optional[Dict[Optional[Tuple[int, int]], Tuple[float, float, float]]] = None,
    x_value_fn=None,
    x_label: str = "Tokens / s",
    y_value_fn=None,
    y_label: str = "Mean GPT Score (Assistant 2)",
) -> None:
    if layer_color_map is None:
        layer_color_map = _build_layer_color_map([p.layer_range for p in points])
    sample_size_map = _build_sample_size_map(points)
    if x_value_fn is None:
        x_value_fn = lambda p: p.avg_generation_tokens_per_s
    if y_value_fn is None:
        y_value_fn = lambda p: p.score_assistant2

    fig, ax = plt.subplots(figsize=(10, 5))
    for point in points:
        color = layer_color_map[point.layer_range]
        size = sample_size_map[point.sample_size]
        ax.scatter(
            x_value_fn(point),
            y_value_fn(point),
            s=size,
            c=[color],
            alpha=1.0 if point.is_baseline else 0.7,
            edgecolors="white",
            linewidths=0.5,
            zorder=5 if point.is_baseline else 3,
        )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(title, fontsize=14)

    layer_handles: List[Line2D] = []
    for layer, color in layer_color_map.items():
        label = "Baseline (Flash Attn)" if layer is None else f"layers[{layer[0]}:{layer[1]}]"
        layer_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=color,
                markeredgecolor="white",
                label=label,
            )
        )

    legend1 = ax.legend(
        handles=layer_handles,
        fontsize=12,
        title_fontsize=12,
        loc="lower left",
    )
    ax.add_artist(legend1)

    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot LLaVA-Bench GPT scores vs tokens/s metrics (generation/total)."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing LLaVA-Bench sweep output folders (default: %(default)s)",
    )
    parser.add_argument(
        "--output-path-generation",
        type=Path,
        default=None,
        help="File path to save GPT score vs generation tokens/s plot.",
    )
    parser.add_argument(
        "--output-path-actual",
        type=Path,
        default=None,
        help="File path to save GPT score vs total tokens/s plot.",
    )
    parser.add_argument(
        "--title-generation",
        type=str,
        default="LLaVA-Bench: GPT Score vs Generation Tokens / s",
        help="Title for the generation tokens/s scatter plot.",
    )
    parser.add_argument(
        "--title-actual",
        type=str,
        default="LLaVA-Bench: GPT Score vs Total Tokens / s",
        help="Title for the total tokens/s scatter plot.",
    )
    parser.add_argument(
        "--show-generation",
        action="store_true",
        help="Display the generation tokens/s scatter plot window.",
    )
    parser.add_argument(
        "--show-actual",
        action="store_true",
        help="Display the total tokens/s scatter plot window.",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=1,
        help="Minimum valid GPT reviews required to include a run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = gather_points(args.results_root, min_reviews=args.min_reviews)

    points.sort(
        key=lambda p: (
            0 if p.is_baseline else 1,
            p.layer_range[0] if p.layer_range else -1,
            p.sample_size or -1,
        )
    )
    layer_color_map = _build_layer_color_map([p.layer_range for p in points])

    if args.output_path_generation or args.show_generation:
        plot_points(
            points,
            args.output_path_generation,
            args.title_generation,
            args.show_generation,
            layer_color_map,
            x_value_fn=lambda p: p.avg_generation_tokens_per_s,
            x_label="Generation Tokens / s",
            y_value_fn=lambda p: p.score_assistant2,
            y_label="Mean GPT Score (Assistant 2)",
        )

    if args.output_path_actual or args.show_actual:
        plot_points(
            points,
            args.output_path_actual,
            args.title_actual,
            args.show_actual,
            layer_color_map,
            x_value_fn=lambda p: p.avg_tokens_per_s,
            x_label="Total Tokens / s",
            y_value_fn=lambda p: p.score_assistant2,
            y_label="Mean GPT Score (Assistant 2)",
        )

    if not (
        args.output_path_generation
        or args.output_path_actual
        or args.show_generation
        or args.show_actual
    ):
        print("[info] No output selected. Use --output-path-generation or --show-generation, etc.")


if __name__ == "__main__":
    main()
