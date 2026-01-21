#!/usr/bin/env python3
"""Plot COCO VQA sweep results (BERT F1 vs. normalized generation throughput).

This utility expects the directory structure emitted by
`experiments/llava/scripts/run_llava_vqa_sweep.sh`:

```
experiments/llava/eval/outputs/coco_vqa/
  ├─ llava_baseline/
  │    └─ metrics.json
  ├─ llava_fna-layers_12_32-samples_64-img_toks/
  │    └─ metrics.json
  └─ ...
```

Only the `metrics.json` files are required. Each point's horizontal
position is the `normalized_avg_generation_tokens_per_s` metric
(throughput), and the vertical axis is the `bertscore_f1` metric.
Layer ranges map to colors, and sample counts map to marker sizes.

In addition to the scatter plot, the script can now derive *generation
speed slopes* by fitting a line through the throughput achieved at
different sample sizes for each FNA layer configuration. The slope of
this line approximates how quickly the model's decoding throughput grows
as more landmarks are sampled, providing an aggregated speed metric per
layer range.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

RESULT_DIR_PATTERN = re.compile(
    r"llava_fna-layers_(?P<start>\d+)_(?P<end>\d+)-samples_(?P<samples>\d+)-img_toks"
)
DEFAULT_RESULTS_ROOT = (
    Path(__file__).resolve().parents[0] / "outputs" / "coco_vqa"
)


@dataclass
class ResultPoint:
    label: str
    layer_range: Optional[Tuple[int, int]]
    sample_size: Optional[int]
    normalized_tokens_per_s: float
    bert_f1: float
    is_baseline: bool = False

    def layer_label(self) -> str:
        return _format_layer_label(self.layer_range)


def _format_layer_label(layer_range: Optional[Tuple[int, int]]) -> str:
    if layer_range is None:
        return "Baseline (FNA disabled)"
    lo, hi = layer_range
    return f"layers[{lo}:{hi}]"


class MetricsLoadError(RuntimeError):
    pass


def _load_metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise MetricsLoadError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _extract_point(dir_path: Path) -> ResultPoint:
    metrics = _load_metrics(dir_path / "metrics.json")
    tokens = metrics.get("normalized_avg_generation_tokens_per_s") or metrics.get(
        "avg_generation_tokens_per_s"
    )
    if tokens is None:
        raise MetricsLoadError(
            f"No generation throughput metric found in {dir_path / 'metrics.json'}"
        )
    bert_f1 = metrics.get("bertscore_f1")
    if bert_f1 is None:
        raise MetricsLoadError(
            f"Missing `bertscore_f1` metric in {dir_path / 'metrics.json'}"
        )

    if dir_path.name == "llava_baseline":
        return ResultPoint(
            label="llava_baseline",
            layer_range=None,
            sample_size=None,
            normalized_tokens_per_s=float(tokens),
            bert_f1=float(bert_f1),
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
        normalized_tokens_per_s=float(tokens),
        bert_f1=float(bert_f1),
    )


def gather_points(results_root: Path) -> List[ResultPoint]:
    points: List[ResultPoint] = []
    if not results_root.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_root}. "
            "Have you run run_llava_vqa_sweep.sh?"
        )

    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            points.append(_extract_point(child))
        except MetricsLoadError as exc:
            print(f"[warning] {exc}")
            continue
    if not points:
        raise RuntimeError(
            f"No valid metrics found under {results_root}. "
            "Ensure the sweep outputs include metrics.json files."
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

    unique_layers = [
        layer
        for layer in dict.fromkeys(layer_ranges)
        if layer is not None
    ]
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
) -> None:
    if layer_color_map is None:
        layer_color_map = _build_layer_color_map([p.layer_range for p in points])
    sample_size_map = _build_sample_size_map(points)

    fig, ax = plt.subplots(figsize=(10, 5))
    for point in points:
        color = layer_color_map[point.layer_range]
        size = sample_size_map[point.sample_size]
        ax.scatter(
            point.normalized_tokens_per_s,
            point.bert_f1,
            s=size,
            c=[color],
            alpha=1.0 if point.is_baseline else 0.7,
            edgecolors="white",
            linewidths=0.5,
            zorder=5 if point.is_baseline else 3,
        )

    ax.set_xlabel("Generation Tokens / s", fontsize=12)
    ax.set_ylabel("BERTScore F1", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(title, fontsize=14)

    # Legend for layer ranges (colors)
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

    # Legend for sample sizes (marker sizes)
    sample_handles: List[Line2D] = []
    for sample, size in sorted(sample_size_map.items(), key=lambda kv: (kv[0] is None, kv[0])):
        if sample is None:
            continue
        markersize = np.sqrt(size) / 2.0
        sample_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=markersize,
                markerfacecolor="#777777",
                markeredgecolor="white",
                label=f"{sample} samples",
            )
        )

    legend1 = ax.legend(
        handles=layer_handles,
        #title="Layer Range",
        fontsize=12,
        title_fontsize=12,
        loc="lower left",
    )
    ax.add_artist(legend1)
    # if sample_handles:
    #     ax.legend(
    #         handles=sample_handles,
    #         title="Sample Size",
    #         fontsize=9,
    #         title_fontsize=10,
    #         loc="upper left",
    #     )

    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()

    # Add a small buffer above the max y datapoint so the highest point isn't
    # flush against the top of the plot. Use 10% of the y-range or a minimum of
    # 0.01 absolute units.
    try:
        ys = [p.bert_f1 for p in points]
        if ys:
            y_min = float(min(ys))
            y_max = float(max(ys))
            y_range = max(1e-6, y_max - y_min)
            pad = max(0.01, 0.10 * y_range)
            ax.set_ylim(bottom=max(0.0, y_min - 0.5 * pad), top=min(1.0, y_max + pad))
    except Exception:
        # If anything goes wrong computing limits, skip and keep defaults.
        pass

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _group_points_by_layer(points: Sequence[ResultPoint]) -> Dict[Optional[Tuple[int, int]], List[ResultPoint]]:
    grouped: Dict[Optional[Tuple[int, int]], List[ResultPoint]] = {}
    for point in points:
        grouped.setdefault(point.layer_range, []).append(point)
    return grouped


def compute_generation_speed_trends(
    points: Sequence[ResultPoint],
    min_points: int = 2,
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Return slope/intercept stats for each layer config with >= min_points."""

    trends: Dict[Tuple[int, int], Dict[str, object]] = {}
    grouped = _group_points_by_layer(points)
    for layer_range, layer_points in grouped.items():
        if layer_range is None:
            continue
        sampled = [p for p in layer_points if p.sample_size is not None]
        if len(sampled) < min_points:
            continue
        sampled.sort(key=lambda p: p.sample_size)
        xs = np.array([p.sample_size for p in sampled], dtype=float)
        ys = np.array([p.normalized_tokens_per_s for p in sampled], dtype=float)
        if np.ptp(xs) == 0:
            # Avoid singular fit when all sample sizes are identical.
            continue
        slope, intercept = np.polyfit(xs, ys, 1)
        trends[layer_range] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "num_points": len(sampled),
            "sample_sizes": [int(p.sample_size) for p in sampled],
            "tokens_per_s": [float(p.normalized_tokens_per_s) for p in sampled],
        }
    return trends


def dump_generation_speed_trends(
    trends: Dict[Tuple[int, int], Dict[str, object]],
    output_path: Path,
) -> None:
    serializable = {
        _format_layer_label(layer_range): stats for layer_range, stats in trends.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(serializable, fp, indent=2)
    print(f"Saved generation speed slopes to {output_path}")


def plot_generation_speed_trends(
    trends: Dict[Tuple[int, int], Dict[str, object]],
    layer_color_map: Dict[Optional[Tuple[int, int]], Tuple[float, float, float]],
    output_path: Optional[Path],
    title: str,
    show: bool,
) -> None:
    if not trends:
        print("[info] No layer configurations have enough sample sizes to fit speed slopes.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for layer_range, stats in sorted(trends.items(), key=lambda item: item[0][0]):
        xs = stats["sample_sizes"]
        ys = stats["tokens_per_s"]
        slope = stats["slope"]
        color = layer_color_map.get(layer_range, (0.25, 0.25, 0.25))
        label = f"{_format_layer_label(layer_range)} (slope={slope:.3f})"
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=color, label=label)

    ax.set_xlabel("Sample Size (num_sample)", fontsize=12)
    ax.set_ylabel("Generation Tokens / s", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved generation speed trend plot to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_slope_estimated_points(
    points: Sequence[ResultPoint],
    trends: Dict[Tuple[int, int], Dict[str, object]],
) -> List[ResultPoint]:
    derived: List[ResultPoint] = []
    for point in points:
        if point.layer_range is None:
            # Baseline: keep actual throughput
            derived.append(point)
            continue
        stats = trends.get(point.layer_range)
        if stats is None or point.sample_size is None:
            # Without slope info, skip entirely to avoid misleading data
            continue
        slope = stats["slope"]
        intercept = stats["intercept"]
        est_tokens = slope * float(point.sample_size) + intercept
        derived.append(
            ResultPoint(
                label=point.label,
                layer_range=point.layer_range,
                sample_size=point.sample_size,
                normalized_tokens_per_s=float(est_tokens),
                bert_f1=point.bert_f1,
                is_baseline=point.is_baseline,
            )
        )
    return derived


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot COCO VQA sweep results (BERT F1 vs normalized generation tokens)."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing sweep output folders (default: %(default)s)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="File path to save the plot (PNG/PDF). When omitted, the plot is only shown.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="COCO VQA: BERTScore vs Generation Throughput",
        help="Matplotlib title for the figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window in addition to saving.",
    )
    parser.add_argument(
        "--speed-lines-output",
        type=Path,
        default=None,
        help=
        "Optional path to save the generation speed vs sample size line plot (one line per layer range).",
    )
    parser.add_argument(
        "--speed-lines-title",
        type=str,
        default="Generation Speed vs Sample Size",
        help="Title for the generation speed trend plot.",
    )
    parser.add_argument(
        "--show-speed-lines",
        action="store_true",
        help="Display the generation speed trend plot window.",
    )
    parser.add_argument(
        "--speed-slopes-json",
        type=Path,
        default=None,
        help="Optional path to dump the fitted slope metrics as JSON.",
    )
    parser.add_argument(
        "--slope-scatter-output",
        type=Path,
        default=None,
        help=
        "Optional path to save a BERT F1 vs throughput plot that uses slope-estimated throughput (one point per sweep run).",
    )
    parser.add_argument(
        "--show-slope-scatter",
        action="store_true",
        help="Display the slope-estimated scatter figure window.",
    )
    parser.add_argument(
        "--slope-scatter-title",
        type=str,
        default="COCO VQA: BERTScore vs Slope-Estimated Throughput",
        help="Title for the slope-estimated scatter plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = gather_points(args.results_root)
    # Sort for deterministic plotting order (baseline first, then by layer and sample size)
    points.sort(
        key=lambda p: (
            0 if p.is_baseline else 1,
            p.layer_range[0] if p.layer_range else -1,
            p.sample_size or -1,
        )
    )
    layer_color_map = _build_layer_color_map([p.layer_range for p in points])
    plot_points(points, args.output_path, args.title, args.show, layer_color_map)

    trends = compute_generation_speed_trends(points)
    if args.speed_slopes_json and trends:
        dump_generation_speed_trends(trends, args.speed_slopes_json)
    elif args.speed_slopes_json:
        print("[info] Skipping slope JSON export; insufficient data across sample sizes.")

    if trends and (args.speed_lines_output or args.show_speed_lines):
        plot_generation_speed_trends(
            trends,
            layer_color_map,
            args.speed_lines_output,
            args.speed_lines_title,
            args.show_speed_lines,
        )
    elif args.speed_lines_output or args.show_speed_lines:
        print(
            "[info] Cannot render generation speed trend plot because no layer range has at least two sample sizes."
        )

    if trends and (args.slope_scatter_output or args.show_slope_scatter):
        slope_points = build_slope_estimated_points(points, trends)
        if slope_points:
            plot_points(
                slope_points,
                args.slope_scatter_output,
                args.slope_scatter_title,
                args.show_slope_scatter,
                layer_color_map,
            )
        else:
            print(
                "[info] Skipping slope-based scatter plot: no points had slope estimates (need at least two sample sizes per layer)."
            )
    elif args.slope_scatter_output or args.show_slope_scatter:
        print(
            "[info] Cannot render slope-based scatter plot because no layer range has at least two sample sizes."
        )


if __name__ == "__main__":
    main()
