"""
This script loads a Llava-Next checkpoint and fine-tunes it on the ScienceQA dataset.
"""

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import logging
from functools import partial
import argparse
from pathlib import Path
from typing import Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple
from datasets import load_dataset, Dataset
from tqdm import tqdm
import random

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fast_nystrom_attention import LlavaNextForConditionalGenerationFNA
from transformers import LlavaNextProcessor

def load_textvqa_dataset(args: argparse.Namespace, split: str) -> Dataset:
    return load_dataset(
        args.textvqa_dataset_name,
        split=split,
        cache_dir=str(args.textvqa_cache_dir) if args.textvqa_cache_dir else None,
    )

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError("--fna-layer-range must be formatted as start:end") from exc
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
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {sorted(mapping)}")
    return mapping[name]



def save_hf_checkpoint(
    model: LlavaNextForConditionalGenerationFNA, 
    processor: LlavaNextProcessor, 
    output_dir: Path, 
    step: int) -> None:
    out_dir = output_dir / f"{checkpoint_{step}}"
    out_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = model.module if hasattr(model, "module") else model"

    unwrapped.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    logging.info("Saved checkpoint %d to %s", step, str(out_dir))


def main():
    args = parse_args()

    set_seed(args.seed)

    dtype = dtype_from_string(args.dtype)
    model, processor = load_model_and_processor(args, dtype)

    train_ds = load_textvqa_dataset(args, split="train")
    test_ds = load_textvqa_dataset(args, split="validation")


if __name__ == "__main__":
    main()