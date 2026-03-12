"""TextVQA evaluation entry-point for LLaVA + Fast Nyström Attention.

This script loads a LLaVA-NeXT checkpoint that has been instrumented with
Fast Nyström Attention (FNA) layers and runs inference on the TextVQA
validation split. It produces:

* ``predictions.jsonl`` – detailed per-question generations.
* ``submission.json`` – minimal list of ``{"question_id", "answer"}`` for
  use with the official VQA evaluation server if desired.
* ``metrics.json`` – locally computed VQA accuracy using the public metric.


"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import string
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple
import sys

from datasets import load_dataset, Dataset

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fast_nystrom_attention import LlavaNextForConditionalGenerationFNA
from transformers import LlavaNextProcessor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "TextVQA Evaluation for Llava + Fast Nyström Attention (FNA)"
    )



def run_textvqa_eval(args: argparse.Namespace):


def main():
    args = parse_args()
    
    run_textvqa_eval(args)



if __name__ == "__main__":
    main()