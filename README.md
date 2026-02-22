# Fast Nyström Attention

Official repository for [Artifacts and Attention Sinks: Structured Approximations for Efficient Vision Transformers](https://arxiv.org/pdf/2507.16018)

## Installation

1. Create an environment:

  ```bash
  conda create -n fna python=3.10 -y
  conda activate fna
  python -m pip install -U pip
  ```

2. Install the PyTorch build for your CUDA:

  ```bash
  # CUDA 12.8 example (adjust for your CUDA)
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
  ```

3. Install [torch-quickfps](https://github.com/Astro-85/torch_quickfps) for your CUDA (example):

  ```bash
  python -m pip install torch-quickfps-cu128
  ```

4. Install project deps:

  ```bash
  python -m pip install -r requirements.txt
  ```

## Experiments

### CLIP

Run the COCO retrieval notebook in [experiments/clip/clip_coco_retrieval.ipynb](experiments/clip/clip_coco_retrieval.ipynb).

### LLaVA

Start with the demo notebook in [experiments/llava/llava_demo.ipynb](experiments/llava/llava_demo.ipynb).

For benchmark sweeps, use the scripts in [experiments/llava/scripts/](experiments/llava/scripts/):

- [Run sweep](experiments/llava/scripts/run_llava_bench_sweep.sh)
- [Run GPT-based evals](experiments/llava/scripts/run_llava_bench_gpt_eval_dir.sh)

Minimum setup for LLaVA-Bench requires the dataset files. Example sweep command (adjust paths and GPU as needed):

```bash
GPU_ID=0 \
QUESTIONS_JSONL=/path/to/questions.jsonl \
IMAGES_ROOT=/path/to/images \
./experiments/llava/scripts/run_llava_bench_sweep.sh
```

Example GPT-based eval over a runs directory:

```bash
EVAL_MODEL=gpt-4o-mini \
QUESTIONS_JSONL=/path/to/questions.jsonl \
ANSWERS_GPT4_JSONL=/path/to/answers_gpt4.jsonl \
CONTEXT_JSONL=/path/to/context.jsonl \
./experiments/llava/scripts/run_llava_bench_gpt_eval_dir.sh ./experiments/llava/eval/outputs_q_fpsample/llava_bench
```

---

## Reference

If you find this repository useful, please cite:

```bibtex
@article{lu2025artifacts,
  title   = {Artifacts and Attention Sinks: Structured Approximations for Efficient Vision Transformers},
  author  = {Lu, Andrew and Liao, Wentinn and Wang, Liuhui and Yang, Huzheng and Shi, Jianbo},
  journal = {arXiv preprint arXiv:2507.16018},
  year    = {2025},
  url     = {https://arxiv.org/abs/2507.16018}
}
```