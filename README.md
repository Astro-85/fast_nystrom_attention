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

5. Sanity check:

  ```bash
  python -c "import torch; import transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
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