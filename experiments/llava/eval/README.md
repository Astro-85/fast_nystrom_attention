# LLaVA COCO VQA evaluation utilities

## Plot sweep results

Use `plot_coco_vqa_results.py` to visualize the BERTScore F1 of each sweep point
against its normalized generation throughput (tokens per second). The script reads
the `metrics.json` files produced by `experiments/llava/scripts/run_llava_vqa_sweep.sh`
and renders a scatter plot where colors encode the masked layer range and marker sizes
reflect the sample count.

The script requires `matplotlib` and `numpy` to be installed in your Python
environment. By default it looks for results under
`experiments/llava/eval/outputs/coco_vqa/` and only shows the plot. To save an
image, pass `--output-path`.

```bash
python experiments/llava/eval/plot_coco_vqa_results.py \
  --output-path experiments/llava/eval/outputs/coco_vqa/bert_f1_vs_tokens.png \
  --show
```

Use `--results-root` to point at an alternate sweep directory (e.g., when running
on a different machine) and `--title` to customize the chart title.
