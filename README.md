# Fast Nyström Attention

Official repository for [Artifacts and Attention Sinks: Structured Approximations for Efficient Vision Transformers](https://arxiv.org/pdf/2507.16018)

## COCO VQA evaluation with LLaVA + FNA

Use `experiments/llava/eval/coco_vqa_eval.py` to benchmark a LLaVA-NeXT
checkpoint instrumented with Fast Nyström Attention. The script expects COCO
images plus the `v2_OpenEnded_mscoco_*` question/annotation JSON files and
produces:

- `predictions.jsonl` – detailed per-question generations
- `submission.json` – ready-to-upload answers in the official VQA format
- `metrics.json` – local accuracy, latency, and token statistics
- `bertscore.jsonl` – optional per-question BERTScore precision/recall/F1 when `--compute-bertscore` is enabled

Example command:

```bash
python experiments/llava/eval/coco_vqa_eval.py \
	--model-id llava-hf/llava-v1.6-vicuna-7b-hf \
	--questions-json /path/to/v2_OpenEnded_mscoco_val2014_questions.json \
	--annotations-json /path/to/v2_mscoco_val2014_annotations.json \
	--images-root /path/to/coco \
	--output-dir outputs/llava_fna_eval \
	--fna-layer-range 18:32 --fna-num-sample 256 \
	--compute-bertscore
```

Useful FNA flags:

- `--fna-resample-every-layer` – resample landmarks before each FNA-enabled layer instead of reusing them.
- `--fna-sampling-strategy {fps,random}` – choose between farthest point sampling (default) and random sampling when selecting landmarks.

> **Note:** BERTScore evaluation relies on the optional [`evaluate`](https://github.com/huggingface/evaluate) and
> [`bert-score`](https://github.com/Tiiiger/bert_score) packages. Install them with `pip install evaluate bert-score`
> before enabling `--compute-bertscore`.

To compare generations against answers from a different model (e.g., GPT) instead of the COCO annotations, supply
`--bertscore-reference-json /path/to/gpt_answers.json`. The JSON file should contain a list of objects with
`question_id` and `answer` fields matching the COCO question ids.
