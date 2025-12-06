#!/usr/bin/env bash
set -euo pipefail

# Optional overrides
GPU_ID=${GPU_ID:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

MODEL_ID="llava-hf/llava-v1.6-vicuna-7b-hf"
QUESTIONS_JSON="/home/andrew/codebases/tmp/COCO/annotations/v2_OpenEnded_mscoco_val2014_questions_shortened_100.json"
ANNOTATIONS_JSON="/home/andrew/codebases/tmp/COCO/annotations/v2_mscoco_val2014_annotations_shortened_100.json"
IMAGES_ROOT="/home/andrew/codebases/tmp/COCO"
EVAL_SCRIPT="experiments/llava/eval/coco_vqa_eval.py"
OUTPUT_ROOT="./experiments/llava/eval/outputs/coco_vqa"

# BERTScore configuration (override via env vars when needed)
BERTSCORE_REFERENCE_JSON=${BERTSCORE_REFERENCE_JSON:-"./experiments/llava/eval/outputs/gpt4_answers_shortened_100.json"}
BERTSCORE_MODEL_TYPE=${BERTSCORE_MODEL_TYPE:-"microsoft/deberta-large-mnli"}
BERTSCORE_BATCH_SIZE=${BERTSCORE_BATCH_SIZE:-16}
BERTSCORE_LANG=${BERTSCORE_LANG:-"en"}
BERTSCORE_RESCALE=${BERTSCORE_RESCALE:-0}
BERTSCORE_REFERENCE_STRATEGY=${BERTSCORE_REFERENCE_STRATEGY:-"majority"}
BERTSCORE_DEVICE=${BERTSCORE_DEVICE:-""}
LAYER_END=32

LAYER_STARTS=(12 14 16 18 20)
NUM_SAMPLES=(16 32 64 128 256 512)
SAMPLING_STRATEGY="fps"

run_eval() {
  local output_dir="$1"
  shift

  echo "============================================================"
  echo "Running eval -> GPU ${GPU_ID} | output: ${output_dir}"
  echo "Extra args: $*"
  echo "============================================================"

  mkdir -p "${output_dir}"

  local bertscore_args=(
    --bertscore-reference-json "${BERTSCORE_REFERENCE_JSON}"
    --bertscore-model-type "${BERTSCORE_MODEL_TYPE}"
    --bertscore-batch-size "${BERTSCORE_BATCH_SIZE}"
    --bertscore-lang "${BERTSCORE_LANG}"
    --bertscore-reference-strategy "${BERTSCORE_REFERENCE_STRATEGY}"
  )
  if [[ "${BERTSCORE_RESCALE}" != "0" ]]; then
    bertscore_args+=(--bertscore-rescale)
  fi
  if [[ -n "${BERTSCORE_DEVICE}" ]]; then
    bertscore_args+=(--bertscore-device "${BERTSCORE_DEVICE}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --model-id "${MODEL_ID}" \
    --questions-json "${QUESTIONS_JSON}" \
    --annotations-json "${ANNOTATIONS_JSON}" \
    --images-root "${IMAGES_ROOT}" \
    --output-dir "${output_dir}" \
    "${bertscore_args[@]}" \
    "$@"
}

# Baseline (FNA disabled)
run_eval "${OUTPUT_ROOT}/llava_baseline" --disable-fna

# FNA sweeps
for layer_start in "${LAYER_STARTS[@]}"; do
  for num_sample in "${NUM_SAMPLES[@]}"; do
    output_dir="${OUTPUT_ROOT}/llava_fna-layers_${layer_start}_${LAYER_END}-samples_${num_sample}-img_toks"
    run_eval "${output_dir}" \
      --fna-layer-range "${layer_start}:${LAYER_END}" \
      --fna-num-sample "${num_sample}" \
      --fna-sampling-strategy "${SAMPLING_STRATEGY}"
  done
done
