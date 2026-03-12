#!/usr/bin/env bash
set -euo pipefail

# Optional overrides
GPU_ID=${GPU_ID:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

MODEL_ID=${MODEL_ID:-"llava-hf/llava-v1.6-vicuna-7b-hf"}
QUESTIONS_JSONL=${QUESTIONS_JSONL:-"/home/andrew/codebases/tmp/llava-bench-in-the-wild/questions.jsonl"}
IMAGES_ROOT=${IMAGES_ROOT:-"/home/andrew/codebases/tmp/llava-bench-in-the-wild/images"}
EVAL_SCRIPT="experiments/llava/eval/llava_bench_eval.py"
OUTPUT_ROOT=${OUTPUT_ROOT:-"./experiments/llava/eval/outputs_q_fpsample/llava_bench"}

# Generation overrides (optional)
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-50}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are a helpful assistant."}
ANSWER_GUIDANCE=${ANSWER_GUIDANCE:-""}
MODEL_ID_FOR_ANSWERS=${MODEL_ID_FOR_ANSWERS:-""}

# FNA sweep settings
LAYER_END=${LAYER_END:-32}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-"fps"}
SAMPLING_FEATURES=${SAMPLING_FEATURES:-"q"}

# Parse sweep arrays (space- or comma-separated) with safe defaults
if [[ -n "${LAYER_STARTS:-}" ]]; then
  IFS=", " read -r -a LAYER_STARTS_ARR <<< "${LAYER_STARTS}"
else
  LAYER_STARTS_ARR=(12 14 16 18 20)
fi

if [[ -n "${NUM_SAMPLES:-}" ]]; then
  IFS=", " read -r -a NUM_SAMPLES_ARR <<< "${NUM_SAMPLES}"
else
  NUM_SAMPLES_ARR=(16 32 64 128 256 512)
fi

run_eval() {
  local output_dir="$1"
  shift

  echo "============================================================"
  echo "Running LLaVA-Bench eval -> GPU ${GPU_ID} | output: ${output_dir}"
  echo "Extra args: $*"
  echo "============================================================"

  mkdir -p "${output_dir}"

  local gen_args=(
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --system-prompt "${SYSTEM_PROMPT}"
  )
  if [[ -n "${ANSWER_GUIDANCE}" ]]; then
    gen_args+=(--answer-guidance "${ANSWER_GUIDANCE}")
  fi
  if [[ -n "${MODEL_ID_FOR_ANSWERS}" ]]; then
    gen_args+=(--model-id-for-answers "${MODEL_ID_FOR_ANSWERS}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --model-id "${MODEL_ID}" \
    --questions-jsonl "${QUESTIONS_JSONL}" \
    --images-root "${IMAGES_ROOT}" \
    --output-dir "${output_dir}" \
    "${gen_args[@]}" \
    "$@"
}

# Baseline (FNA disabled)
run_eval "${OUTPUT_ROOT}/llava_bench_baseline" --disable-fna

# FNA sweeps
for layer_start in "${LAYER_STARTS_ARR[@]}"; do
  for num_sample in "${NUM_SAMPLES_ARR[@]}"; do
    output_dir="${OUTPUT_ROOT}/llava_bench_fna-layers_${layer_start}_${LAYER_END}-samples_${num_sample}-${SAMPLING_FEATURES}"
    run_eval "${output_dir}" \
      --fna-layer-range "${layer_start}:${LAYER_END}" \
      --fna-num-sample "${num_sample}" \
      --fna-sampling-strategy "${SAMPLING_STRATEGY}" \
      --fna-sampling-features "${SAMPLING_FEATURES}"
  done
done
