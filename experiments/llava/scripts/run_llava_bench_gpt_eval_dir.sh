#!/usr/bin/env bash
set -euo pipefail

# Run GPT-based LLaVA-Bench evals over all runs in a directory.
#
# Usage:
#   ./experiments/llava/scripts/run_llava_bench_gpt_eval_dir.sh [RUNS_DIR]
#
# Required env (defaults provided):
#   QUESTIONS_JSONL   Path to questions.jsonl
#   ANSWERS_GPT4_JSONL Path to answers_gpt4.jsonl (reference)
#
# Optional env:
#   CONTEXT_JSONL     Path to context.jsonl (recommended)
#   EVAL_MODEL        Evaluator model id (default: gpt-4o-mini)
#   PYTHON_BIN        Python binary (default: python)
#   USE_IMAGES        Set to 1 to send images to evaluator
#   IMAGES_ROOT       Image folder for questions.jsonl (required if USE_IMAGES=1)
#   IMAGE_DETAIL      low|auto|high (default: low)
#   MAX_TOKENS        Evaluator max tokens (default: 1024)
#   TEMPERATURE       Evaluator temperature (default: 0.2)
#   SKIP_EXISTING     Set to 1 to resume/skip existing reviews (default: 1)

RUNS_DIR=${1:-"./experiments/llava/eval/outputs_q_fpsample/llava_bench"}

PYTHON_BIN=${PYTHON_BIN:-python}
EVAL_SCRIPT="experiments/llava/eval/gpt_llava_bench_eval_reviews.py"

QUESTIONS_JSONL=${QUESTIONS_JSONL:-"/home/andrew/codebases/tmp/llava-bench-in-the-wild/questions.jsonl"}
CONTEXT_JSONL=${CONTEXT_JSONL:-"/home/andrew/codebases/tmp/llava-bench-in-the-wild/context.jsonl"}
ANSWERS_GPT4_JSONL=${ANSWERS_GPT4_JSONL:-"/home/andrew/codebases/tmp/llava-bench-in-the-wild/answers_gpt4.jsonl"}

EVAL_MODEL=${EVAL_MODEL:-"gpt-4o-mini"}
MAX_TOKENS=${MAX_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0.2}

USE_IMAGES=${USE_IMAGES:-0}
IMAGES_ROOT=${IMAGES_ROOT:-""}
IMAGE_DETAIL=${IMAGE_DETAIL:-"low"}

SKIP_EXISTING=${SKIP_EXISTING:-1}

if [[ ! -d "${RUNS_DIR}" ]]; then
  echo "[error] RUNS_DIR does not exist: ${RUNS_DIR}" >&2
  exit 1
fi

if [[ ! -f "${QUESTIONS_JSONL}" ]]; then
  echo "[error] QUESTIONS_JSONL not found: ${QUESTIONS_JSONL}" >&2
  exit 1
fi

if [[ ! -f "${ANSWERS_GPT4_JSONL}" ]]; then
  echo "[error] ANSWERS_GPT4_JSONL not found: ${ANSWERS_GPT4_JSONL}" >&2
  exit 1
fi

if [[ "${USE_IMAGES}" == "1" && -z "${IMAGES_ROOT}" ]]; then
  echo "[error] USE_IMAGES=1 requires IMAGES_ROOT" >&2
  exit 1
fi

mapfile -t ANSWER_FILES < <(find "${RUNS_DIR}" -type f -name "answers.jsonl" | sort)

if [[ ${#ANSWER_FILES[@]} -eq 0 ]]; then
  echo "[warn] No answers.jsonl found under ${RUNS_DIR}" >&2
  exit 0
fi

echo "[info] Found ${#ANSWER_FILES[@]} runs under ${RUNS_DIR}"

for answers_jsonl in "${ANSWER_FILES[@]}"; do
  run_dir=$(dirname "${answers_jsonl}")
  output_jsonl="${run_dir}/gpt_reviews.jsonl"

  echo "============================================================"
  echo "[info] Evaluating: ${answers_jsonl}"
  echo "[info] Output:     ${output_jsonl}"
  echo "============================================================"

  args=(
    --question "${QUESTIONS_JSONL}"
    --answer-list "${ANSWERS_GPT4_JSONL}" "${answers_jsonl}"
    --output "${output_jsonl}"
    --eval-model "${EVAL_MODEL}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
  )

  if [[ -n "${CONTEXT_JSONL}" && -f "${CONTEXT_JSONL}" ]]; then
    args+=(--context "${CONTEXT_JSONL}")
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    args+=(--skip-existing)
  fi

  if [[ "${USE_IMAGES}" == "1" ]]; then
    args+=(--use-images --images-root "${IMAGES_ROOT}" --image-detail "${IMAGE_DETAIL}")
  fi

  "${PYTHON_BIN}" "${EVAL_SCRIPT}" "${args[@]}"

done
