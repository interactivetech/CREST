#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CREST_REPO_ROOT="${REPO_ROOT}"
source "${SCRIPT_DIR}/crest_b300_env.sh"

MODEL="${CREST_MODEL:-meta-llama/Meta-Llama-3-8B}"
DATA="${REPO_ROOT}/resources/data/ReClor-train.jsonl"
OUTPUT="${CREST_SCRATCH_ROOT}/outputs/data_ReClor"
TEMPLATE="${REPO_ROOT}/resources/templates/template_ReClor_inference_CoT.json"
FEWSHOT="${REPO_ROOT}/resources/fewshots/ReClor.json"
LOG="${CREST_SCRATCH_ROOT}/logs/reclor-stage1-$(date +%Y%m%d-%H%M%S).log"
PYTHON="${CREST_ENV}/bin/python"

mkdir -p "${OUTPUT}" "$(dirname "${LOG}")"
if [[ -e "${OUTPUT}/1-rationales.json" ]]; then
    echo "Refusing to overwrite existing full-run output: ${OUTPUT}/1-rationales.json" >&2
    exit 2
fi
exec > >(tee -a "${LOG}") 2>&1

export HF_HUB_OFFLINE=1
START_TIME="$(date +%s)"
EXAMPLE_COUNT="$(wc -l < "${DATA}")"

cat > "${CREST_SCRATCH_ROOT}/outputs/data_ReClor_manifest.txt" <<EOF
repo=${REPO_ROOT}
scratch=${CREST_SCRATCH_ROOT}
model=${MODEL}
commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)
gpu=${CUDA_VISIBLE_DEVICES}
dataset=${DATA}
examples=${EXAMPLE_COUNT}
correct_rationales_per_question=16
output=${OUTPUT}
log=${LOG}
EOF

echo "Starting CREST stage 1: ${EXAMPLE_COUNT} ReClor train questions; 16 rationale generations each."
echo "Output: ${OUTPUT}"
echo "Log: ${LOG}"

cd "${REPO_ROOT}/src/1_rationale_generation"

"${PYTHON}" rationale_generation.py \
    --base_model_name "${MODEL}" \
    --batch_size 48 \
    --max_gen_len 512 \
    --data "${DATA}" \
    --output_directory "${OUTPUT}" \
    --correct_n 16 \
    --incorrect_n 0 \
    --fewshot_samples_path "${FEWSHOT}" \
    --fewshot_count 20 \
    --template_file_path "${TEMPLATE}"

"${PYTHON}" answer_prediction.py \
    --base_model_name "${MODEL}" \
    --batch_size 64 \
    --max_gen_len 5 \
    --data "${DATA}" \
    --rationale_directory "${OUTPUT}" \
    --template_file_path "${TEMPLATE}"

"${PYTHON}" filtering.py --prediction_directory "${OUTPUT}"

"${PYTHON}" consistency_test.py \
    --base_model_name "${MODEL}" \
    --batch_size 96 \
    --max_gen_len 5 \
    --data "${DATA}" \
    --rationale_directory "${OUTPUT}" \
    --template_file_path "${TEMPLATE}"

"${PYTHON}" consistency_evaluation.py --rationale_directory "${OUTPUT}"

"${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import os
import sys

output = sys.argv[1]
expected = [
    "1-rationales.json",
    "2-predictions.json",
    "3-high_quality_rationales.json",
    "3-low_quality_rationales.json",
    "4-consistency-predictions.json",
    "4-consistency-evaluation.json",
    "4-hq_rationale_with_consistency_score.json",
]
missing = [name for name in expected if not os.path.isfile(os.path.join(output, name))]
if missing:
    raise SystemExit(f"Missing stage 1 artifacts: {missing}")
with open(os.path.join(output, "3-high_quality_rationales.json")) as f:
    high_quality = json.load(f)
print(f"Stage 1 complete: {len(high_quality)} high-quality rationales")
PY

END_TIME="$(date +%s)"
ELAPSED="$((END_TIME - START_TIME))"
echo "elapsed_seconds=${ELAPSED}" >> "${CREST_SCRATCH_ROOT}/outputs/data_ReClor_manifest.txt"
echo "Stage 1 finished in ${ELAPSED} seconds."
