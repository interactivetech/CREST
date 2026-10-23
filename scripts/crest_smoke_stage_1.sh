#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CREST_REPO_ROOT="${REPO_ROOT}"
source "${SCRIPT_DIR}/crest_b300_env.sh"

MODEL="${CREST_MODEL:-meta-llama/Meta-Llama-3-8B}"
SMOKE_ROOT="${CREST_SCRATCH_ROOT}/smoke_stage_1"
DATA="${SMOKE_ROOT}/ReClor-smoke.jsonl"
OUTPUT="${SMOKE_ROOT}/outputs/data_ReClor"
TEMPLATE="${REPO_ROOT}/resources/templates/template_ReClor_inference_CoT.json"
FEWSHOT="${REPO_ROOT}/resources/fewshots/ReClor.json"
PYTHON="${CREST_ENV}/bin/python"

mkdir -p "${SMOKE_ROOT}/logs" "${SMOKE_ROOT}/outputs"
head -n 4 "${REPO_ROOT}/resources/data/ReClor-train.jsonl" > "${DATA}"

{
    echo "repo=${REPO_ROOT}"
    echo "scratch=${CREST_SCRATCH_ROOT}"
    echo "model=${MODEL}"
    echo "commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "gpu=${CUDA_VISIBLE_DEVICES}"
    echo "data=${DATA}"
    echo "output=${OUTPUT}"
} > "${SMOKE_ROOT}/manifest.txt"

cd "${REPO_ROOT}/src/1_rationale_generation"

"${PYTHON}" rationale_generation.py \
    --base_model_name "${MODEL}" \
    --batch_size 4 \
    --max_gen_len 128 \
    --data "${DATA}" \
    --output_directory "${OUTPUT}" \
    --correct_n 1 \
    --incorrect_n 0 \
    --fewshot_samples_path "${FEWSHOT}" \
    --fewshot_count 3 \
    --template_file_path "${TEMPLATE}"

"${PYTHON}" answer_prediction.py \
    --base_model_name "${MODEL}" \
    --batch_size 4 \
    --max_gen_len 5 \
    --data "${DATA}" \
    --rationale_directory "${OUTPUT}" \
    --template_file_path "${TEMPLATE}"

"${PYTHON}" filtering.py --prediction_directory "${OUTPUT}"

"${PYTHON}" consistency_test.py \
    --base_model_name "${MODEL}" \
    --batch_size 4 \
    --max_gen_len 8 \
    --data "${DATA}" \
    --rationale_directory "${OUTPUT}" \
    --template_file_path "${TEMPLATE}"

"${PYTHON}" consistency_evaluation.py --rationale_directory "${OUTPUT}"

"${PYTHON}" - "${OUTPUT}" "${SMOKE_ROOT}/manifest.txt" <<'PY'
import json
import os
import sys

output, manifest = sys.argv[1:]
expected = [
    "1-rationales.json",
    "2-predictions.json",
    "3-high_quality_rationales.json",
    "3-low_quality_rationales.json",
    "4-consistency-predictions.json",
    "4-consistency-evaluation.json",
    "4-hq_rationale_with_consistency_score.json",
]
missing = [name for name in expected if not os.path.exists(os.path.join(output, name))]
if missing:
    raise SystemExit(f"Missing smoke artifacts: {missing}")
with open(os.path.join(output, "3-high_quality_rationales.json")) as f:
    high_quality = json.load(f)
if not high_quality:
    raise SystemExit("Smoke run produced no high-quality rationales; inspect 2-predictions.json")
with open(manifest, "a") as f:
    f.write(f"high_quality_rationales={len(high_quality)}\n")
print(f"Smoke test passed: {len(high_quality)} high-quality rationales")
PY
