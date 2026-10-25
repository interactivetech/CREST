#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/crest_b300_env.sh"

REPO_ROOT="${CREST_REPO_ROOT}"
STAGE1_DIR="${CREST_STAGE1_OUTPUT_DIR:-${CREST_SCRATCH_ROOT}/outputs/data_ReClor}"
SOURCE_DATA="${STAGE1_DIR}/4-hq_rationale_with_consistency_score.json"
TRAIN_DATA="${STAGE1_DIR}/4-hq_rationale_with_consistency_score_1.json"
MODEL="${CREST_BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
TEMPLATE="${REPO_ROOT}/resources/templates/template_ReClor_inference_CoT.json"
VAL_DATA="${REPO_ROOT}/resources/data/ReClor-val.jsonl"
TEST_DATA="${REPO_ROOT}/resources/data/ReClor-test.jsonl"
RUN_NAME="reclor_b300_sft"
RUN_DIR="${CREST_SCRATCH_ROOT}/models/ReClor/M_SFT/${RUN_NAME}"
SMOKE_DIR="${CREST_SCRATCH_ROOT}/models/ReClor/M_SFT/${RUN_NAME}_smoke"
LOG_DIR="${CREST_SCRATCH_ROOT}/logs/${RUN_NAME}"
TRAINER_DIR="${REPO_ROOT}/src/2_supervised_fine_tuning"

mkdir -p "${RUN_DIR}" "${SMOKE_DIR}" "${LOG_DIR}"
export WANDB_MODE=offline
export WANDB_DIR="${CREST_SCRATCH_ROOT}/wandb"

if [[ ! -f "${SOURCE_DATA}" ]]; then
    echo "Missing stage 1 rationale file: ${SOURCE_DATA}" >&2
    exit 1
fi

cd "${TRAINER_DIR}"
python filtering.py --stage1_output_dir "${STAGE1_DIR}" | tee "${LOG_DIR}/filtering.log"
python - "${TRAIN_DATA}" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as stream:
    count = sum(1 for line in stream if line.strip())
print(f"Threshold-1 training rows: {count}")
if count != 34593:
    raise SystemExit(f"Expected 34593 rows, found {count}")
PY

COMMON_ARGS=(
    --base_model "${MODEL}"
    --train_on_completion_only True
    --sanity_check 0
    --CoT True
    --per_device_train_batch_size 2
    --simulated_batch 8
    --padding_side left
    --max_length 2048
    --template_dict_path "${TEMPLATE}"
    --logging_frequency 5
)

echo "Starting 32-example one-epoch smoke run"
python train.py "${COMMON_ARGS[@]}" \
    --train_data_path "${TRAIN_DATA}" \
    --output_dir "${SMOKE_DIR}" \
    --lr 8e-4 \
    --num_epochs 1 \
    --sanity_check 32 \
    --key "${RUN_NAME}_smoke" 2>&1 | tee "${LOG_DIR}/smoke_train.log"

python - "${SMOKE_DIR}" <<'PY'
import json, math, os, sys
run_dir = sys.argv[1]
metrics = json.load(open(os.path.join(run_dir, "training_metrics.json")))
loss = metrics["train_metrics"].get("train_loss")
if loss is None or not math.isfinite(float(loss)):
    raise SystemExit(f"Smoke training loss is not finite: {loss}")
if not os.path.isfile(os.path.join(run_dir, "adapter_config.json")):
    raise SystemExit("Smoke run did not save an adapter")
print(f"Smoke training loss: {loss}")
PY

python test.py \
    --base_model "${MODEL}" \
    --eval_data_path "${VAL_DATA}" \
    --model_dir "${SMOKE_DIR}" \
    --output_dir "${SMOKE_DIR}/inference_smoke" \
    --CoT True \
    --batch_size 1 \
    --max_new_tokens 32 \
    --max_eval_samples 1 \
    --template_dict_path "${TEMPLATE}" 2>&1 | tee "${LOG_DIR}/smoke_inference.log"
test -s "${SMOKE_DIR}/inference_smoke/eval_result.json"

echo "Starting four-epoch full training run"
python train.py "${COMMON_ARGS[@]}" \
    --train_data_path "${TRAIN_DATA}" \
    --output_dir "${RUN_DIR}" \
    --lr 8e-4 \
    --num_epochs 4 \
    --key "${RUN_NAME}" 2>&1 | tee "${LOG_DIR}/train.log"

python - "${RUN_DIR}" "${MODEL}" "${VAL_DATA}" "${TEMPLATE}" "${LOG_DIR}" <<'PY'
import json, os, re, subprocess, sys
run_dir, model, val_data, template, log_dir = sys.argv[1:]
checkpoints = [
    os.path.join(run_dir, name) for name in os.listdir(run_dir)
    if os.path.isdir(os.path.join(run_dir, name)) and re.fullmatch(r"checkpoint-\d+", name)
]
checkpoints.sort(key=lambda path: int(os.path.basename(path).split("-")[1]))
if len(checkpoints) != 4:
    raise SystemExit(f"Expected four epoch checkpoints, found {len(checkpoints)}: {checkpoints}")
results = []
for checkpoint in checkpoints:
    val_dir = os.path.join(checkpoint, "val")
    command = [
        sys.executable, "test.py", "--base_model", model,
        "--eval_data_path", val_data, "--model_dir", checkpoint,
        "--output_dir", val_dir, "--CoT", "True", "--batch_size", "16",
        "--max_new_tokens", "512", "--template_dict_path", template,
    ]
    with open(os.path.join(log_dir, os.path.basename(checkpoint) + "_val.log"), "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    metrics_path = os.path.join(val_dir, "eval_second_result_dicts.json")
    with open(metrics_path) as stream:
        metrics = json.load(stream)
    results.append({"checkpoint": checkpoint, "second_pass_accuracy": metrics["accuracy"], "metrics": metrics})
    print(f"{os.path.basename(checkpoint)} second-pass validation accuracy: {metrics['accuracy']}")

# Stable ordering preserves the earlier checkpoint when accuracies tie.
best = max(results, key=lambda result: result["second_pass_accuracy"])
selection = {"selection_metric": "second_pass_validation_accuracy", "checkpoints": results,
             "selected_checkpoint": best["checkpoint"],
             "selected_accuracy": best["second_pass_accuracy"],
             "tie_break": "earlier checkpoint"}
with open(os.path.join(run_dir, "validation_checkpoint_selection.json"), "w") as stream:
    json.dump(selection, stream, indent=2)
print(f"Selected checkpoint: {best['checkpoint']}")
PY

SELECTED_CHECKPOINT="$(python - "${RUN_DIR}/validation_checkpoint_selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_checkpoint"])
PY
)"

echo "Evaluating the held-out test set once using ${SELECTED_CHECKPOINT}"
python test.py \
    --base_model "${MODEL}" \
    --eval_data_path "${TEST_DATA}" \
    --model_dir "${SELECTED_CHECKPOINT}" \
    --output_dir "${RUN_DIR}/final_test" \
    --CoT True \
    --batch_size 16 \
    --max_new_tokens 512 \
    --template_dict_path "${TEMPLATE}" 2>&1 | tee "${LOG_DIR}/test.log"

python - "${RUN_DIR}" <<'PY'
import json, os, sys
run_dir = sys.argv[1]
with open(os.path.join(run_dir, "final_test", "eval_second_result_dicts.json")) as stream:
    metrics = json.load(stream)
with open(os.path.join(run_dir, "final_test", "test_evaluation_summary.json"), "w") as stream:
    json.dump({"evaluations": 1, "second_pass_metrics": metrics}, stream, indent=2)
print(f"Final second-pass test accuracy: {metrics['accuracy']}")
PY
