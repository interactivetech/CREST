#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/crest_b300_env.sh"

REPO_ROOT="${CREST_REPO_ROOT}"
STAGE1_DIR="${CREST_STAGE1_OUTPUT_DIR:-${CREST_SCRATCH_ROOT}/outputs/data_ReClor}"
STAGE3_DATA_DIR="${CREST_SCRATCH_ROOT}/outputs/data_ReClor_stage3"
MODEL="${CREST_BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
TEMPLATE="${REPO_ROOT}/resources/templates/template_ReClor_inference_CoT.json"
VAL_DATA="${REPO_ROOT}/resources/data/ReClor-val.jsonl"
TEST_DATA="${REPO_ROOT}/resources/data/ReClor-test.jsonl"
SFT_RUN_DIR="${CREST_SCRATCH_ROOT}/models/ReClor/M_SFT/reclor_b300_sft"
SFT_SELECTION="${SFT_RUN_DIR}/validation_checkpoint_selection.json"
RUN_DIR="${CREST_STAGE3_RUN_DIR:-${CREST_SCRATCH_ROOT}/models/ReClor/M_CREST/reclor_b300_dpo}"
SMOKE_DIR="${CREST_SCRATCH_ROOT}/models/ReClor/M_CREST/reclor_b300_dpo_smoke_$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${CREST_SCRATCH_ROOT}/logs/reclor_b300_dpo"
TRAINER_DIR="${REPO_ROOT}/src/3_preference_learning"

mkdir -p "${STAGE3_DATA_DIR}" "${RUN_DIR}" "${LOG_DIR}"
export WANDB_MODE=offline
export WANDB_DIR="${CREST_SCRATCH_ROOT}/wandb"

for path in \
    "${STAGE1_DIR}/3-high_quality_rationales.json" \
    "${STAGE1_DIR}/3-low_quality_rationales.json" \
    "${STAGE1_DIR}/4-hq_rationale_with_consistency_score.json" \
    "${SFT_SELECTION}" "${TEMPLATE}" "${VAL_DATA}" "${TEST_DATA}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required stage 3 input is missing: ${path}" >&2
        exit 1
    fi
done

if find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' -print -quit | grep -q .; then
    echo "Stage 3 output already contains epoch directories: ${RUN_DIR}" >&2
    echo "Set CREST_STAGE3_RUN_DIR to a new scratch path to start a separate run." >&2
    exit 1
fi

SFT_ADAPTER="$(python - "${SFT_SELECTION}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_checkpoint"])
PY
)"
if [[ ! -f "${SFT_ADAPTER}/adapter_config.json" ]]; then
    echo "Selected stage 2 adapter is missing: ${SFT_ADAPTER}" >&2
    exit 1
fi

cd "${TRAINER_DIR}"
echo "Generating CREST stage 3 preference pairs from ${STAGE1_DIR}"
python generate_pair_data.py \
    --stage_1_result "${STAGE1_DIR}" \
    --output_dir "${STAGE3_DATA_DIR}" \
    --template_file_path "${TEMPLATE}" 2>&1 | tee "${LOG_DIR}/pair_generation.log"

python - "${STAGE3_DATA_DIR}" "${STAGE1_DIR}" "${TEMPLATE}" "${RUN_DIR}/data_manifest.json" <<'PY'
import json, os, sys

data_dir, stage1_dir, template_path, manifest_path = sys.argv[1:]
template_data = json.load(open(template_path))
manifest = {"stage1_dir": stage1_dir, "datasets": {}}
for name in ("only_z", "only_tz"):
    path = os.path.join(data_dir, name, "qjk_train_data.json")
    rows = []
    with open(path) as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No preference pairs generated for {name}: {path}")
    required = ("id_string", "query", "response_j", "response_k")
    for key in required:
        if any(not row.get(key) for row in rows[:10]):
            raise SystemExit(f"{name} has missing/empty {key} in sample rows")
    prompt = rows[0]["query"]
    if template_data["instruction"] not in prompt:
        raise SystemExit(f"{name} prompt is missing the actual instruction")
    if "[Instruction]" not in prompt or "[Question]" not in prompt or "Answer Choices:" not in prompt:
        raise SystemExit(f"{name} prompt is missing instruction/question/choices markers")
    manifest["datasets"][name] = {"path": path, "rows": len(rows), "sample_id": rows[0]["id_string"]}
    print(f"{name}: {len(rows)} pairs")
os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
with open(manifest_path, "w") as stream:
    json.dump(manifest, stream, indent=2)
PY

COMMON_ARGS=(
    --base_model_name "${MODEL}"
    --sft_peft_path "${SFT_ADAPTER}"
    --train_data_oq "${STAGE3_DATA_DIR}/only_z/qjk_train_data.json"
    --train_data_cs "${STAGE3_DATA_DIR}/only_tz/qjk_train_data.json"
    --ratio_lambda 0.6
    --learning_rate 6e-6
    --dpo_beta 0.1
    --per_device_train_batch_size 8
    --per_device_eval_batch_size 8
    --simulated_batch_size 8
    --max_length 1024
    --max_prompt_length 512
    --gradient_checkpointing True
    --gradient_checkpointing_use_reentrant False
    --optimizer_type paged_adamw_32bit
    --warmup_steps 100
    --weight_decay 0.05
    --lr_scheduler_type cosine
    --seed 42
    --wandb_project_name reclor_b300_dpo
)

cd "${TRAINER_DIR}"
echo "Running one-step stage 3 smoke; output: ${SMOKE_DIR}"
python dpo_training_ratio.py "${COMMON_ARGS[@]}" \
    --working_dir "${SMOKE_DIR}" \
    --max_steps 1 \
    --dpo_epochs 1 \
    --save_steps 1 \
    --logging_steps 1 \
    --sanity_check True 2>&1 | tee "${LOG_DIR}/smoke_train.log"

python - "${SMOKE_DIR}" <<'PY'
import json, math, os, sys
run_dir = sys.argv[1]
metrics = json.load(open(os.path.join(run_dir, "epoch_1", "training_metrics.json")))["train_metrics"]
loss = metrics.get("train_loss")
if loss is None or not math.isfinite(float(loss)):
    raise SystemExit(f"Smoke DPO loss is not finite: {loss}")
if not os.path.isfile(os.path.join(run_dir, "epoch_1", "adapter_config.json")):
    raise SystemExit("DPO smoke did not save the default adapter")
print(f"Smoke DPO loss: {loss}")
PY

python dpo_model_test.py \
    --base_model_name "${MODEL}" \
    --model_dir "${SMOKE_DIR}/epoch_1" \
    --test_data "${VAL_DATA}" \
    --output_dir "${SMOKE_DIR}/inference_smoke" \
    --test_template_path "${TEMPLATE}" \
    --test_batch_size 1 \
    --test_max_new_tokens 32 \
    --max_eval_samples 1 2>&1 | tee "${LOG_DIR}/smoke_inference.log"
python - "${SMOKE_DIR}/inference_smoke/evaluation_summary.json" <<'PY'
import json, math, sys
summary = json.load(open(sys.argv[1]))
metric = summary.get("second_pass_metrics")
if summary["num_examples"] != 1 or summary["scored_examples"] != 1 or not metric:
    raise SystemExit(f"DPO smoke validation did not score one labeled example: {summary}")
if not math.isfinite(metric["accuracy"]):
    raise SystemExit(f"DPO smoke validation accuracy is not finite: {metric}")
print("Smoke adapter reload and one-example validation passed")
PY

echo "Starting four-round DPO training; output: ${RUN_DIR}"
python dpo_training_ratio.py "${COMMON_ARGS[@]}" \
    --working_dir "${RUN_DIR}" \
    --max_steps 3000 \
    --dpo_epochs 4 \
    --save_steps 3000 \
    --logging_steps 10 \
    --sanity_check False 2>&1 | tee "${LOG_DIR}/train.log"

for epoch in 1 2 3 4; do
    adapter="${RUN_DIR}/epoch_${epoch}/adapter_config.json"
    if [[ ! -f "${adapter}" ]]; then
        echo "Missing DPO round ${epoch} adapter: ${adapter}" >&2
        exit 1
    fi
    python dpo_model_test.py \
        --base_model_name "${MODEL}" \
        --model_dir "${RUN_DIR}/epoch_${epoch}" \
        --test_data "${VAL_DATA}" \
        --output_dir "${RUN_DIR}/epoch_${epoch}/valid" \
        --test_template_path "${TEMPLATE}" \
        --test_batch_size 8 \
        --test_max_new_tokens 560 2>&1 | tee "${LOG_DIR}/epoch_${epoch}_val.log"
done

python - "${RUN_DIR}" <<'PY'
import json, math, os, sys
run_dir = sys.argv[1]
results = []
for epoch in range(1, 5):
    checkpoint = os.path.join(run_dir, f"epoch_{epoch}")
    summary_path = os.path.join(checkpoint, "valid", "evaluation_summary.json")
    summary = json.load(open(summary_path))
    metrics = summary.get("second_pass_metrics")
    if summary["num_examples"] != 500 or summary["scored_examples"] != 500 or not metrics:
        raise SystemExit(f"Round {epoch} validation is incomplete or unscored: {summary}")
    accuracy = float(metrics["accuracy"])
    if not math.isfinite(accuracy):
        raise SystemExit(f"Round {epoch} validation accuracy is non-finite: {accuracy}")
    results.append({"epoch": epoch, "checkpoint": checkpoint, "second_pass_accuracy": accuracy, "metrics": metrics})

# max() preserves the earlier round for equal accuracy values.
best = max(results, key=lambda row: row["second_pass_accuracy"])
selection = {
    "selection_metric": "second_pass_validation_accuracy",
    "rounds": results,
    "selected_checkpoint": best["checkpoint"],
    "selected_epoch": best["epoch"],
    "selected_accuracy": best["second_pass_accuracy"],
    "tie_break": "earlier round",
}
with open(os.path.join(run_dir, "validation_checkpoint_selection.json"), "w") as stream:
    json.dump(selection, stream, indent=2)
print(f"Selected DPO round {best['epoch']} at {best['second_pass_accuracy']:.4f} validation accuracy")
PY

SELECTED_CHECKPOINT="$(python - "${RUN_DIR}/validation_checkpoint_selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_checkpoint"])
PY
)"
echo "Generating held-out test predictions once from ${SELECTED_CHECKPOINT}"
python dpo_model_test.py \
    --base_model_name "${MODEL}" \
    --model_dir "${SELECTED_CHECKPOINT}" \
    --test_data "${TEST_DATA}" \
    --output_dir "${RUN_DIR}/final_test" \
    --test_template_path "${TEMPLATE}" \
    --test_batch_size 8 \
    --test_max_new_tokens 560 2>&1 | tee "${LOG_DIR}/test.log"

python - "${RUN_DIR}/final_test/evaluation_summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
if summary["num_examples"] != 1000:
    raise SystemExit(f"Expected 1000 ReClor test predictions, got {summary['num_examples']}")
if summary["scored_examples"] != 0 or summary["labels_available"]:
    raise SystemExit(f"Expected the provided test split to be unlabeled: {summary}")
print("Saved 1000 test predictions; labels are absent, so no test accuracy is reported")
PY

echo "Stage 3 training and evaluation complete. Selected adapter: ${SELECTED_CHECKPOINT}"
echo "Use scripts/upload_hf_adapter.py to publish it after supplying a rotated HF token."
