#!/usr/bin/env bash
set -euo pipefail

# Source this file from the repository root or from another CREST script.
CREST_REPO_ROOT="${CREST_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CREST_USER="${USER:-$(id -un)}"
CREST_SCRATCH_ROOT="${CREST_SCRATCH_ROOT:-/scratch1/${CREST_USER}/CREST_${CREST_USER}_b300}"
CREST_ENV="${CREST_SCRATCH_ROOT}/env"

for required_dir in \
    "${CREST_SCRATCH_ROOT}" \
    "${CREST_ENV}" \
    "${CREST_SCRATCH_ROOT}/hf_cache" \
    "${CREST_SCRATCH_ROOT}/torch_cache" \
    "${CREST_SCRATCH_ROOT}/vllm_cache" \
    "${CREST_SCRATCH_ROOT}/triton_cache" \
    "${CREST_SCRATCH_ROOT}/wandb" \
    "${CREST_SCRATCH_ROOT}/tmp"; do
    if [[ ! -d "${required_dir}" ]]; then
        echo "Missing scratch directory: ${required_dir}" >&2
        return 1 2>/dev/null || exit 1
    fi
done

export CREST_REPO_ROOT CREST_SCRATCH_ROOT CREST_ENV CREST_USER
export CUDA_HOME="${CREST_ENV}/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="${CUDA_HOME}"
export HF_HOME="${CREST_SCRATCH_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${CREST_SCRATCH_ROOT}/hf_cache/hub"
export TRANSFORMERS_CACHE="${CREST_SCRATCH_ROOT}/hf_cache/transformers"
export TORCH_HOME="${CREST_SCRATCH_ROOT}/torch_cache"
export VLLM_CACHE_ROOT="${CREST_SCRATCH_ROOT}/vllm_cache"
export FLASHINFER_WORKSPACE_BASE="${CREST_SCRATCH_ROOT}/flashinfer"
export FLASHINFER_AUTOTUNE_CACHE_DIR="${CREST_SCRATCH_ROOT}/vllm_cache/flashinfer_autotune_cache"
export TRITON_CACHE_DIR="${CREST_SCRATCH_ROOT}/triton_cache"
export PIP_CACHE_DIR="${CREST_SCRATCH_ROOT}/pip_cache"
export WANDB_DIR="${CREST_SCRATCH_ROOT}/wandb"
export TMPDIR="${CREST_SCRATCH_ROOT}/tmp"
export HF_DATASETS_CACHE="${CREST_SCRATCH_ROOT}/hf_cache/datasets"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${PIP_CACHE_DIR}" \
    "${FLASHINFER_WORKSPACE_BASE}" \
    "${WANDB_DIR}" \
    "${TMPDIR}"

if [[ "${HF_HOME}" != "${CREST_SCRATCH_ROOT}"/* || "${TMPDIR}" != "${CREST_SCRATCH_ROOT}"/* ]]; then
    echo "Refusing to run with caches outside scratch: HF_HOME=${HF_HOME}, TMPDIR=${TMPDIR}" >&2
    return 1 2>/dev/null || exit 1
fi

export PATH="${CREST_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
