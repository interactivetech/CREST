# CREST B300 Setup with Scratch-Backed Storage

## Storage Findings

- Home filesystem: 30 GB total, 26 GB used, 4.7 GB free (85% full).
- `/scratch0`: approximately 83 GB free.
- `/scratch1`: approximately 26 TB free.
- The existing Conda installation is under the home directory, so the new CREST environment must not be created there.
- The repository itself is small: roughly 12 MB, so source code can remain in the home project directory.

Use `/scratch1/<username>/CREST_<username>_b300` as the primary storage root. Use the corresponding `/scratch0/<username>/...` path only if scratch1 is unavailable.

## Environment and Cache Layout

Create all large artifacts under the scratch root:

```text
/scratch1/<username>/CREST_<username>_b300/
├── env/                 # Python/Conda environment
├── conda_pkgs/          # Conda package cache
├── pip_cache/           # pip cache
├── hf_cache/            # Hugging Face model/tokenizer cache
├── torch_cache/         # Torch compilation/cache files
├── vllm_cache/         # vLLM compilation/autotune cache
├── triton_cache/        # Triton cache
├── wandb/               # optional W&B files
├── outputs/             # smoke and full-generation outputs
├── models/              # trained checkpoints
├── logs/                # run logs
└── tmp/                 # temporary subsets and intermediate files
```

The setup/run wrapper should export:

- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `TRANSFORMERS_CACHE`
- `TORCH_HOME`
- `PIP_CACHE_DIR`
- `CONDA_PKGS_DIRS`
- `VLLM_CACHE_ROOT`
- `TRITON_CACHE_DIR`
- `WANDB_DIR`
- `TMPDIR`

It should also verify that these variables point to scratch before installing packages or downloading the model.

## Environment Setup

- Create the Python 3.12 environment at `$CREST_SCRATCH_ROOT/env`.
- Do not use the current home-based base environment or the existing empty `vllm_env`.
- Install a current PyTorch/vLLM combination compatible with B300’s compute capability 10.3. B300 is listed by NVIDIA as compute capability 10.3, and current vLLM documentation requires a CUDA 12.8-or-newer-compatible Blackwell stack. [NVIDIA GPU compatibility table](https://developer.nvidia.com/cuda/gpus?lng=en-US), [vLLM GPU installation guidance](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu/)
- Replace or supplement the old requirements with a B300-specific requirements file. The existing `vllm==0.5.4` and `torch==2.4.0` pins should not be installed blindly.
- Include packages imported by the repository but missing from `requirements.txt`, especially `datasets`, `pandas`, `numpy`, `python-dotenv`, `openai`, and `requests`.
- Pin `nvidia-cuda-nvcc==13.0.88` and the B300 FlashInfer precompiled kernel packages; this avoids relying on a host CUDA compiler that may not match the installed PyTorch CUDA runtime.
- Save the resolved package versions and `pip freeze` output under the scratch run directory, not in home.

## Runtime Validation

Run all checks with caches redirected to scratch:

- Confirm the B300 device and available memory with `nvidia-smi`.
- Confirm PyTorch CUDA availability, device name, BF16 support, and a small GPU matrix operation.
- Confirm tokenizer and model access for `meta-llama/Meta-Llama-3-8B`.
- Run a one-prompt vLLM generation test.
- Import every repository module, including analysis modules.
- Confirm no model, package, cache, temporary, or log files are written outside the scratch root except normal Python source-control metadata.

Hugging Face authentication should be completed before model loading. W&B should be disabled or offline for the smoke run.

## Small End-to-End Test

Add a smoke-run wrapper that:

- Uses the repository as source but writes all temporary subsets, outputs, logs, and caches under `$CREST_SCRATCH_ROOT`.
- Sets `home` explicitly to the repository path; do not rely on the existing scripts’ commented-out `home=$(pwd)`.
- Creates a temporary 2–4-example ReClor subset under the scratch `tmp/` directory.
- Runs stage 1 only with:
  - `correct_n=1`
  - `incorrect_n=0`
  - short generation lengths
  - batch sizes no larger than 1–4 initially
  - output under `outputs/smoke_stage1/`
- Executes rationale generation, answer prediction, filtering, consistency testing, and consistency evaluation.
- Adds clear empty-input handling for the case where no rationales survive filtering.
- Stores a concise manifest containing model name, environment path, cache paths, GPU, dataset subset size, git commit, and output locations.

Expected smoke artifacts include:

- `1-rationales.json`
- `2-predictions.json`
- high- and low-quality rationale files
- `4-consistency-predictions.json`
- `4-hq_rationale_with_consistency_score.json`
- consistency evaluation JSON

## Acceptance Criteria

The setup is complete when:

- Home usage remains below the available-space safety threshold and all large files are on scratch.
- PyTorch and vLLM run successfully on the B300.
- Llama 3 8B loads and generates a non-empty response.
- The smoke run completes all stage-1 steps.
- Artifact counts and IDs remain consistent between stages.
- The resolved environment and run manifest are saved under `$CREST_SCRATCH_ROOT`.
- Only after the smoke test passes are full ReClor generation outputs written to the scratch `outputs/` directory.

## Assumptions

- `$CREST_SCRATCH_ROOT` is the authoritative working directory for all large CREST artifacts.
- Source code remains in the CREST repository checkout.
- One visible B300 is used initially.
- Stages 2 and 3 are deferred until stage 1 succeeds.
