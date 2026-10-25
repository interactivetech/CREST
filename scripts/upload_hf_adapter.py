#!/usr/bin/env python3
"""Upload one saved adapter to a newly created private Hugging Face repo."""

import argparse
import getpass
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--validation-selection", required=True)
    parser.add_argument("--scratch-root", required=True)
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise SystemExit(f"Adapter config not found in {adapter_dir}")
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise SystemExit(f"Adapter weights not found in {adapter_dir}")
    selection = json.loads(Path(args.validation_selection).read_text())
    token = getpass.getpass("Hugging Face write token (input hidden): ")
    if not token:
        raise SystemExit("No token supplied.")

    api = HfApi(token=token)
    try:
        account = api.whoami()["name"]
    except Exception:
        raise SystemExit("Hugging Face authentication failed; no token was displayed or saved.")
    owner = args.repo_id.split("/", 1)[0]
    if account.casefold() != owner.casefold():
        raise SystemExit(f"Authenticated HF account {account!r} does not own requested repo namespace {owner!r}.")

    scratch_root = Path(args.scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crest_hf_upload_", dir=scratch_root) as tmp:
        upload_dir = Path(tmp)
        files = ["adapter_config.json", "adapter_model.safetensors"]
        files.extend(
            name for name in (
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"
            ) if (adapter_dir / name).is_file()
        )
        for name in files:
            shutil.copy2(adapter_dir / name, upload_dir / name)

        readme = f"""---
base_model: meta-llama/Meta-Llama-3-8B
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - dpo
  - reclor
  - logical-reasoning
---

# CREST ReClor stage 3 DPO adapter

LoRA adapter for `meta-llama/Meta-Llama-3-8B`, trained with CREST's mixed preference learning on ReClor rationale pairs. The policy was initialized from the CREST stage 2 SFT adapter and selected from four DPO rounds using second-pass validation accuracy.

- Selected round: {selection['selected_epoch']}
- Validation metric: second-pass accuracy, {selection['selected_accuracy']:.4f}
- Test accuracy: not reported because the provided ReClor test split has no labels.

The base model is gated. Request access to `meta-llama/Meta-Llama-3-8B` and accept its license before loading this adapter.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "meta-llama/Meta-Llama-3-8B"
adapter_id = "{args.repo_id}"
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)
```
"""
        (upload_dir / "README.md").write_text(readme)
        try:
            api.create_repo(repo_id=args.repo_id, repo_type="model", private=True, exist_ok=False)
        except Exception:
            raise SystemExit("Could not create the new private repo; it may already exist or the token lacks write access.")
        try:
            api.upload_folder(
                folder_path=str(upload_dir),
                repo_id=args.repo_id,
                repo_type="model",
                commit_message="Upload selected CREST ReClor stage 3 DPO adapter",
                token=token,
            )
        except Exception:
            raise SystemExit("HF upload failed; the token was not displayed or saved.")

    print(f"Uploaded selected adapter to https://huggingface.co/{args.repo_id} (private)")


if __name__ == "__main__":
    main()
