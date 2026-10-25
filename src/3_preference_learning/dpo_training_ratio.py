"""Run CREST's mixed preference-learning rounds with TRL's current DPO API."""

import gc
import json
import os
from pathlib import Path
from typing import Dict, Optional

import fire
import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


def get_paired_dataset(
    data_dir: str,
    sanity_check: bool = False,
    max_datas: Optional[int] = None,
    shuffle: bool = False,
    num_proc: int = 24,
    seed: int = 42,
) -> Dataset:
    dataset = load_dataset("json", data_files=data_dir, split="train")
    if len(dataset) == 0:
        raise ValueError(f"Preference dataset is empty: {data_dir}")
    required = {"id_string", "query", "response_j", "response_k"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{data_dir} is missing columns: {sorted(missing)}")

    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))
    elif max_datas is not None:
        dataset = dataset.select(range(min(len(dataset), max_datas)))

    original_columns = dataset.column_names

    def to_dpo_fields(samples: Dict[str, list]) -> Dict[str, list]:
        return {
            "id_string": samples["id_string"],
            "prompt": samples["query"],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        to_dpo_fields,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


def cap_prompt_length(dataset: Dataset, tokenizer, max_prompt_length: int) -> Dataset:
    """Keep the tail of long prompts, matching the previous DPO truncation behavior."""
    if max_prompt_length < 2:
        raise ValueError("max_prompt_length must be at least 2")

    def trim_prompt(example: Dict[str, str]) -> Dict[str, str]:
        token_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        limit = max_prompt_length - 1  # reserve a BOS token added by DPOTrainer
        if len(token_ids) > limit:
            token_ids = token_ids[-limit:]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            while len(tokenizer(text)["input_ids"]) > max_prompt_length:
                token_ids = tokenizer(text, add_special_tokens=False)["input_ids"][1:]
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
            example["prompt"] = text
        return example

    return dataset.map(trim_prompt, num_proc=1, desc="Capping preference prompt length")


def main(
    base_model_name: str,
    sft_peft_path: str,
    working_dir: str,
    train_data_oq: str,
    train_data_cs: str,
    wandb_project_name: str = "crest_reclor_stage3",
    ratio_lambda: float = 0.6,
    max_steps: int = 3000,
    dpo_epochs: int = 1,
    learning_rate: float = 6e-6,
    dpo_beta: float = 0.1,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    simulated_batch_size: int = 8,
    seed: int = 42,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    restart_epoch: int = 0,
    save_steps: int = 3000,
    logging_steps: int = 10,
    gradient_checkpointing: bool = True,
    gradient_checkpointing_use_reentrant: bool = False,
    optimizer_type: str = "paged_adamw_32bit",
    warmup_steps: int = 100,
    weight_decay: float = 0.05,
    lr_scheduler_type: str = "cosine",
    sanity_check: bool = False,
    report_to: str = "wandb",
):
    if not Path(sft_peft_path).exists():
        raise FileNotFoundError(f"SFT adapter does not exist: {sft_peft_path}")
    if not 0.0 <= ratio_lambda <= 1.0:
        raise ValueError("ratio_lambda must be in [0, 1]")
    if per_device_train_batch_size < 1 or simulated_batch_size < per_device_train_batch_size:
        raise ValueError("simulated_batch_size must be >= per_device_train_batch_size >= 1")
    if simulated_batch_size % per_device_train_batch_size:
        raise ValueError("simulated_batch_size must be divisible by per_device_train_batch_size")
    if max_steps < 1 or dpo_epochs < 1:
        raise ValueError("max_steps and dpo_epochs must be positive")

    working_path = Path(working_dir)
    working_path.mkdir(parents=True, exist_ok=True)
    settings = {
        "base_model_name": base_model_name,
        "sft_peft_path": sft_peft_path,
        "train_data_oq": train_data_oq,
        "train_data_cs": train_data_cs,
        "ratio_lambda": ratio_lambda,
        "max_steps_per_round": max_steps,
        "dpo_rounds": dpo_epochs,
        "learning_rate": learning_rate,
        "dpo_beta": dpo_beta,
        "per_device_train_batch_size": per_device_train_batch_size,
        "simulated_batch_size": simulated_batch_size,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "sanity_check": sanity_check,
        "seed": seed,
    }
    with open(working_path / "setting.json", "w") as stream:
        json.dump(settings, stream, indent=2)

    set_seed(seed)
    os.environ["WANDB_PROJECT"] = wandb_project_name
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    max_datas = max_steps * simulated_batch_size
    max_datas_oq = int(max_datas * (1.0 - ratio_lambda))
    max_datas_cs = int(max_datas * ratio_lambda)
    accelerator = Accelerator()

    for epoch in range(restart_epoch + 1, dpo_epochs + 1):
        print(
            f"Round {epoch}/{dpo_epochs}: selecting at most {max_datas_oq} Z and "
            f"{max_datas_cs} consistency pairs (lambda={ratio_lambda})"
        )
        train_dataset_oq = get_paired_dataset(
            data_dir=train_data_oq,
            sanity_check=sanity_check,
            max_datas=max_datas_oq,
            shuffle=True,
            seed=seed,
        )
        train_dataset_cs = get_paired_dataset(
            data_dir=train_data_cs,
            sanity_check=sanity_check,
            max_datas=max_datas_cs,
            shuffle=True,
            seed=seed,
        )
        if max_datas_oq == 0:
            train_dataset = train_dataset_cs
        elif max_datas_cs == 0:
            train_dataset = train_dataset_oq
        else:
            train_dataset = concatenate_datasets([train_dataset_oq, train_dataset_cs])
        if len(train_dataset) == 0:
            raise ValueError("No preference pairs available for DPO training")
        train_dataset = cap_prompt_length(train_dataset, tokenizer, max_prompt_length)

        if epoch == 1:
            source_adapter = sft_peft_path
        else:
            source_adapter = str(working_path / f"epoch_{epoch - 1}")
        output_dir = working_path / f"epoch_{epoch}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading policy and reference initialization from {source_adapter}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            low_cpu_mem_usage=True,
            device_map={"": accelerator.local_process_index},
            dtype=torch.bfloat16,
        )
        base_model.config.use_cache = False
        model = PeftModel.from_pretrained(base_model, source_adapter, is_trainable=True)
        model.set_adapter("default")

        training_args = DPOConfig(
            output_dir=str(output_dir),
            max_steps=max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=simulated_batch_size // per_device_train_batch_size,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": gradient_checkpointing_use_reentrant},
            learning_rate=learning_rate,
            beta=dpo_beta,
            report_to=report_to,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            optim=optimizer_type,
            weight_decay=weight_decay,
            bf16=True,
            remove_unused_columns=False,
            run_name=f"{wandb_project_name}_round_{epoch}_of_{dpo_epochs}",
            seed=seed,
            max_length=max_length,
            save_steps=save_steps,
            logging_steps=logging_steps,
        )
        # TRL 1.x creates a frozen "ref" adapter copied from the active PEFT adapter.
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        if "ref" not in trainer.model.peft_config:
            raise RuntimeError("TRL did not create the frozen reference adapter")

        result = trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        metrics = {
            key: (float(value) if isinstance(value, (int, float)) else str(value))
            for key, value in result.metrics.items()
        }
        with open(output_dir / "training_metrics.json", "w") as stream:
            json.dump({"train_metrics": metrics, "round": epoch}, stream, indent=2)
        print(f"Saved round {epoch} policy adapter to {output_dir}")

        del trainer, model, base_model, train_dataset, train_dataset_oq, train_dataset_cs
        gc.collect()
        torch.cuda.empty_cache()
        if report_to == "wandb":
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
