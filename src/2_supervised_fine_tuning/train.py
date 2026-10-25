import os 
import json
import warnings
import numpy as np
from typing import Any, Dict, List, Union
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    class DataCollatorForCompletionOnlyLM:
        """Completion-only collator for TRL versions without the legacy class."""

        def __init__(self, response_template: Union[str, List[int]], *args, **kwargs):
            self.tokenizer = kwargs["tokenizer"]
            if isinstance(response_template, str):
                response_template = self.tokenizer.encode(
                    response_template, add_special_tokens=False
                )
            self.response_token_ids = response_template

        def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]):
            import torch

            features = [{key: value for key, value in row.items() if key != "labels"}
                        for row in examples]
            batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100
            marker = self.response_token_ids
            for row in range(len(examples)):
                sequence = batch["input_ids"][row].tolist()
                start = next((idx for idx in range(len(sequence) - len(marker) + 1)
                              if sequence[idx:idx + len(marker)] == marker), None)
                if start is None:
                    warnings.warn("Response template was not found; ignoring this example.")
                    labels[row, :] = -100
                else:
                    labels[row, :start + len(marker)] = -100
            batch["labels"] = labels
            return batch
import fire
import wandb

from utils.utils import generate_prompt_test, generate_option_string, generate_prompt_train

def main(
    base_model: str = "",
    train_data_path: str = None,
    output_dir: str = None,
    train_on_completion_only: bool = True,
    sanity_check: int = 0,
    lr = 5e-5,
    num_epochs = 8,
    CoT = True,
    logging_frequency = 5,
    per_device_train_batch_size = 2,
    simulated_batch = 8,
    key='sft_training_test',
    padding_side='left', 
    template_dict_path = None,
    max_length=2048,
):
    # WANDB_PROJECT
    os.environ["WANDB_PROJECT"] = key
    
    if type(train_on_completion_only) == str:
        train_on_completion_only = train_on_completion_only.lower() == "true"
        
    if type(CoT) == str:
        CoT = CoT.lower() == "true"
        
    if type(lr) == str:
        lr = float(lr)
    
    # cuda availability
    
    print(f"Cuda available: {torch.cuda.is_available()}")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha = 16,
        lora_dropout = 0.05,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    train_data = load_dataset("json", data_files=train_data_path, split='train')
    # For Test
    if int(sanity_check) > 0:
        train_data = train_data.select(range(sanity_check))
    
    template_dict = json.load(open(template_dict_path, "r"))
    
    instruction = template_dict['instruction']
    template = template_dict['first_input'] + '\n\n' + \
        template_dict['oneshot_input'] + template_dict['oneshot_output']
        
    ### Load model
    model = AutoModelForCausalLM.from_pretrained(base_model, 
                                                 device_map="auto", 
                                                 torch_dtype=torch.bfloat16,
                                                # attn_implementation='eager' # for gemma
                                                )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    
    os.makedirs(output_dir, exist_ok=True)

    # TRL 1.x consumes a text column directly. Building it once also avoids
    # passing a batch-shaped formatter to versions that invoke formatters on
    # individual examples.
    def add_text(batch):
        return {"text": generate_prompt_train(
            batch, tokenizer=tokenizer, instruction=instruction, template=template
        )}

    train_data = train_data.map(add_text, batched=True, desc="Formatting prompts")
    train_data = train_data.remove_columns(
        [column for column in train_data.column_names if column != "text"]
    )
    
    if CoT:
        response_template = "[Rationale]\n"
    else:
        response_template = "[Answer]\n"
        
    training_args=SFTConfig(
        output_dir=output_dir,
        num_train_epochs = num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=int(simulated_batch//per_device_train_batch_size),
        optim='adamw_torch',
        warmup_steps=100,
        eval_strategy="no",
        save_strategy="epoch", 
        learning_rate=lr,
        # fp16=True, 
        bf16=True,
        logging_steps=logging_frequency,
        push_to_hub=False,
        report_to='wandb', 
        run_name=key,
        max_length=int(max_length),
        dataset_text_field="text",
    )
    
    if train_on_completion_only:
        # encode without special tokens and convert it into List[int]
        response_template_tokenized =  tokenizer.encode(response_template, add_special_tokens=False)
        print(f"Response template tokenized: {response_template_tokenized}")
        collator = DataCollatorForCompletionOnlyLM(response_template_tokenized, tokenizer=tokenizer)
        
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_data,
            peft_config=lora_config,
            args=training_args,
            data_collator=collator,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_data,
            peft_config=lora_config,
            args=training_args,
        )
    
    train_result = trainer.train()
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump({"train_metrics": train_result.metrics, "log_history": trainer.state.log_history}, f, indent=2)
    trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(main)
