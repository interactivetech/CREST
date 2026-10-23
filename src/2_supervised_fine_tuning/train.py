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
    from transformers import DataCollatorForLanguageModeling

    class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
        """Compatibility collator for TRL releases that removed this export."""

        def __init__(self, response_template: Union[str, List[int]], *args, **kwargs):
            super().__init__(*args, mlm=False, **kwargs)
            if isinstance(response_template, str):
                response_template = self.tokenizer.encode(
                    response_template, add_special_tokens=False
                )
            self.response_token_ids = response_template

        def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]):
            batch = super().torch_call(examples)
            for row in range(len(examples)):
                start = None
                token_ids = self.response_token_ids
                for idx in np.where(batch["labels"][row] == token_ids[0])[0]:
                    if token_ids == batch["labels"][row][idx : idx + len(token_ids)].tolist():
                        start = idx
                if start is None:
                    warnings.warn("Response template was not found; ignoring this example.")
                    batch["labels"][row, :] = -100
                else:
                    batch["labels"][row, : start + len(token_ids)] = -100
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
    
    prompt_list = generate_prompt_train(train_data, tokenizer, instruction=instruction, template=template, test=False)
    os.makedirs(output_dir, exist_ok=True)
    json.dump(prompt_list, open(os.path.join(output_dir,"training_prompt_list.json"), "w"), indent=4)
    print(f"Prompt list saved at {os.path.join(output_dir, 'training_prompt_list.json')}")
    
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
        evaluation_strategy="no",
        save_strategy="epoch", 
        learning_rate=lr,
        # fp16=True, 
        bf16=True,
        logging_steps=logging_frequency,
        push_to_hub=False,
        report_to='wandb', 
        run_name=key,
        max_seq_length=2048,
    )
    
    if train_on_completion_only:
        # encode without special tokens and convert it into List[int]
        response_template_tokenized =  tokenizer.encode(response_template, add_special_tokens=False)
        print(f"Response template tokenized: {response_template_tokenized}")
        collator = DataCollatorForCompletionOnlyLM(response_template_tokenized, tokenizer=tokenizer)
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            peft_config=lora_config,
            args=training_args,
            formatting_func=lambda data: generate_prompt_train(
                data, 
                tokenizer=tokenizer,
                instruction=instruction,
                template=template,
            ),
            data_collator=collator,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            peft_config=lora_config,
            args=training_args,
            formatting_func=lambda data: generate_prompt_train(
                data, 
                tokenizer=tokenizer,
                instruction=instruction,
                template=template,
            ),
        )
    
    trainer.train()
    trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(main)
