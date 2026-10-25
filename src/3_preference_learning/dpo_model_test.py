"""Generate first- and second-pass predictions from a saved DPO adapter."""

import json
import os
from pathlib import Path
from typing import Optional

import fire
import torch
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import combine_query, extract_label


def _adapter_path(model_dir: str) -> str:
    model_path = Path(model_dir)
    candidates = [model_path, model_path / "default", model_path / "dpo_train"]
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file():
            return str(candidate)
    raise FileNotFoundError(f"No PEFT adapter_config.json found under {model_dir}")


def _metrics(results: list[dict]) -> Optional[dict]:
    scored = [row for row in results if row.get("label") not in (None, "", "None")]
    if not scored:
        return None
    predictions = [extract_label(row.get("output", "")) or "X" for row in scored]
    labels = [str(row["label"]).strip().upper() for row in scored]
    extracted = [prediction != "X" for prediction in predictions]
    correct = [prediction == label for prediction, label in zip(predictions, labels)]
    extracted_count = sum(extracted)
    return {
        "accuracy": sum(correct) / len(scored),
        "extracted_ratio": extracted_count / len(scored),
        "extracted_accuracy": (
            sum(is_correct for is_correct, was_extracted in zip(correct, extracted) if was_extracted)
            / extracted_count
            if extracted_count
            else 0.0
        ),
        "total": len(scored),
        "correct": sum(correct),
        "extracted": extracted_count,
    }


def main(
    base_model_name: str,
    model_dir: str,
    test_data: str,
    output_dir: str,
    test_batch_size: int = 8,
    test_template_path: str = None,
    test_max_new_tokens: int = 560,
    max_eval_samples: Optional[int] = None,
):
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"DPO adapter directory does not exist: {model_dir}")
    if not Path(test_data).is_file():
        raise FileNotFoundError(f"Evaluation data does not exist: {test_data}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    adapter_path = _adapter_path(model_dir)
    with open(test_template_path) as stream:
        template_dict = json.load(stream)
    instruction = template_dict["instruction"]
    template = (
        template_dict["first_input"]
        + "\n\n"
        + template_dict["oneshot_input"]
        + template_dict.get("final_output", "")
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    dataset = load_dataset("json", data_files=test_data, split="train")
    if "context" in dataset.column_names and "passage" not in dataset.column_names:
        dataset = dataset.rename_column("context", "passage")
    if max_eval_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_eval_samples)))
    if len(dataset) == 0:
        raise ValueError(f"No evaluation examples found in {test_data}")

    prompts = [
        combine_query(example, template=template, instruction=instruction, tokenizer=tokenizer)
        for example in dataset
    ]

    def collate_texts(texts):
        return tokenizer([text.strip() for text in texts], padding=True, return_tensors="pt")

    def generate(texts, description):
        outputs = []
        loader = DataLoader(texts, batch_size=test_batch_size, collate_fn=collate_texts)
        for batch in tqdm(loader, desc=description):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=test_max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            new_tokens = generated[:, input_ids.shape[1] :]
            outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs

    first_outputs = generate(prompts, "First prediction")
    first_results = [
        {
            "id_string": dataset["id_string"][index],
            "input": prompts[index],
            "label": dataset["label"][index] if "label" in dataset.column_names else None,
            "output": first_outputs[index],
        }
        for index in range(len(first_outputs))
    ]
    with open(output_path / "inference_result_data.json", "w") as stream:
        json.dump(first_results, stream, indent=2, ensure_ascii=False)

    reprompt_indices = [index for index, text in enumerate(first_outputs) if "[Answer]" not in text]
    reprompts = [
        prompts[index]
        + first_outputs[index]
        + "\n\n[Answer]\nTherefore, the answer is "
        for index in reprompt_indices
    ]
    second_outputs = generate(reprompts, "Second prediction") if reprompts else []
    second_results = [dict(row) for row in first_results]
    for index, output in zip(reprompt_indices, second_outputs):
        second_results[index]["input"] = reprompts[reprompt_indices.index(index)]
        second_results[index]["output"] = f"[Answer] The answer is {output}"

    with open(output_path / "inference_result_data-repredicted.json", "w") as stream:
        json.dump(second_results, stream, indent=2, ensure_ascii=False)

    first_metrics = _metrics(first_results)
    second_metrics = _metrics(second_results)
    summary = {
        "adapter_path": adapter_path,
        "num_examples": len(dataset),
        "scored_examples": sum(row.get("label") not in (None, "", "None") for row in first_results),
        "labels_available": any(row.get("label") not in (None, "", "None") for row in first_results),
        "first_pass_metrics": first_metrics,
        "second_pass_metrics": second_metrics,
    }
    with open(output_path / "evaluation_summary.json", "w") as stream:
        json.dump(summary, stream, indent=2)
    with open(output_path / "test_result.json", "w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire(main)
