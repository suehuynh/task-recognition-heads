import random
import torch as t
import tqdm
import argparse
import os
import json
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET_FOLDER = "datasets/abstractive"

def _load_dataset(d_name, dataset_folder=DATASET_FOLDER):
    with open(os.path.join(dataset_folder, f"{d_name}.json"), encoding="utf-8") as f:
        return json.load(f)

def create_instruction_prompts(dataset, instruction):
    assert "{input}" in instruction

    prompts, answers = [], []
    for data in dataset:
        prompts.append(instruction.format(input=data["input"]))
        answers.append(" " + data["output"])
    return prompts, answers

def create_few_shot_prompts(d_name, n_shot, 
                            dataset_folder=DATASET_FOLDER, 
                            delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":"):
    """
    Create clean few-shot prompts from a dataset.
    Format: n_shot demos + a query.
    """
    prompts, answers, query_index = [], [], []
    data = _load_dataset(d_name, dataset_folder)
    for i in range(0, len(data)):
        context = ""
        for j in range(i - n_shot, i):
            example = data[j]
            context += f"{q_bos}{example['input']}{qa_delimiter}{a_bos}{example['output']}{delimiter}"
        query = data[i]
        context += f"{q_bos}{query['input']}{qa_delimiter}"

        prompts.append(context)
        answers.append(a_bos + query["output"])
        query_index.append(i)
    return prompts, answers, query_index

def create_task_corrupt_prompts(original_d_name, corrupt_d_name, n_shot, 
                           dataset_folder=DATASET_FOLDER, 
                           delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":"):
    """
    Create corrupted few-shot prompts where examples are
    from another task and the query are still from original.
    """
    assert original_d_name != corrupt_d_name, "Original and Corrupt datasets must be different from each other."
    original_data = _load_dataset(original_d_name, dataset_folder)
    corrupt_data = _load_dataset(corrupt_d_name, dataset_folder)

    prompts, answers, query_index = [], [], []
    for i in range(len(original_data)):
        context = ""
        for j in range(i - n_shot, i):
            example = corrupt_data[j % len(corrupt_data)]
            context += f"{q_bos}{example['input']}{qa_delimiter}{a_bos}{example['output']}{delimiter}"
        context += f"{q_bos}{original_data[i]['input']}{qa_delimiter}"
        prompts.append(context)
        answers.append(a_bos + original_data[i]["output"])
        query_index.append(i)
    return prompts, answers, query_index

def check_correctness(
    model, prompts: list, answers: list,
    batch_size=10, return_pred_tokens=False, return_answer_tokens=False,
):
    """
    Check correctness and filter only correct generation.
    """
    correct_index = []
    pred_tokens = []

    tokenizer_out = model.tokenizer(
        prompts, padding=True, padding_side="left", return_tensors="pt",
    )
    prompt_tokens = tokenizer_out["input_ids"]
    attention_mask = tokenizer_out["attention_mask"]

    answer_tokens = model.tokenizer(
        answers, add_special_tokens=False, padding=True,
        padding_side="right", return_tensors="pt",
    )["input_ids"][:, 0]

    for i in range(0, len(prompts), batch_size):
        batch_prompt_tokens = prompt_tokens[i : i + batch_size]
        batch_attention_mask = attention_mask[i : i + batch_size]
        batch_answer_tokens = answer_tokens[i : i + batch_size]

        logits = model(
            batch_prompt_tokens,
            attention_mask=batch_attention_mask,
            return_type="logits",
        )
        batch_pred_tokens = logits[:, -1, :].argmax(dim=-1).cpu()

        if return_pred_tokens:
            pred_tokens.extend(batch_pred_tokens.tolist())

        batch_correct = t.where(batch_answer_tokens == batch_pred_tokens)[0].tolist()
        correct_index += [idx + i for idx in batch_correct]

    return_dict = {"correct_index": correct_index, "prompt_tokens": prompt_tokens}
    if return_pred_tokens:
        return_dict["pred_tokens"] = pred_tokens
    if return_answer_tokens:
        return_dict["answer_tokens"] = answer_tokens

    print(f"Accuracy: {len(correct_index)/len(prompts):.3f} ({len(correct_index)}/{len(prompts)})")
    return return_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_name", type=str, default="country-capital",
                        help="task the final query is drawn from")
    parser.add_argument("--corrupt_d_name", type=str, default="present-past",
                        help="task the few-shot demos are drawn from")
    parser.add_argument("--dataset_folder", type=str, default=DATASET_FOLDER)
    parser.add_argument("--n_shot", type=int, default=10)
    parser.add_argument("--n_preview", type=int, default=3)
    args = parser.parse_args()

    clean_prompts, clean_answers, _ = create_few_shot_prompts(d_name=args.d_name, n_shot=args.n_shot)
    corrupt_prompts, corrupt_answers, _ = create_task_corrupt_prompts(
        n_shot=args.n_shot,
        original_d_name=args.d_name,
        corrupt_d_name=args.corrupt_d_name,
        dataset_folder=args.dataset_folder,
    )
    print(f"Length of clean prompts: {len(clean_prompts)}")
    print(f"Length of corrupted prompts: {len(corrupt_prompts)}")
    for i in range(args.n_preview):
        print(f"[{i}] clean:   {clean_prompts[i]!r} -> {clean_answers[i]!r}")
        print(f"[{i}] corrupt: {corrupt_prompts[i]!r} -> {corrupt_answers[i]!r}")