# identify_heads.py
"""
Runs a HookedTransformer over the correct-only prompt subset 
(from behavior_variance.py JSON), caches per-head last-token activations 
via hook_z, projects to vocab space (apply_ln=False), and saves per-example 
MAPS scores to device.
"""
import os
import json
import pickle
import numpy as np
import torch
from transformer_lens import HookedTransformer
import argparse

from utils.build_prompts import (
    _load_dataset, create_instruction_prompts, create_few_shot_prompts,
)
from scoring import maps_scores_for_layer, expand_task_term_token_ids

torch.set_grad_enabled(False)


def load_behavior_correct_index(behavior_path: str, d_name: str, template_key: str) -> list[int]:
    with open(behavior_path, "r") as f:
        behavior = json.load(f)
    return behavior[d_name][str(template_key)]["correct_index"]


def identify_lexical_task_heads(
    model: HookedTransformer,
    prompts: list[str],           # ALREADY filtered to correct_index, same order
    task_term_token_ids: torch.Tensor,
    k: int = 20,
    n_match: int = 1,
    batch_size: int = 20,
) -> np.ndarray:
    """
    Returns: MAPS_scores of shape (n_examples, n_layers, n_heads), binary (0/1).
    """
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    n_examples = len(prompts)
    maps_scores = np.zeros((n_examples, n_layers, n_heads))

    for batch_start in range(0, n_examples, batch_size):
        batch_end = min(batch_start + batch_size, n_examples)
        batch_prompts = prompts[batch_start:batch_end]

        tokens = model.to_tokens(batch_prompts, padding_side="left")
        tokens = tokens.to(model.cfg.device)

        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name.endswith("hook_z"),
        )

        for layer in range(n_layers):
            layer_z = cache[f"blocks.{layer}.attn.hook_z"][:, -1, :, :]  # (batch, n_heads, d_head), LAST TOKEN
            W_O = model.W_O[layer]  # (n_heads, d_head, d_model)
            head_outputs = torch.einsum("bnh,nhm->bnm", layer_z, W_O)  # (batch, n_heads, d_model)
            # NOTE: apply_ln=False default
            head_projections = head_outputs @ model.W_U  # (batch, n_heads, d_vocab)

            layer_scores = maps_scores_for_layer(
                head_projections, task_term_token_ids, k=k, n_match=n_match
            )
            maps_scores[batch_start:batch_end, layer, :] = layer_scores.cpu().numpy()

        print(f"  processed examples {batch_start}-{batch_end}/{n_examples}")

    return maps_scores


def save_maps_scores(maps_scores: np.ndarray, save_root: str, model_name: str,
                      d_name: str, prompt_type: str, template_key, k: int, n_match: int):
    save_dir = os.path.join(save_root, model_name, d_name, "Heads", "MAPS")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"{d_name}_MAPS_k{k}_nmatch{n_match}_{prompt_type}_{template_key}_correct.pkl"
    )
    with open(save_path, "wb") as f:
        pickle.dump(maps_scores, f)
    print(f"Saved MAPS scores to {save_path}")
    return save_path

def build_correct_prompts(prompt_type, template_key, d_name, dataset_folder,
                           behavior_path, project_root):
    """Returns the list of prompt strings, filtered down to correct_index only."""
    correct_index = load_behavior_correct_index(behavior_path, d_name, template_key)

    if prompt_type == "EP":
        prompts, answers, _ = create_few_shot_prompts(
            d_name, n_shot=int(template_key), dataset_folder=dataset_folder,
            delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":",
        )
    elif prompt_type == "IP":
        with open(os.path.join(project_root, "datasets", "dataset_info", "instruction_dict.json")) as f:
            instruction_dict = json.load(f)
        dataset = _load_dataset(d_name, dataset_folder)
        prompts, answers = create_instruction_prompts(dataset, instruction_dict[d_name][str(template_key)])
    else:
        raise ValueError(f"prompt_type {prompt_type} not supported")

    correct_prompts = [prompts[i] for i in correct_index]
    print(f"{prompt_type} template {template_key}: {len(correct_prompts)}/{len(prompts)} correct prompts")
    return correct_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--d_name", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["EP", "IP"])
    parser.add_argument("--template_key", type=str, required=True,
        help="EP: n_shot as string e.g. '5'. IP: instruction index as string e.g. '2'.")
    parser.add_argument("--save_root", type=str, default="output")
    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--dataset_folder", type=str, default="datasets/abstractive")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n_match", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32",
        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_short_name = args.model_name.split("/")[-1]

    print("Loading model...")
    model = HookedTransformer.from_pretrained(args.model_name, device=device, dtype=dtype)

    with open(os.path.join(args.project_root, "datasets", "dataset_info", "task_relation_dict.json")) as f:
        task_relation_dict = json.load(f)
    task_term_token_ids = expand_task_term_token_ids(model, task_relation_dict[args.d_name])
    print(f"Task terms {task_relation_dict[args.d_name]} -> {len(task_term_token_ids)} token variants")

    behavior_file = "EP_vary_n_shot_behavior.json" if args.prompt_type == "EP" else "IP_vary_n_inst_behavior.json"
    behavior_path = os.path.join(args.save_root, model_short_name, "across_tasks", "Behavior", behavior_file)

    correct_prompts = build_correct_prompts(
        args.prompt_type, args.template_key, args.d_name, args.dataset_folder,
        behavior_path, args.project_root,
    )

    maps_scores = identify_lexical_task_heads(
        model, correct_prompts, task_term_token_ids,
        k=args.k, n_match=args.n_match, batch_size=args.batch_size,
    )
    save_maps_scores(
        maps_scores, save_root=args.save_root, model_name=model_short_name,
        d_name=args.d_name, prompt_type=args.prompt_type,
        template_key=args.template_key, k=args.k, n_match=args.n_match,
    )