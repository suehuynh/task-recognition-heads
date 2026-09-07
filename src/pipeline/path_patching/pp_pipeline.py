import os
import sys
import json
import argparse
import numpy as np
import torch
import pickle
from transformer_lens import ActivationCache, HookedTransformer, utilities
from transformer_lens.components import MLP, Embed, LayerNorm, Unembed
from transformer_lens.hook_points import HookPoint

from utils.build_prompts import create_few_shot_prompts, create_task_corrupt_prompts, check_correctness
from metrics import *
from path_patching import (
    get_model_specs_tl, find_earliest_receiver, _resolve_pos, patch_head_input,
    patch_or_freeze_head_vectors, get_path_patch_head_to_heads,
    get_path_patch_head_to_LTH_vocab,
)


from plot import plot_sender_head_effect

torch.set_grad_enabled(False)

def rank_heads(scores_tensor, threshold=None):
    """
    Ranks heads by their score from a 2D tensor of shape [layer, head].

    Args:
        scores_tensor (np.ndarray): A 2D numpy array where the value at
                                    scores_tensor[i, j] is the score for
                                    layer i and head j.

    Returns:
        list: A list of tuples (layer, head, score), sorted in descending
              order of the score.
    """
    # Get the dimensions of the tensor
    num_layers, num_heads = scores_tensor.shape

    # Create an empty list to store the (layer, head, score) tuples
    ranked_head_list = []

    # Iterate through each layer and head to build the list
    for layer in range(num_layers):
        for head in range(num_heads):
            score = scores_tensor[layer, head]
            # If a threshold is provided, only add the head if its score is greater than the threshold
            if threshold is None or score >= threshold:
                ranked_head_list.append((layer, head, score))
            
    ranked_head_list.sort(key=lambda item: item[2], reverse=True)

    return ranked_head_list

class TokDataset:
    """Minimal wrapper so a batch of prompts exposes the `.toks` attribute
    that get_path_patch_head_to_heads expects."""
    def __init__(self, toks):
        self.toks = toks


def convert_task_words_to_token_ids(model, task_words):
    """
    Convert a task's descriptive word list into a set of token IDs, trying
    several casing/spacing variants (leading space, capitalized, all-caps)
    since a word like "capital" and " capital" can be different tokens.
    Only strings that tokenize to exactly one token are kept -- this mirrors
    how the head-output projection can only ever "vote" for single tokens.

    Doing this ONCE up front, rather than decoding每 top-k token back to a
    string at score time, is the main speedup: membership becomes an
    integer tensor comparison instead of a per-token string operation.
    """
    token_ids = set()
    variants = lambda w: [w, " " + w, w.upper(), w.capitalize(), w.lower()]

    for word in task_words:
        for variant in variants(word):
            ids = model.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(ids[0])

    return torch.tensor(sorted(token_ids))

def load_receiver_list(save_root, model_name, d_name, prompt_type, prompt_index, k,
                        component_type="Relation", threshold=0.1):
    folder = os.path.join(
        save_root, model_name, d_name, "Heads", "MAPS", f"{component_type}_across_tasks"  # dropped _vary_k
    )
    fname = f"{d_name}_MAPS_{component_type}_heads_across_tasks_{prompt_type}_{prompt_index}_correct.pkl"
    path = os.path.join(folder, fname)

    if not os.path.exists(path):
        available = os.listdir(folder) if os.path.exists(folder) else []
        raise FileNotFoundError(f"{path} not found. Files present in {folder}: {available}")

    with open(path, "rb") as f:
        maps_scores = pickle.load(f)

    if k not in maps_scores:
        raise KeyError(f"k={k} not found; available keys: {list(maps_scores.keys())}")
    if d_name not in maps_scores[k]:
        raise KeyError(f"d_name={d_name!r} not found under k={k}; available: {list(maps_scores[k].keys())}")

    scores = maps_scores[k][d_name]  # (n_prompts, n_layers, n_heads), values in {0,1}
    per_head_fraction = scores.mean(axis=0)  # (n_layers, n_heads)

    layers, heads = np.where(per_head_fraction >= threshold)
    receiver_list = list(zip(layers.tolist(), heads.tolist()))

    print(f"loaded {len(receiver_list)} receivers at k={k}, threshold={threshold}: {receiver_list}")
    return receiver_list


def load_correct_indices(behavior_json_path, d_name, n_shot):
    """
    Load correct-example indices from behavior_variance.py's
    EP_vary_n_shot_behavior.json for a given task and n_shot.
    """
    with open(behavior_json_path) as f:
        result_dict = json.load(f)

    if d_name not in result_dict:
        raise KeyError(f"d_name={d_name!r} not found; available: {list(result_dict.keys())}")

    key = str(n_shot)
    if key not in result_dict[d_name]:
        raise KeyError(
            f"n_shot={n_shot} not found for {d_name}; available: {list(result_dict[d_name].keys())}"
        )

    return result_dict[d_name][key]["correct_index"]

## EXECUTION
if __name__ == "__main__":
    """
    Path patching: find which upstream heads/MLPs feed the shared few-shots
    lexical-task heads for a given task (the receiver set).
    """
    parser = argparse.ArgumentParser()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--save_root", type=str, default=os.path.join(SCRIPT_DIR, "output"))
    parser.add_argument("--model_name", type=str, required=True,
        help="model name, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--d_name", type=str, default="country-capital")
    parser.add_argument("--corrupt_d_name", type=str, default="present-past",
                        help="task the few-shot demos are drawn from")
    parser.add_argument("--k", type=int, required=True,
        help="which k (20 or 25) to take the lexical-task head MAPS scores from")
    parser.add_argument("--threshold", type=float, default=0.1,
        help="min fraction of prompts a head must fire on (in load_receiver_list) to count as a receiver")
    parser.add_argument("--pp_prompt_type", type=str, default="EP", choices=["EP"],
        help="prompt style used for the path-patching runs")
    parser.add_argument("--pp_prompt_index", type=int, default=10,
        help="EP n_shot count for the path-patching prompts (default matches --ep_index)")
    parser.add_argument("--receiver_input", type=str, nargs="+", default=["q"],
        choices=["q", "k", "v"],
        help="which input stream(s) of the receiver heads to path-patch into. "
             "Normally pass a single value per run (q, k, or v); the list form is "
             "for ad-hoc convenience and multiplies the sweep cost.")
    parser.add_argument("--metric", type=str, nargs="+", default=["l2_norm", "lprr"],
        choices=["l2_norm", "lprr"],
        help="scoring metric(s); each produces its own heatmap/tensor/ranking. "
             "l2_norm = relative delta L2 norm of the receiver q/k/v vector (noising). "
             "lprr = Lexical Probability Recovery Rate on the LTH vocab projection (denoising).")
    parser.add_argument("--task_relation_dict_path", type=str,
        default=os.path.join(SCRIPT_DIR, "datasets", "dataset_info", "task_relation_dict.json"),
        help="JSON mapping each task to its descriptive words (V_task); used by the lprr metric")
    parser.add_argument("--exp_size", type=int, default=50,
        help="number of model-correct prompts to path-patch over")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--include_mlp_receivers", action="store_true",
        help="also add (layer, -1) MLP receivers for each layer that has a shared head")
    parser.add_argument("--dataset", type=str, default=os.path.join(SCRIPT_DIR, "datasets", "abstractive"))
    parser.add_argument("--component_type", type=str, default="Relation")
    parser.add_argument("--behavior_json_path", type=str, default=None,
        help="path to EP_vary_n_shot_behavior.json from behavior_variance_tl.py "
             "(default: <save_root>/<model_name>/across_tasks/Behavior/EP_vary_n_shot_behavior.json)")

    args = parser.parse_args()

    # Resolve relative paths against this script's directory, not the current
    # working directory, so the script works regardless of where it's launched
    # from (repo root, fewshot_pp/, a SLURM job, ...). The canonical data tree is
    # fewshot_pp/output next to this file.
    def _rooted(p):
        return p if (p is None or os.path.isabs(p)) else os.path.join(SCRIPT_DIR, p)
    args.save_root = _rooted(args.save_root)
    args.dataset = _rooted(args.dataset)
    args.task_relation_dict_path = _rooted(args.task_relation_dict_path)
    args.behavior_json_path = _rooted(args.behavior_json_path)

    model_name_short = args.model_name.split("/")[-1]
    behavior_json_path = args.behavior_json_path or os.path.join(
        args.save_root, model_name_short, "across_tasks", "Behavior", "EP_vary_n_shot_behavior.json")

    print("model_name", args.model_name)
    print("d_name", args.d_name)
    print("pp_prompt_index", args.pp_prompt_index, "k", args.k)

    receiver_list = load_receiver_list(
        save_root=args.save_root, model_name=model_name_short, d_name=args.d_name,
        prompt_type=args.pp_prompt_type, prompt_index=args.pp_prompt_index, k=args.k,
        component_type=args.component_type, threshold=args.threshold,
    )

    if not receiver_list:
        raise ValueError(
            f"No lexical-task heads found for {args.d_name} at k={args.k}, threshold={args.threshold}. "
            "Re-run identify_heads_tl.py or lower --threshold."
        )
    print(f"receiver_list ({len(receiver_list)} heads): {receiver_list}")

    if args.include_mlp_receivers:
        raise NotImplementedError(
            "--include_mlp_receivers is not supported by get_path_patch_head_to_heads yet -- "
            "it only patches q/k/v inputs of attention heads."
        )

    # Load model
    model = HookedTransformer.from_pretrained(args.model_name)

    correct_index = load_correct_indices(behavior_json_path, args.d_name, args.pp_prompt_index)[:args.exp_size]
    print(f"using {len(correct_index)} model-correct examples")

    clean_prompts, clean_answers, _ = create_few_shot_prompts(d_name=args.d_name, n_shot=args.n_shot)
    corrupt_prompts, corrupt_answers, _ = create_task_corrupt_prompts(
            n_shot=args.n_shot,
            original_d_name=args.d_name,
            corrupt_d_name=args.corrupt_d_name,
            dataset_folder=args.dataset_folder,
        )

    # Filter to only the examples the model actually answers correctly
    clean_prompts = [clean_prompts[i] for i in correct_index]
    corrupt_prompts = [corrupt_prompts[i] for i in correct_index]

    # Tokenize clean/corrupt together so both batches share one padded length
    # get_path_patch_head_to_heads indexes new_cache/orig_cache positionally and
    # needs matching tensor shapes.
    n_clean = len(clean_prompts)
    all_toks = model.to_tokens(clean_prompts + corrupt_prompts, padding_side="left")
    clean_dataset = TokDataset(all_toks[:n_clean])
    corrupt_dataset = TokDataset(all_toks[n_clean:])

    z_filter = lambda name: name.endswith("z")
    _, clean_z_cache = model.run_with_cache(clean_dataset.toks, names_filter=z_filter, return_type=None)
    _, corrupt_z_cache = model.run_with_cache(corrupt_dataset.toks, names_filter=z_filter, return_type=None)

    task_token_ids = None
    if "lprr" in args.metric:
        with open(args.task_relation_dict_path) as f:
            task_relation_dict = json.load(f)
        if args.d_name not in task_relation_dict:
            raise KeyError(f"{args.d_name!r} not in {args.task_relation_dict_path}")
        task_token_ids = convert_task_words_to_token_ids(
            model, task_relation_dict[args.d_name]
        ).to(model.cfg.device)
        print(f"V_task ({len(task_token_ids)} token ids): {task_relation_dict[args.d_name]}")

    n_heads = get_model_specs_tl(model)["n_heads"]
    save_dir = os.path.join(args.save_root, model_name_short, args.d_name, "Heads", "causal_mediation", "path_patching")
    os.makedirs(save_dir, exist_ok=True)
    tag = f"{args.pp_prompt_type}{args.pp_prompt_index}_k{args.k}"

    for receiver_input in args.receiver_input:
        for metric in args.metric:
            print(f"\n=== path patching: sender -> LTH.{receiver_input}  |  metric={metric} ===")
            if metric == "l2_norm":
                results = get_path_patch_head_to_heads(
                    receiver_heads=receiver_list,
                    receiver_input=receiver_input,
                    model=model,
                    patching_metric=l2_norm_rel,
                    new_dataset=corrupt_dataset,
                    orig_dataset=clean_dataset,
                    new_cache=corrupt_z_cache,
                    orig_cache=clean_z_cache,
                )  # [layer, head]
            else:  # lprr
                results = get_path_patch_head_to_LTH_vocab(
                    receiver_heads=receiver_list,
                    receiver_input=receiver_input,
                    model=model,
                    task_token_ids=task_token_ids,
                    clean_dataset=clean_dataset,
                    corrupt_dataset=corrupt_dataset,
                    clean_z_cache=clean_z_cache,
                    corrupt_z_cache=corrupt_z_cache,
                )  # [layer, head]

            stem = f"sender_to_shared_lexical_heads_{tag}_{receiver_input}_{metric}"
            torch.save(results, os.path.join(save_dir, f"{stem}.pt"))

            plot_path = os.path.join(save_dir, f"{stem}_heatmap.html")
            plot_sender_head_effect(
                results, receiver_list, receiver_input, save_path=plot_path, metric=metric
            )

            heads_ranked = rank_heads(results.cpu().numpy(), threshold=None)
            heads_ranked = [
                (int(layer), int(head), float(score))
                for layer, head, score in heads_ranked
                if not np.isnan(score)
            ]

            ranked = {
                "meta": {
                    "model_name": model_name_short, "d_name": args.d_name, "k": args.k,
                    "threshold": args.threshold,
                    "pp_prompt_type": args.pp_prompt_type, "pp_prompt_index": args.pp_prompt_index,
                    "receiver_input": receiver_input, "metric": metric,
                    "n_examples": len(clean_prompts),
                    "receiver_list": receiver_list,
                },
                "heads_ranked": heads_ranked,
            }
            print(f"Top 15 upstream heads (sender -> {receiver_input}, {metric}):")
            for layer, head, score in heads_ranked[:15]:
                print(f"  L{layer}H{head}: {score:.4f}")

            ranked_path = os.path.join(save_dir, f"{stem}_ranked.json")
            with open(ranked_path, "w") as f:
                json.dump(ranked, f, indent=2)

    print("\nsaved to", save_dir)