import os
import json
import argparse
import numpy as np
import torch
import pickle
from transformer_lens import HookedTransformer

from utils.build_prompts import create_few_shot_prompts, create_task_corrupt_prompts
from metrics import l2_norm_rel, l2_norm_abs
from path_patching import (
    get_path_patch_head_to_heads,
    get_path_patch_head_to_LTH_vocab,
)
from plot import plot_sender_head_effect

torch.set_grad_enabled(False)

# .../src/pipeline/path_patching/pp_pipeline.py -> repo root is three levels up.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))


def rank_heads(scores_tensor, threshold=None):
    """
    Ranks heads by their score from a 2D tensor of shape [layer, head].

    Returns:
        list of (layer, head, score) tuples, sorted by descending score.
    """
    num_layers, num_heads = scores_tensor.shape
    ranked_head_list = []
    for layer in range(num_layers):
        for head in range(num_heads):
            score = scores_tensor[layer, head]
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
    """
    token_ids = set()
    variants = lambda w: [w, " " + w, w.upper(), w.capitalize(), w.lower()]

    for word in task_words:
        for variant in variants(word):
            ids = model.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(ids[0])

    return torch.tensor(sorted(token_ids))


def load_receiver_list(save_root, model_name, d_name, k, threshold,
                       receiver_key="shared_heads", receiver_list_path=None):
    """
    Load the receiver head set (the shared lexical-task heads) written by
    identify_heads/lth_pipeline.py at

        <save_root>/<model_name>/<d_name>/Heads/shared_heads_p{threshold}_k{k}.pkl

    The pickle is a dict with keys "shared_heads", "ip_heads", "ep_heads"
    (each a list of (layer, head) tuples). `receiver_key` picks which list to
    use as the receivers; the default is the IP-EP intersection.
    """
    path = receiver_list_path or os.path.join(
        save_root, model_name, d_name, "Heads", f"shared_heads_p{threshold}_k{k}.pkl"
    )

    if not os.path.exists(path):
        folder = os.path.dirname(path)
        available = os.listdir(folder) if os.path.isdir(folder) else []
        raise FileNotFoundError(
            f"{path} not found. Run identify_heads/lth_pipeline.py first "
            f"(p={threshold}, k={k}). Files present in {folder}: {available}"
        )

    with open(path, "rb") as f:
        data = pickle.load(f)

    if receiver_key not in data:
        raise KeyError(f"{receiver_key!r} not in {path}; available: {list(data)}")

    receiver_list = [tuple(h) for h in data[receiver_key]]
    print(f"loaded {len(receiver_list)} receivers ({receiver_key}) from {path}: {receiver_list}")
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
    Path patching: find which upstream heads feed the shared few-shot
    lexical-task heads for a given task (the receiver set).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
        help="model name, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--d_name", type=str, default="country-capital")
    parser.add_argument("--corrupt_d_name", type=str, default="present-past",
        help="task the few-shot demos are drawn from in the corrupt prompts")
    parser.add_argument("--project_root", type=str, default=PROJECT_ROOT,
        help="repo root; relative --save_root/--dataset_folder/... are resolved against it")
    parser.add_argument("--save_root", type=str, default=os.path.join(PROJECT_ROOT, "output"),
        help="output tree root, shared with the behavior/identify_heads stages")
    parser.add_argument("--dataset_folder", type=str,
        default=os.path.join(PROJECT_ROOT, "datasets", "abstractive"))
    parser.add_argument("--task_relation_dict_path", type=str,
        default=os.path.join(PROJECT_ROOT, "datasets", "dataset_info", "task_relation_dict.json"),
        help="JSON mapping each task to its descriptive words (V_task); used by the lprr metric")
    parser.add_argument("--behavior_json_path", type=str, default=None,
        help="path to EP_vary_n_shot_behavior.json from behavior_variance.py "
             "(default: <save_root>/<model_name>/across_tasks/Behavior/EP_vary_n_shot_behavior.json)")
    parser.add_argument("--k", type=int, required=True,
        help="which k the shared-heads file was written at (matches identify_heads --k)")
    parser.add_argument("--threshold", type=float, default=0.1,
        help="the p threshold the shared-heads file was written at (matches lth_pipeline --threshold)")
    parser.add_argument("--receiver_key", type=str, default="shared_heads",
        choices=["shared_heads", "ip_heads", "ep_heads"],
        help="which head list in the shared-heads pickle to use as the receivers")
    parser.add_argument("--receiver_list_path", type=str, default=None,
        help="explicit path to a shared-heads pickle (overrides the derived path)")
    parser.add_argument("--pp_prompt_type", type=str, default="EP", choices=["EP"],
        help="prompt style used for the path-patching runs")
    parser.add_argument("--pp_prompt_index", type=int, default=10,
        help="EP n_shot count for the path-patching prompts; must be a key present "
             "in EP_vary_n_shot_behavior.json for this task")
    parser.add_argument("--receiver_input", type=str, nargs="+", default=["q"],
        choices=["q", "k", "v"],
        help="which input stream(s) of the receiver heads to path-patch into. "
             "Normally pass a single value per run (q, k, or v); the list form "
             "multiplies the sweep cost.")
    parser.add_argument("--metric", type=str, nargs="+", default=["l2_norm_rel", "lprr"],
        choices=["l2_norm_rel", "l2_norm_abs", "lprr"],
        help="scoring metric(s); each produces its own heatmap/tensor/ranking. "
             "l2_norm_rel = relative delta L2 norm of the receiver q/k/v vector (noising). "
             "lprr = Lexical Probability Recovery Rate on the LTH vocab projection (denoising).")
    parser.add_argument("--exp_size", type=int, default=50,
        help="number of model-correct prompts to path-patch over")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    # Resolve relative paths against the repo root, not the current working
    # directory, so the script works regardless of where it's launched from.
    def _rooted(p):
        return p if (p is None or os.path.isabs(p)) else os.path.join(args.project_root, p)
    args.save_root = _rooted(args.save_root)
    args.dataset_folder = _rooted(args.dataset_folder)
    args.task_relation_dict_path = _rooted(args.task_relation_dict_path)
    args.behavior_json_path = _rooted(args.behavior_json_path)
    args.receiver_list_path = _rooted(args.receiver_list_path)

    model_name_short = args.model_name.split("/")[-1]
    behavior_json_path = args.behavior_json_path or os.path.join(
        args.save_root, model_name_short, "across_tasks", "Behavior", "EP_vary_n_shot_behavior.json")

    print("model_name", args.model_name)
    print("d_name", args.d_name)
    print("pp_prompt_index", args.pp_prompt_index, "k", args.k)
    print("save_root", args.save_root)

    receiver_list = load_receiver_list(
        save_root=args.save_root, model_name=model_name_short, d_name=args.d_name,
        k=args.k, threshold=args.threshold, receiver_key=args.receiver_key,
        receiver_list_path=args.receiver_list_path,
    )

    if not receiver_list:
        raise ValueError(
            f"No {args.receiver_key} found for {args.d_name} at k={args.k}, p={args.threshold}. "
            "Re-run identify_heads/lth_pipeline.py or lower --threshold."
        )
    print(f"receiver_list ({len(receiver_list)} heads): {receiver_list}")

    # Load model
    model = HookedTransformer.from_pretrained(args.model_name)

    correct_index = load_correct_indices(behavior_json_path, args.d_name, args.pp_prompt_index)[:args.exp_size]
    print(f"using {len(correct_index)} model-correct examples")

    n_shot = args.pp_prompt_index
    clean_prompts, clean_answers, _ = create_few_shot_prompts(
        d_name=args.d_name, n_shot=n_shot, dataset_folder=args.dataset_folder,
    )
    corrupt_prompts, corrupt_answers, _ = create_task_corrupt_prompts(
        original_d_name=args.d_name,
        corrupt_d_name=args.corrupt_d_name,
        n_shot=n_shot,
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

    save_dir = os.path.join(args.save_root, model_name_short, args.d_name, "Heads", "causal_mediation", "path_patching")
    os.makedirs(save_dir, exist_ok=True)
    tag = f"{args.pp_prompt_type}{args.pp_prompt_index}_k{args.k}"

    for receiver_input in args.receiver_input:
        for metric in args.metric:
            print(f"\n=== path patching: sender -> LTH.{receiver_input}  |  metric={metric} ===")
            if metric == "l2_norm_rel":
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
            elif metric == "l2_norm_abs":
                results = get_path_patch_head_to_heads(
                    receiver_heads=receiver_list,
                    receiver_input=receiver_input,
                    model=model,
                    patching_metric=l2_norm_abs,
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
