"""
Reads the per-example MAPS score array saved by identify_heads.py 
and applies the p% threshold (the Per-Prompt-Style criterion) to 
produce the final lexical task head list for one (prompt_type, template) run.
"""
import os
import argparse
import pickle
import numpy as np


def maps_pickle_path(save_root, model_name, d_name, prompt_type, template_key, k, n_match):
    return os.path.join(
        save_root, model_name, d_name, "Heads", "MAPS",
        f"{d_name}_MAPS_k{k}_nmatch{n_match}_{prompt_type}_{template_key}_correct.pkl",
    )


def load_maps_scores(pkl_path: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def threshold_heads(maps_scores: np.ndarray, threshold: float):
    """
    Args:
        maps_scores: (n_examples, n_layers, n_heads), binary per-prompt scores
        threshold: p — min fraction of prompts a head must match
    Returns:
        head_list: sorted list of (layer, head) tuples
        fraction_matched: (n_layers, n_heads) array — the actual match rate per head
    """
    fraction_matched = maps_scores.mean(axis=0)
    indices = np.argwhere(fraction_matched >= threshold)
    head_list = sorted((int(layer), int(head)) for layer, head in indices)
    return head_list, fraction_matched


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
        help="full or short model name; only the part after the last '/' is used for paths")
    parser.add_argument("--d_name", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["EP", "IP"])
    parser.add_argument("--template_key", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="output")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n_match", type=int, default=1)
    parser.add_argument("--threshold", "-p", type=float, default=0.1,
        help="p: minimum fraction of prompts a head must match to count as a lexical task head")
    args = parser.parse_args()

    model_short_name = args.model_name.split("/")[-1]
    pkl_path = maps_pickle_path(
        args.save_root, model_short_name, args.d_name,
        args.prompt_type, args.template_key, args.k, args.n_match,
    )
    maps_scores = load_maps_scores(pkl_path)
    head_list, fraction_matched = threshold_heads(maps_scores, args.threshold)

    print(f"Loaded {pkl_path}")
    print(f"{args.prompt_type} template {args.template_key}, p={args.threshold}: {len(head_list)} heads")
    for layer, head in head_list:
        print(f"  Layer {layer}, Head {head}: matched {fraction_matched[layer, head]:.1%} of prompts")

    save_dir = os.path.join(args.save_root, model_short_name, args.d_name, "Heads", "MAPS")
    save_path = os.path.join(
        save_dir,
        f"{args.d_name}_heads_p{args.threshold}_k{args.k}_nmatch{args.n_match}_{args.prompt_type}_{args.template_key}.pkl",
    )
    with open(save_path, "wb") as f:
        pickle.dump(
            {"head_list": head_list, "fraction_matched": fraction_matched, "threshold": args.threshold},
            f,
        )
    print(f"Saved head list to {save_path}")