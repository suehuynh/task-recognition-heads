"""
Pipeline: best-IP-template + fixed-EP-template -> identify heads for
each -> threshold each at p -> intersect -> shared lexical task heads.
"""
import os
import json
import argparse
import pickle
import numpy as np
import torch
from transformer_lens import HookedTransformer

from identify_heads import identify_lexical_task_heads, save_maps_scores, build_correct_prompts
from scoring import expand_task_term_token_ids
from select_heads_p import threshold_heads
from intersection import intersect_head_lists


def select_best_ip_template(behavior_path: str, d_name: str, n_templates: int = 5) -> str:
    with open(behavior_path, "r") as f:
        behavior = json.load(f)
    accuracies = [behavior[d_name][str(i)]["accuracy"] for i in range(n_templates)]
    best_index = int(np.argmax(accuracies))
    print(f"Best IP template for {d_name}: index {best_index} (accuracy {accuracies[best_index]:.3f})")
    return str(best_index)


def run_one_style(model, task_term_token_ids, prompt_type, template_key,
                   behavior_path, d_name, dataset_folder, project_root,
                   save_root, model_name, k, n_match, batch_size):
    # Reuses identify_heads.py's own prompt-building + correct-filtering
    correct_prompts = build_correct_prompts(
        prompt_type, template_key, d_name, dataset_folder, behavior_path, project_root,
    )
    maps_scores = identify_lexical_task_heads(
        model, correct_prompts, task_term_token_ids, k=k, n_match=n_match, batch_size=batch_size,
    )
    return save_maps_scores(
        maps_scores, save_root=save_root, model_name=model_name, d_name=d_name,
        prompt_type=prompt_type, template_key=template_key, k=k, n_match=n_match,
    )


def save_head_list(head_list, fraction_matched, save_root, model_name, d_name,
                    prompt_type, template_key, threshold, k, n_match):
    save_dir = os.path.join(save_root, model_name, d_name, "Heads", "MAPS")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(
        save_dir, f"{d_name}_heads_p{threshold}_k{k}_nmatch{n_match}_{prompt_type}_{template_key}.pkl",
    )
    with open(path, "wb") as f:
        pickle.dump({"head_list": head_list, "fraction_matched": fraction_matched, "threshold": threshold}, f)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--d_name", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="output")
    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--dataset_folder", type=str, default="datasets/abstractive")
    parser.add_argument("--ip_n_templates", type=int, default=5)
    parser.add_argument("--ep_template_key", type=str, default="5")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n_match", type=int, default=1)
    parser.add_argument("--threshold", "-p", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_short_name = args.model_name.split("/")[-1]

    behavior_root = os.path.join(args.save_root, model_short_name, "across_tasks", "Behavior")
    ip_behavior_path = os.path.join(behavior_root, "IP_vary_n_inst_behavior.json")
    ep_behavior_path = os.path.join(behavior_root, "EP_vary_n_shot_behavior.json")

    print("Loading model...")
    model = HookedTransformer.from_pretrained(args.model_name, device=device, dtype=dtype)
    print("Model loaded!")
    with open(os.path.join(args.project_root, "datasets", "dataset_info", "task_relation_dict.json")) as f:
        task_relation_dict = json.load(f)
    task_term_token_ids = expand_task_term_token_ids(model, task_relation_dict[args.d_name])
    print(f"Task terms {task_relation_dict[args.d_name]} -> {len(task_term_token_ids)} token variants")

    best_ip_template = select_best_ip_template(ip_behavior_path, args.d_name, args.ip_n_templates)

    ip_maps_path = run_one_style(
        model, task_term_token_ids, "IP", best_ip_template, ip_behavior_path,
        args.d_name, args.dataset_folder, args.project_root, args.save_root,
        model_short_name, args.k, args.n_match, args.batch_size,
    )
    ep_maps_path = run_one_style(
        model, task_term_token_ids, "EP", args.ep_template_key, ep_behavior_path,
        args.d_name, args.dataset_folder, args.project_root, args.save_root,
        model_short_name, args.k, args.n_match, args.batch_size,
    )

    with open(ip_maps_path, "rb") as f:
        ip_scores = pickle.load(f)
    with open(ep_maps_path, "rb") as f:
        ep_scores = pickle.load(f)

    ip_heads, ip_fraction = threshold_heads(ip_scores, args.threshold)
    ep_heads, ep_fraction = threshold_heads(ep_scores, args.threshold)
    print(f"IP heads (p={args.threshold}): {len(ip_heads)} -> {ip_heads}")
    print(f"EP heads (p={args.threshold}): {len(ep_heads)} -> {ep_heads}")

    save_head_list(ip_heads, ip_fraction, args.save_root, model_short_name, args.d_name,
                    "IP", best_ip_template, args.threshold, args.k, args.n_match)
    save_head_list(ep_heads, ep_fraction, args.save_root, model_short_name, args.d_name,
                    "EP", args.ep_template_key, args.threshold, args.k, args.n_match)

    shared_heads = intersect_head_lists(ip_heads, ep_heads)
    print(f"\nShared lexical task heads (p={args.threshold}): {len(shared_heads)}")
    print(shared_heads)

    result_dir = os.path.join(args.save_root, model_short_name, args.d_name, "Heads")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"shared_heads_p{args.threshold}_k{args.k}.pkl")
    with open(result_path, "wb") as f:
        pickle.dump({
            "shared_heads": shared_heads, "ip_heads": ip_heads, "ep_heads": ep_heads,
            "threshold": args.threshold, "k": args.k, "n_match": args.n_match,
            "best_ip_template": best_ip_template, "ep_template_key": args.ep_template_key,
        }, f)
    print(f"Saved to {result_path}")