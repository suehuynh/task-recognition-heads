"""
Path patching into the EARLIEST identified LTH's q/k/v input, from every head
above it. One receiver only, so every candidate sender is causally reachable
by construction -- no aggregation, no masking.

direction="noising" (default): clean baseline, sender's z swapped to corrupt.
direction="denoising": corrupt baseline, sender's z restored to clean.

Metrics: l2_norm_rel / l2_norm_abs (receiver's raw q/k/v change) and lprr
(recovery/degradation of the receiver's own task-vocabulary verbalization).

Usage (run from src/pipeline/path_patching, PYTHONPATH=<repo>/src):
    python earliest_lth_patch.py --model_name meta-llama/Llama-3.2-1B-Instruct \
        --d_name country-capital --k 20 --threshold 0.2 --pp_prompt_index 10
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils

from utils.build_prompts import create_few_shot_prompts, create_task_corrupt_prompts
from metrics import l2_norm_abs, l2_norm_rel, task_vocab_prob_mass
from path_patching import (
    _overwrite_receiver_input,
    _remap_receivers_for_input,
    device,
    find_earliest_receiver,
    patch_or_freeze_head_vectors,
)
from pp_pipeline import (
    PROJECT_ROOT,
    TokDataset,
    convert_task_words_to_token_ids,
    load_correct_indices,
    load_receiver_list,
    rank_heads,
)
from plot import plot_sender_head_effect

Direction = Literal["noising", "denoising"]
Metric = Literal["l2_norm_rel", "l2_norm_abs", "lprr"]

_L2_FNS = {"l2_norm_rel": l2_norm_rel, "l2_norm_abs": l2_norm_abs}


@dataclass
class Context:
    model: HookedTransformer
    receiver_layer: int
    receiver_head: int
    clean_dataset: TokDataset
    corrupt_dataset: TokDataset
    clean_z_cache: dict
    corrupt_z_cache: dict
    task_token_ids: Tensor | None


def load_context(args: argparse.Namespace) -> Context:
    model_short = args.model_name.split("/")[-1]
    receiver_list = load_receiver_list(
        save_root=args.save_root, model_name=model_short, d_name=args.d_name,
        k=args.k, threshold=args.threshold, receiver_key=args.receiver_key,
    )
    receiver_layer, receiver_head = find_earliest_receiver(receiver_list)
    print(f"earliest receiver: L{receiver_layer}H{receiver_head} "
          f"(of {len(receiver_list)} {args.receiver_key})")

    model = HookedTransformer.from_pretrained(args.model_name)

    task_token_ids = None
    if "lprr" in args.metric:
        with open(args.task_relation_dict_path) as f:
            task_relation_dict = json.load(f)
        task_token_ids = convert_task_words_to_token_ids(
            model, task_relation_dict[args.d_name]
        ).to(model.cfg.device)

    behavior_json_path = args.behavior_json_path or os.path.join(
        args.save_root, model_short, "across_tasks", "Behavior", "EP_vary_n_shot_behavior.json")
    correct_index = load_correct_indices(
        behavior_json_path, args.d_name, args.pp_prompt_index)[:args.exp_size]

    clean_prompts, _, _ = create_few_shot_prompts(
        d_name=args.d_name, n_shot=args.pp_prompt_index, dataset_folder=args.dataset_folder)
    corrupt_prompts, _, _ = create_task_corrupt_prompts(
        original_d_name=args.d_name, corrupt_d_name=args.corrupt_d_name,
        n_shot=args.pp_prompt_index, dataset_folder=args.dataset_folder)
    clean_prompts = [clean_prompts[i] for i in correct_index]
    corrupt_prompts = [corrupt_prompts[i] for i in correct_index]
    print(f"using {len(clean_prompts)} model-correct examples")

    n = len(clean_prompts)
    toks = model.to_tokens(clean_prompts + corrupt_prompts, padding_side="left")
    clean_dataset, corrupt_dataset = TokDataset(toks[:n]), TokDataset(toks[n:])

    z_filter = lambda name: name.endswith("z")
    _, clean_z_cache = model.run_with_cache(clean_dataset.toks, names_filter=z_filter, return_type=None)
    _, corrupt_z_cache = model.run_with_cache(corrupt_dataset.toks, names_filter=z_filter, return_type=None)

    return Context(model, receiver_layer, receiver_head, clean_dataset, corrupt_dataset,
                   clean_z_cache, corrupt_z_cache, task_token_ids)


def _vocab_mass(z_cache: dict, model: HookedTransformer, layer: int, head: int,
                 task_token_ids: Tensor) -> Tensor:
    """P(V_task) of the receiver's own output, mean over prompts. Scalar tensor."""
    z_slice = z_cache[tl_utils.get_act_name("z", layer)][:, -1, head]  # [batch, d_head]
    head_out = z_slice @ model.W_O[layer, head]                       # [batch, d_model]
    return task_vocab_prob_mass(head_out, model, task_token_ids).mean()


def l2_effect(ctx: Context, receiver_input: str, direction: Direction, metric: Metric) -> Tensor:
    """[sender_layer, sender_head] tensor of l2_norm_rel/abs on the receiver's own input."""
    model, layer, head = ctx.model, ctx.receiver_layer, ctx.receiver_head
    head_for_input = _remap_receivers_for_input(model, [(layer, head)], receiver_input)[0][1]
    hook_name = tl_utils.get_act_name(receiver_input, layer)
    hook_filter = lambda name: name == hook_name
    z_filter = lambda name: name.endswith("z")
    metric_fn = _L2_FNS[metric]

    if direction == "noising":
        run_toks, new_cache, orig_cache = ctx.clean_dataset.toks, ctx.corrupt_z_cache, ctx.clean_z_cache
    else:
        run_toks, new_cache, orig_cache = ctx.corrupt_dataset.toks, ctx.clean_z_cache, ctx.corrupt_z_cache

    _, baseline_cache = model.run_with_cache(run_toks, names_filter=hook_filter, return_type=None)
    baseline_vec = baseline_cache[hook_name][:, -1, head_for_input].unsqueeze(0)  # [1, batch, d_head]

    results = torch.zeros(layer, model.cfg.n_heads, device=device)
    for sender_layer in tqdm(range(layer), desc=f"{receiver_input}/{metric}/{direction}"):
        for sender_head in range(model.cfg.n_heads):
            model.reset_hooks()
            model.add_hook(z_filter, partial(
                patch_or_freeze_head_vectors, new_cache=new_cache, orig_cache=orig_cache,
                head_to_patch=(sender_layer, sender_head),
            ), level=1)
            _, patched_cache = model.run_with_cache(run_toks, names_filter=hook_filter, return_type=None)
            patched_vec = patched_cache[hook_name][:, -1, head_for_input].unsqueeze(0)
            results[sender_layer, sender_head] = metric_fn(baseline_vec, patched_vec)

    model.reset_hooks()
    return results


def lprr_effect(ctx: Context, receiver_input: str, direction: Direction) -> Tensor:
    """[sender_layer, sender_head] tensor of LPRR for the single earliest receiver."""
    model, layer, head = ctx.model, ctx.receiver_layer, ctx.receiver_head
    head_for_input = _remap_receivers_for_input(model, [(layer, head)], receiver_input)[0][1]
    input_hook = tl_utils.get_act_name(receiver_input, layer)
    z_hook = tl_utils.get_act_name("z", layer)
    z_filter = lambda name: name.endswith("z")

    p_clean = _vocab_mass(ctx.clean_z_cache, model, layer, head, ctx.task_token_ids)
    p_corrupt = _vocab_mass(ctx.corrupt_z_cache, model, layer, head, ctx.task_token_ids)
    denom = (p_clean - p_corrupt).item()

    results = torch.full((layer, model.cfg.n_heads), float("nan"), device=device)
    if abs(denom) < 1e-6:
        print(f"[LPRR] WARNING: P_clean ({p_clean.item():.4g}) and P_corrupt "
              f"({p_corrupt.item():.4g}) don't separate V_task; returning NaN.")
        return results

    if direction == "denoising":
        run_toks, new_cache, orig_cache = ctx.corrupt_dataset.toks, ctx.clean_z_cache, ctx.corrupt_z_cache
    else:
        run_toks, new_cache, orig_cache = ctx.clean_dataset.toks, ctx.corrupt_z_cache, ctx.clean_z_cache

    for sender_layer in tqdm(range(layer), desc=f"{receiver_input}/lprr/{direction}"):
        for sender_head in range(model.cfg.n_heads):
            # Run B: isolate the sender -> receiver.{receiver_input} edge.
            model.reset_hooks()
            model.add_hook(z_filter, partial(
                patch_or_freeze_head_vectors, new_cache=new_cache, orig_cache=orig_cache,
                head_to_patch=(sender_layer, sender_head),
            ), level=1)
            _, run_b_cache = model.run_with_cache(
                run_toks, names_filter=lambda name: name == input_hook, return_type=None)
            edge_vec = run_b_cache[input_hook][:, :, head_for_input]  # [batch, pos, d_head]

            # Run C: restore only that edge into the otherwise-unpatched run.
            model.reset_hooks()
            model.add_hook(input_hook, partial(
                _overwrite_receiver_input, edge_by_layer={layer: [(head_for_input, edge_vec)]},
            ), level=1)
            _, run_c_cache = model.run_with_cache(
                run_toks, names_filter=lambda name: name == z_hook, return_type=None)

            p_patched = _vocab_mass(run_c_cache, model, layer, head, ctx.task_token_ids)
            numer = (p_patched - p_corrupt) if direction == "denoising" else (p_clean - p_patched)
            results[sender_layer, sender_head] = numer / denom

    model.reset_hooks()
    return results


def save_result(results: Tensor, ctx: Context, save_dir: str, stem: str,
                 receiver_input: str, metric: Metric, direction: Direction) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(results, os.path.join(save_dir, f"{stem}.pt"))

    # plot.py has a distinct "lprr_noising" label; l2 metrics read the same
    # either direction (the delta itself, not a recovery/degradation framing).
    metric_label = "lprr_noising" if (metric == "lprr" and direction == "noising") else metric
    plot_sender_head_effect(
        results, [(ctx.receiver_layer, ctx.receiver_head)], receiver_input,
        save_path=os.path.join(save_dir, f"{stem}_heatmap.html"), metric=metric_label,
    )

    ranked = [
        (int(l), int(h), float(s)) for l, h, s in rank_heads(results.cpu().numpy(), threshold=None)
        if not math.isnan(s)
    ]
    print(f"Top 10 senders ({stem}):")
    for l, h, s in ranked[:10]:
        print(f"  L{l}H{h}: {s:.4f}")
    with open(os.path.join(save_dir, f"{stem}_ranked.json"), "w") as f:
        json.dump({"receiver": [ctx.receiver_layer, ctx.receiver_head], "heads_ranked": ranked}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--d_name", type=str, default="country-capital")
    parser.add_argument("--corrupt_d_name", type=str, default="present-past")
    parser.add_argument("--project_root", type=str, default=PROJECT_ROOT)
    parser.add_argument("--save_root", type=str, default=os.path.join(PROJECT_ROOT, "output"))
    parser.add_argument("--dataset_folder", type=str, default=os.path.join(PROJECT_ROOT, "datasets", "abstractive"))
    parser.add_argument("--task_relation_dict_path", type=str,
        default=os.path.join(PROJECT_ROOT, "datasets", "dataset_info", "task_relation_dict.json"))
    parser.add_argument("--behavior_json_path", type=str, default=None)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--receiver_key", type=str, default="ep_heads",
        choices=["ep_heads", "ip_heads", "shared_heads"])
    parser.add_argument("--pp_prompt_index", type=int, default=10)
    parser.add_argument("--exp_size", type=int, default=50)
    parser.add_argument("--receiver_input", type=str, nargs="+", default=["q", "k", "v"], choices=["q", "k", "v"])
    parser.add_argument("--direction", type=str, default="noising", choices=["noising", "denoising"])
    parser.add_argument("--metric", type=str, nargs="+", default=["l2_norm_rel", "l2_norm_abs", "lprr"],
        choices=["l2_norm_rel", "l2_norm_abs", "lprr"])
    args = parser.parse_args()

    def rooted(path: str | None) -> str | None:
        return path if (path is None or os.path.isabs(path)) else os.path.join(args.project_root, path)
    args.save_root = rooted(args.save_root)
    args.dataset_folder = rooted(args.dataset_folder)
    args.task_relation_dict_path = rooted(args.task_relation_dict_path)
    args.behavior_json_path = rooted(args.behavior_json_path)

    torch.set_grad_enabled(False)
    ctx = load_context(args)

    model_short = args.model_name.split("/")[-1]
    save_dir = os.path.join(args.save_root, model_short, args.d_name, "Heads", "causal_mediation", "earliest_lth")
    tag = f"EP{args.pp_prompt_index}_k{args.k}_p{args.threshold}"
    receiver_tag = f"L{ctx.receiver_layer}H{ctx.receiver_head}"

    for receiver_input in args.receiver_input:
        for metric in args.metric:
            print(f"\n=== {receiver_tag}.{receiver_input}  metric={metric}  direction={args.direction} ===")
            results = (
                l2_effect(ctx, receiver_input, args.direction, metric) if metric in _L2_FNS
                else lprr_effect(ctx, receiver_input, args.direction)
            )
            stem = f"{receiver_tag}_{tag}_{receiver_input}_{args.direction}_{metric}"
            save_result(results, ctx, save_dir, stem, receiver_input, metric, args.direction)

    print("\nsaved to", save_dir)


if __name__ == "__main__":
    main()
