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

from scoring import maps_scores_for_layer

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