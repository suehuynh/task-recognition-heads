# scoring.py
"""
Adapted from identify_heads_early_decode.py's calc_MAPS_score.
"""
import torch

def maps_score_binary(
    projection: torch.Tensor,   # (batch, d_vocab)
    task_term_token_ids: torch.Tensor,  # (n_options,)
    k: int = 20,
    n_match: int = 1,
) -> torch.Tensor:
    """
    Per-prompt binary lexical-task-head criterion:
    1 if at least `n_match` of the top-`k` decoded token indices for that
    example are in `task_term_token_ids`, else 0.

    Returns: FloatTensor of shape (batch,), each entry 0.0 or 1.0
    """
    task_term_token_ids = task_term_token_ids.to(projection.device)
    _, top_k_indices = torch.topk(projection, k, dim=-1)  # (batch, k)

    # (batch, k, 1) == (1, 1, n_options) -> (batch, k, n_options)
    matches = top_k_indices.unsqueeze(-1) == task_term_token_ids.view(1, 1, -1)
    matched_per_position = matches.any(dim=-1)          # (batch, k)
    n_matches_per_example = matched_per_position.sum(dim=-1)  # (batch,)

    return (n_matches_per_example >= n_match).float()


def maps_scores_for_layer(
    head_outputs_projected: torch.Tensor,  # (batch, n_heads, d_vocab)
    task_term_token_ids: torch.Tensor,
    k: int = 20,
    n_match: int = 1,
) -> torch.Tensor:
    """Vectorized across heads in one layer. Returns (batch, n_heads)."""
    n_heads = head_outputs_projected.shape[1]
    scores = torch.zeros(head_outputs_projected.shape[0], n_heads)
    for h in range(n_heads):
        scores[:, h] = maps_score_binary(
            head_outputs_projected[:, h, :], task_term_token_ids, k=k, n_match=n_match
        )
    return scores