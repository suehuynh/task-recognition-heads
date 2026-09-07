import torch as t
from torch import Tensor
from collections import defaultdict
from jaxtyping import Float, Int
from typing import Callable
from tqdm.auto import tqdm
import functools
from functools import partial
from itertools import product

from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

from metrics import task_vocab_prob_mass

# PATH PATCHING
# Step 1: Run and cache the head activation through 
# clean and corrupted prompts
# Step 2: Run clean prompt with sender node's output
# is patched from corrupted prompts and other heads/MLP
# activations are forced to freeze and cache receiver's
# inputs
# Step 3: Run through clean prompts with receiver's
# inputs are patched from previous run while upstream
# activations are frozen. Record the outputs of run 2.
# Step 4: Measure changes in metrics from orginal clean
# run with one from receiver-corrupt run.

device = "cuda" if t.cuda.is_available() else "cpu"

def find_earliest_receiver(receiver_list: list[tuple[int, int]]) -> tuple[int, int]:
    """
    Finds the computationally earliest component in a list of receivers.
    Order: Layer index first, then Attention Heads < MLP within a layer.
    """

    def compare_receivers(item1: tuple[int, int], item2: tuple[int, int]) -> int:
        layer1, comp1 = item1
        layer2, comp2 = item2
        if layer1 < layer2:
            return -1
        elif layer1 > layer2:
            return 1
        else:
            if comp1 >= 0 and comp2 == -1:
                return -1  # Attn < MLP
            elif comp1 == -1 and comp2 >= 0:
                return 1  # MLP > Attn
            else:
                return 0  # Order between heads doesn't matter

    if not receiver_list:
        raise ValueError("receiver_list cannot be empty for find_earliest_receiver")
    return sorted(receiver_list, key=functools.cmp_to_key(compare_receivers))[0]


def _resolve_pos(pos: int, seq_len: int) -> int:
    return pos if pos >= 0 else seq_len + pos

def patch_head_input(
    orig_activation: Float[Tensor, "batch pos head_idx d_head"],
    hook: HookPoint,
    patched_cache: ActivationCache,
    head_list: list[tuple[int, int]],
) -> Float[Tensor, "batch pos head_idx d_head"]:
    """
    Function which can patch any combination of heads in layers,
    according to the heads in head_list.
    """
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
    return orig_activation

def patch_or_freeze_head_vectors(
    orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: tuple[int, int],
) -> Float[Tensor, "batch pos head_index d_head"]:
    """
    Freeze all head outputs (i.e. set them to their values in orig_cache), 
    except for head_to_patch (if it's in this layer) which we patch with the
    value from new_cache.

    head_to_patch: tuple of (layer, head)
    """
    # Setting using ..., otherwise changing orig_head_vector will edit cache value too
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector

def _remap_receivers_for_input(
    model: HookedTransformer,
    receiver_heads: list[tuple[int, int]],
    receiver_input: str,
) -> list[tuple[int, int]]:
    """
    receiver_heads is in query/z-head space (0..n_heads-1). Under grouped-query
    attention, hook_k / hook_v are indexed by KV-head (fewer heads), so remap
    the head ids into KV-head space when the receiver input is k or v.
    hook_q keeps the raw query-head id.
    """
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", None) or model.cfg.n_heads
    if receiver_input in ("k", "v") and n_kv_heads != model.cfg.n_heads:
        group_size = model.cfg.n_heads // n_kv_heads
        return [(layer, head // group_size) for layer, head in receiver_heads]
    return list(receiver_heads)


def get_path_patch_head_to_heads(
    receiver_heads: list[tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset,
    orig_dataset,
    new_cache: ActivationCache | None = None,
    orig_cache: ActivationCache | None = None,
) -> Float[Tensor, "layer head"]:
    """
    Performs path patching (see algorithm at the top), with:

        sender head = each head above the LTHs, loop through one at a time
        receiver node = input to a set of LTHs

    The receiver node is specified by receiver_heads and receiver_input, for example if
    receiver_input = "v" and receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)], we're doing path
    patching from each head to the value inputs of the LTHs.

    Returns:
        tensor of metric values for every possible sender head
    """
    model.reset_hooks()

    assert receiver_input in ("k", "q", "v")
    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    # Under grouped-query attention, hook_k/hook_v are indexed by KV-head (fewer
    # heads than hook_q/hook_z). receiver_heads is in query/z-head space (0..n_heads-1),
    # so remap into KV-head space before indexing k/v tensors with it.
    receiver_heads_for_patch = _remap_receivers_for_input(model, receiver_heads, receiver_input)

    results = t.zeros(max(receiver_layers), model.cfg.n_heads, device=device, dtype=t.float32)

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we
    # only cache the things we need (in this case, just attn head outputs).
    z_name_filter = lambda name: name.endswith("z")
    if new_cache is None:
        _, new_cache = model.run_with_cache(new_dataset.toks, names_filter=z_name_filter, return_type=None)
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(orig_dataset.toks, names_filter=z_name_filter, return_type=None)

    # Clean baseline for the receiver's input.
    _, clean_receiver_cache = model.run_with_cache(
        orig_dataset.toks, names_filter=receiver_hook_names_filter, return_type=None
    )

    def _gather_receiver_vec(cache):
        return t.stack(
            [cache[utils.get_act_name(receiver_input, layer)][:, :, head]
             for layer, head in receiver_heads_for_patch],
            dim=0,
        )  # [n_receiver_heads, batch, pos, d_head]

    clean_receiver_vec = _gather_receiver_vec(clean_receiver_cache)

    for sender_layer, sender_head in tqdm(list(product(range(max(receiver_layers)), range(model.cfg.n_heads)))):
        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen.
        # This directly gives us the receiver's input under this single-sender
        # intervention -- exactly the "patched_head_output" patching_metric wants,
        # so there's no need for a further step 3 run through the rest of the model.

        model.reset_hooks()
        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn, level=1)

        _, patched_cache = model.run_with_cache(
            orig_dataset.toks, names_filter=receiver_hook_names_filter, return_type=None
        )
        assert set(patched_cache.keys()) == set(receiver_hook_names)

        patched_receiver_vec = _gather_receiver_vec(patched_cache)

        # Save the results
        results[sender_layer, sender_head] = patching_metric(clean_receiver_vec, patched_receiver_vec)

    model.reset_hooks()
    return results


def _overwrite_receiver_input(
    orig_input: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    edge_by_layer: dict[int, list[tuple[int, Tensor]]],
) -> Float[Tensor, "batch pos head_index d_head"]:
    """
    overwrite the receiver heads' q/k/v in this layer with the values 
    captured in step 2 (the single restored sender -> receiver edge).
    """
    for head, vec in edge_by_layer.get(hook.layer(), []):
        orig_input[:, :, head] = vec
    return orig_input


def get_path_patch_head_to_LTH_vocab(
    receiver_heads: list[tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    task_token_ids: Int[Tensor, "n_task_tokens"],
    clean_dataset,
    corrupt_dataset,
    clean_z_cache: ActivationCache | None = None,
    corrupt_z_cache: ActivationCache | None = None,
) -> Float[Tensor, "layer head"]:
    """
    Path patching in the denoising direction, scored by LPRR (Lexical
    Probability Recovery Rate): how much restoring the single edge
    `sender_head -> LTH.W_{receiver_input}` recovers the LTH's verbalization of
    the task-descriptive vocabulary V_task.

        LPRR(s) = (P_patched(V_task) - P_corrupt(V_task))
                  / (P_clean(V_task) - P_corrupt(V_task))

    P_*(V_task) is the softmax-prob mass on `task_token_ids` in the logit lens
    of each LTH's own output at the last position, averaged over prompts then
    over LTH heads.

    Returns a [layer, head] tensor of LPRR for every candidate sender head
    (senders range over all heads in layers 0 .. max(receiver_layer) - 1).
    NaN everywhere if the clean/corrupt runs don't separate V_task (denominator
    ~ 0).
    """
    model.reset_hooks()
    assert receiver_input in ("k", "q", "v")

    receiver_layers = set(next(zip(*receiver_heads)))
    input_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    input_hook_filter = lambda name: name in input_hook_names
    z_hook_names = [utils.get_act_name("z", layer) for layer in receiver_layers]
    z_hook_filter = lambda name: name in z_hook_names
    all_z_filter = lambda name: name.endswith("z")

    # k/v are indexed in KV-head space; q/z stay in query-head space.
    receiver_heads_for_input = _remap_receivers_for_input(model, receiver_heads, receiver_input)

    # ========== Baselines ==========
    if clean_z_cache is None:
        _, clean_z_cache = model.run_with_cache(clean_dataset.toks, names_filter=all_z_filter, return_type=None)
    if corrupt_z_cache is None:
        _, corrupt_z_cache = model.run_with_cache(corrupt_dataset.toks, names_filter=all_z_filter, return_type=None)

    def _lth_vocab_mass(z_cache: ActivationCache) -> Tensor:
        """mean-over-prompts, mean-over-LTH P(V_task) from a hook_z cache -> scalar."""
        per_head = []
        for layer, head in receiver_heads:  # original query-head indexing for z / W_O
            z_slice = z_cache[utils.get_act_name("z", layer)][:, -1, head]  # [batch, d_head]
            head_out = z_slice @ model.W_O[layer, head]                     # [batch, d_model]
            per_head.append(task_vocab_prob_mass(head_out, model, task_token_ids).mean())
        return t.stack(per_head).mean()

    p_clean = _lth_vocab_mass(clean_z_cache)
    p_corrupt = _lth_vocab_mass(corrupt_z_cache)
    denom = (p_clean - p_corrupt).item()

    results = t.full((max(receiver_layers), model.cfg.n_heads), float("nan"), device=device, dtype=t.float32)
    if abs(denom) < 1e-6:
        print(f"[LPRR] WARNING: P_clean ({p_clean.item():.4g}) and P_corrupt "
              f"({p_corrupt.item():.4g}) don't separate V_task; returning NaN.")
        return results

    for sender_layer, sender_head in tqdm(list(product(range(max(receiver_layers)), range(model.cfg.n_heads)))):
        # ---- Run B: isolate the sender -> LTH.{input} edge ----
        # corrupt input, every head frozen to its corrupt value except the one
        # sender head which is restored to its clean value.
        model.reset_hooks()
        model.add_hook(
            all_z_filter,
            partial(
                patch_or_freeze_head_vectors,
                new_cache=clean_z_cache,
                orig_cache=corrupt_z_cache,
                head_to_patch=(sender_layer, sender_head),
            ),
            level=1,
        )
        _, run_b_cache = model.run_with_cache(
            corrupt_dataset.toks, names_filter=input_hook_filter, return_type=None
        )

        edge_by_layer: dict[int, list[tuple[int, Tensor]]] = defaultdict(list)
        for (layer, head_for_input) in receiver_heads_for_input:
            vec = run_b_cache[utils.get_act_name(receiver_input, layer)][:, :, head_for_input]
            edge_by_layer[layer].append((head_for_input, vec))

        # ---- Run C: restore only that edge into the fully-corrupt run ----
        model.reset_hooks()
        model.add_hook(
            input_hook_filter,
            partial(_overwrite_receiver_input, edge_by_layer=edge_by_layer),
            level=1,
        )
        _, run_c_cache = model.run_with_cache(
            corrupt_dataset.toks, names_filter=z_hook_filter, return_type=None
        )

        p_patched = _lth_vocab_mass(run_c_cache)
        results[sender_layer, sender_head] = (p_patched - p_corrupt) / (p_clean - p_corrupt)

    model.reset_hooks()
    return results