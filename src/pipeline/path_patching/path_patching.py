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
    collapse_receivers: bool = True,
) -> Float[Tensor, "layer head"]:
    """
    Performs path patching (see algorithm at the top), with:

        sender head = each head above the LTHs, loop through one at a time
        receiver node = input to a set of LTHs

    The receiver node is specified by receiver_heads and receiver_input, for example if
    receiver_input = "v" and receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)], we're doing path
    patching from each head to the value inputs of the LTHs.

    collapse_receivers=True (default): returns [sender_layer, sender_head] --
        one score per sender, aggregated only over the receivers that sender
        can *causally reach*, i.e. receivers strictly downstream of the
        sender's own layer (layer > sender_layer). A sender in the same layer
        as a receiver, or a later one, has zero causal path to it (that
        receiver's q/k/v is computed from resid_pre, before its own layer's
        attention runs), so including it would dilute the aggregate with a
        structural zero rather than a measured non-effect. A sender_layer with
        no reachable receiver is skipped (left at its initialised 0) -- this
        cannot currently happen since senders only range up to
        max(receiver_layer) - 1, but the guard is kept for safety if that
        range is ever widened.
    collapse_receivers=False: keeps one score per receiver head, unmasked
        (each receiver's own entry is already correct on its own -- for an
        unreachable sender it is exactly 0, not diluted, since nothing else is
        averaged into it); returns [n_receiver_heads, sender_layer,
        sender_head], in the same order as `receiver_heads`.

    Same forward passes either way -- this only changes how the per-receiver
    values (always computed) are combined into `results`.

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
    # Layer of each receiver, same order as receiver_heads/receiver_heads_for_patch --
    # used to mask which receivers a given sender_layer can causally reach.
    receiver_layers_by_index = [layer for layer, _ in receiver_heads]

    if collapse_receivers:
        results = t.zeros(max(receiver_layers), model.cfg.n_heads, device=device, dtype=t.float32)
    else:
        results = t.zeros(len(receiver_heads), max(receiver_layers), model.cfg.n_heads,
                           device=device, dtype=t.float32)

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
        # Final (query) position only. Task recognition happens at the last
        # token; taking all positions would let the demo region -- where clean
        # and corrupt prompts differ simply because they are different
        # sentences -- dominate the norm.
        return t.stack(
            [cache[utils.get_act_name(receiver_input, layer)][:, -1, head]
             for layer, head in receiver_heads_for_patch],
            dim=0,
        )  # [n_receiver_heads, batch, d_head]

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

        # Per-receiver score, always -- reduction/masking happens here, not
        # inside the metric, so it can be aware of which receivers this
        # particular sender_layer can causally reach.
        per_receiver_score = patching_metric(clean_receiver_vec, patched_receiver_vec, reduce=False)  # [n_receivers]

        if collapse_receivers:
            reachable = [layer > sender_layer for layer in receiver_layers_by_index]
            if not any(reachable):
                continue  # no receiver is downstream of this sender; leave as 0
            mask = t.tensor(reachable, device=device)
            results[sender_layer, sender_head] = per_receiver_score[mask].mean()
        else:
            results[:, sender_layer, sender_head] = per_receiver_score

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
    collapse_receivers: bool = True,
    direction: str = "denoising",
) -> Float[Tensor, "layer head"]:
    """
    Path patching scored by LPRR (Lexical Probability Recovery Rate): how much
    the single edge `sender_head -> LTH.W_{receiver_input}` accounts for the
    LTH's verbalization of the task-descriptive vocabulary V_task.

    P_*(V_task) is the softmax-prob mass on `task_token_ids` in the logit lens
    of each LTH's own output at the last position, averaged over prompts, and
    (when collapse_receivers=True) over LTH heads too. Both directions share
    the same denominator (P_clean - P_corrupt), so the masking and instability
    diagnostics below apply identically to either.

    direction="denoising" (default, unchanged behaviour): corrupt input,
        restore just this one sender->receiver.{q,k,v} edge, everything else
        stays corrupt. Scores how much that recovers P(V_task) toward clean:

            LPRR(s) = (P_patched(V_task) - P_corrupt(V_task))
                      / (P_clean(V_task) - P_corrupt(V_task))

        0 = this sender explains none of the clean->corrupt drop; 1 = it alone
        explains all of it.
    direction="noising": the mirror image -- clean input, corrupt just this
        one sender->receiver.{q,k,v} edge, everything else stays clean. Scores
        how much that degrades P(V_task) toward corrupt:

            LPRR_noising(s) = (P_clean(V_task) - P_patched(V_task))
                               / (P_clean(V_task) - P_corrupt(V_task))

        0 = knocking out this one sender has no effect on the clean run; 1 =
        it alone accounts for the entire clean->corrupt gap.

    collapse_receivers=True (default): returns [sender_layer, sender_head].
        For each sender_layer, LPRR is computed as a ratio of means using only
        the receivers that sender can *causally reach* -- strictly downstream
        of the sender's own layer (layer > sender_layer). A sender at or after
        a receiver's layer cannot affect it (that receiver's q/k/v is fixed
        before its own layer's attention runs), so its P_patched for that
        receiver is identical to P_corrupt; including it would dilute the
        aggregate with a structural non-effect rather than a measured one. A
        sender_layer with no reachable receiver is skipped (left as NaN) --
        cannot currently happen given the sender range, kept as a guard.
    collapse_receivers=False: keeps one LPRR per receiver head, unmasked (each
        receiver's own value is already correct on its own -- for an
        unreachable sender it is exactly 0/denom = 0, not diluted); returns
        [n_receiver_heads, sender_layer, sender_head], same receiver order as
        `receiver_heads`. A per-receiver denominator that is individually near
        zero shows up as a large/inf/NaN LPRR for that one receiver.

    Returns a tensor of LPRR for every candidate sender head (senders range
    over all heads in layers 0 .. max(receiver_layer) - 1).
    """
    model.reset_hooks()
    assert receiver_input in ("k", "q", "v")
    assert direction in ("denoising", "noising")

    receiver_layers = set(next(zip(*receiver_heads)))
    input_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    input_hook_filter = lambda name: name in input_hook_names
    z_hook_names = [utils.get_act_name("z", layer) for layer in receiver_layers]
    z_hook_filter = lambda name: name in z_hook_names
    all_z_filter = lambda name: name.endswith("z")

    # k/v are indexed in KV-head space; q/z stay in query-head space.
    receiver_heads_for_input = _remap_receivers_for_input(model, receiver_heads, receiver_input)
    # Layer of each receiver, same order as receiver_heads -- used to mask
    # which receivers a given sender_layer can causally reach.
    receiver_layers_by_index = [layer for layer, _ in receiver_heads]

    # ========== Baselines ==========
    if clean_z_cache is None:
        _, clean_z_cache = model.run_with_cache(clean_dataset.toks, names_filter=all_z_filter, return_type=None)
    if corrupt_z_cache is None:
        _, corrupt_z_cache = model.run_with_cache(corrupt_dataset.toks, names_filter=all_z_filter, return_type=None)

    def _lth_vocab_mass(z_cache: ActivationCache, collapse: bool = True) -> Tensor:
        """mean-over-prompts P(V_task) from a hook_z cache, per LTH head.

        collapse=True (default): also mean over LTH heads -> scalar tensor.
        collapse=False: keeps the receiver axis -> [n_receiver_heads] tensor,
            same order as `receiver_heads`.
        """
        per_head = []
        for layer, head in receiver_heads:  # original query-head indexing for z / W_O
            z_slice = z_cache[utils.get_act_name("z", layer)][:, -1, head]  # [batch, d_head]
            head_out = z_slice @ model.W_O[layer, head]                     # [batch, d_model]
            per_head.append(task_vocab_prob_mass(head_out, model, task_token_ids).mean())
        stacked = t.stack(per_head)  # [n_receiver_heads]
        return stacked.mean() if collapse else stacked

    # Always compute the per-receiver P_clean/P_corrupt -- cheap (no forward
    # pass, reuses the already-cached clean_z_cache/corrupt_z_cache) and
    # needed both for per-sender-layer masking and for the aggregate gate.
    p_clean_per = _lth_vocab_mass(clean_z_cache, collapse=False)      # [n_receiver_heads]
    p_corrupt_per = _lth_vocab_mass(corrupt_z_cache, collapse=False)  # [n_receiver_heads]
    per_receiver_denom = p_clean_per - p_corrupt_per

    p_clean = p_clean_per.mean()
    p_corrupt = p_corrupt_per.mean()
    denom = (p_clean - p_corrupt).item()

    if collapse_receivers:
        results = t.full((max(receiver_layers), model.cfg.n_heads), float("nan"), device=device, dtype=t.float32)
    else:
        results = t.full((len(receiver_heads), max(receiver_layers), model.cfg.n_heads),
                          float("nan"), device=device, dtype=t.float32)

    if abs(denom) < 1e-6:
        print(f"[LPRR] WARNING: P_clean ({p_clean.item():.4g}) and P_corrupt "
              f"({p_corrupt.item():.4g}) don't separate V_task; returning NaN.")
        return results

    unstable = [
        receiver_heads[i] for i in range(len(receiver_heads))
        if abs(per_receiver_denom[i].item()) < 1e-6
    ]
    if unstable:
        print(f"[LPRR] NOTE: these receivers individually don't separate V_task "
              f"(P_clean - P_corrupt near 0 -- will show as extreme/inf/NaN LPRR): {unstable}")

    # denoising: corrupt input, restore sender's *clean* value into an
    #     otherwise-corrupt run.
    # noising: clean input, inject sender's *corrupt* value into an
    #     otherwise-clean run. Exact mirror -- same two-run edge-isolation
    #     trick, with the run dataset and the freeze/patch caches swapped.
    if direction == "denoising":
        run_toks = corrupt_dataset.toks
        edge_new_cache, edge_orig_cache = clean_z_cache, corrupt_z_cache
    else:
        run_toks = clean_dataset.toks
        edge_new_cache, edge_orig_cache = corrupt_z_cache, clean_z_cache

    for sender_layer, sender_head in tqdm(list(product(range(max(receiver_layers)), range(model.cfg.n_heads)))):
        # ---- Run B: isolate the sender -> LTH.{input} edge ----
        model.reset_hooks()
        model.add_hook(
            all_z_filter,
            partial(
                patch_or_freeze_head_vectors,
                new_cache=edge_new_cache,
                orig_cache=edge_orig_cache,
                head_to_patch=(sender_layer, sender_head),
            ),
            level=1,
        )
        _, run_b_cache = model.run_with_cache(
            run_toks, names_filter=input_hook_filter, return_type=None
        )

        edge_by_layer: dict[int, list[tuple[int, Tensor]]] = defaultdict(list)
        for (layer, head_for_input) in receiver_heads_for_input:
            vec = run_b_cache[utils.get_act_name(receiver_input, layer)][:, :, head_for_input]
            edge_by_layer[layer].append((head_for_input, vec))

        # ---- Run C: restore only that edge into the otherwise-unpatched run ----
        model.reset_hooks()
        model.add_hook(
            input_hook_filter,
            partial(_overwrite_receiver_input, edge_by_layer=edge_by_layer),
            level=1,
        )
        _, run_c_cache = model.run_with_cache(
            run_toks, names_filter=z_hook_filter, return_type=None
        )

        p_patched_per = _lth_vocab_mass(run_c_cache, collapse=False)  # [n_receiver_heads]
        # denoising numerator moves p_patched toward p_clean (recovery);
        # noising numerator moves p_patched toward p_corrupt (degradation).
        numer_per = (p_patched_per - p_corrupt_per) if direction == "denoising" else (p_clean_per - p_patched_per)

        if collapse_receivers:
            reachable = [layer > sender_layer for layer in receiver_layers_by_index]
            if not any(reachable):
                continue  # no receiver is downstream of this sender; leave as NaN
            mask = t.tensor(reachable, device=device)
            masked_denom = per_receiver_denom[mask].mean()
            results[sender_layer, sender_head] = numer_per[mask].mean() / masked_denom
        else:
            results[:, sender_layer, sender_head] = numer_per / per_receiver_denom

    model.reset_hooks()
    return results