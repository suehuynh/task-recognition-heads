def l2_norm_abs(clean_receiver_input, patched_receiver_input, reduce: bool = True):
    """
    Absolute L2 magnitude of the change a single patched sender induces in the
    receiver heads' input vector (q/k/v) at the final query position.

    Both args have shape [n_receiver_heads, batch, d_head] (the last-position
    slice produced by get_path_patch_head_to_heads._gather_receiver_vec).

    Computes ||q_patched - q_clean||_2 per (receiver head, prompt), then means
    over prompts. This is the norm of the difference vector -- not the
    difference of norms -- so it responds to *directional* changes in the
    receiver input (which tokens the head attends to), not only to rescaling.

    reduce=True (default, unchanged behaviour): also means over the receiver
        axis and returns a python float, as before.
    reduce=False: keeps the receiver axis and returns a [n_receiver_heads]
        tensor (mean over prompts only) -- one score per receiver head, for
        per-receiver path patching.
    """
    delta = (patched_receiver_input - clean_receiver_input).norm(p=2, dim=-1)  # [n_heads, batch]
    per_head = delta.mean(dim=-1)  # [n_heads]
    return per_head.mean().item() if reduce else per_head

def l2_norm_rel(clean_receiver_input, patched_receiver_input, reduce: bool = True):
    """
    Same change as l2_norm_abs, but each (receiver head, prompt) delta is
    divided by that receiver's clean input norm before averaging -- a
    fractional change that is comparable across heads of different magnitude
    and not dominated by a few high-norm receivers/prompts.

    reduce: see l2_norm_abs.
    """
    delta = (patched_receiver_input - clean_receiver_input).norm(p=2, dim=-1)   # [n_heads, batch]
    clean = clean_receiver_input.norm(p=2, dim=-1).clamp_min(1e-8)              # [n_heads, batch]
    per_head = (delta / clean).mean(dim=-1)  # [n_heads]
    return per_head.mean().item() if reduce else per_head

def early_decode(head_output, model):
    """
    Project a single head's output (z_h @ W_O_h, shape [..., d_model]) through
    the model's final layernorm and unembedding.

    head_output: tensor, shape [..., d_model]: one head's contribution at
        one sequence position
    Returns: logits over vocab, shape [..., d_vocab].
    """
    normalized = model.ln_final(head_output)
    return model.unembed(normalized)

def topk_match_count(logits, task_relation_words, tokenizer, k=10):
    """
    Number of the top-k decoded tokens that appear in task_relation_words
    (the paper's task-descriptive term list, e.g. antonym -> ["opposite",
    "reverse", "antonym", ...]). [n]

    logits: tensor, shape [d_vocab]
    task_relation_words: set[str] -- lowercased, from task_relation_dict.json.
    """
    topk_token_ids = logits.topk(k).indices.tolist()
    topk_strs = {tokenizer.decode([tid]).strip().lower() for tid in topk_token_ids}
    return len(topk_strs & set(w.lower() for w in task_relation_words))


def task_vocab_prob_mass(head_output, model, task_token_ids):
    """
    Sum of softmax-probability mass on task_token_ids, read from the logit
    lens of one head's output.

    head_output: tensor, shape [..., d_model] -- one head's contribution
        (z_h @ W_O_h) at one sequence position.
    task_token_ids: 1D LongTensor of vocab ids for the task-descriptive words
        (V_task), e.g. from convert_task_words_to_token_ids.

    Returns: tensor, shape [...] -- P(V_task) per leading index.
    """
    probs = early_decode(head_output, model).softmax(dim=-1)
    return probs[..., task_token_ids].sum(dim=-1)

def n_match_effect(clean_head_output, patched_head_output, model, task_relation_words, tokenizer, k=10):
    """
    Signed change in n (top-k task-term match count) between the clean and
    sender-patched runs, for one receiver head.

    Negative = patching this sender degraded the receiver's task
    verbalization (expected if sender causally feeds the LTH).
    Zero = no effect at this (sender, receiver) pair.
    """
    clean_logits = early_decode(clean_head_output, model)
    patched_logits = early_decode(patched_head_output, model)

    n_clean = topk_match_count(clean_logits, task_relation_words, tokenizer, k=k)
    n_patched = topk_match_count(patched_logits, task_relation_words, tokenizer, k=k)

    return n_patched - n_clean