def l2_norm_rel(clean_head_output, patched_head_output):
    """
    Relative change in a receiver head's output magnitude between the clean
    and sender-patched runs.
    """
    clean_norm = clean_head_output.norm(p=2)
    patched_norm = patched_head_output.norm(p=2)
    return ((patched_norm - clean_norm) / clean_norm).item()

def l2_norm_abs(clean_head_output, patched_head_output):
    """
    Absoluate change in a receiver head's output magnitude between the clean
    and sender-patched runs.
    """
    clean_norm = clean_head_output.norm(p=2)
    patched_norm = patched_head_output.norm(p=2)
    return (abs(patched_norm - clean_norm) / clean_norm).item()

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