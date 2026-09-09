import os
import torch
import plotly.express as px


_METRIC_LABELS = {
    "l2_norm_rel": "receiver-input change ||Δ|| / ||clean||",
    "l2_norm_abs": "receiver-input change ||Δ||",
    "lprr": "LPRR",
}

# l2_norm_* are non-negative (norm of the difference vector); lprr is signed
# and centred at 0.
_DIVERGING_METRICS = {"lprr"}


def plot_sender_head_effect(scores, receiver_list, receiver_input, save_path=None,
                            title=None, metric="l2_norm_rel"):
    """
    Layer x head heatmap of the path-patching effect of each candidate sender
    head on the receiver LTHs' `receiver_input` (as produced by
    get_path_patch_head_to_heads / get_path_patch_head_to_LTH_vocab).

    scores: tensor/array, shape [n_sender_layers, n_heads] -- results tensor.
    receiver_list: list[(layer, head)] -- the LTHs being patched into, shown
        in the title.
    receiver_input: "q", "k", or "v" -- which receiver input stream was patched.
    metric: "l2_norm_rel" or "l2_norm_abs" or "lprr" -- sets the colorbar label and title suffix.
    save_path: if given, write an interactive HTML file here.

    Returns the plotly Figure.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    color_label = _METRIC_LABELS.get(metric, metric)

    if title is None:
        title = (f"Sender-head effect ({color_label}) on receiver_input='{receiver_input}' "
                 f"of LTHs {receiver_list}")

    diverging = metric in _DIVERGING_METRICS
    fig = px.imshow(
        scores,
        labels={"x": "Head", "y": "Layer", "color": color_label},
        title=title,
        color_continuous_scale="RdBu_r" if diverging else "Reds",
        color_continuous_midpoint=0 if diverging else None,
        aspect="auto",
    )
    fig.update_layout(width=800, height=500)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"saved heatmap to {save_path}")

    return fig