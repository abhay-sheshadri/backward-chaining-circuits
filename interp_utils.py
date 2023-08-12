import plotly.express as px
import numpy as np
from functools import partial
import tqdm.auto as tqdm_auto
import matplotlib.pyplot as plt
import transformer_lens.utils as utils
import torch
import matplotlib.pyplot as plt


def display_head(cache, labels, layer, head, show=True):
    average_patterns = cache[f"blocks.{layer}.attn.hook_pattern"]
    last_idx = average_patterns.shape[-1] - 1
    while labels[last_idx] == ",":
        last_idx -= 1
    last_idx += 1
    matrix = average_patterns[0, head, :last_idx, :last_idx].cpu()
    labels = labels[:last_idx]
    fig = px.imshow(
        matrix,
        labels=dict(x="AttendedPos", y="CurrentPos", color="Value"),
    )
    layout = dict(
        width=800,
        height=800,
        xaxis=dict(
            tickmode="array",
            tickvals=np.arange(len(labels)),
            ticktext=labels,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=np.arange(len(labels)),
            ticktext=labels,
        )
    )
    fig.update_layout(layout)
    if show:
        fig.show()
    else:
        return fig


def mask_model_att(model, index_start, index_end, layer_start, layer_end):
    # Define hook function
    def attention_score_hook_edges(resid_pre, hook, position_start, position_end):
        resid_pre[:, :, position_start:position_end, :] = 0
        return resid_pre
    # Add hook to every layer
    for layer in range(layer_start, layer_end):
        temp_hook_fn = partial(attention_score_hook_edges, position_start=index_start, position_end=index_end)
        model.blocks[layer].attn.hook_pattern.add_hook(temp_hook_fn)


def logits_to_logit_diff(clean_tokens, corrupted_tokens, logits, comparison_index):
    correct_index = clean_tokens[comparison_index]
    incorrect_index = corrupted_tokens[comparison_index]
    return logits[0, comparison_index-1, correct_index] - logits[0, comparison_index-1, incorrect_index]


def activation_patching(model, dataset, clean_tokens, corrupted_tokens, comparison_index):
    # We run on the clean prompt with the cache so we store activations to patch in later.
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    clean_logit_diff = logits_to_logit_diff(clean_tokens, corrupted_tokens, clean_logits, comparison_index)
    print(f"Clean logit difference: {clean_logit_diff.item():.3f}")

    # We don't need to cache on the corrupted prompt.
    corrupted_logits = model(corrupted_tokens)
    corrupted_logit_diff = logits_to_logit_diff(clean_tokens, corrupted_tokens, corrupted_logits, comparison_index)
    print(f"Corrupted logit difference: {corrupted_logit_diff.item():.3f}")
    print(f"Positive Direction: {dataset.idx2tokens[clean_tokens[comparison_index]]}")
    print(f"Negative Direction: {dataset.idx2tokens[corrupted_tokens[comparison_index]]}")

    def residual_stream_patching_hook(
        resid_pre,
        hook,
        position):
        # Each HookPoint has a name attribute giving the name of the hook.
        clean_resid_pre = clean_cache[hook.name]
        resid_pre[:, position, :] = clean_resid_pre[:, position, :]
        return resid_pre
    # We make a tensor to store the results for each patching run. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
    num_positions = clean_tokens.shape[0]
    patching_result = torch.zeros((model.cfg.n_layers, num_positions), device=model.cfg.device)
    for layer in tqdm_auto.tqdm(range(model.cfg.n_layers)):
        for position in range(num_positions):
            # Use functools.partial to create a temporary hook function with the position fixed
            temp_hook_fn = partial(residual_stream_patching_hook, position=position)
            # Run the model with the patching hook
            patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[
                (utils.get_act_name("resid_pre", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            patched_logit_diff = logits_to_logit_diff(clean_tokens, corrupted_tokens, patched_logits, comparison_index).detach()
            # Store the result, normalizing by the clean and corrupted logit difference so it's between 0 and 1 (ish)
            normalize_ratio = (clean_logit_diff - corrupted_logit_diff)
            if normalize_ratio == 0:
                normalize_ratio = 1
            patching_result[layer, position] = (patched_logit_diff - corrupted_logit_diff) / normalize_ratio
    return patching_result


def plot_activations(patching_result, clean_tokens, dataset):
    # Add the index to the end of the label, because plotly doesn't like duplicate labels
    token_labels = [f"{dataset.idx2tokens[token]}_{index}" for index, token in enumerate(clean_tokens)]
    plt.imshow(patching_result, x=token_labels, xaxis="Position", yaxis="Layer", title="Activation patching")
