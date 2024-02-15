from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import tqdm.auto as tqdm_auto
import transformer_lens.utils as tl_util
from neel_plotly import imshow, line, scatter
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from .utils import *


def kl_divergence(old_dist, new_dist):
    divs = torch.sum(old_dist * torch.log(old_dist / new_dist), dim=1)
    return divs.max().item()


def add_attention_blockout(model, layer, head, current, attending):
    # Define hook function
    def attention_score_hook_edges(resid_pre, hook, r, c):
        # Prevent pos r from attending to c
        resid_pre[:, head, r, c] = 0
        return resid_pre
    # Add hook to every layer
    temp_hook_fn = partial(attention_score_hook_edges, r=current, c=attending)
    model.blocks[layer].attn.hook_pattern.add_hook(temp_hook_fn)


def add_attention_blockout_parallel(model, layer, indices):
    # indices is a list of (current, attending) 
    indices = torch.tensor(indices).t()
    # Define hook function
    def attention_score_hook_edges(resid_pre, hook, indices):
        # Prevent pos r from attending to c
        resid_pre[:, indices[0], indices[1], indices[2]] = 0
        return resid_pre
    # Add hook to every layer
    temp_hook_fn = partial(attention_score_hook_edges, indices=indices)
    model.blocks[layer].attn.hook_pattern.add_hook(temp_hook_fn)


def attention_knockout_discovery(model, dataset, test_graph, threshold=None, prev_ablated_edges=None):
    # Evaluate model on test_graph
    correct, base_probs = is_model_correct(model, dataset, test_graph, return_probs=True)
    assert correct
    labels, cache = get_example_cache(test_graph, model, dataset)
    
    # Iterate over heads in each layer
    important_edges = {i: [] for i in range(model.cfg.n_layers)}
    if prev_ablated_edges is None:
        ablated_edges = {i: [] for i in range(model.cfg.n_layers)}
    else:
        ablated_edges = prev_ablated_edges
    
    for l in range(model.cfg.n_layers-1, -1, -1):
        for h in range(model.cfg.n_heads):
            # Check if we can remove attention edges without affecting the correctness of the output
            for i in range(model.cfg.n_ctx - 1, -1, -1):
                for j in range(i, -1, -1):
                    # Skip if this edge was already pruned
                    if (h, i, j) in ablated_edges[l]:
                        continue
                    
                    model.reset_hooks()                  
                    # Add already ablated edges
                    for L in ablated_edges.keys():
                        inds = ablated_edges[L]
                        if len(inds) == 0:
                            continue
                        add_attention_blockout_parallel(model, L, inds)
                        
                    # Block new edge
                    add_attention_blockout(model, l, h, i, j)
                    
                    # If correct, add it to the ablations list
                    correct, new_probs = is_model_correct(model, dataset, test_graph, return_probs=True)
                    score = kl_divergence(base_probs, new_probs)
                    
                    if correct and (threshold is None or score < threshold):
                        ablated_edges[l].append((h, i, j))
                    else:
                        important_edges[l].append((h, i, j))
                        print(f"Breaking: Layer {l} head {h}, labels[{i}] attending to labels[{j}], {labels[i]} attending to {labels[j]}")
                        
    return ablated_edges, important_edges


def attention_knockout_discovery_multiple(model, dataset, multiple_test_graphs, threshold=None, prev_ablated_edges=None):
    # Evaluate model on test_graph
    correct, base_probs = is_model_correct_multiple(model, dataset, multiple_test_graphs, return_probs=True)
    assert correct
    
    # Iterate over heads in each layer
    important_edges = {i: [] for i in range(model.cfg.n_layers)}
    if prev_ablated_edges is None:
        ablated_edges = {i: [] for i in range(model.cfg.n_layers)}
    else:
        ablated_edges = prev_ablated_edges
    
    for l in range(model.cfg.n_layers-1, -1, -1):
        for h in range(model.cfg.n_heads):
            # Check if we can remove attention edges without affecting the correctness of the output
            for i in range(model.cfg.n_ctx - 1, -1, -1):
                for j in range(i, -1, -1):
                    # Skip if this edge was already pruned
                    if (h, i, j) in ablated_edges[l]:
                        continue
                    
                    model.reset_hooks()                  
                    # Add already ablated edges
                    for L in ablated_edges.keys():
                        inds = ablated_edges[L]
                        if len(inds) == 0:
                            continue
                        add_attention_blockout_parallel(model, L, inds)
                    
                    # Block new edge
                    add_attention_blockout(model, l, h, i, j)
                    
                    # Check if abl# Check if ablations allow the model to get the correct input on all of the graphs
                    all_correct, new_probs = is_model_correct_multiple(model, dataset, multiple_test_graphs, return_probs=True)
                    score = kl_divergence(base_probs, new_probs)
                    
                    # If correct, add it to the ablations list
                    if all_correct and (threshold is None or score < threshold):
                        ablated_edges[l].append((h, i, j))
                    else:
                        important_edges[l].append((h, i, j))
                        print(f"Breaking: Layer {l} head {h}, labels[{i}] attending to labels[{j}]")

    return ablated_edges, important_edges
