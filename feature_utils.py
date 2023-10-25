import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformer_lens import HookedTransformer, HookedTransformerConfig
import transformer_lens.utils as utils

from tree_generation import *
from utils import *
from interp_utils import *
from probing import *
from sparse_coding import *

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict
from einops import rearrange
from PIL import Image


def return_probing_dataset(acts, graphs, dataset):
    X = {key: [] for key in acts.keys()}
    y = []
    for gidx, graph in enumerate(graphs):
        # Get output labels
        tokens = dataset.tokenize(graph)[:-1]
        start_idx = np.where(tokens == dataset.start_token)[0].item() + 2
        labels = [dataset.idx2tokens[idx] for idx in tokens]
        end_idx = num_last(labels, ",") + 1
        # Find neighboring nodes for each pos in edgelist
        path = [int(x.replace(">", "")) for x in labels[start_idx-1:end_idx-1]]
        path_arr = np.zeros((16,))
        path_arr[path] = 1.
        y.append(path_arr)
        # Iterate over all layers residual streams
        for key in X.keys():
            streams = acts[key][gidx][0, start_idx-1:start_idx]
            X[key].append(streams)
    # Convert everything to np arrays
    for key in X.keys():
        X[key] = torch.cat(X[key], dim=0).detach().cpu().numpy()
    y = np.array(y)
    return X, y


def get_dictionary_activations(model, dataset, cache_name, autoencoder, max_seq_length, batch_size=32, device='cuda'):
    num_features, d_model = autoencoder.model.W_e.shape
    datapoints = len(dataset)
    dictionary_activations = torch.zeros((datapoints*max_seq_length, num_features))
    token_list = torch.zeros((datapoints*max_seq_length), dtype=torch.int64)
    with torch.no_grad():
        dl = DataLoader(dataset, batch_size=batch_size)
        for i, batch in enumerate(tqdm(dl)):
            batch = batch[0][:, :-1].to(device)
            token_list[i*batch_size*max_seq_length:(i+1)*batch_size*max_seq_length] = rearrange(batch, "b s -> (b s)")
            _, cache = model.run_with_cache(batch.to(device))
            batched_neuron_activations = rearrange(cache[cache_name], "b s n -> (b s) n" )
            batched_dictionary_activations, _ = autoencoder.model(batched_neuron_activations.cuda())
            dictionary_activations[i*batch_size*max_seq_length:(i+1)*batch_size*max_seq_length,:] = batched_dictionary_activations.cpu()
    return dictionary_activations, token_list


def get_dictionary_activations_at_pos(model, dataset, cache_name, autoencoder, batch_size=32, device='cuda'):
    num_features, d_model = autoencoder.model.W_e.shape
    datapoints = len(dataset)
    dictionary_activations = torch.zeros((datapoints, num_features))
    with torch.no_grad():
        dl = DataLoader(dataset, batch_size=batch_size)
        for i, batch in enumerate(tqdm(dl)):
            batch = batch[0][:, :-1].to(device)
            _, cache = model.run_with_cache(batch.to(device))
            if len(cache[cache_name].shape) == 3:
                batched_neuron_activations = cache[cache_name][:, 47, :]
            if len(cache[cache_name].shape) == 4: # For activations in attn
                batched_neuron_activations = cache[cache_name][:, 47, 0, :]
            batched_dictionary_activations, _ = autoencoder.model(batched_neuron_activations.cuda())
            dictionary_activations[i*batch_size:(i+1)*batch_size, :] = batched_dictionary_activations.cpu()
    return dictionary_activations


def get_feature_indices(feature_index, dictionary_activations, k=10, setting="max"):
    best_feature_activations = dictionary_activations[:, feature_index]
    # Sort the features by activation, get the indices
    if setting=="max":
        found_indices = torch.argsort(best_feature_activations, descending=True)[:k]
    elif setting=="uniform":
        # min_value = torch.min(best_feature_activations)
        min_value = torch.min(best_feature_activations)
        max_value = torch.max(best_feature_activations)

        # Define the number of bins
        num_bins = k

        # Calculate the bin boundaries as linear interpolation between min and max
        bin_boundaries = torch.linspace(min_value, max_value, num_bins + 1)

        # Assign each activation to its respective bin
        bins = torch.bucketize(best_feature_activations, bin_boundaries)

        # Initialize a list to store the sampled indices
        sampled_indices = []

        # Sample from each bin
        for bin_idx in torch.unique(bins):
            if(bin_idx==0): # Skip the first one. This is below the median
                continue
            # Get the indices corresponding to the current bin
            bin_indices = torch.nonzero(bins == bin_idx, as_tuple=False).squeeze(dim=1)
            
            # Randomly sample from the current bin
            sampled_indices.extend(np.random.choice(bin_indices, size=1, replace=False))

        # Convert the sampled indices to a PyTorch tensor & reverse order
        found_indices = torch.tensor(sampled_indices).long().flip(dims=[0])
    else: # random
        # get nonzero indices
        nonzero_indices = torch.nonzero(best_feature_activations)[:, 0]
        # shuffle
        shuffled_indices = nonzero_indices[torch.randperm(nonzero_indices.shape[0])]
        found_indices = shuffled_indices[:k]
    return found_indices


def get_feature_datapoints(found_indices, best_feature_activations, max_seq_length, dataset):
    num_datapoints = len(dataset)
    datapoint_indices = [np.unravel_index(i, (num_datapoints, max_seq_length)) for i in found_indices]
    all_activations = best_feature_activations.reshape(num_datapoints, max_seq_length).tolist()
    full_activations = []
    partial_activations = []
    text_list = []
    full_text = []
    token_list = []
    local_activations = []
    full_token_list = []
    for i, (md, s_ind) in enumerate(datapoint_indices):
        md = int(md)
        s_ind = int(s_ind)
        full_tok = torch.tensor(dataset[md][0])
        full_text.append(dataset.untokenize(full_tok))
        tok = dataset[md][0][:s_ind+1]
        full_activations.append(all_activations[md])
        partial_activations.append(all_activations[md][:s_ind+1])
        local_activations.append(all_activations[md][s_ind])
        text = dataset.untokenize(tok)
        text_list.append(text)
        token_list.append(tok)
        full_token_list.append(full_tok)
    return text_list, full_text, token_list, full_token_list, partial_activations, full_activations, local_activations


def fixed_plot(strings):
    G = nx.Graph()
    for s in strings:
        nodes = s.split('-')
        G.add_edge(nodes[0], nodes[1])
    nx.draw(G, with_labels=True)
    plt.axis('off')


def save_plots_as_images(strings_list, filenames):
    for idx, strings in enumerate(strings_list):
        parse_example(strings)
        plt.savefig(filenames[idx])
        plt.close()


def concat_images_vertically(filenames):
    images = [Image.open(filename) for filename in filenames]
    total_width = max(im.width for im in images)
    total_height = sum(im.height for im in images)
    new_image = Image.new('RGB', (total_width, total_height))
    
    y_offset = 0
    for im in images:
        new_image.paste(im, (0, y_offset))
        y_offset += im.height
    return new_image


def save_feature_plots(dictionary_activations, feature, num_feature_datapoints=1000, save_path="./features"): 

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    uniform_indices = get_feature_indices(feature, dictionary_activations, k=num_feature_datapoints, setting="uniform")
    text_list, full_text, token_list, full_token_list, partial_activations, full_activations, local_activations = get_feature_datapoints(uniform_indices, dictionary_activations[:, feature], max_seq_length, dataset)

    if len(text_list) > 0:
        
        # sample the 20 most interesting ones
        if len(text_list) > 20:
            merged = [list(a) for a in zip(full_text, local_activations)]
            def sort(sub_li):
                sub_li.sort(key = lambda x: x[1])
                return sub_li
            merged_sorted = sort(merged)
            subset = merged_sorted[-20:]
            full_text = [s[0] for s in subset]

        # generate plots    
        filenames = [f"temp_plot_{i}.png" for i in range(len(full_text))]    
        save_plots_as_images(full_text, filenames)
        concatenated_image = concat_images_vertically(filenames)

        # Save concatenated image as a PDF
        concatenated_image.save(os.path.join(save_path, f'feature_{feature}.pdf'), "PDF", resolution=100.0)

        # delete temp images
        for filename in filenames:
            import os
            os.remove(filename)

    else: 

        print(f"No examples for feature {feature} found.")


def plot_feature(dictionary_activations, feature, num_feature_datapoints=1000): 

    uniform_indices = get_feature_indices(feature, dictionary_activations, k=num_feature_datapoints, setting="uniform")
    text_list, full_text, token_list, full_token_list, partial_activations, full_activations, local_activations = get_feature_datapoints(uniform_indices, dictionary_activations[:, feature], max_seq_length, dataset)

    if len(text_list) > 0:
        
        # sample the 20 most interesting ones
        if len(text_list) > 20:
            merged = [list(a) for a in zip(full_text, local_activations)]
            def sort(sub_li):
                sub_li.sort(key = lambda x: x[1])
                return sub_li
            merged_sorted = sort(merged)
            subset = merged_sorted[-20:]
            full_text = [s[0] for s in subset]

        for graph in text_list:
            print(graph)
            parse_example(graph)
            plt.show()

    else: 

        print(f"No examples for feature {feature} found.")