import torch
import torch.nn.functional as F
from fancy_einsum import einsum

def apply_causal_mask_and_softmax(matrix):
    # Assuming matrix is a PyTorch tensor of shape (65, 65)
    # Step 1: Create a causal mask
    rows, cols = matrix.size()
    causal_mask = torch.tril(torch.ones(rows, cols, dtype=torch.bool))

    # Step 2: Apply the causal mask
    # Use a very large negative number for masked values
    masked_matrix = matrix.masked_fill(~causal_mask, float('-inf'))
    

    # Step 3: Compute softmax per row
    softmax_matrix = F.softmax(masked_matrix, dim=1)

    return masked_matrix

def delete_non_paths(input_dict):
    keys_to_delete = [key for key, value in input_dict.items() if len(value) <= 2]
    for key in keys_to_delete:
        del input_dict[key]
    return input_dict

special_chars = [",", ":", "|"]
def extract_subpaths(model, graph, cache, labels, up_to_layer=3, up_to_position=48, threshold=0.9):

    edge_list = graph.split("|")[0].split(",")
    edge_list = [i.split(">") for i in edge_list]
    edge_list = [(int(i), int(j)) for i, j in edge_list]

    paths = {}
    for layer in range(1, up_to_layer):

        # compute information-weighted attention patterns
        with torch.no_grad():
            info_weighting = einsum(
                "batch pos head_index d_head, \
                    head_index d_head d_model -> \
                    batch pos d_model",
                cache[f"blocks.{layer}.attn.hook_v"],
                model.blocks[layer].attn.W_O,
            )
        attn_pattern = (cache[f"blocks.{layer}.attn.hook_pattern"][0, 0, :, :] * info_weighting[0, :, :].norm(dim=1, p=2)).cpu()[:, :]
        attn_pattern = apply_causal_mask_and_softmax(attn_pattern)
        seq_len, _ = attn_pattern.shape

        for current_pos in range(10, up_to_position):  # start at 10  
            current_token = labels[current_pos]     
            for attended_pos in range(seq_len):
                attn_value = attn_pattern[current_pos, attended_pos]
                if attn_value > threshold:
                    attended_token = labels[attended_pos].replace(">", "")
                    previous_token = labels[attended_pos - 1].replace(">", "")
                    if not attended_token in special_chars and not previous_token in special_chars:
                        if (int(previous_token), int(attended_token)) in edge_list:
                            identifier = (current_pos, current_token)
                            if identifier in paths.keys():
                                if paths[identifier][-1] != previous_token and paths[identifier][-1] == attended_token:
                                    paths[identifier].append(previous_token)
                            else:
                                paths[identifier] = [attended_token, previous_token]
    
    paths = delete_non_paths(paths)
    return paths