import os

import numpy as np
import torch
from torch.utils.data import Dataset

from . import generate_example, parse_example


def generate_dataset_file(n_states, file_name, n_examples):
    """Generate dataset file if it does not exist

    Args:
        n_states (int): Number of nodes per tree
        file_name (str): Name of file to save dataset in
        n_examples (int): Number of different examples to sample
    """
    if os.path.exists(file_name):
        print("Loading contents from file...")
    else:
        print("Generating file...")
        with open(file_name, "w") as f:
            for seed in range(n_examples):
                line = generate_example(
                    n_states=n_states,
                    seed=seed
                )
                f.write(line+"\n")


class GraphDataset(Dataset):
    
    def __init__(self, n_states, file_name, n_examples):
        # Create a list of vocab
        number_tokens = sorted([str(i) for i in range(n_states)], key=lambda x: len(x), reverse=True)
        self.n_states = n_states
        self.idx2tokens = [",", ":", "|"] + [f">{t}" for t in number_tokens] + number_tokens
        self.tokens2idx = {token: idx for idx, token in enumerate(self.idx2tokens)}
        self.max_seq_length = n_states * 4 + 2
        self.pad_token = self.tokens2idx[","]
        self.start_token = 1
        # Open up dataset file and load+tokenize strings
        self.X = []
        self.masks = []
        generate_dataset_file(n_states, file_name, n_examples)
        with open(file_name, "r") as f:
            for line in f.readlines():
                # Tokenize string
                tokens = self.tokenize(line.rstrip())
                self.X.append(tokens)
                # Run checks
                assert self.untokenize(self.X[-1]) == line.rstrip()
                assert len(self.X[-1]) <= self.max_seq_length
                # Create a mask
                start_idx = np.where(tokens == 1)[0].item()
                index_tensor = np.arange(tokens.shape[0])
                mask = np.zeros_like(tokens, dtype=bool)
                mask[start_idx+2:len(tokens)+2] = True
                self.masks.append(mask)
        # Stack arrays
        self.X = torch.from_numpy(np.stack(self.X))
        self.masks = torch.from_numpy(np.stack(self.masks))
        
    def tokenize(self, text):
        # Convert to token list
        tokens = []
        i = 0
        while i < len(text):
            for idx, word in enumerate(self.idx2tokens):
                if text.startswith(word, i):
                    tokens.append(idx)
                    i += len(word)
                    break
            else:
                i += 1
        # Convert to fixed length numpy array
        tokens_arr = np.array(tokens)
        padding_length = self.max_seq_length - len(tokens)
        tokens_arr = np.pad(tokens_arr, ((0, padding_length),), mode='constant', constant_values=0)
        return tokens_arr

    def untokenize(self, tokens):
        substrings = [self.idx2tokens[idx] for idx in tokens]
        return "".join(substrings).rstrip(",")
    
    def visualize_example(self, index):
        string = self.untokenize(self[index][0])
        parse_example(string)

    def __getitem__(self, index):
        return self.X[index], self.masks[index]

    def __len__(self):
        return len(self.X)
