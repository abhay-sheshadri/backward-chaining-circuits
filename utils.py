import os
import time

import numpy as np
import torch
from tqdm import tqdm
import networkx as nx

import wandb


def get_loaders(dataset, batch_size, train_test_split=0.9):
    # Split the dataset into train and test sets
    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                    [train_size, test_size])
    # Collate function
    def collate(data):
        tokens, masks = zip(*data)
        tokens = torch.stack(tokens, dim=0)
        masks = torch.stack(masks, dim=0)
        return tokens, masks
    # Create data loaders for train and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate)
    return train_loader, test_loader


def train(model, train_loader, test_loader, n_epochs, learning_rate=3e-4, betas=(0.9, 0.99), wd=0.01, use_wandb=True):
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, betas=betas, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, 2e-6)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    if use_wandb:
        run_name = f"CoT_{int(time.time())}"
        opt_kwargs = {
            "lr": learning_rate,
            "n_epochs": n_epochs,
            "betas": betas
        }
        wandb.init(
            project="planning-in-transformers",
            name=run_name,
            config=({
                **model.cfg.__dict__, **opt_kwargs
            })
        )

    # Start training
    for epoch in range(n_epochs):

        pbar = tqdm(total=len(train_loader))
        model.train()
        losses = []
        accs = []

        for idx, (tokens, mask) in enumerate(train_loader):
            optimizer.zero_grad()
            
            tokens = tokens.cuda().to(torch.long)
            inputs = tokens[:, :-1]
            output_mask = mask[:, 1:].cuda()
            targets = tokens[:, 1:][output_mask]
            
            outputs = model(inputs)[output_mask]
            loss = loss_fn(outputs, targets)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            losses.append(loss.item())
            acc = torch.mean( (outputs.argmax(-1) == targets).to(torch.float)) * 100
            accs.append(acc)

            pbar.set_description(f"TRAIN - Epoch: {epoch+1}, Loss: {sum(losses)/len(losses):.4f}, Acc: {sum(accs)/len(accs):.4f}%")
            pbar.update(1)

        scheduler.step()
        pbar.close()
        
        if use_wandb:        
            wandb.log({"train/loss": sum(losses) / len(losses)}, step=epoch)
            wandb.log({"train/acc": sum(accs)/len(accs)}, step=epoch)
    
        model.eval()
        with torch.no_grad():

            pbar = tqdm(total=len(test_loader))
            losses = []
            accs = []

            for idx, (tokens, mask) in enumerate(test_loader):

                tokens = tokens.cuda().to(torch.long)
                inputs = tokens[:, :-1]
                output_mask = mask[:, 1:].cuda()
                targets = tokens[:, 1:][output_mask]
                
                outputs = model(inputs)[output_mask]
                loss = loss_fn(outputs, targets)
                
                losses.append(loss.item())
                acc = torch.mean( (outputs.argmax(-1) == targets).to(torch.float)) * 100
                accs.append(acc)
                
                pbar.set_description(f"TEST  - Epoch: {epoch+1}, Loss: {sum(losses)/len(losses):.4f}, Acc: {sum(accs)/len(accs):.4f}%")
                pbar.update(1)
        
            pbar.close()
        
        if use_wandb:
            torch.save(model.state_dict(), f"checkpoint_{epoch}.pt")
            wandb.log({"test/loss": sum(losses)/len(losses)}, step=epoch)
            wandb.log({"test/acc": sum(accs)/len(accs)}, step=epoch)
            
            # Save model
            artifact = wandb.Artifact(run_name, type="model")
            artifact.add_file(local_path="model.pt",
                            name=f"checkpoint_{epoch}.pt")
            wandb.log_artifact(artifact)
            os.remove(f"checkpoint_{epoch}.pt")


def get_example_cache(example, model, dataset):
    # Output tokens and forward cache
    tokens = dataset.tokenize(example)[:-1]
    inputs = torch.from_numpy(tokens).unsqueeze(0).cuda()
    _, cache = model.run_with_cache(inputs)
    labels = [dataset.idx2tokens[idx] for idx in tokens]
    return labels, cache


def extract_adj_matrix(example_str, power=None):
    # Extract edgelist
    graph = example_str.split("|")[0]
    graph = graph.split(",")
    edgelist = []
    nodes = set()
    for e in graph:
        out_node = int(e.split(">")[0])
        in_node = int(e.split(">")[1])
        edgelist.append((out_node, in_node))
        nodes.add(out_node)
        nodes.add(in_node)
    # Extract path
    goal = example_str.split("|")[1]
    goal_node = int(goal.split(":")[0])
    path = goal.split(":")[1].split(">")
    path = [int(p) for p in path]
    path_edges = list(zip(path[:-1], path[1:]))
    # Make sure every edge in the path is valid
    for edge in path_edges:
        assert edge in edgelist
    # Create networkx graph
    G = nx.DiGraph()
    G.add_nodes_from(range(len(nodes)))
    for edge in edgelist:
        if edge in path_edges:
            color = "red"
        else:
            color = "black"
        G.add_edge(edge[0], edge[1], color=color)
    # Convert to numpy adjacency matrix
    adjacency_matrix_sparse = nx.adjacency_matrix(G)
    adjacency = adjacency_matrix_sparse.toarray()
    if power is not None:
        # Change all leaf nodes to self-loops
        row_sums = np.sum(adjacency, axis=1)
        for i in range(len(row_sums)):
            if row_sums[i] == 0:
                adjacency[i, i] = 1
        # Exponentiate matrix
        return np.linalg.matrix_power(adjacency, power)
    else:
        return adjacency


def num_last(arr, char):
    fidx = len(arr) - 1
    while arr[fidx] == char and fidx >= 0:
        fidx -= 1
    return fidx + 1


def eval_model(model, dataset, test_graph):
    model.eval()
    
    # Initialize counters
    test_graph_tokens = dataset.tokenize(test_graph)
    start_idx = np.where(test_graph_tokens == dataset.start_token)[0].item() + 2
    curr_idx = start_idx

    flag = False
    while not flag and curr_idx < dataset.max_seq_length - 1:
        # Convert to pytorch
        input_tokens = torch.from_numpy(test_graph_tokens).to(torch.long).cuda()
        input_tokens[curr_idx:] = 0
        input_tokens = input_tokens.unsqueeze(0)[:, :-1]
        # Run model
        with torch.no_grad():
            outputs = model(input_tokens).argmax(-1)
            pred = outputs[0, curr_idx-1]
            test_graph_tokens[curr_idx] = pred.item()
            if pred.item() == dataset.pad_token:  # Check if we reached the goal
                flag = True
        curr_idx += 1

    final_path = dataset.untokenize(test_graph_tokens[:curr_idx])
    return final_path, test_graph == final_path


def is_model_correct(model, dataset, test_graph, return_probs=False):
    model.eval()
    # Initialize counters
    test_graph_tokens = dataset.tokenize(test_graph)
    start_idx = np.where(test_graph_tokens == dataset.start_token)[0].item() + 1
    end_idx = num_last([dataset.idx2tokens[i] for i in test_graph_tokens], ",")
    input_tokens = torch.from_numpy(test_graph_tokens).to(torch.long).cuda()
    input_tokens = input_tokens.unsqueeze(0)[:, :-1]
    # Run model
    with torch.no_grad():
        probs = model(input_tokens).softmax(-1)
        outputs = probs.argmax(-1)
    correct = torch.all(outputs[:, start_idx:end_idx] == input_tokens[:, start_idx+1:end_idx+1]).item()
    if return_probs:
        return correct, probs[0, torch.arange(start_idx, end_idx), input_tokens[0, start_idx+1:end_idx+1]]
    return correct
