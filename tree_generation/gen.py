import networkx as nx
import numpy as np
import random


def topological_sort_edges(graph):
    try:
        node_order = list(nx.topological_sort(graph))
        sorted_edges = []
        for i in range(len(node_order) - 1):
            for j in range(i + 1, len(node_order)):
                if graph.has_edge(node_order[i], node_order[j]):
                    sorted_edges.append((node_order[i], node_order[j]))
        return sorted_edges
    except nx.NetworkXUnfeasible:
        print("The graph is not a directed acyclic graph (DAG)!")
        return None


def sample_tree_graph(n_states, seed):
    tree = nx.random_tree(
        n_states,
        seed=seed,
        create_using=nx.DiGraph
    )
    return topological_sort_edges(tree)


def shortest_path(edgelist, n_nodes, start, end):
    # BFS algorithm to extract the shortest path to a goal node
    visited = np.zeros((n_nodes,), dtype=bool)
    queue = [(start, [start],  0)]  # (node, distance)
    while queue:
        current_node, path, current_distance = queue.pop(0)
        if current_node == end:
            return path
        if not visited[current_node]:
            visited[current_node] = True
            neighbors = [edgelist[i][1] for i in range(len(edgelist)) if edgelist[i][0] == current_node]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    copy_path = path.copy()
                    copy_path.append(neighbor)
                    queue.append((neighbor, copy_path, current_distance + 1))
    return -1  # Path not found, should not be possible since graphs are strongly-connected


def sample_random_leaf(edgelist, rng):
    source_nodes = set([source for source, target in edgelist])
    target_nodes = set([target for source, target in edgelist])
    leaf_nodes = target_nodes - source_nodes
    return rng.choice(list(leaf_nodes))  # Sample a random leaf node.


def sample_random_leaf(edgelist, rng):
    source_nodes = set([source for source, target in edgelist])
    target_nodes = set([target for source, target in edgelist])
    leaf_nodes = target_nodes - source_nodes
    return rng.choice(list(leaf_nodes))  # Sample a random leaf node.


def generate_example(n_states, seed, order="random"):
    assert order in ["forward", "backward", "random"]
    # Sample random edge list
    rng = np.random.default_rng(seed=seed)
    edgelist = sample_tree_graph(n_states, seed)
    shuffled_nodes = np.arange(n_states)
    rng.shuffle(shuffled_nodes)
    edgelist = [(shuffled_nodes[i], shuffled_nodes[j]) for i, j in edgelist]
    if order == "random":
      rng.shuffle(edgelist)
    elif order == "backward":
      edgelist = edgelist[::-1]
    # Sample Shortest path
    start_node = shuffled_nodes[0]
    end_node = sample_random_leaf(edgelist, rng) # We should instead sample random leaf node
    if start_node == end_node:
        end_node = (end_node+1) % n_states
    path = shortest_path(edgelist, n_states, start_node, end_node)
    # Convert to a series of tokens
    string = ",".join([f"{i}>{j}" for i, j in edgelist])
    string = string + f"|{end_node}:"
    return (string + ">".join([str(p) for p in path]))

