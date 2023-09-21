import random
from typing import List, Tuple

import networkx as nx
import numpy as np


def sample_tree_graph(
    n_states: int,
    path_length: int,
    rng: np.random.Generator,
    is_binary: bool = False
) -> Tuple[nx.DiGraph, int, int]:
    """Generate a random tree with the specified params

    Args:
        n_states (int): Number of nodes in tree
        path_length (int): Lnegth of path
        rng (np.random.Generator): Random number generator
        is_binary (bool, optional): Whether or not each node can have a 
        max of two children. Defaults to False.

    Returns:
        Tuple[nx.DiGraph, int, int]: Return the graph, the root, and the goal node
    """
    # Create adjacency matrix
    adj_matrix = np.zeros((n_states, n_states))
    source_node = None
    destination_node = None
    nodes = [i for i in range(n_states)]
    nodes_in_tree = []
    intermediate_node = None
    # Generate path of defined 'path_length'
    for i in range(path_length + 1):  # '+ 1' as we have to sample the source node first 
        # sample example and remove it from the list of nodes
        sample_node = rng.choice(nodes, 1)[0]
        nodes.remove(sample_node)
        nodes_in_tree.append(sample_node)
        # construct the edge and add it to the edge list
        if source_node is None:
            source_node = sample_node
        else:
            adj_matrix[intermediate_node, sample_node] = 1
        intermediate_node = sample_node
    destination_node = intermediate_node
    # Add other nodes until requested 'n_states'
    nodes_in_tree.remove(destination_node)  # remove destination to ensure we don't increase path length
    for n in nodes:
        # Sample a position in the tree
        if is_binary:
            valid_parents = [node for node in nodes_in_tree if adj_matrix[node].sum() < 2]
        else:
            valid_parents = nodes_in_tree
        connection_node = rng.choice(valid_parents, 1)[0]
        # Integrate node
        adj_matrix[connection_node, n] = 1
        nodes_in_tree.append(n)
    # Create networkx object
    tree = nx.DiGraph(incoming_graph_data=adj_matrix)
    return tree, source_node, destination_node


def topological_sort_edges(graph: nx.DiGraph) -> List[Tuple[int, int]]:
    """Takes in a DAG in NetworkX format, and creates an edgelist where 
    the edges are topologically sorted by the first node in each edge.

    Args:
        graph (nx.DiGraph): Input graph

    Returns:
        List[Tuple[int, int]]: Sorted edgelist
    """
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


def shortest_path(
    edgelist: List[Tuple[int, int]],
    n_nodes: int,
    start: int,
    end: int
):
    """Does BFS over graph to find the shortest path from
    start to end

    Args:
        edgelist (List[Tuple[int, int]]): List of edges
        n_nodes (int): Total number of nodes in graph
        start (int): First node in path
        end (int): Last node in path

    Returns:
        Returns a list of nodes in the path if found, returns -1 otherwise
    """
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


def generate_example(
    n_states: int,
    seed: int,
    order: str = "random",
    path_length: int = None,
    return_all_leafs: bool = False,
    is_binary: bool = True
):
    """Generates a random example involving an edgelist of a tree, a leaf node, and a path from the root node to the leaf

    Args:
        n_states (int): Number of nodes in graph
        seed (int): Seed for rng that generates graph
        order (str, optional): The order of the edges in the edgelist. Defaults to "random".
        path_length (int, optional): Distance between root and goal in example. Defaults to None.
        return_all_leafs (bool, optional): Whether to return all possible path. Defaults to False.
        is_binary (bool, optional): Whether or not the tree should be binary. Defaults to False.

    Returns:
        Returns single string by default
        Returns a list of all possible example strings if return_all_leafs is True
    """
    assert n_states >= 2
    assert path_length is None or path_length < n_states
    assert order in ["forward", "backward", "random"]
    # Sample tree and path with path_length
    rng = np.random.default_rng(seed=seed)
    if path_length is None:
        path_length = rng.integers(1, n_states)
    graph, start_node, goal = sample_tree_graph(n_states, path_length, rng, is_binary)
    edgelist = topological_sort_edges(graph)
    if order == "random":
        rng.shuffle(edgelist)
    elif order == "backward":
        edgelist = edgelist[::-1]
    # Create all possible examples from edgelist
    source_nodes = set([source for source, target in edgelist])
    target_nodes = set([target for source, target in edgelist])
    leaf_nodes = target_nodes - source_nodes
    # Sample all possible paths
    examples = {}
    for end_node in list(leaf_nodes):
        path = shortest_path(edgelist, n_states, start_node, end_node)
        # Convert to a series of tokens
        string = ",".join([f"{i}>{j}" for i, j in edgelist])
        string = string + f"|{end_node}:"
        string = string + ">".join([str(p) for p in path])
        examples[end_node] = string
    # Return all leafs if specified
    if return_all_leafs:
        return list(examples.values())
    else:
        return examples[goal]
