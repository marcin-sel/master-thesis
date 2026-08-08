import warnings

import networkx as nx
import numpy as np
import pandas as pd


def graph_from_matrix(matrix: pd.DataFrame, threshold: float) -> nx.Graph:
    """Undirected graph with an edge for every off-diagonal entry > threshold.

    When the matrix has been mapped through an empirical-CDF transform (values
    in ``[0, 1]``), the boundary thresholds behave as: ``threshold=0`` keeps
    every pair (each has CDF >= 1/n > 0), giving the full graph, while
    ``threshold=1`` keeps nothing (no entry exceeds the maximum CDF value of 1),
    giving the empty graph.
    """
    columns = list(matrix.columns)
    graph = nx.Graph()
    graph.add_nodes_from(columns)
    for i, c1 in enumerate(columns):
        for c2 in columns[i + 1 :]:
            if matrix.loc[c1, c2] > threshold:
                graph.add_edge(c1, c2)
    return graph


def permute_nodes(graph: nx.Graph, seed: int = 42) -> nx.Graph:
    """Return a new graph with the same structure but permuted node labels.

    The permutation is deterministic given the same ``seed``.
    """
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes)
    permuted_nodes = rng.permutation(nodes)
    mapping = dict(zip(nodes, permuted_nodes))
    return nx.relabel_nodes(graph, mapping)


def graph_from_matrix_top_n(matrix: pd.DataFrame, n_edges: int) -> nx.Graph:
    """Undirected graph keeping the ``n_edges`` highest off-diagonal pairs."""
    columns = list(matrix.columns)
    graph = nx.Graph()
    graph.add_nodes_from(columns)
    pairs = [
        (c1, c2, matrix.loc[c1, c2])
        for i, c1 in enumerate(columns)
        for c2 in columns[i + 1 :]
    ]
    if n_edges >= len(pairs):
        warnings.warn(
            f"n_edges={n_edges} >= number of possible pairs ({len(pairs)}) for "
            f"{len(columns)} nodes: the ii graph will be FULLY CONNECTED and "
            f"indistinguishable from the 'fully_connected' graph. "
            f"Lower n_edges below {len(pairs)}."
        )
    pairs.sort(key=lambda t: t[2], reverse=True)
    for c1, c2, _ in pairs[:n_edges]:
        graph.add_edge(c1, c2)
    return graph


def graph_stats(graphs, true_edges):
    """Edge precision/recall of each graph against the true edges."""
    true_set = {frozenset(e) for e in true_edges}
    info = {}
    for name, g in graphs.items():
        edges = {frozenset(e) for e in g.edges()}
        found = len(edges & true_set)
        info[name] = {
            "edges_len": len(edges),
            "true_edges_in_set": found,
            "precision": found / len(edges) if edges else 0,
            "recall": found / len(true_set) if true_set else 0,
            "true_edges_len": len(true_set),
        }
    return info
