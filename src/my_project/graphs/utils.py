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
