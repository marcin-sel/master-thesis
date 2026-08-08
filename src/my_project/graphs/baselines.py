"""
Baseline feature graphs: fully connected and edgeless.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from my_project.graphs.base import GraphBuilder


class FullGraphBuilder(GraphBuilder):
    """Connect every pair of features (complete graph over the columns of ``X``)."""

    def build(self, X: pd.DataFrame, y: pd.Series | None = None) -> nx.Graph:
        return nx.complete_graph(list(X.columns))


class EmptyGraphBuilder(GraphBuilder):
    """No edges between features (one isolated node per column of ``X``)."""

    def build(self, X: pd.DataFrame, y: pd.Series | None = None) -> nx.Graph:
        return nx.empty_graph(list(X.columns))


def build_true_graph(reference_graph, true_edges):
    """Oracle graph on the same node set as ``reference_graph``."""
    g = nx.empty_graph(reference_graph)
    for v1, v2 in true_edges:
        g.add_edge(v1, v2)
        g.add_edge(v2, v1)
    return nx.to_undirected(g)


def build_base_graphs(graph_true):
    """Threshold-independent graphs (same for every threshold).

    Skip the ``"oracle"`` graph when there are no true edges (e.g. real
    datasets): an edgeless oracle carries no ground-truth structure and would
    just duplicate ``"empty"``.
    """
    graphs = {
        "empty": nx.empty_graph(graph_true),
        "fully_connected": nx.complete_graph(graph_true),
    }
    if graph_true.number_of_edges() > 0:
        graphs = {"oracle": graph_true, **graphs}
    return graphs
