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
