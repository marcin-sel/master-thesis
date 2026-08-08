from abc import ABC, abstractmethod

import networkx as nx
import pandas as pd


class GraphBuilder(ABC):
    """Build a feature graph from data.

    Concrete builders derive a ``networkx`` graph over the columns of ``X``
    (optionally informed by the target ``y``) using a specific strategy
    (interaction information, structure learning, semantic similarity, ...).
    Keeping a single ``build`` entry point lets callers swap strategies without
    knowing how the underlying graph is computed.
    """

    @abstractmethod
    def build(self, X: pd.DataFrame, y: pd.Series | None = None) -> nx.Graph:
        """Return a graph whose nodes are the columns of ``X``."""
        raise NotImplementedError
