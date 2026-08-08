"""Interaction-information based feature graphs."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import pandas as pd

from my_project.graphs.base import GraphBuilder
from my_project.graphs.utils import (
    graph_from_matrix,
    graph_from_matrix_top_n,
    permute_nodes,
)
from my_project.information_theory.estimators import MEASURES
from my_project.information_theory.matrices import (
    compute_information_matrices,
    to_probability,
)


class InformationGraphBuilder(GraphBuilder):
    """Build a feature graph from a pairwise information measure.

    A single builder for the three pairwise measures, selected by ``measure``:

    - ``"interaction_information"`` (default): interaction information
      ``II(Xi; Xj; y)``; following pyitlib's convention, positive means the pair
      is synergistic about ``y``, so higher values correspond to stronger
      synergy.
    - ``"joint_target_mi"``: joint mutual information ``I((Xi, Xj); y)`` -- how
      much the pair *together* tells about the target.
    - ``"feature_mi"``: feature-feature mutual information ``I(Xi; Xj)``, which
      ignores ``y`` and captures redundancy / collinearity between features.

    Parameters
    ----------
    threshold:
        Edge threshold in ``[0, 1]``. An edge is added when the processed
        measure exceeds it.
    measure:
        Which pairwise measure to use (see above). ``"feature_mi"`` ignores
        ``y``; the other two require it.
    encode:
        Ordinal-encode the features before computing information measures.
    probability:
        Map the matrix through its empirical CDF so the threshold behaves like a
        quantile (``threshold=0`` => full graph, ``threshold=1`` => empty).
    estimator:
        Mutual-information estimator. ``"numpy"`` (default) is the fast
        vectorized plug-in (relative-frequency / maximum-likelihood) estimator;
        any other value is forwarded to :mod:`pyitlib` (e.g. ``"ML"``, or a
        shrinkage estimator such as ``"PERKS"`` / ``"MINIMAX"``).
    matrix:
        Optional precomputed *raw* (unprocessed) matrix for the chosen measure,
        e.g. an entry returned by :func:`compute_information_matrices`. When
        given, :meth:`fit` skips the expensive computation and only applies the
        probability transform, so the matrix can be computed once up front and
        reused across thresholds.

    After :meth:`fit` (or :meth:`build`) the processed matrix is available as
    ``matrix_``; :meth:`graph_at` reuses it to build graphs at several
    thresholds without recomputing.
    """

    def __init__(
        self,
        threshold: float = 0.9,
        *,
        measure: str = "interaction_information",
        encode: bool = True,
        n_bins: int | None = None,
        probability: bool = True,
        estimator: str = "numpy",
        matrix: pd.DataFrame | None = None,
        n_jobs: int | None = None,
    ):
        if measure not in MEASURES:
            raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}.")
        self.threshold = threshold
        self.measure = measure
        self.encode = encode
        self.n_bins = n_bins
        self.probability = probability
        self.estimator = estimator
        self.matrix = matrix
        self.n_jobs = n_jobs

    def _process(self, matrix: pd.DataFrame) -> pd.DataFrame:
        if self.probability:
            matrix = to_probability(matrix)
        return matrix

    def _compute_matrix(self, X: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
        if X is None:
            raise ValueError("Provide X, or a precomputed matrix.")
        return compute_information_matrices(
            X,
            y,
            measures=self.measure,
            encode=self.encode,
            n_bins=self.n_bins,
            estimator=self.estimator,
            n_jobs=self.n_jobs,
        )[self.measure]

    def fit(
        self, X: pd.DataFrame | None = None, y: pd.Series | None = None
    ) -> InformationGraphBuilder:
        """Cache the processed measure matrix.

        If a precomputed ``matrix`` was passed to the constructor it is processed
        directly and ``X``/``y`` are ignored; otherwise the matrix for the chosen
        ``measure`` is computed from ``X`` (and ``y`` unless ``measure`` is
        ``"feature_mi"``).
        """
        raw = self.matrix if self.matrix is not None else self._compute_matrix(X, y)
        self.matrix_ = self._process(raw)
        return self

    def graph_at(self, threshold: float) -> nx.Graph:
        """Build the graph at ``threshold`` from the cached matrix."""
        return graph_from_matrix(self.matrix_, threshold)

    def build(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        matrix: pd.DataFrame | None = None,
        threshold: float | None = None,
    ) -> nx.Graph:
        """Build the graph, optionally overriding ``matrix`` and ``threshold``.

        A ``matrix`` passed here takes precedence over both ``X``/``y`` and the
        precomputed matrix from the constructor; ``threshold`` defaults to the
        constructor value when omitted.
        """
        if matrix is not None:
            self.matrix = matrix
        if threshold is not None:
            self.threshold = threshold

        self.fit(X, y)
        return self.graph_at(self.threshold)


def build_ii_graphs(
    ii,
    permute_seeds: Sequence[int],
    threshold=None,
    n_bins=None,
    n_edges=None,
):
    """Graphs built from the interaction-information matrix.

    Provide either ``threshold`` (a quantile in [0, 1] applied to the
    CDF-processed matrix, i.e. the fraction of edges to drop) or ``n_edges``
    (keep exactly the top-N pairs). ``n_edges`` takes precedence when both are
    given. Also returns one permuted copy per seed in ``permute_seeds``.
    """
    if n_edges is not None:
        graph_ii = graph_from_matrix_top_n(ii, n_edges)
    else:
        graph_ii = InformationGraphBuilder(n_bins=n_bins).build(
            matrix=ii, threshold=threshold
        )
    graphs = {"ii": graph_ii}
    for s in permute_seeds:
        graphs[f"ii_permuted_{s}"] = permute_nodes(graph_ii, seed=s)
    return graphs
