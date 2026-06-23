"""
Interaction-information based feature graphs.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import OrdinalEncoder

from my_project.graphs.base import GraphBuilder
from my_project.graphs.utils import graph_from_matrix

MEASURES = (
    "feature_mi",
    "joint_target_mi",
    "interaction_information",
)


def _mi_joint(stacked: np.ndarray, y: np.ndarray, *, estimator: str = "ML") -> float:
    """I((X1, X2, ...); y) for a stack of variables (one row per variable)."""
    return float(
        drv.entropy_joint(stacked, estimator=estimator)
        + drv.entropy(y, estimator=estimator)
        - drv.entropy_joint(np.vstack([stacked, y]), estimator=estimator)
    )


def _mi_pair(x1: np.ndarray, x2: np.ndarray, *, estimator: str = "ML") -> float:
    """I(x1; x2): mutual information between two features, independent of y."""
    return float(drv.information_mutual(x1, x2, estimator=estimator))


def _pair_measures(
    x1: np.ndarray,
    x2: np.ndarray,
    y_codes: np.ndarray | None,
    *,
    want_joint: bool,
    want_feature_mi: bool,
    estimator: str,
) -> tuple[float | None, float | None]:
    """Compute the per-pair measures for one ``(x1, x2)`` pair.

    Returns ``(joint_mi, feature_mi)``; entries that were not requested are
    ``None`` so an unrequested value can never be silently used as a real
    score. This is the unit of work dispatched to each parallel job.
    """
    joint_mi = (
        _mi_joint(np.vstack([x1, x2]), y_codes, estimator=estimator)
        if want_joint
        else None
    )
    feature_mi = _mi_pair(x1, x2, estimator=estimator) if want_feature_mi else None
    return joint_mi, feature_mi


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode every column to integer codes pyitlib can consume.

    Missing values are encoded as their own category (code ``0``) so that
    "missingness" is treated as an informative symbol instead of crashing the
    integer cast. Known categories are shifted to ``1..K``.
    """
    encoder = OrdinalEncoder(encoded_missing_value=-1)
    encoded = np.asarray(encoder.fit_transform(X)) + 1
    return pd.DataFrame(encoded.astype(int), columns=X.columns, index=X.index)


def compute_information_matrices(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    *,
    measures: str | Sequence[str] = MEASURES,
    encode: bool = True,
    estimator: str = "ML",
    n_bins: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Return the requested pairwise information matrices keyed by measure name.

    ``measures`` selects any subset of:

    - ``"joint_target_mi"``: joint mutual information ``I((Xi, Xj); y)`` -- how much the
      pair together tells about the target.
    - ``"interaction_information"``: interaction information ``II(Xi; Xj; y)`` (pyitlib
      convention, positive => synergy).
    - ``"feature_mi"``: feature-feature mutual information ``I(Xi; Xj)``, which
      ignores ``y`` and captures redundancy / collinearity between features.

    Each returned matrix is a symmetric ``DataFrame`` indexed by the columns of
    ``X`` with a zero diagonal. ``y`` is required unless ``measures`` requests
    only ``"feature_mi"``. ``estimator`` is forwarded to pyitlib. Pass a single
    measure name to get a one-entry dict. ``n_jobs`` is forwarded to
    :class:`joblib.Parallel`, spreading the per-pair work across processes
    (``None``/``1`` runs serially, ``-1`` uses all cores).
    """
    if isinstance(measures, str):
        measures = (measures,)
    measures = tuple(measures)
    if not measures:
        raise ValueError("Provide at least one measure.")
    unknown = set(measures) - set(MEASURES)
    if unknown:
        raise ValueError(f"Unknown measures {sorted(unknown)}; choose from {MEASURES}.")

    if n_bins is not None:
        X = (
            X.apply(pd.qcut, q=n_bins, axis=0, labels=False).copy()
            if n_bins
            else X.copy()
        )

    if encode:
        X = encode_features(X)

    columns = list(X.columns)
    arrays = {column: X[column].to_numpy() for column in columns}

    needs_y = any(m in ("joint_target_mi", "interaction_information") for m in measures)
    if needs_y:
        if y is None:
            raise ValueError(f"measures {measures} require y.")
        y_codes = np.asarray(pd.factorize(y)[0])

    # The expensive term I((Xi, Xj); y) is shared by both target-aware
    # measures: interaction information is just that joint MI minus the two
    # cheap single-feature terms, II = I((Xi, Xj); y) - I(Xi; y) - I(Xj; y).
    # So we compute the pairwise joint MI once and, when interaction is also
    # requested, precompute each feature's I(Xi; y) (n cheap calls) and derive
    # interaction by subtraction instead of recomputing the joint entropy.
    needs_joint = "joint_target_mi" in measures or "interaction_information" in measures
    want_feature_mi = "feature_mi" in measures
    if "interaction_information" in measures:
        feature_target_mi = {
            column: _mi_pair(arrays[column], y_codes, estimator=estimator)
            for column in columns
        }

    results = {
        measure: pd.DataFrame(0.0, index=columns, columns=columns)
        for measure in measures
    }

    # All unordered feature pairs are the units of work; compute each pair's
    # measures in parallel, then scatter the results into the matrices.
    pairs = [(i, j) for i in range(len(columns)) for j in range(i + 1, len(columns))]
    values = Parallel(n_jobs=n_jobs)(
        delayed(_pair_measures)(
            arrays[columns[i]],
            arrays[columns[j]],
            y_codes if needs_y else None,
            want_joint=needs_joint,
            want_feature_mi=want_feature_mi,
            estimator=estimator,
        )
        for i, j in pairs
    )

    for (i, j), (joint_mi, feature_mi) in zip(pairs, values):
        c1, c2 = columns[i], columns[j]
        if "joint_target_mi" in results:
            results["joint_target_mi"].iat[i, j] = results["joint_target_mi"].iat[
                j, i
            ] = joint_mi
        if "interaction_information" in results:
            value = joint_mi - feature_target_mi[c1] - feature_target_mi[c2]
            results["interaction_information"].iat[i, j] = results[
                "interaction_information"
            ].iat[j, i] = value
        if "feature_mi" in results:
            results["feature_mi"].iat[i, j] = results["feature_mi"].iat[j, i] = (
                feature_mi
            )

    return results


def to_probability(matrix: pd.DataFrame) -> pd.DataFrame:
    """Map each entry to its empirical CDF rank (quantile) in ``[0, 1]``.

    The matrix is symmetric with a zero diagonal, so the reference distribution
    is built from the upper triangle only (each pair counted once, diagonal
    excluded); otherwise every value would be double-counted and the diagonal
    zeros would skew the quantiles. The resulting matrix stays symmetric.
    """
    values = matrix.to_numpy()
    upper = values[np.triu_indices_from(values, k=1)]
    flat = np.sort(upper)
    n = len(flat)

    def cdf(value: float) -> float:
        return np.searchsorted(flat, value, side="right") / n

    return matrix.map(cdf)


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
        Probability estimator forwarded to :mod:`pyitlib` (``"ML"`` for the
        plug-in maximum-likelihood / relative-frequency estimator, or a
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
        estimator: str = "ML",
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
