"""Pairwise information matrices over a feature set."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from my_project.information_theory.estimators import (
    MEASURES,
    _mi_pair,
    _pair_measures,
    encode_features,
)


def compute_information_matrices(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    *,
    measures: str | Sequence[str] = MEASURES,
    encode: bool = True,
    estimator: str = "numpy",
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
    only ``"feature_mi"``. ``estimator`` selects the mutual-information
    estimator: ``"numpy"`` (default) is a fast vectorized plug-in
    (relative-frequency / maximum-likelihood) estimator computed via contingency
    tables; any other value is forwarded to :mod:`pyitlib` as its ``estimator``
    (e.g. ``"ML"``, ``"JAMES-STEIN"``). Values are in bits. Pass a single
    measure name to get a one-entry dict. ``n_jobs`` is forwarded to
    :class:`joblib.Parallel`, spreading the per-pair work across processes
    (``None``/``1`` runs serially, ``-1`` uses all cores).
    """

    X = X.copy()
    y = y.copy() if y is not None else None

    if isinstance(measures, str):
        measures = (measures,)
    measures = tuple(measures)
    if not measures:
        raise ValueError("Provide at least one measure.")
    unknown = set(measures) - set(MEASURES)
    if unknown:
        raise ValueError(f"Unknown measures {sorted(unknown)}; choose from {MEASURES}.")

    if n_bins is not None:
        X_nunique = X.nunique()
        to_bin_cols = X_nunique[X_nunique > n_bins].index
        if not to_bin_cols.any():
            raise ValueError(
                f"n_bins={n_bins} is too large; all {X.shape[1]} columns have "
                f"fewer unique values than that."
            )
        X[to_bin_cols] = (
            X[to_bin_cols]
            .apply(pd.qcut, q=n_bins, axis=0, labels=False, duplicates="drop")
            .copy()
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


def compute_info(X_train, y_train, n_bins=None):
    """Interaction-information matrix over the train split."""
    matrices = compute_information_matrices(X_train, y_train, n_bins=n_bins)
    return matrices["interaction_information"]


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

    result = matrix.map(cdf)
    # np.fill_diagonal(result.values, 0.0)

    return result
