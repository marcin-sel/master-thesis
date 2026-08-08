"""Low-level information-theoretic estimators for feature pairs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import OrdinalEncoder

MEASURES = (
    "feature_mi",
    "joint_target_mi",
    "interaction_information",
)


def _combine_codes(rows: Sequence[np.ndarray]) -> np.ndarray:
    """Fold several integer code arrays (one variable each) into a single code.

    Each row is shifted to start at ``0`` and mixed in with a positional radix
    (``combined * cardinality + row``), so the result is a unique integer code
    per joint outcome -- i.e. the code of the tuple ``(row_0, row_1, ...)``.
    """
    combined: np.ndarray | None = None
    for row in rows:
        row = np.asarray(row)
        row = row - row.min()
        if combined is None:
            combined = row.astype(np.int64)
        else:
            combined = combined * (int(row.max()) + 1) + row
    return combined


def _mi_from_codes(a: np.ndarray, b: np.ndarray) -> float:
    """I(a; b) in bits from a plug-in (relative-frequency) contingency table.

    Both inputs must be integer code arrays. This is the vectorized
    maximum-likelihood ("ML") estimator: it matches pyitlib's ``estimator="ML"``
    up to floating point, but is computed from a single ``np.bincount`` instead
    of per-call entropy estimation, which is dramatically faster.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a = a - a.min()
    b = b - b.min()
    ca = int(a.max()) + 1
    cb = int(b.max()) + 1
    n = a.shape[0]
    joint = np.bincount(a * cb + b, minlength=ca * cb).astype(np.float64)
    joint = joint.reshape(ca, cb)
    pij = joint / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    mask = joint > 0
    outer = pi @ pj
    return float(np.sum(pij[mask] * np.log2(pij[mask] / outer[mask])))


def _mi_joint(stacked: np.ndarray, y: np.ndarray, *, estimator: str = "numpy") -> float:
    """I((X1, X2, ...); y) for a stack of variables (one row per variable)."""
    if estimator == "numpy":
        return _mi_from_codes(_combine_codes(stacked), y)
    return float(
        drv.entropy_joint(stacked, estimator=estimator)
        + drv.entropy(y, estimator=estimator)
        - drv.entropy_joint(np.vstack([stacked, y]), estimator=estimator)
    )


def _mi_pair(x1: np.ndarray, x2: np.ndarray, *, estimator: str = "numpy") -> float:
    """I(x1; x2): mutual information between two features, independent of y."""
    if estimator == "numpy":
        return _mi_from_codes(x1, x2)
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
