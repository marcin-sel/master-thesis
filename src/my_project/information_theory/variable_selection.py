import numpy as np
import pandas as pd

from my_project.information_theory.estimators import encode_features


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_features: int,
    method: str = "CIFE",
    n_bins: int | None = 5,
) -> list[str]:
    """Select an informative subset of features (top-``k`` by a MI criterion).

    Standalone preprocessing step, meant to run *before* any graph
    construction: it just picks which columns to keep. It delegates to
    :mod:`skfeature`'s information-theoretic feature selectors, which implement
    the greedy, redundancy-aware criteria from Brown et al. (2012) -- so besides
    plain joint mutual information you also get CIFE and friends out of the box.

    Supported ``method`` values (case-insensitive):

    - ``"JMI"``  -- Joint Mutual Information (default).
    - ``"JMIM"`` -- alias of ``"JMI"`` (kept for backwards compatibility).
    - ``"CIFE"`` -- Conditional Infomax Feature Extraction (double-counts the
      class-conditional redundancy term; strong at picking up interactions).
    - ``"MRMR"`` -- Max-Relevance Min-Redundancy.
    - ``"MIM"``  -- Mutual Information Maximisation (relevance only).
    - ``"DISR"`` -- Double Input Symmetrical Relevance.
    - ``"CMIM"`` -- Conditional Mutual Information Maximisation.
    - ``"ICAP"`` -- Interaction Capping.
    - ``"MIFS"`` -- Mutual Information Feature Selection.

    Parameters
    ----------
    X, y:
        Feature frame and target. ``skfeature`` estimates mutual information from
        contingency tables, so it needs *discrete* inputs; continuous columns
        (e.g. Madelon) are binned automatically -- see ``n_bins``.
    n_features:
        Number of features to keep (top-``k`` by the chosen criterion). Clamped
        to the number of available columns.
    method:
        Which selection criterion to use (see the list above).
    n_bins:
        Quantile bins used to discretize columns that have more than ``n_bins``
        distinct values. Already-discrete columns (binary flags, low-cardinality
        categoricals) are left untouched. Pass ``None`` to skip binning entirely
        (only safe when every column is already discrete).

    Returns
    -------
    list[str]
        The selected column names in selection order (most informative first).
    """
    # Imported lazily so importing this module (and the graph builders) doesn't
    # pull in skfeature unless feature selection is actually used.
    import importlib

    # "JMIM" isn't a distinct skfeature criterion; treat it as JMI so existing
    # configs keep working.
    key = method.strip().upper()
    key = "JMI" if key == "JMIM" else key
    available = ["JMI", "JMIM", "CIFE", "MRMR", "MIM", "DISR", "CMIM", "ICAP", "MIFS"]
    try:
        # Each criterion lives in its own submodule, e.g.
        # ``skfeature.function.information_theoretical_based.CIFE``.
        module = importlib.import_module(
            f"skfeature.function.information_theoretical_based.{key}"
        )
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Unknown method {method!r}; choose one of {available}."
        ) from exc
    selector = getattr(module, key.lower())

    n_keep = min(int(n_features), X.shape[1])

    # skfeature needs integer-coded, discrete features: bin the high-cardinality
    # (continuous) columns into quantile bins, then ordinal-encode everything to
    # contiguous integer codes.
    X_disc = X.copy()
    if n_bins is not None:
        to_bin = X_disc.nunique()
        to_bin = to_bin[to_bin > n_bins].index
        if len(to_bin):
            X_disc[to_bin] = X_disc[to_bin].apply(
                pd.qcut, q=n_bins, labels=False, duplicates="drop"
            )
    X_disc = encode_features(X_disc)

    y_codes = np.asarray(pd.factorize(y)[0])
    # ``mode="index"`` returns the selected column indices in selection order
    # (most informative first).
    indices = np.asarray(
        selector(X_disc.to_numpy(), y_codes, mode="index", n_selected_features=n_keep)
    )[:n_keep]
    return [X.columns[i] for i in indices]
