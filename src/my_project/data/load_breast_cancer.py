import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_breast_cancer_data() -> dict[str, pd.DataFrame | pd.Series | list[str]]:
    """Load the Wisconsin Diagnostic Breast Cancer dataset from scikit-learn.

    This is the ``sklearn.datasets.load_breast_cancer`` benchmark: 569 samples,
    30 continuous features and a binary target (malignant vs. benign). It ships
    with scikit-learn, so no download or network access is required.

    The 30 features are 10 real-valued nuclear measurements (radius, texture,
    perimeter, area, smoothness, compactness, concavity, concave points,
    symmetry, fractal dimension), each reported as its ``mean``, standard error
    (``error``) and ``worst`` value. Several of these are functionally related
    (e.g. ``perimeter`` ~ ``2*pi*radius`` and ``area`` ~ ``pi*radius**2``), so
    the dataset carries genuine, non-linear feature interactions that were not
    imposed by us -- unlike the synthetic generators.

    The return value mirrors the shape of the other loaders (``X``, ``y`` plus a
    little metadata), so it can be used interchangeably by downstream tooling.

    Returns
    -------
    dict[str, pandas.DataFrame | pandas.Series | list[str]]
        Dictionary with keys ``"X"``, ``"y"`` and ``"feature_names"``.

    X : pandas.DataFrame
        Feature matrix of shape ``(569, 30)`` with the original scikit-learn
        feature names (spaces replaced by underscores).
    y : pandas.Series
        Binary target named ``"y"`` (0 = malignant, 1 = benign, matching the
        scikit-learn convention).
    feature_names : list[str]
        The 30 column names, for convenience.
    """
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    X.columns = [name.replace(" ", "_") for name in X.columns]

    y = pd.Series(bunch.target.to_numpy(), name="y")

    return {
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
    }


def generate_breast_cancer_data(
    n_samples: int | None = None,
    *,
    random_state: int | None = None,
    **_ignored,
) -> dict[str, pd.DataFrame | pd.Series | list]:
    """Load Breast Cancer behind the synthetic-generator interface.

    Drop-in replacement for the synthetic generators: it returns the same dict
    contract (``"X"``, ``"y"``, ``"true_interactions"``) so the same experiment
    pipeline can train on the real Breast Cancer benchmark.

    Breast Cancer is a *fixed* dataset (569 labelled rows), so the
    generator-style arguments are handled as follows:

    - ``n_samples``: if given, a **stratified** subsample of exactly that many
      rows is drawn (using ``random_state``); ``None`` returns all 569 rows.
      Requesting more than 569 rows raises ``ValueError`` (the split arithmetic
      in the training pipeline assumes the frame has exactly ``n_samples`` rows).
    - ``random_state``: seeds the subsample only.
    - any other keyword is accepted and ignored, so the generic
      ``generate(**GENERATOR_KWARGS)`` wrapper can stay generator-agnostic.

    ``true_interactions`` is an empty list: the dataset has genuine feature
    interactions but no known *pairwise* ground-truth interaction graph, so
    there is no oracle edge set to return.

    Returns
    -------
    dict
        ``{"X", "y", "coef", "cov", "true_interactions"}``. ``coef`` and ``cov``
        are ``None`` (no generative parameters exist for a real dataset); the
        keys are kept for parity with the synthetic generators.
    """
    data = load_breast_cancer_data()
    X = data["X"]
    y = data["y"]

    if n_samples is not None:
        if n_samples > len(X):
            raise ValueError(
                f"Breast Cancer has only {len(X)} labelled rows; cannot draw "
                f"n_samples={n_samples}."
            )
        if n_samples < len(X):
            from sklearn.model_selection import train_test_split

            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=n_samples,
                stratify=y,
                random_state=random_state,
            )
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

    return {
        "X": X,
        "y": y,
        "coef": None,
        "cov": None,
        "true_interactions": [],
    }
