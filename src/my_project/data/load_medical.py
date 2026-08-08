import copy
import os

import pandas as pd
import yaml
from sklearn import config_context
from sklearn.preprocessing import OrdinalEncoder

from my_project.data.pipelines import discretization_pipeline
from my_project.data.utils import load_column_config, select_modeling_features


def load_medical_data(
    *, data_dir: str | None = None, configs_dir: str | None = None
) -> dict[str, pd.DataFrame | pd.Series | list[str]]:
    """Load the cleaned LBBB/RBBB medical dataset (features + binary target).

    Reads the preprocessed pickle produced by ``notebooks/data_cleaning`` (path
    from ``configs/data.yaml``), keeps the modeling features listed in
    ``configs/columns.py`` and the binary mortality target ``zgon_binary``. Paths
    default to the ``DATA_DIR`` / ``CONFIGS_DIR`` environment variables.

    Returns
    -------
    dict
        ``{"X", "y", "feature_names"}``. ``X`` still has the raw mixed dtypes
        (categorical / boolean / numeric); ``y`` is the 0/1 target named ``"y"``.
    """
    data_dir = data_dir or os.environ["DATA_DIR"]
    configs_dir = configs_dir or os.environ["CONFIGS_DIR"]

    with open(os.path.join(configs_dir, "data.yaml")) as f:
        data_config = yaml.safe_load(f)
    col_cfg = load_column_config(configs_dir)

    df = pd.read_pickle(
        os.path.join(data_dir, data_config["paths"]["preprocessed_data_path"])
    )

    X = select_modeling_features(df, configs_dir).reset_index(drop=True)
    y = pd.Series(df[col_cfg.TARGET].astype(int).to_numpy(), name="y")

    return {"X": X, "y": y, "feature_names": list(X.columns)}


def generate_medical_data(
    n_samples: int | None = None,
    *,
    random_state: int | None = None,
    apply_pipeline: bool = False,
    data_dir: str | None = None,
    configs_dir: str | None = None,
    **_ignored,
) -> dict[str, pd.DataFrame | pd.Series | list]:
    """Load the medical dataset behind the synthetic-generator interface.

    Drop-in replacement for the synthetic generators: it returns the same dict
    contract (``"X"``, ``"y"``, ``"true_interactions"``) so the same experiment
    pipeline can train on the real LBBB/RBBB data.

    The medical data is a *fixed* dataset (5553 labelled rows), so the
    generator-style arguments are handled as follows:

    - ``n_samples``: if given, a **stratified** subsample of exactly that many
      rows is drawn (using ``random_state``); ``None`` returns all rows.
      Requesting more rows than available raises ``ValueError`` (the split
      arithmetic in the training pipeline assumes the frame has exactly
      ``n_samples`` rows).
    - ``apply_pipeline``: when ``False`` (default) the **cleaned but not encoded**
      feature frame is returned with its original mixed dtypes
      (``category``/``object``/``bool`` + numeric). This is what the training
      pipeline expects: it detects categorical columns by dtype and gives them
      learned embeddings (via ``preprocessing_pipeline_nn``), while the graph
      side discretizes separately. Set ``True`` only if you want a ready-made
      all-integer discrete matrix (``discretization_pipeline`` + ordinal
      encoding of leftover categoricals), e.g. for standalone information
      measures without the datamodule.
    - ``random_state`` seeds the subsample only; any other keyword (e.g. ``cov``
      / ``noise_level``) is accepted and ignored so the generic
      ``generate(**GENERATOR_KWARGS)`` wrapper stays generator-agnostic.

    ``true_interactions`` is an empty list: the medical data has genuine feature
    interactions but no known *pairwise* ground-truth interaction graph, so there
    is no oracle edge set to return.

    Returns
    -------
    dict
        ``{"X", "y", "coef", "cov", "true_interactions"}``. ``coef`` and ``cov``
        are ``None`` (no generative parameters exist for a real dataset); the
        keys are kept for parity with the synthetic generators.
    """
    data = load_medical_data(data_dir=data_dir, configs_dir=configs_dir)
    X = data["X"]
    y = data["y"]

    if n_samples is not None:
        if n_samples > len(X):
            raise ValueError(
                f"Medical dataset has only {len(X)} labelled rows; cannot draw "
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

    if apply_pipeline:
        pipe = copy.deepcopy(discretization_pipeline)
        # The pipeline includes custom transformers (e.g. HighMissingDiscretizer)
        # that don't implement `set_output`, so calling `pipe.set_output(...)`
        # raises. Force pandas output through the global config context instead
        # (the same mechanism the GNN notebooks rely on via `set_config`).
        with config_context(transform_output="pandas"):
            X = pipe.fit_transform(X, y)
        # After discretization the numeric columns are ordinal bins, but the
        # categorical/boolean columns are still labels; ordinal-encode them so
        # the whole frame is a discrete integer matrix.
        cat_cols = X.select_dtypes(exclude="number").columns
        if len(cat_cols):
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols]).astype(int)

    return {
        "X": X,
        "y": y,
        "coef": None,
        "cov": None,
        "true_interactions": [],
    }
