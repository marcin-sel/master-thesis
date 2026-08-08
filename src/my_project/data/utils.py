import importlib.util
import os

import numpy as np
from sklearn.model_selection import train_test_split


def load_column_config(configs_dir=None):
    """Load ``configs/columns.py`` as a module (column whitelists)."""
    configs_dir = configs_dir or os.environ["CONFIGS_DIR"]
    path = os.path.join(configs_dir, "columns.py")
    spec = importlib.util.spec_from_file_location("project_columns", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_modeling_features(df, configs_dir=None):
    """Return the ``MODELING_FEATURES`` columns present in ``df`` (df order kept)."""
    features = set(load_column_config(configs_dir).MODELING_FEATURES)
    return df[[c for c in df.columns if c in features]]


def columns_info_func(df, sort_by_column="missing_pct", ascending=False):
    columns_info = df.isna().mean().rename("missing_pct").to_frame()
    columns_info["dtype"] = df.dtypes.values
    columns_info["n_unique"] = df.nunique().values

    columns_info = columns_info.sort_values(by=sort_by_column, ascending=ascending)
    columns_info["cumsum_pct"] = (np.arange(len(columns_info)) + 1) / len(columns_info)

    min_fraction = df.apply(
        lambda x: x.value_counts(normalize=True, dropna=False)
        .sort_values()
        .reset_index()
        .iloc[0]
        .rename({x.name: "min_var", "proportion": "min_proportion"}),
        axis=0,
    ).T
    max_fraction = df.apply(
        lambda x: x.value_counts(normalize=True, dropna=False)
        .sort_values()
        .reset_index()
        .iloc[-1]
        .rename({x.name: "max_var", "proportion": "max_proportion"}),
        axis=0,
    ).T
    columns_info = columns_info.merge(
        min_fraction, left_index=True, right_index=True, how="left"
    )
    columns_info = columns_info.merge(
        max_fraction, left_index=True, right_index=True, how="left"
    )
    columns_info["is_categorical"] = columns_info.dtype.astype(str).isin(
        ["object", "category"]
    )

    return columns_info


def split_data(X, y, random_state=42, train_size=0.7, valid_size=0.10, test_size=0.20):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval,
        y_trainval,
        test_size=valid_size / (train_size + valid_size),
        random_state=random_state,
        stratify=y_trainval,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def shuffle_X(X, pct, seed):
    X_shuffled = X.copy()
    rng = np.random.default_rng(seed)
    for col in X.columns:
        mask = rng.random(len(X)) < pct
        X_shuffled.loc[mask, col] = rng.permutation(X_shuffled.loc[mask, col].values)
    return X_shuffled
