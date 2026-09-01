import importlib.util
import os
import warnings
from typing import Any, Callable

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


def prepare_data(
    generate: Callable[..., dict[str, Any]],
    n_samples: int,
    data_seed: int,
    *,
    test_size: int,
    valid_size: int | None = None,
    noise_level: float | None = None,
    feature_selection: dict | None = None,
    **gen_kwargs,
):
    """Generate one dataset and split it into train/valid/test.

    ``generate`` is a bound generator callable (see
    ``my_project.data.generators.build_generator``). Extra ``gen_kwargs``
    (e.g. ``cov``) are forwarded straight to it. When ``feature_selection`` is a
    kwargs dict it is fitted on TRAIN only and applied to valid/test (no
    leakage); true-interaction edges whose endpoints were dropped are removed so
    graph stats stay consistent with the kept features.
    """
    data_dict = generate(n_samples=n_samples, random_state=data_seed, **gen_kwargs)
    X = data_dict["X"]
    y = data_dict["y"]
    true_edges = data_dict["true_interactions"]
    if noise_level and noise_level > 0:
        X = shuffle_X(X, pct=noise_level, seed=data_seed)
    if valid_size is None:
        valid_size = int(0.15 * (n_samples - test_size))
    train_size = n_samples - test_size - valid_size
    splits = split_data(
        X,
        y,
        random_state=data_seed,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
    )
    if feature_selection:
        splits, true_edges = _apply_feature_selection(
            splits, true_edges, feature_selection
        )
    return true_edges, splits


def _apply_feature_selection(splits, true_edges, feature_selection):
    """Fit feature selection on TRAIN only, apply to valid/test (no leakage).

    True-interaction edges whose endpoints were dropped are removed so graph
    stats stay consistent with the kept features.
    """
    from my_project.information_theory.variable_selection import select_features

    X_train, X_valid, X_test, y_train, y_valid, y_test = splits
    selected = select_features(X_train, y_train, **feature_selection)
    splits = (
        X_train[selected],
        X_valid[selected],
        X_test[selected],
        y_train,
        y_valid,
        y_test,
    )
    selected_set = set(selected)
    true_edges = [e for e in true_edges if all(v in selected_set for v in e)]
    return splits, true_edges


def prepare_data_fixed_test(
    generate: Callable[..., dict[str, Any]],
    train_size: int,
    data_seed: int,
    *,
    test_size: int,
    valid_size: int,
    noise_level: float | None = None,
    feature_selection: dict | None = None,
    test_seed: int = 0,
    **gen_kwargs,
):
    """Split with a held-out test set that is shared across train sizes.

    Drop-in variant of :func:`prepare_data` for sweeps over ``train_size``:

    - The test set is determined by ``(test_seed, **gen_kwargs)``. Calls using
        the same values therefore reuse an identical test set across train sizes.
        A caller may derive ``test_seed`` from ``data_seed`` to use a different,
        but still train-size-independent, test set for each data replicate.
    - Train/valid are drawn as an **independent** sample per ``train_size``
        (their own ``generate`` call of ``train_size + valid_size`` rows), so
        different train sizes are NOT nested subsamples of one another.

    Disjointness: the synthetic generators are i.i.d. draws, so test and train
    never share observations. Finite real-data loaders (e.g. HIGGS) that accept
    ``exclude_indexes`` are handed the test rows' pool indices so those rows are
    kept out of the train/valid draw, guaranteeing no leakage there too.
    """
    if test_seed == data_seed:
        # Same seed => the test and train/valid draws share one RNG stream: for
        # synthetic generators the rows overlap (identical when the sizes match).
        # Real loaders using ``exclude_indexes`` are unaffected.
        warnings.warn(
            f"test_seed ({test_seed}) == data_seed ({data_seed}): synthetic "
            "generators will draw overlapping test and train/valid sets. Use a "
            "test_seed different from data_seed.",
            stacklevel=2,
        )
    # Held-out test set determined only by the supplied test seed and generator args.
    test_dict = generate(n_samples=test_size, random_state=test_seed, **gen_kwargs)
    X_test = test_dict["X"]
    y_test = test_dict["y"]

    # Independent train/valid draw for this train_size. ``exclude_indexes`` keeps
    # the test rows out of it for finite real datasets (synthetic gens ignore it).
    train_val_dict = generate(
        n_samples=train_size + valid_size,
        random_state=data_seed,
        exclude_indexes=X_test.index.to_list(),
        **gen_kwargs,
    )
    X_train_val = train_val_dict["X"]
    y_train_val = train_val_dict["y"]
    true_edges = train_val_dict["true_interactions"]

    if noise_level and noise_level > 0:
        X_train_val = shuffle_X(X_train_val, pct=noise_level, seed=data_seed)
        X_test = shuffle_X(X_test, pct=noise_level, seed=test_seed)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val,
        y_train_val,
        test_size=valid_size / (train_size + valid_size),
        random_state=data_seed,
        stratify=y_train_val,
    )
    # Give every split a clean integer index that is *globally unique* across
    # train/valid/test (not three overlapping 0-based ranges): downstream code
    # concatenates the splits into one frame and selects rows by label
    # (``X_all.loc[fold["test"]]``), so colliding labels would pull rows from the
    # wrong split and leak train/valid rows into the test set.
    n_tr, n_va, n_te = len(X_train), len(X_valid), len(X_test)
    train_idx = np.arange(n_tr)
    valid_idx = np.arange(n_tr, n_tr + n_va)
    test_idx = np.arange(n_tr + n_va, n_tr + n_va + n_te)

    def _reindex(obj, idx):
        obj = obj.copy()
        obj.index = idx
        return obj

    splits = (
        _reindex(X_train, train_idx),
        _reindex(X_valid, valid_idx),
        _reindex(X_test, test_idx),
        _reindex(y_train, train_idx),
        _reindex(y_valid, valid_idx),
        _reindex(y_test, test_idx),
    )
    if feature_selection:
        splits, true_edges = _apply_feature_selection(
            splits, true_edges, feature_selection
        )
    return true_edges, splits
