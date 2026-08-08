import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import zscore

INTERACTION_DICTS = {
    1: {
        ("x1", "x2"): lambda a, b: np.log(np.abs(a)) * b,
        ("x3", "x4"): lambda a, b: a * b,
        ("x5", "x6"): lambda a, b: a * b,
        ("x7", "x8"): lambda a, b: int(a > 0) ** int(b > 0),
        ("x8", "x9"): lambda a, b: np.log(a**2 + b**2 + 1.0),
    },
    2: {
        ("x1", "x2"): lambda a, b: np.power(np.pi, a * b),
        ("x1", "x3"): lambda a, b: np.power(np.pi, a * b),
        ("x2", "x3"): lambda a, b: np.power(np.pi, a * b),
        ("x4", "x5"): lambda a, b: -np.arcsin(0.5 * a) - np.arcsin(0.5 * b),
        ("x3", "x5"): lambda a, b: np.log(np.abs(a + b) + 1.0),
        ("x7", "x8"): lambda a, b: (a * b) / (1.0 + np.abs(b)),
        ("x7", "x9"): lambda a, b: (a * b) / (1.0 + np.abs(b)),
        ("x7", "x10"): lambda a, b: (a * b) / (1.0 + np.abs(b)),
        ("x8", "x9"): lambda a, b: (a * b) / (1.0 + np.abs(b)),
        ("x8", "x10"): lambda a, b: (a * b) / (1.0 + np.abs(b)),
        ("x9", "x10"): lambda a, b: (a * b) / (1.0 + np.abs(b)),
        ("x2", "x7"): lambda a, b: -a * b,
    },
    3: {
        ("x1", "x2"): lambda a, b: np.exp(np.abs(a - b)),
        ("x2", "x3"): lambda a, b: np.abs(a * b),
        ("x3", "x4"): lambda a, b: -np.power(a**2, np.abs(b)),
        ("x4", "x5"): lambda a, b: np.log(a**2 + b**2 + 1.0),
        ("x4", "x7"): lambda a, b: np.log(a**2 + b**2 + 1.0),
        ("x4", "x8"): lambda a, b: np.log(a**2 + b**2 + 1.0),
        ("x5", "x7"): lambda a, b: np.log(a**2 + b**2 + 1.0),
        ("x5", "x8"): lambda a, b: np.log(a**2 + b**2 + 1.0),
        ("x7", "x8"): lambda a, b: np.log(a**2 + b**2 + 1.0),
    },
    4: {
        ("x1", "x2"): lambda a, b: np.exp(np.abs(a - b)),
        ("x2", "x3"): lambda a, b: np.abs(a * b),
        ("x3", "x4"): lambda a, b: -np.power(a**2, np.abs(b)),
        ("x1", "x4"): lambda a, b: (a * b) ** 2,
        ("x4", "x5"): lambda a, b: np.log(a**2 + b**2 + 1.0),
        ("x4", "x7"): lambda a, b: np.log(a**2 + b**2 + 1.0),
        ("x4", "x8"): lambda a, b: np.log(a**2 + b**2 + 1.0),
        ("x5", "x7"): lambda a, b: np.log(a**2 + b**2 + 1.0),
        ("x5", "x8"): lambda a, b: np.log(a**2 + b**2 + 1.0),
        ("x7", "x8"): lambda a, b: np.log(a**2 + b**2 + 1.0),
    },
    5: {
        ("x1", "x2"): lambda a, b: 1.0 / (1.0 + a**2 + b**2),
        ("x1", "x3"): lambda a, b: 1.0 / (1.0 + a**2 + b**2),
        ("x2", "x3"): lambda a, b: 1.0 / (1.0 + a**2 + b**2),
        ("x4", "x5"): lambda a, b: np.sqrt(np.exp(a + b)),
        ("x6", "x7"): lambda a, b: np.abs(a + b),
        ("x8", "x9"): lambda a, b: a * b,
        ("x8", "x10"): lambda a, b: a * b,
        ("x9", "x10"): lambda a, b: a * b,
    },
    6: {
        ("x1", "x2"): lambda a, b: np.exp(np.abs(a * b) + 1.0),
        ("x3", "x4"): lambda a, b: -np.exp(np.abs(a + b) + 1.0),
        ("x5", "x6"): lambda a, b: np.cos(a + b),
        ("x5", "x8"): lambda a, b: np.cos(a - b),
        ("x6", "x8"): lambda a, b: np.cos(a - b),
        ("x8", "x9"): lambda a, b: np.sqrt(a**2 + b**2),
        ("x8", "x10"): lambda a, b: np.sqrt(a**2 + b**2),
        ("x9", "x10"): lambda a, b: np.sqrt(a**2 + b**2),
    },
    7: {
        ("x1", "x2"): lambda a, b: (np.arctan(a) + np.arctan(b)) ** 2,
        ("x3", "x4"): lambda a, b: np.maximum(a * b, 0.0),
        ("x3", "x6"): lambda a, b: np.maximum(a * b, 0.0),
        ("x4", "x6"): lambda a, b: np.maximum(a * b, 0.0),
        ("x4", "x5"): lambda a, b: -1.0 / (1.0 + (a * b) ** 2),
        ("x4", "x7"): lambda a, b: -1.0 / (1.0 + (a * b) ** 2),
        ("x4", "x8"): lambda a, b: -1.0 / (1.0 + (a * b) ** 2),
        ("x5", "x6"): lambda a, b: -1.0 / (1.0 + (a * b) ** 2),
        ("x5", "x7"): lambda a, b: -1.0 / (1.0 + (a * b) ** 2),
        ("x5", "x8"): lambda a, b: -1.0 / (1.0 + (a * b) ** 2),
        ("x6", "x7"): lambda a, b: -1.0 / (1.0 + (a * b) ** 2),
        ("x6", "x8"): lambda a, b: -1.0 / (1.0 + (a * b) ** 2),
        ("x7", "x8"): lambda a, b: -1.0 / (1.0 + (a * b) ** 2),
        ("x7", "x9"): lambda a, b: (np.abs(a) / (1.0 + np.abs(b))) ** 5,
    },
    8: {
        ("x1", "x2"): lambda a, b: a * b,
        ("x3", "x5"): lambda a, b: np.power(2.0, a + b),
        ("x3", "x6"): lambda a, b: np.power(2.0, a + b),
        ("x5", "x6"): lambda a, b: np.power(2.0, a + b),
        ("x3", "x4"): lambda a, b: np.power(2.0, a + b),
        ("x3", "x7"): lambda a, b: np.power(2.0, a + b),
        ("x4", "x5"): lambda a, b: np.power(2.0, a + b),
        ("x4", "x7"): lambda a, b: np.power(2.0, a + b),
        ("x5", "x7"): lambda a, b: np.power(2.0, a + b),
        ("x7", "x8"): lambda a, b: np.sin(a * np.sin(b)),
        ("x7", "x9"): lambda a, b: np.sin(a * np.sin(b)),
        ("x8", "x9"): lambda a, b: np.sin(a + b),
    },
    9: {
        ("x1", "x2"): lambda a, b: np.tanh(a * b),
        ("x1", "x3"): lambda a, b: np.tanh(a * b),
        ("x1", "x4"): lambda a, b: np.tanh(a * b),
        ("x1", "x5"): lambda a, b: np.tanh(a * b),
        ("x2", "x3"): lambda a, b: np.tanh(a * b),
        ("x2", "x4"): lambda a, b: np.tanh(a * b),
        ("x2", "x5"): lambda a, b: np.tanh(a * b),
        ("x3", "x4"): lambda a, b: np.tanh(a * b),
        ("x3", "x5"): lambda a, b: np.tanh(a * b),
        ("x4", "x5"): lambda a, b: np.tanh(a * b),
        ("x5", "x6"): lambda a, b: np.exp(a + b),
        ("x6", "x7"): lambda a, b: np.log((a * b) ** 2 + 1.0),
        ("x6", "x8"): lambda a, b: np.log((a * b) ** 2 + 1.0),
        ("x7", "x8"): lambda a, b: np.log((a * b) ** 2 + 1.0),
        ("x9", "x10"): lambda a, b: a * b,
    },
    10: {
        ("x1", "x2"): lambda a, b: np.sinh(a + b),
        ("x3", "x5"): lambda a, b: np.arccos(np.tanh(a + b)),
        ("x3", "x7"): lambda a, b: np.arccos(np.tanh(a + b)),
        ("x5", "x7"): lambda a, b: np.arccos(np.tanh(a + b)),
        ("x4", "x5"): lambda a, b: np.cos(a + b),
        ("x7", "x9"): lambda a, b: 1.0 / np.cos(a * b),
    },
}


def f_function(X, interaction_dict):
    """Evaluate each interaction term, one column per feature tuple.

    ``X`` is indexed by feature name (DataFrame or mapping). Every
    ``(name_1, ..., name_k) -> func`` entry produces a column
    ``"name_1_..._name_k"`` equal to ``func(X[name_1], ..., X[name_k])``,
    evaluated row by row, so ``func`` must accept exactly ``k`` scalar
    arguments (the arity matches the length of the key tuple).
    """
    X_interactions = pd.DataFrame(index=getattr(X, "index", None))
    for names, func in interaction_dict.items():
        column = "_".join(names)
        X_interactions[column] = X.apply(
            lambda row, names=names, func=func: func(*(row[name] for name in names)),
            axis=1,
        )
    return X_interactions


def _resolve_interactions(interactions):
    """Normalise the ``interactions`` argument to a ``{(i, j): callable}`` dict.

    Accepts:
    - ``None`` -> no interactions;
    - an ``int`` -> the matching entry of :data:`INTERACTION_DICTS`;
    - a ``dict`` ``{(name_i, name_j): callable}`` -> used as-is;
    - any iterable of ``(name_i, name_j)`` pairs -> each defaults to the
      product ``a * b``.
    """
    if interactions is None:
        return {}
    if isinstance(interactions, int):
        return dict(INTERACTION_DICTS[interactions])
    if isinstance(interactions, dict):
        return dict(interactions)
    return {tuple(pair): (lambda a, b: a * b) for pair in interactions}


def _make_covariance(cov, p, df=1, V=None, return_corr=True, shrinkage=0.0, rng=None):
    if isinstance(cov, (int, float)) and not isinstance(cov, bool):
        # A single scalar ``rho`` builds an equicorrelation matrix: unit
        # diagonal with every off-diagonal equal to ``rho``. Positive definite
        # for ``-1/(p-1) < rho < 1``.
        rho = float(cov)
        if p > 1 and not -1.0 / (p - 1) < rho < 1.0:
            raise ValueError(
                f"Scalar correlation must satisfy -1/(p-1) < rho < 1 "
                f"(p={p}); got {rho}."
            )
        cov = np.full((p, p), rho, dtype=float)
        np.fill_diagonal(cov, 1.0)
        return cov

    if cov is None:
        # ``df`` controls the Wishart degrees of freedom; ``df >= p`` keeps the
        # drawn matrix full-rank (positive definite). Default to ``p`` when not
        # provided so a ``None`` doesn't break the ``np.eye(p) / df`` scaling.
        if df is None:
            df = p

        if V is None:
            V = np.eye(p) / df
        else:
            V = np.asarray(V, dtype=float)
            if V.shape != (p, p):
                raise ValueError(f"V must have shape ({p}, {p}), " f"got {V.shape}.")
            if not np.allclose(V, V.T):
                raise ValueError("V must be symmetric.")
            if np.any(np.linalg.eigvalsh(V) <= 0):
                raise ValueError("V must be positive definite.")

        a = rng.multivariate_normal(mean=np.zeros(p), cov=V, size=df)
        cov = a.T @ a

        if shrinkage > 0.0:
            cov = (1.0 - shrinkage) * cov + shrinkage * np.eye(p)

    else:
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (p, p):
            raise ValueError(f"cov must have shape ({p}, {p}), " f"got {cov.shape}.")
        if not np.allclose(cov, cov.T):
            raise ValueError("cov must be symmetric.")
        if np.any(np.linalg.eigvalsh(cov) <= 0):
            raise ValueError("cov must be positive definite.")

    if return_corr:
        d = np.sqrt(np.diag(cov))
        cov = cov / np.outer(d, d)

    return cov


def generate_pairwise_interaction_data(
    n_samples: int = 5000,
    *,
    n_informative: int = 5,
    interactions=None,
    cov: float | pd.DataFrame | None = None,
    cov_shrinkage: float = 0.0,
    cov_df: int | None = None,
    coef: float | None = None,
    n_redundant: int = 0,
    n_repeated: int = 0,
    n_noise: int = 0,
    coef_loc: float = 0.0,
    coef_scale: float = 1.0,
    intercept: float = 0.0,
    flip_y: float = 0.0,
    main_effects: bool = True,
    standardize: bool = True,
    shuffle: bool = True,
    random_state: int | None = None,
    return_interactions: bool = False,
) -> dict[str, pd.DataFrame | pd.Series | list[str]]:
    """Generate a classification problem with explicit pairwise interactions.

    A ``make_classification``-style generator where the label is a noisy logit
    of a random linear combination of the *informative* features and a set of
    *pairwise interaction* terms.


    Parameters
    ----------
    interactions:
        Pairwise interactions. ``None`` for none, an ``int`` to pick an entry of
        :data:`INTERACTION_DICTS`, a ``{(name_i, name_j): callable}`` dict, or an
        iterable of ``(name_i, name_j)`` pairs (each defaulting to ``a * b``).
        Referenced names must lie in ``x1..x{n_informative}``.
    coef_loc:
        Mean of the normal distribution for the informative feature coefficients.
    coef_scale:
        Standard deviation of the normal distribution for the informative feature coefficients.
    cov:
        Covariance matrix ``(n_informative, n_informative)`` for the informative
        features; must be symmetric positive-definite. A single scalar ``rho``
        builds an equicorrelation matrix (unit diagonal, every off-diagonal
        equal to ``rho``), valid for ``-1/(n_informative-1) < rho < 1``.
        ``None`` draws a random correlation matrix.
    standardize:
        If ``True`` (default), the informative and interaction columns are
        z-scored (zero mean, unit variance) before forming the logit, and the
        returned ``X`` contains those standardized columns. If ``False``, raw
        columns are used both for the logit and in ``X``.
    main_effects:
        If ``True`` (default), the label depends on a linear combination of the
        informative features in addition to the interaction terms. If ``False``,
        the informative main effects are dropped from the logit (their
        coefficients are zeroed), so the label is driven *purely* by the pairwise
        interaction terms (plus the intercept). The informative features still
        appear in ``X``.
    return_interactions:
        If ``True``, the interaction columns are appended to ``X`` (otherwise
        they only drive the label and stay hidden).

    Returns
    -------
    dict[str, pd.DataFrame | pd.Series | list[str]]
        Dictionary with keys ``"X"``, ``"y"``, ``"coef"``, ``"cov"``, and ``"true_interactions"``.

    X : pandas.DataFrame
        Feature matrix (informative ``x*`` + ``red_*`` + ``rep_*`` + ``noise_*``,
        plus the interaction columns when ``return_interactions=True``). Columns
        are identified by name and survive shuffling.
    y : pandas.Series
        Binary target.
    coef : pandas.Series
        Generative coefficients indexed by name, in the order
        ``["intercept", x1..xN, interaction columns in dict order]``.
    cov : pandas.DataFrame
        Covariance matrix of the informative features, indexed by name.
    cov_shrinkage : float
        Shrinkage factor applied to the covariance matrix (default 0.0).
    true_interactions : list[str]
        List of the interaction column names that were used to generate the label.
    """
    rng = np.random.default_rng(random_state)

    if n_informative < 1:
        raise ValueError("n_informative must be at least 1.")

    interaction_dict = _resolve_interactions(interactions)

    names = [f"x{i}" for i in range(1, n_informative + 1)]
    name_set = set(names)
    for feature_names in interaction_dict:
        outside = [name for name in feature_names if name not in name_set]
        if outside:
            raise ValueError(
                f"Interaction {tuple(feature_names)!r} references features "
                f"outside x1..x{n_informative}: {outside}."
            )

    cov = _make_covariance(
        cov=cov,
        p=n_informative,
        df=cov_df,
        return_corr=True,
        shrinkage=cov_shrinkage,
        rng=rng,
    )

    X_inf = pd.DataFrame(
        rng.multivariate_normal(np.zeros(n_informative), cov, size=n_samples),
        columns=names,
    )

    cov = pd.DataFrame(cov, index=names, columns=names)

    X_int = f_function(X_inf, interaction_dict)
    if not np.all(np.isfinite(X_int.to_numpy())):
        bad = int(np.count_nonzero(~np.isfinite(X_int.to_numpy())))
        raise FloatingPointError(
            f"Interactions produced {bad} non-finite values; check that the "
            "interaction functions match the feature domain."
        )

    if standardize:
        X_inf = pd.DataFrame(
            zscore(X_inf.to_numpy(), axis=0, ddof=0),
            columns=X_inf.columns,
            index=X_inf.index,
        )
        if interaction_dict:
            X_int = pd.DataFrame(
                zscore(X_int.to_numpy(), axis=0, ddof=0),
                columns=X_int.columns,
                index=X_int.index,
            )

    w_inf = rng.normal(coef_loc, coef_scale, size=n_informative)
    if not main_effects:
        w_inf = np.zeros(n_informative)
    logit = float(intercept) + X_inf.to_numpy() @ w_inf

    if interaction_dict:
        w_int = rng.normal(coef_loc, coef_scale, size=X_int.shape[1])
        logit = logit + X_int.to_numpy() @ w_int
    else:
        w_int = np.empty(0)

    y = rng.binomial(1, expit(logit))

    if flip_y > 0:
        flip_mask = rng.random(n_samples) < flip_y
        y[flip_mask] = 1 - y[flip_mask]

    X = X_inf.copy()

    if n_redundant > 0:
        B = 2.0 * rng.uniform(size=(n_informative, n_redundant)) - 1.0
        redundant = X_inf.to_numpy() @ B
        for k in range(n_redundant):
            X[f"red_{k + 1}"] = redundant[:, k]

    if n_repeated > 0:
        pool = list(X.columns)  # informative + redundant
        picks = rng.integers(0, len(pool), size=n_repeated)
        for k, pick in enumerate(picks):
            X[f"rep_{k + 1}"] = X[pool[pick]].to_numpy()

    if n_noise > 0:
        noise = rng.standard_normal(size=(n_samples, n_noise))
        for k in range(n_noise):
            X[f"noise_{k + 1}"] = noise[:, k]

    if return_interactions:
        for col in X_int.columns:
            X[col] = X_int[col].to_numpy()

    y = pd.Series(y, name="y")

    if shuffle:
        row_perm = rng.permutation(n_samples)
        X = X.iloc[row_perm].reset_index(drop=True)
        y = y.iloc[row_perm].reset_index(drop=True)

    coef = pd.Series(
        np.concatenate([[float(intercept)], w_inf, w_int]),
        index=["intercept", *names, *X_int.columns],
        name="coef",
    )

    return {
        "X": X,
        "y": y,
        "coef": coef,
        "cov": cov,
        "true_interactions": list(interaction_dict.keys()),
    }
