import numpy as np
import pandas as pd
from scipy.special import expit


def _unpack_features(X):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[1] != 10:
        raise ValueError("X has to have shape (n_samples, 10).")

    return X.T


def f1(X):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = _unpack_features(X)

    truth = [
        ("x1", "x2"),
        ("x1", "x3"),
        ("x2", "x3"),
        ("x3", "x5"),
        ("x7", "x8"),
        ("x7", "x9"),
        ("x7", "x10"),
        ("x8", "x9"),
        ("x8", "x10"),
        ("x9", "x10"),
        ("x2", "x7"),
    ]
    y = (
        np.power(np.pi, x1 * x2) * np.sqrt(2.0 * x3)
        - np.arcsin(x4)
        + np.log(x3 + x5)
        - (x9 / x10) * np.sqrt(x7 / x8)
        - x2 * x7
    )
    return y, truth


def f2(X):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = _unpack_features(X)

    truth = [
        ("x1", "x2"),
        ("x1", "x3"),
        ("x2", "x3"),
        ("x3", "x5"),
        ("x7", "x8"),
        ("x7", "x9"),
        ("x7", "x10"),
        ("x8", "x9"),
        ("x8", "x10"),
        ("x9", "x10"),
        ("x2", "x7"),
    ]
    y = (
        np.power(np.pi, x1 * x2) * np.sqrt(2.0 * np.abs(x3))
        - np.arcsin(0.5 * x4)
        + np.log(np.abs(x3 + x5) + 1.0)
        + (x9 / (1.0 + np.abs(x10))) * np.sqrt(np.abs(x7) / (1.0 + np.abs(x8)))
        - x2 * x7
    )
    return y, truth


def f3(X):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = _unpack_features(X)

    truth = [
        ("x1", "x2"),
        ("x2", "x3"),
        ("x3", "x4"),
        ("x4", "x5"),
        ("x4", "x7"),
        ("x4", "x8"),
        ("x5", "x7"),
        ("x5", "x8"),
        ("x7", "x8"),
    ]
    y = (
        np.exp(np.abs(x1 - x2))
        + np.abs(x2 * x3)
        - np.power(x3**2, np.abs(x4))
        + np.log(x4**2 + x5**2 + x7**2 + x8**2)
        + x9
        + 1.0 / (1.0 + x10**2)
    )
    return y, truth


def f4(X):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = _unpack_features(X)

    truth = [
        ("x1", "x2"),
        ("x2", "x3"),
        ("x3", "x4"),
        ("x1", "x4"),
        ("x4", "x5"),
        ("x4", "x7"),
        ("x4", "x8"),
        ("x5", "x7"),
        ("x5", "x8"),
        ("x7", "x8"),
    ]
    y = (
        np.exp(np.abs(x1 - x2))
        + np.abs(x2 * x3)
        - np.power(x3**2, np.abs(x4))
        + (x1 * x4) ** 2
        + np.log(x4**2 + x5**2 + x7**2 + x8**2)
        + x9
        + 1.0 / (1.0 + x10**2)
    )
    return y, truth


def f5(X):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = _unpack_features(X)

    truth = [
        ("x1", "x2"),
        ("x1", "x3"),
        ("x2", "x3"),
        ("x4", "x5"),
        ("x6", "x7"),
        ("x8", "x9"),
        ("x8", "x10"),
        ("x9", "x10"),
    ]
    y = (
        1.0 / (1.0 + x1**2 + x2**2 + x3**2)
        + np.sqrt(np.exp(x4 + x5))
        + np.abs(x6 + x7)
        + x8 * x9 * x10
    )
    return y, truth


def f6(X):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = _unpack_features(X)

    truth = [
        ("x1", "x2"),
        ("x3", "x4"),
        ("x5", "x6"),
        ("x5", "x8"),
        ("x6", "x8"),
        ("x8", "x9"),
        ("x8", "x10"),
        ("x9", "x10"),
    ]
    y = (
        np.exp(np.abs(x1 * x2) + 1.0)
        - np.exp(np.abs(x3 + x4) + 1.0)
        + np.cos(x5 + x6 - x8)
        + np.sqrt(x8**2 + x9**2 + x10**2)
    )
    return y, truth


def f7(X):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = _unpack_features(X)

    truth = [
        ("x1", "x2"),
        ("x3", "x4"),
        ("x3", "x6"),
        ("x4", "x6"),
        ("x4", "x5"),
        ("x4", "x7"),
        ("x4", "x8"),
        ("x5", "x6"),
        ("x5", "x7"),
        ("x5", "x8"),
        ("x6", "x7"),
        ("x6", "x8"),
        ("x7", "x8"),
        ("x7", "x9"),
    ]
    y = (
        (np.arctan(x1) + np.arctan(x2)) ** 2
        + np.maximum(x3 * x4 + x6, 0.0)
        - 1.0 / (1.0 + (x4 * x5 * x6 * x7 * x8) ** 2)
        + (np.abs(x7) / (1.0 + np.abs(x9))) ** 5
        + np.sum(np.asarray(X), axis=1)
    )
    return y, truth


def f8(X):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = _unpack_features(X)

    truth = [
        ("x1", "x2"),
        ("x3", "x5"),
        ("x3", "x6"),
        ("x5", "x6"),
        ("x3", "x4"),
        ("x3", "x7"),
        ("x4", "x5"),
        ("x4", "x7"),
        ("x5", "x7"),
        ("x7", "x8"),
        ("x7", "x9"),
        ("x8", "x9"),
    ]
    y = (
        x1 * x2
        + np.power(2.0, x3 + x5 + x6)
        + np.power(2.0, x3 + x4 + x5 + x7)
        + np.sin(x7 * np.sin(x8 + x9))
        + np.arccos(0.9 * x10)
    )
    return y, truth


def f9(X):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = _unpack_features(X)

    truth = [
        ("x1", "x2"),
        ("x1", "x3"),
        ("x1", "x4"),
        ("x1", "x5"),
        ("x2", "x3"),
        ("x2", "x4"),
        ("x2", "x5"),
        ("x3", "x4"),
        ("x3", "x5"),
        ("x4", "x5"),
        ("x5", "x6"),
        ("x6", "x7"),
        ("x6", "x8"),
        ("x7", "x8"),
        ("x9", "x10"),
    ]
    y = (
        np.tanh(x1 * x2 + x3 * x4) * np.sqrt(np.abs(x5))
        + np.exp(x5 + x6)
        + np.log((x6 * x7 * x8) ** 2 + 1.0)
        + x9 * x10
        + 1.0 / (1.0 + np.abs(x10))
    )
    return y, truth


def f10(X):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = _unpack_features(X)

    truth = [
        ("x1", "x2"),
        ("x3", "x5"),
        ("x3", "x7"),
        ("x5", "x7"),
        ("x4", "x5"),
        ("x7", "x9"),
    ]
    y = (
        np.sinh(x1 + x2)
        + np.arccos(np.tanh(x3 + x5 + x7))
        + np.cos(x4 + x5)
        + 1.0 / np.cos(x7 * x9)
    )
    return y, truth


FUNCTIONS = {
    1: f1,
    2: f2,
    3: f3,
    4: f4,
    5: f5,
    6: f6,
    7: f7,
    8: f8,
    9: f9,
    10: f10,
}


def generate_f_data(
    n_samples: int,
    function_id: int,
    random_state: int = 42,
    classification: bool = True,
    normalize: bool = True,
    noise_std: float = 0.0,
):
    if function_id not in FUNCTIONS:
        raise ValueError("function_id must be in the range 1-10.")

    rng = np.random.default_rng(random_state)

    if function_id == 1:
        X = np.empty((n_samples, 10), dtype=float)

        # x1, x2, x3, x6, x7, x9 ~ U(0, 1)
        zero_one_columns = [0, 1, 2, 5, 6, 8]
        X[:, zero_one_columns] = rng.uniform(
            0.0,
            1.0,
            size=(n_samples, len(zero_one_columns)),
        )

        # x4, x5, x8, x10 ~ U(0.6, 1)
        restricted_columns = [3, 4, 7, 9]
        X[:, restricted_columns] = rng.uniform(
            0.6,
            1.0,
            size=(n_samples, len(restricted_columns)),
        )
    else:
        # F2-F10
        X = rng.uniform(-1.0, 1.0, size=(n_samples, 10))

    y_raw, true_edges = FUNCTIONS[function_id](X)

    if not np.all(np.isfinite(y_raw)):
        bad_count = np.count_nonzero(~np.isfinite(y_raw))
        raise FloatingPointError(
            f"F{function_id} generated {bad_count} NaN or inf values."
        )

    if normalize:
        y_raw = (y_raw - np.mean(y_raw)) / np.std(y_raw)

    if classification:
        logits = y_raw
        probabilities = expit(logits)
        y = rng.binomial(1, probabilities)
    else:
        y = y_raw + rng.normal(
            0.0,
            noise_std,
            size=n_samples,
        )

    X = pd.DataFrame(
        X,
        columns=[f"x{i}" for i in range(1, 11)],
    )
    y = pd.Series(y, name="y")

    return X, y, true_edges


def _correlated_binary_block(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    rho: float,
) -> np.ndarray:
    if n_features == 0:
        return np.empty((n_samples, 0))

    shared = rng.standard_normal(size=(n_samples, 1))
    individual = rng.standard_normal(size=(n_samples, n_features))
    latent = np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * individual
    return (latent > 0).astype(int)


def generate_xor_with_main_effects(
    n_samples: int = 5000,
    n_interactions: int = 2,
    n_main_features: int = 2,
    main_corr: float = 0.5,
    coef_scale: float = 1.0,
    xor_strength: float = 3.0,
    xor_vars_as_main=True,
    n_noise_features: int = 5,
    intercept: float = 0.0,
    random_state: int = 42,
    weights=None,
) -> tuple[pd.DataFrame, pd.Series, list[tuple[str, str]]]:
    rng = np.random.default_rng(random_state)

    data: dict[str, np.ndarray] = {}
    true_edges: list[tuple[str, str]] = []
    logit = np.full(n_samples, float(intercept))

    for k in range(n_interactions):
        a_name = f"x{k + 1}_a"
        b_name = f"x{k + 1}_b"
        a = rng.binomial(1, 0.5, size=n_samples)
        b = rng.binomial(1, 0.5, size=n_samples)
        xor = np.logical_xor(a, b).astype(int)

        data[a_name] = a
        data[b_name] = b
        true_edges.append((a_name, b_name))

        logit = logit + xor_strength * (2 * xor - 1)
        if xor_vars_as_main:
            for var in [a, b]:
                logit = logit + coef_scale * (2 * var - 1)

    main = _correlated_binary_block(rng, n_samples, n_main_features, main_corr)
    if not weights:
        weights = rng.normal(0.0, coef_scale, size=n_main_features)

    for j in range(n_main_features):
        m_j = main[:, j]
        data[f"m{j + 1}"] = m_j
        logit = logit + weights[j] * (2 * m_j - 1)

    prob = expit(logit)
    y = rng.binomial(1, prob)

    for i in range(n_noise_features):
        data[f"noise_{i + 1}"] = rng.binomial(1, 0.5, size=n_samples)

    y = pd.Series(y, name="y")

    X = pd.DataFrame(data)

    return X, y, true_edges
