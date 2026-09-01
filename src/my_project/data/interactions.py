"""Ground-truth pairwise interaction primitives and their metadata.

Single source of truth for the synthetic-data interaction functions, the default
interaction set, their math-form labels (:data:`PRETTY_NAMES`) and their
simulated population moments (:data:`INTERACTION_MOMENTS`). Kept in a dedicated,
dependency-light module so both ``generators`` (the registry/presets) and
``generate_synthetic_data_pairwise_interaction`` (the generator) can import from
one place without an import cycle.
"""

from __future__ import annotations

import numpy as np


# --- Ground-truth pairwise interaction functions --------------------------
def sign_times_another(a, b):
    return b if a > 0 else -b


def multiplication(a, b):
    return a * b


def xor(a, b):
    return int(a > 0) ^ int(b > 0)


def log_of_sum_of_abs_plus_one(a, b):
    return np.log(np.abs(a) + np.abs(b) + 1.0)


def sin_of_sum(a, b):
    return np.sin(np.pi * (a + b))


def sinusoidal_interaction(a, b):
    return np.sin(np.pi * a) * np.sin(np.pi * b)


def or_xor(a, b, c, d):
    return int(xor(int(a > 0), int(b > 0)) | xor(int(c > 0), int(d > 0)))


def quadratic_magnitude_interaction(a, b):
    return (a**2 - 1.0) * (b**2 - 1.0)


def similarity_interaction(a, b, sigma=1.0):
    return np.exp(-0.5 * ((a - b) / sigma) ** 2)


# Default pairwise interactions: every informative feature takes part in
# exactly one interaction (mirrors the notebook's active `pairwise` config).
DEFAULT_INTERACTIONS = {
    ("x1", "x2"): sign_times_another,
    ("x3", "x4"): multiplication,
    ("x5", "x6"): sinusoidal_interaction,
    ("x6", "x7"): xor,
    ("x6", "x8"): quadratic_magnitude_interaction,
    ("x7", "x8"): similarity_interaction,
}


# Human-readable (math-form) labels for each ground-truth interaction function,
# used for plot/table axis labels instead of the raw ``__name__``.
PRETTY_NAMES: dict[str, str] = {
    "sign_times_another": "sgn(a)\u00b7b",
    "multiplication": "a\u00b7b",
    "sinusoidal_interaction": "sin(\u03c0a)\u00b7sin(\u03c0b)",
    "xor": "\U0001d7d9(a>0)\u2295\U0001d7d9(b>0)",
    "quadratic_magnitude_interaction": "(a\u00b2\u22121)(b\u00b2\u22121)",
    "similarity_interaction": "exp(\u2212\u00bd(a\u2212b)\u00b2)",
    "log_of_sum_of_abs_plus_one": "log(|a|+|b|+1)",
    "sin_of_sum": "sin(\u03c0(a+b))",
    "or_xor": "(\U0001d7d9(a>0)\u2295\U0001d7d9(b>0))\u2228(\U0001d7d9(c>0)\u2295\U0001d7d9(d>0))",
}


# Simulated population moments ``{func_name: {rho: (mean, std)}}`` for the
# default pairwise interactions, estimated on standard-normal features with
# equicorrelation ``rho``. Used to standardize interaction columns with fixed
# (sample-independent) statistics instead of the empirical per-draw z-score, so
# the interaction scale is reproducible across sample sizes and random seeds.
INTERACTION_MOMENTS: dict[str, dict[float, tuple[float, float]]] = {
    "quadratic_magnitude_interaction": {
        0.0: (0.0, 1.98),
        0.25: (0.12, 2.47),
        0.5: (0.5, 3.65),
    },
    "multiplication": {0.0: (0.0, 1.0), 0.25: (0.25, 1.03), 0.5: (0.5, 1.12)},
    "similarity_interaction": {
        0.0: (0.58, 0.34),
        0.25: (0.63, 0.32),
        0.5: (0.71, 0.28),
    },
    "sign_times_another": {0.0: (0.0, 1.0), 0.25: (0.2, 0.98), 0.5: (0.4, 0.92)},
    "sinusoidal_interaction": {0.0: (0.0, 0.5), 0.25: (0.0, 0.5), 0.5: (0.0, 0.5)},
    "xor": {0.0: (0.5, 0.5), 0.25: (0.42, 0.49), 0.5: (0.33, 0.47)},
}


def lookup_interaction_moment(func, cov_scalar):
    """Return the simulated ``(mean, std)`` for ``func`` at correlation ``cov_scalar``.

    ``None`` when the correlation is not scalar or the ``(function, rho)`` pair
    has no tabulated moment, signalling the caller to fall back to the empirical
    z-score.
    """
    if cov_scalar is None:
        return None
    per_cov = INTERACTION_MOMENTS.get(getattr(func, "__name__", ""))
    if not per_cov:
        return None
    for rho, moment in per_cov.items():
        if abs(rho - cov_scalar) < 1e-9:
            return moment
    return None


def interactions_name_map(interactions: dict) -> dict:
    """Readable ``{pair: function_name}`` map for MLflow tags / logging."""
    return {
        pair: getattr(func, "__name__", str(func))
        for pair, func in interactions.items()
    }


def pretty_interaction_name(func) -> str:
    """Math-form label for one interaction function (falls back to ``__name__``)."""
    name = getattr(func, "__name__", str(func))
    return PRETTY_NAMES.get(name, name)


def interaction_label_map(interactions: dict) -> dict:
    """Readable ``{pair: math-form label}`` map for plot/table axis labels."""
    return {pair: pretty_interaction_name(func) for pair, func in interactions.items()}
