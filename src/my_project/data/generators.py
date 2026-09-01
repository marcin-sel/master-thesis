"""Synthetic-data generators, ground-truth interactions and config hashing.

Bundles the interaction functions, the generator registry, a call-agnostic
``build_generator`` factory and ``compute_config_hash`` (a stable fingerprint of
the data-generating config used to keep Optuna studies from different configs
apart).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from my_project.data.generate_synthetic_data import (
    generate_f_data,
    generate_xor_with_main_effects,
)
from my_project.data.generate_synthetic_data_pairwise_interaction import (
    generate_pairwise_interaction_data,
)

# Ground-truth interaction primitives live in ``interactions``; re-exported here
# so existing ``from my_project.data.generators import ...`` call sites keep working.
from my_project.data.interactions import (  # noqa: F401
    DEFAULT_INTERACTIONS,
    INTERACTION_MOMENTS,
    PRETTY_NAMES,
    interaction_label_map,
    interactions_name_map,
    log_of_sum_of_abs_plus_one,
    lookup_interaction_moment,
    multiplication,
    or_xor,
    pretty_interaction_name,
    quadratic_magnitude_interaction,
    sign_times_another,
    similarity_interaction,
    sin_of_sum,
    sinusoidal_interaction,
    xor,
)
from my_project.data.load_breast_cancer import generate_breast_cancer_data
from my_project.data.load_higgs import generate_higgs_data
from my_project.data.load_madelon import generate_madelon_data
from my_project.data.load_medical import generate_medical_data

# Map each generator kind to its callable; kind-specific static arguments live
# in the config's ``generator_kwargs``.
GENERATORS: dict[str, Callable[..., Any]] = {
    "f": generate_f_data,
    "xor": generate_xor_with_main_effects,
    "pairwise": generate_pairwise_interaction_data,
    "madelon": generate_madelon_data,
    "breast_cancer": generate_breast_cancer_data,
    "higgs": generate_higgs_data,
    "medical": generate_medical_data,
}


def build_generator(kind: str, static_kwargs: dict[str, Any]) -> Callable[..., Any]:
    """Return a pure ``generate(**kwargs)`` bound to one generator kind.

    Call-time kwargs (e.g. ``n_samples``, ``random_state``, ``cov``) are merged
    over the kind's static ``static_kwargs`` so callers stay generator-agnostic.
    """
    generator = GENERATORS[kind]

    def generate(**kwargs):
        return generator(**{**static_kwargs, **kwargs})

    return generate


def compute_config_hash(
    generator_kind, generator_kwargs, noise_level, n_bins, length=8
):
    """Short, stable fingerprint of the data-generating config.

    The generator name alone does NOT distinguish, e.g., two ``pairwise`` runs
    with different interactions, ``noise_level``, ``n_bins`` or other generator
    kwargs: the Optuna study names would collide and their trials would be
    merged. This hash folds all of those into a stable id so every distinct
    config (including a distinct ``noise_level``) gets its own study.
    ``interactions`` holds callables, so hash their names.
    """
    payload = {
        "generator_kind": generator_kind,
        "noise_level": float(noise_level or 0.0),
        "n_bins": n_bins,
        "generator_kwargs": {
            k: (
                {str(pair): getattr(f, "__name__", str(f)) for pair, f in v.items()}
                if k == "interactions" and isinstance(v, dict)
                else v
            )
            for k, v in generator_kwargs.items()
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class GeneratorPreset:
    """One dataset's default wiring for the sweep.

    Bundles the static ``generator_kwargs`` forwarded to the generator with the
    sweep defaults that make sense for that dataset (edge selection, grids,
    binning, feature selection, ...). Every field is only a *default*: the CLI /
    ``build_config`` can override any of them. To add a new dataset, register a
    generator in ``GENERATORS`` and add a matching preset here.
    """

    kind: str
    name: str
    generator_kwargs: dict = field(default_factory=dict)
    n_bins: int = 5
    test_size: int = 5000
    valid_size: int = 5000
    edge_mode: str = "n_edges"
    n_edges_grid: tuple = (6, 12)
    threshold_grid: tuple = (0.8, 0.9, 0.95)
    train_grid: tuple = (1000, 5000)
    cov_grid: tuple = (0.0,)
    feature_selection: dict | None = None


# Per-dataset presets. `--generator <kind>` (or build_config(generator_kind=...))
# looks the kind up here to fill in dataset-specific defaults. Real datasets
# (higgs/madelon/breast_cancer) have no ground-truth interactions and no `cov`,
# so they default to a single cov=0.0 cell and threshold-based edge selection.
GENERATOR_PRESETS: dict[str, GeneratorPreset] = {
    "pairwise": GeneratorPreset(
        kind="pairwise",
        name="pairwise",
        generator_kwargs={
            "n_informative": 10,
            "coef_scale": 0.0,
            "coef_loc": 1.0,
            "interactions": DEFAULT_INTERACTIONS,
        },
        edge_mode="n_edges",
        n_edges_grid=(6, 12),
        train_grid=(1000, 5000),
        cov_grid=(0.0, 0.25, 0.5),
    ),
    "higgs": GeneratorPreset(
        kind="higgs",
        name="higgs",
        generator_kwargs={"feature_set": "low_level"},
        edge_mode="threshold",
        threshold_grid=(0.6, 0.7, 0.8, 0.9),
        train_grid=(15000,),
        cov_grid=(0.0,),
    ),
    "madelon": GeneratorPreset(
        kind="madelon",
        name="madelon",
        generator_kwargs={},
        test_size=100,
        valid_size=500,
        edge_mode="threshold",
        threshold_grid=(0.9, 0.95),
        train_grid=(2000,),
        cov_grid=(0.0,),
        feature_selection={"n_features": 100, "method": "CIFE", "n_bins": 5},
    ),
    "breast_cancer": GeneratorPreset(
        kind="breast_cancer",
        name="breast_cancer",
        generator_kwargs={},
        test_size=100,
        valid_size=100,
        edge_mode="threshold",
        threshold_grid=(0.9, 0.95),
        train_grid=(369,),
        cov_grid=(0.0,),
    ),
}


def get_preset(kind: str) -> GeneratorPreset:
    """Return the preset for a generator kind (or a bare default if unknown)."""
    return GENERATOR_PRESETS.get(kind, GeneratorPreset(kind=kind, name=kind))
