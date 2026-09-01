"""Pipelines and shared Optuna search spaces for the SYNTHETIC sweep notebook.

Used by ``notebooks/gnn/synthetic_data_clean.ipynb`` (higgs / pairwise / xor / f
/ madelon / breast_cancer). Search spaces come from
``my_project.experiments.search_space`` and are shared with the medical experiment.
"""

from __future__ import annotations

from my_project.experiments.search_space import (
    GNN_SEARCH_SPACE,
    MLP_SEARCH_SPACE,
    XGB_SEARCH_SPACE,
)

# Synthetic datasets are generated in-memory and already numeric/binary, so no
# sklearn preprocessing/graph pipeline is applied (``run_sweep`` bins via
# ``n_bins``). Kept explicit for symmetry with the medical experiment and so a
# pipeline can be added later without changing the call site.
PREPROCESSING_PIPELINE = None
GRAPH_PREPROCESSING_PIPELINE = None

__all__ = [
    "PREPROCESSING_PIPELINE",
    "GRAPH_PREPROCESSING_PIPELINE",
    "GNN_SEARCH_SPACE",
    "MLP_SEARCH_SPACE",
    "XGB_SEARCH_SPACE",
]
