"""Pipelines and Optuna search space for the SYNTHETIC sweep notebook.

Used by ``notebooks/gnn/synthetic_data_clean.ipynb`` (higgs / pairwise / xor / f
/ madelon / breast_cancer). Split out from ``my_project.gnn.search_space`` so the
synthetic sweep can be tuned independently of the medical experiment. The grid is
an own literal copy (currently identical to the medical one).

``run_sweep`` feeds ``SEARCH_SPACE`` to both the MLP and GNN configs, so this is
a single grid (the former ``BASE_SEARCH_SPACE``).
"""

from __future__ import annotations

# Synthetic datasets are generated in-memory and already numeric/binary, so no
# sklearn preprocessing/graph pipeline is applied (``run_sweep`` bins via
# ``n_bins``). Kept explicit for symmetry with the medical experiment and so a
# pipeline can be added later without changing the call site.
PREPROCESSING_PIPELINE = None
GRAPH_PREPROCESSING_PIPELINE = None

SEARCH_SPACE = {
    "emb_dim": {"type": "int", "values": [4, 16], "step": 4},
    "hidden_dim": {"type": "int", "values": [8, 48], "step": 4},
    "n_layers": {"type": "int", "values": [1, 3]},
    "dropout": {"type": "float", "values": [0.0, 0.4]},
    "lr": {"type": "float", "values": [1e-3, 1e-2], "log": True},
    "weight_decay": {"type": "float", "values": [1e-6, 1e-2], "log": True},
    "batch_size": {"type": "categorical", "values": [256, 512, 1024]},
    "encoder_post_norm": {"type": "categorical", "values": ["batchnorm"]},
}

# XGBoost baseline grid (passed to run_sweep as xgb_search_space). Own literal
# copy so it can diverge from the medical one.
XGB_SEARCH_SPACE = {
    "learning_rate": {"type": "float", "values": [1e-3, 0.2], "log": True},
    "max_depth": {"type": "int", "values": [2, 8]},
    "min_child_weight": {"type": "int", "values": [1, 10]},
    "subsample": {"type": "float", "values": [0.5, 1.0]},
    "colsample_bytree": {"type": "float", "values": [0.5, 1.0]},
    "reg_alpha": {"type": "float", "values": [1e-3, 20.0], "log": True},
    "reg_lambda": {"type": "float", "values": [1e-3, 20.0], "log": True},
}

__all__ = [
    "PREPROCESSING_PIPELINE",
    "GRAPH_PREPROCESSING_PIPELINE",
    "SEARCH_SPACE",
    "XGB_SEARCH_SPACE",
]
