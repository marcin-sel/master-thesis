"""Pipelines and Optuna search spaces for the MEDICAL (real-data) tuning notebook.

Used by ``notebooks/gnn/tuning.ipynb``. Split out from the shared modules
(``my_project.data.pipelines`` / ``my_project.gnn.search_space``) so the medical
experiment can be tuned independently of the synthetic sweep. The grids below are
own literal copies (currently identical to the synthetic ones) so editing them
here does not affect the synthetic experiment.
"""

from __future__ import annotations

from my_project.data.pipelines import (
    discretization_pipeline,
    preprocessing_pipeline,
    preprocessing_pipeline_nn,
)

# --- Pipelines ---------------------------------------------------------------
# NN input (impute/encode -> ordinal codes for embeddings), XGBoost input
# (impute/encode, no ordinal shift needed) and graph input (discretize numerics
# so the interaction matrix sees a discrete representation).
NN_PREPROCESSING_PIPELINE = preprocessing_pipeline_nn
XGBOOST_PREPROCESSING_PIPELINE = preprocessing_pipeline
GRAPH_PREPROCESSING_PIPELINE = discretization_pipeline

# --- Search spaces -----------------------------------------------------------
# GNN uses the full grid; MLP is the same grid (kept separate so it can drop
# node-embedding params later without touching the GNN grid).
GNN_SEARCH_SPACE = {
    "emb_dim": {"type": "int", "values": [4, 16], "step": 4},
    "hidden_dim": {"type": "int", "values": [8, 48], "step": 4},
    "n_layers": {"type": "int", "values": [1, 3]},
    "dropout": {"type": "float", "values": [0.0, 0.4]},
    "lr": {"type": "float", "values": [1e-3, 1e-2], "log": True},
    "weight_decay": {"type": "float", "values": [1e-6, 1e-2], "log": True},
    "batch_size": {"type": "categorical", "values": [256, 512, 1024]},
    "encoder_post_norm": {"type": "categorical", "values": ["batchnorm"]},
}

MLP_SEARCH_SPACE = {
    "emb_dim": {"type": "int", "values": [4, 16], "step": 4},
    "hidden_dim": {"type": "int", "values": [8, 48], "step": 4},
    "n_layers": {"type": "int", "values": [1, 3]},
    "dropout": {"type": "float", "values": [0.0, 0.4]},
    "lr": {"type": "float", "values": [1e-3, 1e-2], "log": True},
    "weight_decay": {"type": "float", "values": [1e-6, 1e-2], "log": True},
    "batch_size": {"type": "categorical", "values": [256, 512, 1024]},
    "encoder_post_norm": {"type": "categorical", "values": ["batchnorm"]},
}

# XGBoost baseline grid (passed to run_xgboost_fixed_folds_study as
# xgb_search_space). Own literal copy so it can diverge from the synthetic one.
XGB_SEARCH_SPACE = {
    "learning_rate": {"type": "float", "values": [1e-3, 0.2], "log": True},
    "max_depth": {"type": "int", "values": [3, 10]},
    "min_child_weight": {"type": "float", "values": [1.0, 20.0]},
    "subsample": {"type": "float", "values": [0.5, 1.0]},
    "colsample_bytree": {"type": "float", "values": [0.5, 1.0]},
    "gamma": {"type": "float", "values": [0.0, 5.0]},
    "reg_alpha": {"type": "float", "values": [1e-8, 10.0], "log": True},
    "reg_lambda": {"type": "float", "values": [1e-8, 10.0], "log": True},
}

__all__ = [
    "NN_PREPROCESSING_PIPELINE",
    "XGBOOST_PREPROCESSING_PIPELINE",
    "GRAPH_PREPROCESSING_PIPELINE",
    "GNN_SEARCH_SPACE",
    "MLP_SEARCH_SPACE",
    "XGB_SEARCH_SPACE",
]
