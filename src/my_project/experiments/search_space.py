"""Shared hyperparameter search space for the GNN / MLP / XGBoost Optuna sweeps."""

from __future__ import annotations

GNN_SEARCH_SPACE = {
    "emb_dim": {"type": "int", "values": [1, 9], "step": 2},
    "emb_num_hidden_ratio": {"type": "categorical", "values": [1]},
    "emb_num_batch_norm": {"type": "categorical", "values": [False]},
    "encoder_post_norm": {"type": "categorical", "values": [None]},
    "encoder_post_activation": {"type": "categorical", "values": [None]},
    "hidden_dim": {"type": "int", "values": [4, 48], "step": 4},
    "n_layers": {"type": "int", "values": [1, 3]},
    "dropout": {"type": "float", "values": [0.0, 0.4]},
    "lr": {"type": "float", "values": [1e-3, 1e-2], "log": True},
    "weight_decay": {"type": "float", "values": [1e-6, 1e-2], "log": True},
    "batch_size": {"type": "categorical", "values": [256, 512, 1024]},
}

MLP_SEARCH_SPACE = GNN_SEARCH_SPACE.copy()

XGB_SEARCH_SPACE = {
    "learning_rate": {"type": "float", "values": [1e-4, 0.2], "log": True},
    "max_depth": {"type": "int", "values": [2, 10]},
    "min_child_weight": {"type": "float", "values": [1.0, 30.0], "log": True},
    "subsample": {"type": "float", "values": [0.4, 1.0]},
    "colsample_bytree": {"type": "float", "values": [0.4, 1.0]},
    "gamma": {"type": "float", "values": [0.0, 8.0]},
    "reg_alpha": {"type": "float", "values": [1e-8, 20.0], "log": True},
    "reg_lambda": {"type": "float", "values": [1e-8, 50.0], "log": True},
}

BASE_SEARCH_SPACE = GNN_SEARCH_SPACE
SEARCH_SPACE = GNN_SEARCH_SPACE
XGB_BASE_SEARCH_SPACE = XGB_SEARCH_SPACE
