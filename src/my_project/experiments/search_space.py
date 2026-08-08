"""Shared hyperparameter search space for the GNN / MLP / XGBoost Optuna sweeps."""

from __future__ import annotations

BASE_SEARCH_SPACE = {
    "emb_dim": {"type": "int", "values": [4, 32], "step": 4},
    "hidden_dim": {"type": "int", "values": [8, 48], "step": 4},
    "n_layers": {"type": "int", "values": [1, 3]},
    "dropout": {"type": "float", "values": [0.0, 0.4]},
    "lr": {"type": "float", "values": [1e-3, 1e-2], "log": True},
    "weight_decay": {"type": "float", "values": [1e-6, 1e-2], "log": True},
    "batch_size": {"type": "categorical", "values": [256, 512, 1024]},
    "emb_dim": {"type": "int", "values": [4, 16], "step": 4},
    # "encoder_post_activation": {"type": "categorical", "values": [
    #     # None,
    #     "relu"
    # ]},
    "encoder_post_norm": {
        "type": "categorical",
        "values": [
            # None,
            "batchnorm",
            # "layernorm",
        ],
    },
}


MLP_SEARCH_SPACE = BASE_SEARCH_SPACE.copy()
# MLP_SEARCH_SPACE.pop("emb_dim")
# MLP_SEARCH_SPACE.pop("num_emb_hidden")
# MLP_SEARCH_SPACE.pop("encoder_post_activation")


GNN_SEARCH_SPACE = dict(BASE_SEARCH_SPACE)
SEARCH_SPACE = BASE_SEARCH_SPACE

XGB_BASE_SEARCH_SPACE = {
    "learning_rate": {
        "type": "float",
        "values": [1e-3, 0.2],
        "log": True,
    },
    "max_depth": {
        "type": "int",
        "values": [2, 8],
    },
    "min_child_weight": {
        "type": "int",
        "values": [1, 10],
    },
    "subsample": {
        "type": "float",
        "values": [0.5, 1.0],
    },
    "colsample_bytree": {
        "type": "float",
        "values": [0.5, 1.0],
    },
    "reg_alpha": {
        "type": "float",
        "values": [1e-3, 20.0],
        "log": True,
    },
    "reg_lambda": {
        "type": "float",
        "values": [1e-3, 20.0],
        "log": True,
    },
}

XGB_SEARCH_SPACE = dict(XGB_BASE_SEARCH_SPACE)
