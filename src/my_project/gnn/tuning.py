import copy

import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from my_project.gnn.data_module import GNNDataModule
from my_project.gnn.training import train_gnn


def suggest_gnn_params(trial, *, base_params=None):
    if base_params is None:
        base_params = {}
    else:
        base_params = base_params.copy()

    emb_dim = trial.suggest_categorical("emb_dim", [32, 64, 128])

    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 4)
    n_mlp_layers = trial.suggest_int("n_mlp_layers", 1, 4)

    add_skip = False
    # add_skip = trial.suggest_categorical("add_skip", [False, True])

    if add_skip:
        conv_hidden_dim = [emb_dim] * n_conv_layers
    else:
        conv_hidden_dim = [
            trial.suggest_categorical(f"conv_hidden_dim_{layer_idx}", [16, 32, 64, 128])
            for layer_idx in range(n_conv_layers)
        ]

    mlp_hidden_dim = [
        trial.suggest_categorical(f"mlp_hidden_dim_{layer_idx}", [16, 32, 64, 128])
        for layer_idx in range(n_mlp_layers)
    ]

    trial_suggested_params = {
        "emb_dim": emb_dim,
        "add_skip": add_skip,
        "num_emb_hidden": trial.suggest_int("num_emb_hidden", 4, 16),
        "conv_hidden_dim": conv_hidden_dim,
        "mlp_hidden_dim": mlp_hidden_dim,
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 384, 512]),
    }

    suggested_params = base_params.copy()
    suggested_params.update(trial_suggested_params)
    return suggested_params


def objective(trial, model_cls, data, technical_settings=None, base_params=None):
    data = copy.deepcopy(data)

    if technical_settings is None:
        technical_settings = {}

    if base_params is None:
        base_params = {}

    trial.set_user_attr("model_cls", model_cls.__name__)

    if data.__class__ == GNNDataModule:
        data = [data]

    params = suggest_gnn_params(trial, base_params=base_params)

    fold_scores = []
    fold_best_epochs = []
    fold_best_checkpoints = []

    trial.set_user_attr("params", params)

    for fold_idx, data_fold in enumerate(data):
        trainer = train_gnn(
            params=params,
            model_cls=model_cls,
            trial_id=trial.number,
            fold_id=fold_idx,
            data=data_fold,
            **technical_settings,
        )
        best_score = trainer.callback_metrics[
            technical_settings["monitor_kwargs"]["monitor"]
        ].item()
        fold_scores.append(best_score)

        best_epoch = _extract_best_epoch(trainer)
        fold_best_epochs.append(best_epoch)

        checkpoint_callback = None
        best_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                checkpoint_callback = cb
                best_path = checkpoint_callback.best_model_path
                break

        fold_best_checkpoints.append(best_path)

        intermediate_value = float(np.mean(fold_scores))
        trial.report(intermediate_value, step=fold_idx)

        mean_score = np.mean(fold_scores)

        trial.set_user_attr("best_epochs", fold_best_epochs)
        trial.set_user_attr("best_checkpoints", fold_best_checkpoints)

        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_valid_epochs", fold_best_epochs)
        trial.set_user_attr("fold_score", np.mean(fold_scores))
        trial.set_user_attr("valid_epoch", np.mean(fold_best_epochs))

        if trial.should_prune():
            raise optuna.TrialPruned()

    return mean_score


def _extract_best_epoch(trainer) -> int | None:
    """Return epoch index stored in the best checkpoint, if available."""
    checkpoint_callback = next(
        (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)), None
    )
    if checkpoint_callback is None:
        return None

    best_path = checkpoint_callback.best_model_path
    if not best_path:
        return None

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    return int(epoch) if epoch is not None else None
