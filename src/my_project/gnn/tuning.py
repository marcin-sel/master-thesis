import copy

import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from mlflow.tracking import MlflowClient
from my_project.gnn.training import train_gnn


def suggest_mlp_params(trial, base_params=None):
    if base_params is None:
        base_params = {}
    else:
        base_params = base_params.copy()

    emb_dim = trial.suggest_categorical("emb_dim", [32, 64, 96, 128])

    n_mlp_layers = trial.suggest_int("n_mlp_layers", 1, 5)

    mlp_hidden_dim = [
        trial.suggest_categorical(f"mlp_hidden_dim_{layer_idx}", [16, 32, 64, 96, 128])
        for layer_idx in range(n_mlp_layers)
    ]

    trial_suggested_params = {
        "emb_dim": emb_dim,
        # "num_emb_hidden": trial.suggest_int("num_emb_hidden", 4, 16),
        "mlp_hidden_dim": mlp_hidden_dim,
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 384, 512]),
    }

    suggested_params = base_params.copy()
    suggested_params.update(trial_suggested_params)
    return suggested_params


def suggest_gnn_params(trial, base_params=None):
    if base_params is None:
        base_params = {}
    else:
        base_params = base_params.copy()

    emb_dim = trial.suggest_categorical("emb_dim", [32, 64, 96, 128])

    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 4)

    add_skip = base_params.get("add_skip", False)
    # add_skip = trial.suggest_categorical("add_skip", [False, True])

    conv_hidden_dim = [emb_dim] * n_conv_layers
    # if add_skip:
    #     conv_hidden_dim = [emb_dim] * n_conv_layers
    # else:
    #     conv_hidden_dim = [
    #         trial.suggest_categorical(f"conv_hidden_dim_{layer_idx}", [16, 32, 64, 96, 128])
    #         for layer_idx in range(n_conv_layers)
    #     ]

    # n_mlp_layers = trial.suggest_int("n_mlp_layers", 1, 2)
    n_mlp_layers = 1

    # mlp_hidden_dim = [
    #     trial.suggest_categorical(f"mlp_hidden_dim_{layer_idx}", [16, 32, 64, 96, 128])
    #     for layer_idx in range(n_mlp_layers)
    # ]

    trial_suggested_params = {
        "emb_dim": emb_dim,
        "add_skip": add_skip,
        # "num_emb_hidden": trial.suggest_int("num_emb_hidden", 4, 16),
        "conv_hidden_dim": conv_hidden_dim,
        # "mlp_hidden_dim": mlp_hidden_dim,
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 384, 512]),
    }

    suggested_params = base_params.copy()
    suggested_params.update(trial_suggested_params)
    return suggested_params


def objective(
    trial,
    model_cls,
    cv_folds,
    technical_settings=None,
    base_params=None,
    suggest_params_func=suggest_gnn_params,
):
    if technical_settings is None:
        technical_settings = {}
    else:
        technical_settings = technical_settings.copy()

    if base_params is None:
        base_params = {}
    else:
        base_params = base_params.copy()

    trial.set_user_attr("model_cls", model_cls.__name__)

    params = suggest_params_func(trial, base_params=base_params)

    fold_scores = []
    fold_best_epochs = []
    fold_best_checkpoints = []

    trial.set_user_attr("params", params)

    fold_seeds = technical_settings.pop("fold_seeds", None)

    # Create one MLflow parent run per trial so that folds become nested children.
    logger_kwargs = technical_settings.get("logger_kwargs", {})
    experiment_name = logger_kwargs.get("experiment_name")
    tracking_uri = logger_kwargs.get("tracking_uri")

    mlflow_client = None
    parent_run_id = None
    if experiment_name is not None:
        mlflow_client = MlflowClient(tracking_uri=tracking_uri)
        exp = mlflow_client.get_experiment_by_name(experiment_name)
        if exp is None:
            experiment_id = mlflow_client.create_experiment(experiment_name)
        else:
            experiment_id = exp.experiment_id

        parent_run = mlflow_client.create_run(
            experiment_id=experiment_id,
            run_name=f"{experiment_name}_trial_{trial.number}",
            tags={"trial_id": str(trial.number)},
        )
        parent_run_id = parent_run.info.run_id

    try:
        for fold_idx, fold_results in enumerate(cv_folds):
            fold_base_params = fold_results.get("params").copy()
            data_fold = fold_results.get("data")
            data_fold = copy.deepcopy(data_fold)
            data_fold.keep_on_gpu = technical_settings.get("keep_on_gpu", False)
            data_fold.setup()

            params_fold = params.copy()
            params_fold.update(fold_base_params)

            fold_seed = None
            if fold_seeds is not None and fold_idx < len(fold_seeds):
                fold_seed = fold_seeds[fold_idx]

            tags = technical_settings.get("tags", {}).copy()
            # Backward-compat: older callers passed labels via `to_log`.
            tags.update(technical_settings.get("to_log", {}))
            if fold_seed is not None:
                tags["seed"] = fold_seed

            trainer = train_gnn(
                params=params_fold,
                model_cls=model_cls,
                data=data_fold,
                trial_id=trial.number,
                fold_id=fold_idx,
                tags=tags,
                parent_run_id=parent_run_id,
                **{
                    k: v
                    for k, v in technical_settings.items()
                    if k not in ("tags", "to_log", "keep_on_gpu")
                },
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

        # Log aggregated metrics on the parent run.
        if mlflow_client is not None and parent_run_id is not None and fold_scores:
            mlflow_client.log_metric(
                parent_run_id, "fold_score_mean", float(np.mean(fold_scores))
            )
            mlflow_client.log_metric(
                parent_run_id, "fold_score_std", float(np.std(fold_scores))
            )
            valid_epochs = [e for e in fold_best_epochs if e is not None]
            if valid_epochs:
                mlflow_client.log_metric(
                    parent_run_id, "valid_epoch_mean", float(np.mean(valid_epochs))
                )

        parent_status = "FINISHED"
        parent_trial_status = "finished"
        return mean_score

    except optuna.TrialPruned:
        # MLflow has no PRUNED status; use KILLED + a tag to stay filterable.
        parent_status = "KILLED"
        parent_trial_status = "pruned"
        raise
    except Exception:
        parent_status = "FAILED"
        parent_trial_status = "failed"
        raise
    finally:
        if mlflow_client is not None and parent_run_id is not None:
            mlflow_client.set_tag(parent_run_id, "trial_status", parent_trial_status)
            mlflow_client.set_terminated(parent_run_id, status=parent_status)


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
