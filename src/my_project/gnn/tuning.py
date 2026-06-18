import copy
import json
import os
import traceback

import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from mlflow.tracking import MlflowClient
from my_project.gnn.training import train_gnn


def _suggest_from_grid(trial, search_space):
    """Pull every grid dimension through the trial.

    The values come straight from ``search_space`` so it stays the single
    source of truth: a ``GridSampler`` walks the combinations exhaustively,
    while a sampler like TPE samples within the same categorical choices.
    """
    return {
        key: trial.suggest_categorical(key, values)
        for key, values in search_space.items()
    }


def suggest_mlp_params(trial, search_space, base_params=None):
    if base_params is None:
        base_params = {}
    else:
        base_params = base_params.copy()

    sampled = _suggest_from_grid(trial, search_space)

    # `hidden_dim` (int) + `n_layers` is expanded by the model into a per-layer
    # list, so reuse the sampled embedding width here.
    derived = {
        # "hidden_dim": sampled["hidden_dim"],
    }

    suggested_params = base_params.copy()
    suggested_params.update(sampled)
    suggested_params.update(derived)
    return suggested_params


def suggest_gnn_params(trial, search_space, base_params=None):
    if base_params is None:
        base_params = {}
    else:
        base_params = base_params.copy()

    sampled = _suggest_from_grid(trial, search_space)

    derived = {
        # "emb_dim": sampled["hidden_dim"],
        # "add_skip": base_params.get("add_skip", False),
    }

    suggested_params = base_params.copy()
    suggested_params.update(sampled)
    suggested_params.update(derived)
    return suggested_params


def objective(
    trial,
    model_cls,
    cv_folds,
    search_space,
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

    params = suggest_params_func(trial, search_space, base_params=base_params)

    # Where to drop a failure traceback that does not depend on MLflow being
    # writable. We reuse the trial checkpoint dir's parent so errors land next
    # to the study results.
    checkpoint_dir = technical_settings.get("trainer_kwargs", {}).get("checkpoint_dir")
    error_log_dir = (
        os.path.join(os.path.dirname(checkpoint_dir), "errors")
        if checkpoint_dir
        else None
    )

    fold_scores = []
    fold_best_epochs = []
    fold_best_checkpoints = []
    fold_metrics = []
    fold_run_ids = []
    fold_run_metrics = []

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
            tags={
                "trial_id": str(trial.number),
                "run_type": "parent",
                "experiment_name": experiment_name,
            },
        )
        parent_run_id = parent_run.info.run_id

        # Log shared hyperparameters and labels on the parent run. They are also
        # repeated on each fold (child) run so individual folds are inspectable.
        # model_cls is logged as a param (its name) on the fold runs by
        # train_gnn, so mirror it here too for a chartable axis on the parent.
        mlflow_client.log_param(parent_run_id, "model_cls", model_cls.__name__)
        for key, value in params.items():
            mlflow_client.log_param(
                parent_run_id,
                key,
                json.dumps(value) if isinstance(value, dict) else value,
            )
        for key, value in technical_settings.get("tags", {}).items():
            if value is not None:
                mlflow_client.set_tag(parent_run_id, key, str(value))
                # Labels are also logged as params so MLflow charts (e.g. box
                # plots) can use them on the X axis, which tags cannot do.
                mlflow_client.log_param(parent_run_id, key, value)

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
            tags["run_type"] = "fold"
            if experiment_name is not None:
                tags["experiment_name"] = experiment_name
            if fold_seed is not None:
                tags["seed"] = fold_seed

            trainer = train_gnn(
                params=params_fold,
                model_cls=model_cls,
                data=data_fold,
                trial_id=trial.number,
                fold_id=fold_idx,
                tags=tags,
                # `graph_name` is not visible inside train_gnn, so pass it as a
                # param to make it usable as a chart axis on the fold runs.
                extra_params={
                    key: value
                    for key, value in technical_settings.get("tags", {}).items()
                    if value is not None
                },
                parent_run_id=parent_run_id,
                log_params=True,
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

            # Collect every numeric callback metric so the parent can hold means.
            fold_metric_values = {}
            for metric_name, metric_value in trainer.callback_metrics.items():
                try:
                    fold_metric_values[metric_name] = float(metric_value.item())
                except (AttributeError, ValueError, TypeError):
                    continue
            fold_metrics.append(fold_metric_values)

            # Track the child run so trial-level means can be mirrored onto it.
            fold_run_ids.append(getattr(trainer.logger, "run_id", None))

            # Run-level metrics (e.g. model size) are logged by train_gnn; pick
            # them up generically so the parent does not hardcode their names.
            fold_run_metrics.append(dict(getattr(trainer, "logged_run_metrics", {})))

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

        # Aggregate per-fold metrics into trial-level means so callbacks (e.g. a
        # progress bar) and other consumers can read them straight from the
        # trial, independent of whether MLflow logging is enabled.
        if fold_metrics:
            metric_names = {name for fold in fold_metrics for name in fold}
            mean_metrics = {}
            for metric_name in metric_names:
                values = [
                    fold[metric_name] for fold in fold_metrics if metric_name in fold
                ]
                if values:
                    mean_metrics[metric_name] = float(np.mean(values))
            trial.set_user_attr("mean_metrics", mean_metrics)

        # Log aggregated metrics on the parent run.
        if mlflow_client is not None and parent_run_id is not None and fold_scores:
            valid_epochs = [e for e in fold_best_epochs if e is not None]
            if valid_epochs:
                mlflow_client.log_metric(
                    parent_run_id, "mean_best_epoch", float(np.mean(valid_epochs))
                )
                # Mirror the best epoch under the child-facing "epoch" name too.
                mlflow_client.log_metric(
                    parent_run_id, "epoch", float(np.mean(valid_epochs))
                )

            # Run-level metrics logged per child by train_gnn (e.g. model size)
            # are not part of callback_metrics; mirror their mean onto the parent.
            if fold_run_metrics:
                run_metric_names = {name for fold in fold_run_metrics for name in fold}
                for metric_name in sorted(run_metric_names):
                    values = [
                        fold[metric_name]
                        for fold in fold_run_metrics
                        if metric_name in fold
                    ]
                    if values:
                        mlflow_client.log_metric(
                            parent_run_id, metric_name, float(np.mean(values))
                        )

            # Trial-level mean/std for every metric. The suffixed values are
            # mirrored onto each child so that sorting by `<metric>_mean` keeps
            # the whole family (parent + folds) grouped in MLflow's flat view.
            child_run_ids = [rid for rid in fold_run_ids if rid is not None]
            metric_names = {name for fold in fold_metrics for name in fold}
            for metric_name in sorted(metric_names):
                values = [
                    fold[metric_name] for fold in fold_metrics if metric_name in fold
                ]
                if not values:
                    continue
                metric_mean = float(np.mean(values))
                metric_std = float(np.std(values))
                # Mean under the original name (kept for backward-compat).
                mlflow_client.log_metric(parent_run_id, metric_name, metric_mean)
                for run_id in [parent_run_id, *child_run_ids]:
                    mlflow_client.log_metric(run_id, f"mean_{metric_name}", metric_mean)
                    mlflow_client.log_metric(run_id, f"std_{metric_name}", metric_std)

        parent_status = "FINISHED"
        parent_trial_status = "finished"
        parent_error = None
        return mean_score

    except optuna.TrialPruned:
        # MLflow has no PRUNED status; use KILLED + a tag to stay filterable.
        parent_status = "KILLED"
        parent_trial_status = "pruned"
        parent_error = None
        raise
    except Exception as exc:
        parent_status = "FAILED"
        parent_trial_status = "failed"
        # Keep the failure reason so it is inspectable in the MLflow UI instead
        # of just a bare "failed" status.
        error_tb = traceback.format_exc()
        parent_error = (exc, error_tb)

        # Persist the traceback independently of MLflow: a failure is often
        # caused by MLflow itself being unwritable (e.g. "readonly database"),
        # in which case logging the error *to* MLflow would silently fail too.
        # The Optuna storage and the local filesystem are separate sinks.
        try:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}"[:5000])
            trial.set_user_attr("error_traceback", error_tb)
        except Exception:
            pass
        if error_log_dir is not None:
            try:
                os.makedirs(error_log_dir, exist_ok=True)
                error_path = os.path.join(error_log_dir, f"trial_{trial.number}.txt")
                with open(error_path, "w", encoding="utf-8") as fh:
                    fh.write(error_tb)
            except Exception:
                pass
        raise
    finally:
        if mlflow_client is not None and parent_run_id is not None:
            # Guard every MLflow write: if the backend is down we still want the
            # parent run terminated rather than left hanging in RUNNING.
            try:
                mlflow_client.set_tag(
                    parent_run_id, "trial_status", parent_trial_status
                )
                if parent_error is not None:
                    exc, error_tb = parent_error
                    # Short, searchable message in the runs table; full traceback
                    # as an artifact for the details view.
                    error_message = f"{type(exc).__name__}: {exc}"
                    mlflow_client.set_tag(parent_run_id, "error", error_message[:5000])
                    mlflow_client.log_text(
                        parent_run_id, error_tb, "error_traceback.txt"
                    )
            except Exception:
                pass
            try:
                mlflow_client.set_terminated(parent_run_id, status=parent_status)
            except Exception:
                pass


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
