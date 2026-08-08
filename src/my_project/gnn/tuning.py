from __future__ import annotations

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


def _parse_spec(spec):
    """Normalize one search-space entry to ``(type, config)``."""
    if isinstance(spec, dict):
        return spec["type"], spec

    if isinstance(spec, (tuple, list)):
        if not spec:
            raise ValueError("Empty search-space spec.")

        ptype = spec[0]

        if ptype == "categorical":
            return "categorical", {"values": list(spec[1:])}

        if len(spec) < 2:
            raise ValueError(f"Malformed spec: {spec!r}")

        if ptype in {"int", "float"}:
            low = spec[1]
            high = spec[2] if len(spec) >= 3 else spec[1]
            return ptype, {"values": [low, high]}

        return "categorical", {"values": list(spec)}

    raise TypeError(f"Unsupported search-space spec: {spec!r}")


def _suggest_one(trial, name, spec):
    """Suggest a single hyperparameter from its spec."""
    ptype, cfg = _parse_spec(spec)

    if ptype == "categorical":
        return trial.suggest_categorical(name, cfg["values"])

    low, high = cfg["values"]

    if ptype == "int":
        return trial.suggest_int(
            name,
            int(low),
            int(high),
            step=int(cfg.get("step", 1)),
            log=cfg.get("log", False),
        )

    if ptype == "float":
        return trial.suggest_float(
            name,
            float(low),
            float(high),
            step=cfg.get("step"),
            log=cfg.get("log", False),
        )

    raise ValueError(f"Unknown param type {ptype!r} for {name!r}")


def _suggest_from_grid(trial, search_space):
    """Suggest all parameters from a search-space mapping."""
    return {
        name: _suggest_one(trial, name, spec) for name, spec in search_space.items()
    }


def grid_values(spec):
    """Return the discrete values represented by one search-space spec."""
    ptype, cfg = _parse_spec(spec)
    values = list(cfg["values"])

    if ptype == "categorical":
        return values

    low, high = values
    if ptype == "int":
        step = int(cfg.get("step", 1))
        return list(range(int(low), int(high) + 1, step))

    step = cfg.get("step")
    if step is not None:
        n_steps = int(round((float(high) - float(low)) / float(step)))
        return [float(low) + index * float(step) for index in range(n_steps + 1)]

    return [float(low), float(high)]


def param_distribution(spec):
    """Return the Optuna distribution corresponding to one search-space spec."""
    ptype, cfg = _parse_spec(spec)

    if ptype == "categorical":
        return optuna.distributions.CategoricalDistribution(cfg["values"])

    low, high = cfg["values"]
    if ptype == "int":
        return optuna.distributions.IntDistribution(
            low=int(low),
            high=int(high),
            step=int(cfg.get("step", 1)),
            log=bool(cfg.get("log", False)),
        )

    return optuna.distributions.FloatDistribution(
        low=float(low),
        high=float(high),
        step=cfg.get("step"),
        log=bool(cfg.get("log", False)),
    )


def value_in_spec(value, spec):
    """Return whether a concrete parameter value fits the given spec."""
    ptype, cfg = _parse_spec(spec)

    if ptype == "categorical":
        return value in cfg["values"]

    low, high = cfg["values"]
    if value < low or value > high:
        return False

    step = cfg.get("step")
    if step is None:
        return True

    if ptype == "int":
        return (int(value) - int(low)) % int(step) == 0

    ratio = (float(value) - float(low)) / float(step)
    return np.isclose(ratio, round(ratio))


def suggest_from_search_space(
    trial,
    search_space,
    base_params=None,
    derived_params=None,
):
    """Suggest params from a search-space spec and merge base/derived values."""
    if base_params is None:
        base_params = {}
    else:
        base_params = base_params.copy()

    if derived_params is None:
        derived_params = {}
    else:
        derived_params = derived_params.copy()

    sampled = _suggest_from_grid(trial, search_space)

    suggested_params = base_params.copy()
    suggested_params.update(sampled)
    suggested_params.update(derived_params)
    return suggested_params


def objective(
    trial,
    model_cls,
    cv_folds,
    search_space,
    technical_settings=None,
    base_params=None,
    suggest_params_func=suggest_from_search_space,
    run_test=True,
    pruning_mode="epoch",
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

    n_folds = len(cv_folds)
    logger_kwargs = technical_settings.get("logger_kwargs", {})
    experiment_name = logger_kwargs.get("experiment_name")
    tracking_uri = logger_kwargs.get("tracking_uri")

    settings_tags = technical_settings.get("tags", {})
    run_name_parts = [
        str(settings_tags.get("model_cls") or model_cls.__name__),
        settings_tags.get("conv_layer"),
        settings_tags.get("graph_name"),
    ]
    run_name_base = "__".join(str(part) for part in run_name_parts if part)
    trial_run_name = f"{run_name_base}__trial_{trial.number}"
    # Group all trials of one model/graph combo (mirrors XGBoost's run_group).
    run_group = run_name_base
    # Put them in settings_tags so they land on fold runs as both tags and
    # params (via extra_params), not only on the parent run.
    settings_tags["run_group"] = run_group
    settings_tags["parent_run_name"] = trial_run_name

    mlflow_client = None
    parent_run_id = None
    if experiment_name is not None and n_folds > 1:
        mlflow_client = MlflowClient(tracking_uri=tracking_uri)
        exp = mlflow_client.get_experiment_by_name(experiment_name)
        if exp is None:
            experiment_id = mlflow_client.create_experiment(experiment_name)
        else:
            experiment_id = exp.experiment_id

        parent_run = mlflow_client.create_run(
            experiment_id=experiment_id,
            run_name=trial_run_name,
            tags={
                "trial_id": str(trial.number),
                "run_type": "parent",
                "experiment_name": experiment_name,
                "run_group": run_group,
                "parent_run_name": trial_run_name,
            },
        )
        parent_run_id = parent_run.info.run_id

        mlflow_client.log_param(parent_run_id, "model_cls", model_cls.__name__)
        mlflow_client.log_param(parent_run_id, "parent_run_name", trial_run_name)
        mlflow_client.log_param(parent_run_id, "run_group", run_group)
        for key, value in params.items():
            mlflow_client.log_param(
                parent_run_id,
                key,
                json.dumps(value) if isinstance(value, dict) else value,
            )
        for key, value in settings_tags.items():
            if value is not None:
                mlflow_client.set_tag(parent_run_id, key, str(value))
                mlflow_client.log_param(parent_run_id, key, value)

    try:
        for fold_idx, fold_results in enumerate(cv_folds):
            fold_base_params = fold_results.get("params").copy()
            data_fold = copy.deepcopy(fold_results.get("data"))
            data_fold.keep_on_gpu = technical_settings.get("keep_on_gpu", False)
            data_fold.setup()

            params_fold = params.copy()
            params_fold.update(fold_base_params)

            fold_seed = None
            if fold_seeds is not None and fold_idx < len(fold_seeds):
                fold_seed = fold_seeds[fold_idx]

            tags = settings_tags.copy()
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
                run_name=(
                    trial_run_name
                    if n_folds == 1
                    else f"{trial_run_name}__fold_{fold_idx}"
                ),
                extra_params={
                    key: value
                    for key, value in settings_tags.items()
                    if value is not None and key not in params_fold
                },
                parent_run_id=parent_run_id,
                log_params=True,
                pruning_trial=trial if pruning_mode == "epoch" else None,
                **{
                    key: value
                    for key, value in technical_settings.items()
                    if key not in ("tags", "to_log", "keep_on_gpu")
                },
            )
            best_score = trainer.callback_metrics[
                technical_settings["monitor_kwargs"]["monitor"]
            ].item()
            fold_scores.append(best_score)

            # Snapshot val metrics before test(); Lightning drops them from
            # callback_metrics after a separate test loop, so mean_val/* would
            # otherwise never be aggregated onto the parent run.
            fold_metric_values = {}
            for metric_name, metric_value in trainer.callback_metrics.items():
                try:
                    fold_metric_values[metric_name] = float(metric_value.item())
                except (AttributeError, ValueError, TypeError):
                    continue

            test_dataloader = data_fold.test_dataloader()
            if run_test and test_dataloader is not None:
                trainer.test(
                    dataloaders=test_dataloader,
                    ckpt_path="best",
                    verbose=False,
                )

            for metric_name, metric_value in trainer.callback_metrics.items():
                try:
                    fold_metric_values[metric_name] = float(metric_value.item())
                except (AttributeError, ValueError, TypeError):
                    continue
            fold_metrics.append(fold_metric_values)

            fold_run_ids.append(getattr(trainer.logger, "run_id", None))
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

            if pruning_mode == "fold":
                intermediate_value = float(np.mean(fold_scores))
                trial.report(intermediate_value, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = float(np.mean(fold_scores))

            trial.set_user_attr("best_epochs", fold_best_epochs)
            trial.set_user_attr("best_checkpoints", fold_best_checkpoints)
            trial.set_user_attr("fold_scores", fold_scores)
            trial.set_user_attr("fold_valid_epochs", fold_best_epochs)
            trial.set_user_attr("fold_score", mean_score)
            valid_epochs = [epoch for epoch in fold_best_epochs if epoch is not None]
            if valid_epochs:
                trial.set_user_attr("valid_epoch", float(np.mean(valid_epochs)))

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

        if mlflow_client is not None and parent_run_id is not None and fold_scores:
            valid_epochs = [epoch for epoch in fold_best_epochs if epoch is not None]
            if valid_epochs:
                mlflow_client.log_metric(
                    parent_run_id, "mean_best_epoch", float(np.mean(valid_epochs))
                )
                mlflow_client.log_metric(
                    parent_run_id, "epoch", float(np.mean(valid_epochs))
                )

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

            child_run_ids = [run_id for run_id in fold_run_ids if run_id is not None]
            metric_names = {name for fold in fold_metrics for name in fold}
            for metric_name in sorted(metric_names):
                values = [
                    fold[metric_name] for fold in fold_metrics if metric_name in fold
                ]
                if not values:
                    continue
                metric_mean = float(np.mean(values))
                mlflow_client.log_metric(parent_run_id, metric_name, metric_mean)
                metric_std = float(np.std(values))
                for run_id in [parent_run_id, *child_run_ids]:
                    mlflow_client.log_metric(run_id, f"mean_{metric_name}", metric_mean)
                    mlflow_client.log_metric(run_id, f"std_{metric_name}", metric_std)

        parent_status = "FINISHED"
        parent_trial_status = "finished"
        parent_error = None
        return mean_score

    except optuna.TrialPruned:
        parent_status = "KILLED"
        parent_trial_status = "pruned"
        parent_error = None
        raise
    except Exception as exc:
        parent_status = "FAILED"
        parent_trial_status = "failed"
        error_tb = traceback.format_exc()
        parent_error = (exc, error_tb)

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
            try:
                mlflow_client.set_tag(
                    parent_run_id, "trial_status", parent_trial_status
                )
                if parent_error is not None:
                    exc, error_tb = parent_error
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
