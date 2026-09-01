from __future__ import annotations

import copy
import json
import os
import re
from inspect import signature
from pathlib import Path
from typing import Any, Sequence

import lightning as L
import mlflow
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from mlflow.tracking import MlflowClient
from sklearn import config_context
from xgboost import XGBClassifier

from my_project.gnn.trainer import GNNLightningModule

from .run_optuna_study import build_cv_datamodules
from .training_helpers import resolve_artifact_location
from .xgboost_training import _align_xgb_categoricals, binary_metrics


def final_split_indices(folds: Sequence[dict[str, Any]]) -> tuple[pd.Index, pd.Index]:
    """Return the common train+valid and held-out test indices from CV folds."""
    if not folds:
        raise ValueError("At least one fold is required")

    train_sets = [set(fold["train"]) | set(fold["valid"]) for fold in folds]
    test_sets = [set(fold["test"]) for fold in folds]
    if any(indices != train_sets[0] for indices in train_sets[1:]):
        raise ValueError("CV folds do not share one train+valid population")
    if any(indices != test_sets[0] for indices in test_sets[1:]):
        raise ValueError("CV folds do not share one held-out test set")
    if train_sets[0] & test_sets[0]:
        raise ValueError("Final train and test indices overlap")

    full_order = list(folds[0]["train"]) + list(folds[0]["valid"])
    train_index = pd.Index(list(dict.fromkeys(full_order)))
    test_index = pd.Index(list(folds[0]["test"]))
    return train_index, test_index


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _final_epochs(best_epochs: Sequence[int | float | None]) -> int:
    valid = [int(epoch) for epoch in best_epochs if epoch is not None]
    if not valid:
        raise ValueError("The best trial has no CV-selected best_epochs")
    return max(1, int(round(float(np.median(valid)))) + 1)


def _save_graph_artifact(graph, graph_name: str, path: Path) -> None:
    nodes = list(graph.nodes())
    node_positions = {node: index for index, node in enumerate(nodes)}
    edges = [
        [source, target]
        if node_positions[source] <= node_positions[target]
        else [target, source]
        for source, target in graph.edges()
    ]
    edges.sort(key=lambda edge: (node_positions[edge[0]], node_positions[edge[1]]))
    payload = {
        "graph_name": graph_name,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "nodes": [str(node) for node in nodes],
        "edges": [[str(source), str(target)] for source, target in edges],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _log_final_run(
    *,
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    tags: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: Sequence[Path],
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=resolve_artifact_location(tracking_uri),
        )
    else:
        experiment_id = experiment.experiment_id

    string_tags = {key: str(value) for key, value in tags.items() if value is not None}
    run = client.create_run(
        experiment_id,
        tags={**string_tags, "mlflow.runName": run_name},
    )
    run_id = run.info.run_id
    for key, value in params.items():
        logged_value = (
            json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
        )
        client.log_param(run_id, key, logged_value)
    for key, value in metrics.items():
        client.log_metric(run_id, key, float(value))
    for path in artifact_paths:
        client.log_artifact(run_id, str(path))
    client.set_terminated(run_id, status="FINISHED")
    return run_id


def _existing_final_run(
    *,
    tracking_uri: str,
    experiment_name: str,
    model_cls: str,
    graph_name: str,
    final_protocol: str,
    predictions_saved: bool,
    source_study_name: str,
    source_trial_number: int,
) -> Any | None:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            "tags.run_type = 'final_refit' AND "
            f"tags.model_cls = '{model_cls}' AND "
            f"tags.graph_name = '{graph_name}' AND "
            f"tags.final_protocol = '{final_protocol}' AND "
            f"tags.predictions_saved = '{predictions_saved}' AND "
            f"tags.source_study_name = '{source_study_name}' AND "
            f"tags.source_trial_number = '{source_trial_number}'"
        ),
        max_results=1,
    )
    return runs[0] if runs else None


def fit_final_gnn(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    train_index: pd.Index,
    valid_index: pd.Index,
    test_index: pd.Index,
    graph,
    graph_name: str,
    model_cls,
    source_study_name: str,
    source_trial_number: int,
    best_params: dict[str, Any],
    best_epochs: Sequence[int | float | None],
    preprocessing_pipeline,
    trainer_kwargs: dict[str, Any],
    monitor_kwargs: dict[str, Any],
    early_stopping_kwargs: dict[str, Any],
    tracking_uri: str,
    experiment_name: str,
    output_dir: str | os.PathLike[str],
    seed: int,
    keep_on_gpu: bool = True,
    save_predictions: bool = True,
) -> dict[str, Any]:
    """Refit one GNN/MLP with a fixed validation set, then test exactly once."""
    if (
        train_index.intersection(valid_index).size
        or train_index.intersection(test_index).size
        or valid_index.intersection(test_index).size
    ):
        raise ValueError("Final train, validation, and test indices must be disjoint")

    model_name = model_cls.__name__
    cv_median_best_epochs = _final_epochs(best_epochs)
    artifact_dir = Path(output_dir) / _slug(f"{model_name}__{graph_name}")
    checkpoint_path = artifact_dir / "final.ckpt"
    predictions_path = artifact_dir / "test_predictions.csv"
    graph_path = artifact_dir / "final_graph.json"
    existing_run = _existing_final_run(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        model_cls=model_name,
        graph_name=graph_name,
        final_protocol="fixed_validation",
        predictions_saved=save_predictions,
        source_study_name=source_study_name,
        source_trial_number=source_trial_number,
    )
    if existing_run is not None:
        return {
            "model_cls": model_name,
            "graph_name": graph_name,
            "cv_median_best_epochs": cv_median_best_epochs,
            "run_id": existing_run.info.run_id,
            "model_path": str(checkpoint_path),
            "predictions_path": str(predictions_path) if save_predictions else None,
            "source_study_name": source_study_name,
            "source_trial_number": source_trial_number,
            "reused": True,
            **existing_run.data.metrics,
        }
    L.seed_everything(seed, workers=True)
    final_fold = {
        "train": train_index,
        "valid": valid_index,
        "test": test_index,
    }
    built = build_cv_datamodules(
        X=X,
        y=y,
        folds=[final_fold],
        graph=graph,
        preprocessing_pipeline=copy.deepcopy(preprocessing_pipeline),
        num_workers=0,
        keep_on_gpu=keep_on_gpu,
    )[0]
    data = built["data"]
    params = {**copy.deepcopy(best_params), **built["params"]}
    data.batch_size = int(params.get("batch_size", data.batch_size))

    model_param_names = set(signature(model_cls.__init__).parameters)
    model_params = {
        key: value for key, value in params.items() if key in model_param_names
    }
    module_param_names = set(signature(GNNLightningModule.__init__).parameters)
    module_params = {
        key: value for key, value in params.items() if key in module_param_names
    }
    module = GNNLightningModule(
        model_cls=model_cls,
        model_kwargs=model_params,
        **module_params,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    _save_graph_artifact(graph, graph_name, graph_path)
    checkpoint_callback = ModelCheckpoint(
        **monitor_kwargs,
        save_top_k=1,
        dirpath=artifact_dir,
        filename="final",
        enable_version_counter=False,
    )
    early_stopping_callback = EarlyStopping(**early_stopping_kwargs)
    fit_kwargs = copy.deepcopy(trainer_kwargs)
    fit_kwargs.update(
        {
            "logger": False,
            "callbacks": [early_stopping_callback, checkpoint_callback],
            "enable_checkpointing": True,
        }
    )
    trainer = L.Trainer(**fit_kwargs)
    trainer.fit(module, datamodule=data)
    epochs_trained = trainer.current_epoch
    global_steps = trainer.global_step

    split_predictions = {}
    metrics = {}
    for split_name, dataloader in (
        ("train", data.train_dataloader()),
        ("valid", data.val_dataloader()),
        ("test", data.test_dataloader()),
    ):
        prediction_batches = trainer.predict(
            module,
            dataloaders=dataloader,
            ckpt_path=checkpoint_callback.best_model_path,
        )
        predictions = {
            key: torch.cat([batch[key] for batch in prediction_batches]).numpy()
            for key in ("graph_ids", "y_true", "y_pred", "y_score", "logit")
        }
        split_predictions[split_name] = predictions
        metrics.update(
            binary_metrics(
                predictions["y_true"],
                predictions["y_score"],
                prefix=split_name,
                threshold=module.threshold,
            )
        )
        metrics[f"{split_name}/n_samples"] = len(predictions["y_true"])

    test_predictions = split_predictions["test"]
    observation_index = test_predictions["graph_ids"]
    y_true = test_predictions["y_true"]
    if len(y_true) != len(test_index):
        raise RuntimeError("Prediction count does not match the held-out test size")
    if not np.array_equal(observation_index, test_index.to_numpy()):
        raise RuntimeError("Prediction indices do not match the held-out test indices")

    metrics.update(
        {
            "final/epochs_trained": epochs_trained,
            "final/global_steps": global_steps,
            "final/best_monitor_score": float(checkpoint_callback.best_model_score),
        }
    )
    if save_predictions:
        pd.DataFrame(
            {
                "index": observation_index,
                "y_true": y_true,
                "logit": test_predictions["logit"],
                "y_pred": test_predictions["y_pred"],
                "y_score": test_predictions["y_score"],
            }
        ).to_csv(predictions_path, index=False)

    run_id = _log_final_run(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=f"final__{model_name}__{graph_name}",
        params={
            **params,
            "cv_median_best_epochs": cv_median_best_epochs,
            "final_max_epochs": trainer.max_epochs,
        },
        tags={
            "run_type": "final_refit",
            "model_cls": model_name,
            "graph_name": graph_name,
            "final_protocol": "fixed_validation",
            "test_evaluation": "single_final",
            "predictions_saved": save_predictions,
            "source_study_name": source_study_name,
            "source_trial_number": source_trial_number,
        },
        metrics=metrics,
        artifact_paths=[checkpoint_path, graph_path]
        + ([predictions_path] if save_predictions else []),
    )
    return {
        "model_cls": model_name,
        "graph_name": graph_name,
        "cv_median_best_epochs": cv_median_best_epochs,
        "final_max_epochs": trainer.max_epochs,
        "final_epochs_trained": epochs_trained,
        "final_best_score": float(checkpoint_callback.best_model_score),
        "run_id": run_id,
        "model_path": str(checkpoint_path),
        "predictions_path": str(predictions_path) if save_predictions else None,
        "source_study_name": source_study_name,
        "source_trial_number": source_trial_number,
        "reused": False,
        **metrics,
    }


def fit_final_xgboost(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    train_index: pd.Index,
    test_index: pd.Index,
    source_study_name: str,
    source_trial_number: int,
    best_params: dict[str, Any],
    best_n_estimators: int,
    preprocessing_pipeline,
    tracking_uri: str,
    experiment_name: str,
    output_dir: str | os.PathLike[str],
    save_predictions: bool = True,
) -> dict[str, Any]:
    """Refit XGBoost on all development data, then test exactly once."""
    if train_index.intersection(test_index).size:
        raise ValueError("Final train and test indices overlap")
    if best_n_estimators is None:
        raise ValueError("The best XGBoost trial has no CV-selected estimator count")

    artifact_dir = Path(output_dir) / "XGBoost__xgboost_full"
    model_path = artifact_dir / "final.json"
    predictions_path = artifact_dir / "test_predictions.csv"
    existing_run = _existing_final_run(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        model_cls="XGBoost",
        graph_name="xgboost_full",
        final_protocol="train_valid_refit",
        predictions_saved=save_predictions,
        source_study_name=source_study_name,
        source_trial_number=source_trial_number,
    )
    if existing_run is not None:
        return {
            "model_cls": "XGBoost",
            "graph_name": "xgboost_full",
            "best_n_estimators": int(best_n_estimators),
            "run_id": existing_run.info.run_id,
            "model_path": str(model_path),
            "predictions_path": str(predictions_path) if save_predictions else None,
            "source_study_name": source_study_name,
            "source_trial_number": source_trial_number,
            "reused": True,
            **existing_run.data.metrics,
        }
    X_train = X.loc[train_index]
    y_train = y.loc[train_index].astype(int)
    X_test = X.loc[test_index]
    y_test = y.loc[test_index].astype(int)
    if preprocessing_pipeline is not None:
        pipeline = copy.deepcopy(preprocessing_pipeline)
        with config_context(transform_output="pandas"):
            X_train = pipeline.fit_transform(X_train, y_train)
            X_test = pipeline.transform(X_test)
    X_train, (X_test,) = _align_xgb_categoricals(X_train, [X_test])

    params = copy.deepcopy(best_params)
    params["n_estimators"] = int(best_n_estimators)
    params["enable_categorical"] = True
    params.pop("early_stopping_rounds", None)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    train_score = model.predict_proba(X_train)[:, 1]
    logits = model.predict(X_test, output_margin=True)
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    metrics = {
        **binary_metrics(y_train.to_numpy(), train_score, prefix="train_valid"),
        **binary_metrics(y_test.to_numpy(), y_score, prefix="test"),
        "train_valid/n_samples": len(y_train),
        "test/n_samples": len(y_test),
        "final/n_estimators": int(best_n_estimators),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    if save_predictions:
        pd.DataFrame(
            {
                "index": y_test.index,
                "y_true": y_test.to_numpy(),
                "logit": logits,
                "y_pred": y_pred,
                "y_score": y_score,
            }
        ).to_csv(predictions_path, index=False)

    run_id = _log_final_run(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name="final__XGBoost__xgboost_full",
        params=params,
        tags={
            "run_type": "final_refit",
            "model_cls": "XGBoost",
            "graph_name": "xgboost_full",
            "final_protocol": "train_valid_refit",
            "test_evaluation": "single_final",
            "predictions_saved": save_predictions,
            "source_study_name": source_study_name,
            "source_trial_number": source_trial_number,
        },
        metrics=metrics,
        artifact_paths=[model_path] + ([predictions_path] if save_predictions else []),
    )
    return {
        "model_cls": "XGBoost",
        "graph_name": "xgboost_full",
        "best_n_estimators": int(best_n_estimators),
        "run_id": run_id,
        "model_path": str(model_path),
        "predictions_path": str(predictions_path) if save_predictions else None,
        "source_study_name": source_study_name,
        "source_trial_number": source_trial_number,
        "reused": False,
        **metrics,
    }


__all__ = ["final_split_indices", "fit_final_gnn", "fit_final_xgboost"]
