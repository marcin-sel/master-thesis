from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import optuna
import pandas as pd
from my_project.gnn.data_module import GNNDataModule
from my_project.gnn.tuning import objective
from optuna.trial import TrialState


def build_cv_datamodules(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    folds: Sequence[dict[str, Any]],
    graph: Any | None = None,
    graphs: Sequence[Any] | None = None,
    preprocessing_pipeline,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Create one GNNDataModule per CV fold after fitting preprocessing on train only.

    You can pass either:
    - graph: one graph reused for all folds
    - graphs: a sequence of graphs with the same length as folds
    """

    if graphs is not None:
        fold_graphs = list(graphs)
        if len(fold_graphs) != len(folds):
            raise ValueError(
                f"Length mismatch: got {len(fold_graphs)} graphs for {len(folds)} folds."
            )
    else:
        fold_graphs = [graph] * len(folds)

    data = []

    for fold, fold_graph in zip(folds, fold_graphs):
        X_train = X.loc[fold["train"]]
        y_train = y.loc[fold["train"]]
        X_valid = X.loc[fold["valid"]]
        y_valid = y.loc[fold["valid"]]

        if preprocessing_pipeline is not None:
            preprocessing_pipeline.fit(X_train)

            X_train = preprocessing_pipeline.transform(X_train)
            X_valid = preprocessing_pipeline.transform(X_valid)

        data_fold = GNNDataModule(
            graph=fold_graph,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            num_workers=num_workers,
        )
        data_fold.setup()
        data.append(data_fold)

    return {
        "data": data,
        "preprocessing_pipeline": preprocessing_pipeline,
    }


def run_optuna_study_for_gnn(
    *,
    model_cls,
    data: GNNDataModule | Sequence[GNNDataModule],
    study_name: str,
    storage_url: str | None,
    n_trials: int,
    trainer_kwargs: dict[str, Any],
    monitor_mode: str,
    show_progress_bar: bool = False,
    base_params: dict[str, Any] | None = None,
    optuna_n_jobs: int = 1,
) -> optuna.Study:
    direction = "maximize" if monitor_mode == "max" else "minimize"

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )

    finished_states = {TrialState.COMPLETE, TrialState.PRUNED}
    finished_trials = sum(1 for trial in study.trials if trial.state in finished_states)
    remaining_trials = max(0, n_trials - finished_trials)

    if remaining_trials == 0:
        return study

    study.optimize(
        lambda trial: objective(
            trial,
            model_cls=model_cls,
            data=data,
            trainer_kwargs=copy.deepcopy(trainer_kwargs),
            base_params=copy.deepcopy(base_params),
        ),
        n_jobs=optuna_n_jobs,
        n_trials=remaining_trials,
        show_progress_bar=show_progress_bar,
        gc_after_trial=True,
    )
    return study


def get_best_trial_checkpoint(study: optuna.Study, monitor_mode: str) -> dict[str, Any]:
    """Return best checkpoint metadata for inference from an Optuna study."""
    best_trial = study.best_trial
    best_checkpoints = best_trial.user_attrs.get("best_checkpoints", [])
    fold_scores = best_trial.user_attrs.get("fold_scores", [])

    if not best_checkpoints:
        raise ValueError(
            "No checkpoints found in study.best_trial.user_attrs['best_checkpoints']"
        )

    if fold_scores and len(fold_scores) == len(best_checkpoints):
        if monitor_mode == "max":
            best_fold_idx = int(np.argmax(fold_scores))
        else:
            best_fold_idx = int(np.argmin(fold_scores))
    else:
        best_fold_idx = 0

    return {
        "best_trial": best_trial,
        "best_fold_idx": best_fold_idx,
        "best_model_path": best_checkpoints[best_fold_idx],
        "fold_scores": fold_scores,
        "best_checkpoints": best_checkpoints,
    }
