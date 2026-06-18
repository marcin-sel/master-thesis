from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import optuna
import pandas as pd
from my_project.gnn.data_module import GNNDataModule
from my_project.gnn.tuning import objective, suggest_gnn_params
from my_project.gnn.utils import feature_indexes, feature_n_classes
from optuna.trial import TrialState
from sklearn.pipeline import Pipeline


def build_cv_datamodules(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    folds: Sequence[dict[str, Any]],
    graph: Any | None = None,
    graphs: Sequence[Any] | None = None,
    graph_builder: Callable[[pd.DataFrame, pd.Series], Any] | None = None,
    graph_preprocessing_pipeline=None,
    preprocessing_pipeline=None,
    num_workers: int = 0,
    keep_on_gpu: bool = False,
    device: str | None = None,
) -> list[tuple[GNNDataModule, dict[str, Any]]]:
    """Create one GNNDataModule per CV fold after fitting preprocessing on train only.

    You can pass exactly one of:
    - graph: one graph reused for all folds
    - graphs: a sequence of graphs with the same length as folds
    - graph_builder: a callable ``(X_train, y_train) -> graph`` invoked once per
      fold so the graph is learned from train data only. By default it receives
      the same *preprocessed* training slice the model sees. Pass
      ``graph_preprocessing_pipeline`` to instead feed it a separately processed
      view of the fold's raw training rows (e.g. a discretizer, since
      interaction-information builders need discrete inputs); the pipeline is
      fit on train only and the resulting graph's nodes still line up with the
      model features because both share the same column names.
    """

    provided = [
        name
        for name, value in (
            ("graph", graph),
            ("graphs", graphs),
            ("graph_builder", graph_builder),
        )
        if value is not None
    ]
    if len(provided) > 1:
        raise ValueError(
            f"Pass only one of graph/graphs/graph_builder, got: {provided}."
        )

    if graph_preprocessing_pipeline is not None and graph_builder is None:
        raise ValueError(
            "graph_preprocessing_pipeline requires graph_builder to be set."
        )

    if graphs is not None:
        fold_graphs = list(graphs)
        if len(fold_graphs) != len(folds):
            raise ValueError(
                f"Length mismatch: got {len(fold_graphs)} graphs for {len(folds)} folds."
            )
    else:
        fold_graphs = [graph] * len(folds)

    results = []

    for fold_index, fold in enumerate(folds):
        X_train = X.loc[fold["train"]]
        y_train = y.loc[fold["train"]]
        X_valid = X.loc[fold["valid"]]
        y_valid = y.loc[fold["valid"]]

        # Build the graph from the raw train slice (before the model's
        # preprocessing rescales it) when a dedicated graph pipeline is given.
        X_train_graph = X_train
        if graph_preprocessing_pipeline is not None:
            graph_preprocessing_pipeline.fit(X_train)
            X_train_graph = graph_preprocessing_pipeline.transform(X_train)

        categorical_dtypes = ["category", "object", "bool", "boolean"]

        if preprocessing_pipeline is not None:
            # preprocessing_pipeline = copy.deepcopy(preprocessing_pipeline)
            preprocessing_pipeline.fit(X_train)

            # Detect categorical features from the preprocessed-but-not-encoded
            # view, so quantile-binned high-missing columns (now categorical
            # with a "Missing" level) are recognized alongside the original
            # categoricals before the encoder turns everything into integers.
            if "encoder" in preprocessing_pipeline.named_steps:
                pre_encoder = Pipeline(preprocessing_pipeline.steps[:-1])
                encoder = preprocessing_pipeline.named_steps["encoder"]
                X_train_pre = pre_encoder.transform(X_train)
                X_valid_pre = pre_encoder.transform(X_valid)
                categorical_features = X_train_pre.select_dtypes(
                    include=categorical_dtypes
                ).columns.to_list()
                X_train = encoder.transform(X_train_pre)
                X_valid = encoder.transform(X_valid_pre)
            else:
                X_train = preprocessing_pipeline.transform(X_train)
                X_valid = preprocessing_pipeline.transform(X_valid)
                categorical_features = X_train.select_dtypes(
                    include=categorical_dtypes
                ).columns.to_list()
        else:
            categorical_features = X.select_dtypes(
                include=categorical_dtypes
            ).columns.to_list()

        if graph_builder is not None:
            graph_input = (
                X_train_graph if graph_preprocessing_pipeline is not None else X_train
            )
            fold_graph = graph_builder(graph_input, y_train)
        else:
            fold_graph = fold_graphs[fold_index]

        columns = list(X_train.columns)

        if categorical_features:
            categorical_features_indexes = pd.Series(
                feature_indexes(categorical_features, columns)
            )[categorical_features]
            categorical_features_n_classes = pd.Series(
                feature_n_classes(X_train[categorical_features])
            )[categorical_features]

            categorical_features_index_n_classes_map = dict(
                zip(categorical_features_indexes, categorical_features_n_classes)
            )
            categorical_features_index_n_classes_map = {
                int(idx): int(n_classes)
                for idx, n_classes in sorted(
                    categorical_features_index_n_classes_map.items(),
                    key=lambda item: item[0],
                )
            }
        else:
            categorical_features_index_n_classes_map = {}

        data_fold = GNNDataModule(
            graph=fold_graph,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            num_workers=num_workers,
            keep_on_gpu=keep_on_gpu,
            device=device,
        )
        data_fold.setup()
        results.append(
            {
                "data": data_fold,
                "params": {
                    "categorical_features_index_n_classes_map": categorical_features_index_n_classes_map,
                    "n_nodes": len(columns),
                    "n_edges": fold_graph.number_of_edges(),
                },
            }
        )

    return results


def run_optuna_study_for_gnn(
    *,
    model_cls,
    cv_folds: dict[str, Any] | Sequence[dict[str, Any]],
    study_name: str,
    storage_url: str | None,
    search_space: dict[str, Any],
    n_trials: int | None = None,
    base_params: dict[str, Any] | None = None,
    show_progress_bar: bool = False,
    optuna_n_jobs: int = 1,
    direction: str = "minimize",
    technical_settings: dict[str, Any],
    suggest_params_func: callable = suggest_gnn_params,
    pruner: optuna.pruners.BasePruner | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    callbacks: Sequence[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]]
    | None = None,
) -> optuna.Study:
    if pruner is None:
        pruner = optuna.pruners.MedianPruner()

    grid_keys = list(search_space.keys())
    grid_distributions = {
        key: optuna.distributions.CategoricalDistribution(values)
        for key, values in search_space.items()
    }

    storage = optuna.storages.get_storage(storage_url)

    # Optuna's storage locks each parameter's categorical choices to the values
    # seen in the first trial. Extending the grid (e.g. adding a `dropout`
    # value) and reusing the study would therefore raise
    # "CategoricalDistribution does not support dynamic value space". To keep a
    # *stable* study name and still avoid recomputing points we already
    # evaluated, we rebuild the study in place when the grid changed: salvage
    # every COMPLETE trial that still falls inside the new grid, drop the study,
    # recreate it under the same name, and re-insert the salvaged trials
    # carrying the new (wider) distributions.
    salvaged_trials = []
    try:
        existing_study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        existing_study = None

    if existing_study is not None:
        stored_distributions = {}
        for trial in existing_study.get_trials(deepcopy=False):
            stored_distributions.update(trial.distributions)

        grid_changed = bool(stored_distributions) and any(
            stored_distributions.get(key) != grid_distributions[key]
            for key in grid_keys
        )

        if grid_changed:
            for trial in existing_study.get_trials(
                deepcopy=False, states=[TrialState.COMPLETE]
            ):
                if set(trial.params) == set(grid_keys) and all(
                    trial.params[key] in search_space[key] for key in grid_keys
                ):
                    salvaged_trials.append(
                        optuna.trial.create_trial(
                            state=TrialState.COMPLETE,
                            params=dict(trial.params),
                            distributions=grid_distributions,
                            value=trial.value,
                            user_attrs=dict(trial.user_attrs),
                        )
                    )
            optuna.delete_study(study_name=study_name, storage=storage)

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )

    for salvaged_trial in salvaged_trials:
        study.add_trial(salvaged_trial)

    if n_trials is None:
        # --- Exhaustive grid mode (resumable) ---
        # We do not rely on GridSampler's own resume logic: it keys "visited"
        # cells on an exact search-space match, so extending the grid (changing
        # the length of any dimension) makes it forget every past trial and
        # re-run the whole grid, duplicates included. Instead we build the grid
        # ourselves from `search_space`, skip combinations already covered by a
        # COMPLETE trial, and explicitly enqueue only the missing ones. This
        # lets us extend the grid and continue, and retries failed combinations.
        all_combinations = [
            dict(zip(grid_keys, values))
            for values in itertools.product(*search_space.values())
        ]

        def _combo_key(params: dict[str, Any]) -> tuple:
            return tuple(params[key] for key in grid_keys)

        completed = {
            _combo_key(trial.params)
            for trial in study.trials
            if trial.state == TrialState.COMPLETE
            and all(key in trial.params for key in grid_keys)
        }

        missing_combinations = [
            combo for combo in all_combinations if _combo_key(combo) not in completed
        ]

        if not missing_combinations:
            return study

        for combo in missing_combinations:
            study.enqueue_trial(combo, skip_if_exists=True)

        n_trials_to_run = len(missing_combinations)
    else:
        # --- Sampling mode (TPE / Random / CMA-ES / ...) ---
        # The sampler draws points from the same categorical choices, so
        # `search_space` still defines the space; `n_trials` is the budget.
        # Resuming counts already-finished trials toward that budget.
        finished_trials = sum(
            1
            for trial in study.trials
            if trial.state in {TrialState.COMPLETE, TrialState.PRUNED}
        )
        n_trials_to_run = max(0, n_trials - finished_trials)

        if n_trials_to_run == 0:
            return study

    study.optimize(
        lambda trial: objective(
            trial,
            model_cls=model_cls,
            cv_folds=cv_folds,
            search_space=search_space,
            technical_settings=technical_settings,
            base_params=copy.deepcopy(base_params),
            suggest_params_func=suggest_params_func,
        ),
        n_jobs=optuna_n_jobs,
        n_trials=n_trials_to_run,
        show_progress_bar=show_progress_bar,
        gc_after_trial=True,
        callbacks=callbacks,
    )
    return study


def get_best_trial_checkpoint(
    study: optuna.Study, monitor_mode: str = "min"
) -> dict[str, Any]:
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
