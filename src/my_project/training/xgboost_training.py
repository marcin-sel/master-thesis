"""XGBoost baseline tuning helpers used by sweep orchestration."""

import copy
import os

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn import config_context
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from my_project.experiments.search_space import XGB_SEARCH_SPACE

from .training_helpers import _build_study_name, resolve_artifact_location
from .tuning import suggest_from_search_space


def _xgb_result_from_study(study):
    """Standardized XGBoost result payload used by sweep aggregations."""
    return {
        "graph_name": "xgboost_full",
        "model_cls": "XGBoost",
        "best_value": study.best_value,
        **study.best_trial.user_attrs.get("test_metrics", {}),
    }


def binary_metrics(y_true, proba, *, prefix, threshold=0.5):
    """Compute the same metric set the GNN logs, under the same names/prefix."""
    pred = (proba >= threshold).astype(int)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        f"{prefix}/accuracy": accuracy_score(y_true, pred),
        f"{prefix}/auc": roc_auc_score(y_true, proba),
        f"{prefix}/avg_precision": average_precision_score(y_true, proba),
        f"{prefix}/f1": f1_score(y_true, pred, zero_division=0),
        f"{prefix}/precision": precision_score(y_true, pred, zero_division=0),
        f"{prefix}/recall": recall_score(y_true, pred, zero_division=0),
        f"{prefix}/specificity": specificity,
    }


def _norm_cat_series(s):
    """Normalize categorical-like series to string values with explicit missing."""
    arr = s.astype("object").to_numpy(copy=True)
    arr[pd.isna(arr)] = "missing"
    return arr.astype(str)


def _align_xgb_categoricals(X_train, X_others):
    """Align pandas categorical dtypes across train/valid/test for XGBoost."""
    X_train = X_train.copy()
    outs = [x.copy() for x in X_others]
    cat_cols = X_train.select_dtypes(
        include=["object", "string", "boolean", "bool", "category"]
    ).columns
    for col in cat_cols:
        train_col = pd.Categorical(_norm_cat_series(X_train[col]))
        X_train[col] = train_col
        cats = train_col.categories
        for xo in outs:
            xo[col] = pd.Categorical(_norm_cat_series(xo[col]), categories=cats)
    return X_train, outs


def _run_xgboost_folded_study(
    *,
    X,
    y,
    folds,
    study_name,
    n_trials,
    seed,
    optuna_storage,
    n_startup_trials,
    experiment_name,
    run_tags,
    suggest_params_func,
    xgb_early_stopping_rounds,
    preprocessing_pipeline=None,
    mlflow_tracking_uri=None,
    enqueue_params=None,
):
    """Shared Optuna+MLflow runner for XGBoost over predefined folds."""
    loggable_tags = {k: v for k, v in run_tags.items() if v is not None}

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()
    if mlflow_client.get_experiment_by_name(experiment_name) is None:
        mlflow_client.create_experiment(
            experiment_name,
            artifact_location=resolve_artifact_location(
                mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
            ),
        )
    mlflow.set_experiment(experiment_name)

    def objective(trial):
        params = suggest_params_func(trial)
        graph_name = run_tags.get("graph_name", "xgboost_full")
        parent_run_name = f"XGBoost__{graph_name}__trial_{trial.number}"

        fold_val_scores = []
        fold_val_metrics = []
        fold_test_metrics = []
        fold_train_metrics = []
        fold_run_ids = []
        fold_best_ns = []

        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            parent_run_id = parent_run.info.run_id
            mlflow.set_tags(
                {**run_tags, "run_type": "parent", "trial_id": trial.number}
            )
            mlflow.log_params(loggable_tags)
            mlflow.log_param("parent_run_name", parent_run_name)
            mlflow.log_params(params)

            for fold_idx, fold in enumerate(folds):
                X_tr, y_tr = X.loc[fold["train"]], y.loc[fold["train"]].astype(int)
                X_va, y_va = X.loc[fold["valid"]], y.loc[fold["valid"]].astype(int)
                X_te, y_te = X.loc[fold["test"]], y.loc[fold["test"]].astype(int)

                if preprocessing_pipeline is not None:
                    pre = copy.deepcopy(preprocessing_pipeline)
                    with config_context(transform_output="pandas"):
                        X_tr = pre.fit_transform(X_tr, y_tr)
                        X_va = pre.transform(X_va)
                        X_te = pre.transform(X_te)

                X_tr, (X_va, X_te) = _align_xgb_categoricals(X_tr, [X_va, X_te])

                scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                model = XGBClassifier(
                    **params,
                    scale_pos_weight=scale_pos_weight,
                    early_stopping_rounds=xgb_early_stopping_rounds,
                    enable_categorical=True,
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

                best_iter = getattr(model, "best_iteration", None)
                best_n = (
                    int(best_iter + 1)
                    if best_iter is not None
                    else int(params.get("n_estimators", 2000))
                )

                val_m = binary_metrics(
                    y_va.to_numpy(), model.predict_proba(X_va)[:, 1], prefix="val"
                )
                test_m = binary_metrics(
                    y_te.to_numpy(), model.predict_proba(X_te)[:, 1], prefix="test"
                )
                train_m = binary_metrics(
                    y_tr.to_numpy(), model.predict_proba(X_tr)[:, 1], prefix="train"
                )

                with mlflow.start_run(
                    run_name=f"{parent_run_name}__fold_{fold_idx}",
                    nested=True,
                ) as fold_run:
                    mlflow.set_tags(
                        {
                            **run_tags,
                            "run_type": "fold",
                            "trial_id": trial.number,
                            "fold_id": fold_idx,
                        }
                    )
                    mlflow.log_params(loggable_tags)
                    mlflow.log_param("parent_run_name", parent_run_name)
                    mlflow.log_params(params)
                    mlflow.log_param("scale_pos_weight", float(scale_pos_weight))
                    mlflow.log_param("best_n_estimators", best_n)
                    mlflow.log_metrics(val_m)
                    mlflow.log_metrics(test_m)
                    mlflow.log_metrics(train_m)

                fold_run_ids.append(fold_run.info.run_id)
                fold_val_scores.append(val_m["val/auc"])
                fold_val_metrics.append(val_m)
                fold_test_metrics.append(test_m)
                fold_train_metrics.append(train_m)
                fold_best_ns.append(best_n)

            all_fold_metrics = [
                {**v, **t, **tr}
                for v, t, tr in zip(
                    fold_val_metrics, fold_test_metrics, fold_train_metrics
                )
            ]
            metric_names = {
                name for fold_metrics in all_fold_metrics for name in fold_metrics
            }
            mean_metrics = {}
            for name in sorted(metric_names):
                values = [
                    fold_metrics[name]
                    for fold_metrics in all_fold_metrics
                    if name in fold_metrics
                ]
                if not values:
                    continue
                metric_mean = float(np.mean(values))
                metric_std = float(np.std(values))
                mean_metrics[name] = metric_mean
                mlflow_client.log_metric(parent_run_id, name, metric_mean)
                for rid in [parent_run_id, *fold_run_ids]:
                    mlflow_client.log_metric(rid, f"mean_{name}", metric_mean)
                    mlflow_client.log_metric(rid, f"std_{name}", metric_std)

            best_n_mean = float(np.mean(fold_best_ns))
            best_n_std = float(np.std(fold_best_ns))
            mlflow_client.log_metric(
                parent_run_id, "mean_best_n_estimators", best_n_mean
            )
            for rid in [parent_run_id, *fold_run_ids]:
                mlflow_client.log_metric(rid, "mean_best_n_estimators", best_n_mean)
                mlflow_client.log_metric(rid, "std_best_n_estimators", best_n_std)

        trial.set_user_attr("mean_metrics", mean_metrics)
        trial.set_user_attr("model_cls", "XGBoost")
        if len(fold_test_metrics) == 1:
            trial.set_user_attr("test_metrics", fold_test_metrics[0])
        else:
            trial.set_user_attr(
                "test_metrics",
                {k: v for k, v in mean_metrics.items() if k.startswith("test/")},
            )
        return float(np.mean(fold_val_scores))

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        n_startup_trials=n_startup_trials,
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=optuna_storage,
        load_if_exists=True,
        sampler=sampler,
    )

    if enqueue_params is not None:
        for params in enqueue_params:
            study.enqueue_trial(params, skip_if_exists=True)
        budget = len(enqueue_params)
    else:
        budget = n_trials

    finished = sum(
        1
        for trial in study.trials
        if trial.state
        in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    )
    remaining = max(0, budget - finished)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    return study


def tune_xgboost(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
    *,
    n_trials,
    seed,
    optuna_storage,
    n_startup_trials=10,
    experiment_name="test_synthetic_tuning",
    extra_tags=None,
    top_k=None,
    main_data_seed=None,
    preprocessing_pipeline=None,
    xgb_search_space=None,
):
    X_all = pd.concat([X_train, X_valid, X_test])
    y_all = pd.concat([y_train, y_valid, y_test])
    folds = [{"train": X_train.index, "valid": X_valid.index, "test": X_test.index}]

    tags = {
        "graph_name": "xgboost_full",
        "model_cls": "XGBoost",
        "n_trials": n_trials,
        "sampler": "TPESampler",
        **(extra_tags or {}),
    }

    def suggest_xgb_params(trial):
        base_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 2000,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        return suggest_from_search_space(
            trial,
            xgb_search_space if xgb_search_space is not None else XGB_SEARCH_SPACE,
            base_params=base_params,
        )

    name_tags = extra_tags or {}

    def _study_name_for(data_seed):
        return _build_study_name(
            name_tags.get("generator_name"),
            f"cfg{name_tags['config_hash']}"
            if name_tags.get("config_hash") is not None
            else None,
            "XGBoost",
            "xgboost_full",
            f"seed{data_seed}" if data_seed is not None else None,
            f"train{name_tags['train_size']}"
            if name_tags.get("train_size") is not None
            else None,
            f"cov{name_tags['cov']}" if name_tags.get("cov") is not None else None,
            f"noise{name_tags['noise_level']}"
            if name_tags.get("noise_level") is not None
            else None,
            f"nedges{name_tags['n_edges']}"
            if name_tags.get("n_edges") is not None
            else None,
            f"thr{name_tags['threshold']}"
            if name_tags.get("threshold") is not None
            else None,
        )

    data_seed = name_tags.get("data_seed")
    study_name = _study_name_for(data_seed)

    enqueue_params = None
    if top_k is not None and main_data_seed is not None and data_seed != main_data_seed:
        main_study_name = _study_name_for(main_data_seed)
        try:
            main_study = optuna.load_study(
                study_name=main_study_name, storage=optuna_storage
            )
        except KeyError:
            main_study = None

        if main_study is not None:
            completed = main_study.get_trials(
                deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
            )
            completed.sort(key=lambda t: t.value, reverse=True)
            top = completed[:top_k]
            if top:
                enqueue_params = [dict(t.params) for t in top]

        if enqueue_params is None:
            print(
                f"[top-k] brak ukończonych triali w studium głównym "
                f"{main_study_name!r}; pełny tuning XGBoost dla seed={data_seed}"
            )

    study = _run_xgboost_folded_study(
        X=X_all,
        y=y_all,
        folds=folds,
        study_name=study_name,
        n_trials=n_trials,
        seed=seed,
        optuna_storage=optuna_storage,
        n_startup_trials=n_startup_trials,
        experiment_name=experiment_name,
        run_tags=tags,
        suggest_params_func=suggest_xgb_params,
        xgb_early_stopping_rounds=20,
        preprocessing_pipeline=preprocessing_pipeline,
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        enqueue_params=enqueue_params,
    )

    return _xgb_result_from_study(study)


def run_xgboost_fixed_folds_study(
    *,
    X,
    y,
    folds,
    n_trials,
    seed,
    optuna_storage,
    experiment_name,
    data_tags,
    mlflow_tracking_uri=None,
    n_startup_trials=10,
    xgb_max_rounds=2000,
    xgb_early_stopping_rounds=15,
    xgb_preprocessing_pipeline=None,
    xgb_search_space=None,
    return_study=False,
):
    """Run XGBoost baseline on predefined folds with MLflow logging."""

    graph_name = "xgboost_full"
    study_name = _build_study_name(experiment_name, "XGBoost", graph_name)
    run_group = _build_study_name("XGBoost", graph_name)

    run_tags = {
        "graph_name": graph_name,
        "model_cls": "XGBoost",
        "run_group": run_group,
        "n_trials": n_trials,
        "search_strategy": "tpe",
        "experiment_name": experiment_name,
        **(data_tags or {}),
    }

    def suggest_xgb_params(trial):
        base_params = {
            "objective": "binary:logistic",
            "eval_metric": ["aucpr", "logloss", "error", "auc"],
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": -1,
            "n_estimators": xgb_max_rounds,
        }
        if xgb_search_space is not None:
            return suggest_from_search_space(
                trial, xgb_search_space, base_params=base_params
            )
        return {
            **base_params,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    study = _run_xgboost_folded_study(
        X=X,
        y=y,
        folds=folds,
        study_name=study_name,
        n_trials=n_trials,
        seed=seed,
        optuna_storage=optuna_storage,
        n_startup_trials=n_startup_trials,
        experiment_name=experiment_name,
        run_tags=run_tags,
        suggest_params_func=suggest_xgb_params,
        xgb_early_stopping_rounds=xgb_early_stopping_rounds,
        preprocessing_pipeline=xgb_preprocessing_pipeline,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    if return_study:
        return study
    return _xgb_result_from_study(study)


__all__ = [
    "binary_metrics",
    "run_xgboost_fixed_folds_study",
    "tune_xgboost",
]
