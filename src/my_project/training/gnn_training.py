"""GNN-centric tuning and fixed-fold orchestration helpers."""

import copy

import networkx as nx
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from my_project.gnn.models import MyMLP
from my_project.training import training_helpers, xgboost_training
from my_project.training.optuna_runner import MetricsProgressCallback
from my_project.training.tuning import suggest_from_search_space

_build_study_name = training_helpers._build_study_name
_extract_graph_permute_seed = training_helpers._extract_graph_permute_seed
tune_xgboost = xgboost_training.tune_xgboost


def run_gnn_tuning(
    graph,
    graph_name,
    model_cls,
    suggest_params_func,
    X_all,
    y_all,
    folds,
    *,
    optuna_storage,
    search_space,
    technical_settings,
    direction,
    n_trials,
    sampler_seed,
    standardize=True,
    keep_on_gpu=True,
    n_startup_trials=10,
    conv_layer=None,
    sampler=None,
    extra_tags=None,
    experiment_name="test_synthetic_tuning",
    pruning_mode="epoch",
    top_k=None,
    main_data_seed=None,
    permuted_top_k=None,
    preprocessing_pipeline=None,
    show_metrics_progress=False,
    pruning_warmup_steps=None,
):
    from .run_optuna_study import build_cv_datamodules, run_optuna_study_for_gnn

    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            seed=sampler_seed,
            multivariate=True,
            n_startup_trials=n_startup_trials,
        )

    if isinstance(graph, (list, tuple)):
        graphs_per_fold = list(graph)
        if not graphs_per_fold:
            raise ValueError("`graph` list cannot be empty")
    else:
        graphs_per_fold = [graph]

    if len(graphs_per_fold) == 1 and len(folds) > 1:
        graphs_per_fold = graphs_per_fold * len(folds)
    elif len(graphs_per_fold) != len(folds):
        raise ValueError(
            f"graphs/folds length mismatch: {len(graphs_per_fold)} graphs for {len(folds)} folds"
        )

    graph_template = graphs_per_fold[0]
    columns = list(graph_template.nodes())

    if preprocessing_pipeline is None and standardize:
        preprocessing_pipeline = Pipeline([("scaler", StandardScaler())]).set_output(
            transform="pandas"
        )

    graph_keep_on_gpu = keep_on_gpu and graph_name != "fully_connected"

    cv_folds = build_cv_datamodules(
        X=X_all[columns],
        y=y_all,
        folds=folds,
        graphs=graphs_per_fold,
        preprocessing_pipeline=preprocessing_pipeline,
        num_workers=0,
        keep_on_gpu=graph_keep_on_gpu,
    )

    base_params = {}
    if conv_layer is not None:
        base_params["conv_layer"] = conv_layer

    name_tags = extra_tags or {}

    def _study_name_for(seed, graph=None):
        graph = graph_name if graph is None else graph
        # experiment_name namespaces the study, so runs with the same
        # generator/data_type but a different experiment (e.g. a changed feature
        # set) don't collide into one Optuna study. Drop the generator token
        # when it just duplicates experiment_name (synthetic sweeps set them equal).
        generator_name = name_tags.get("generator_name")
        generator_token = generator_name if generator_name != experiment_name else None
        return _build_study_name(
            experiment_name,
            generator_token,
            f"cfg{name_tags['config_hash']}"
            if name_tags.get("config_hash") is not None
            else None,
            model_cls.__name__,
            conv_layer,
            graph,
            f"seed{seed}" if seed is not None else None,
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

    def _top_params_from(source_study_name, k):
        try:
            source_study = optuna.load_study(
                study_name=source_study_name, storage=optuna_storage
            )
        except KeyError:
            return None
        completed = source_study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
        )
        completed.sort(key=lambda t: t.value, reverse=(direction == "maximize"))
        top = completed[:k] if k is not None else completed
        return [dict(t.params) for t in top] or None

    enqueue_params = None
    study_storage = optuna_storage
    # In "fold" mode the pruner steps over fold indices (0..n_folds-1), so a
    # warmup of 15 would exceed the number of folds and never prune. When
    # pruning_warmup_steps is given it wins; otherwise fall back to a
    # mode-appropriate default (below the fold count for fold pruning).
    if pruning_warmup_steps is not None:
        n_warmup_steps = pruning_warmup_steps
    elif pruning_mode == "fold":
        n_warmup_steps = max(1, len(folds) // 2)
    else:
        n_warmup_steps = 15
    pruner = optuna.pruners.PercentilePruner(
        percentile=50, n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps
    )
    if top_k is not None and main_data_seed is not None and data_seed != main_data_seed:
        source_name = _study_name_for(main_data_seed)
        enqueue_params = _top_params_from(source_name, top_k)
        if enqueue_params is not None:
            pruner = optuna.pruners.NopPruner()
        else:
            print(
                f"[top-k] brak ukończonych triali w studium głównym "
                f"{source_name!r}; pełny tuning dla seed={data_seed}"
            )

    # Permuted graphs reuse the best hyperparameters from their source
    # (non-permuted) graph study instead of a fresh search: enqueue the source's
    # top-k combos and skip pruning. Requires the source study to have finished
    # first (guaranteed by the graph ordering in the sweep).
    permute_seed = _extract_graph_permute_seed(graph_name)
    if (
        permuted_top_k is not None
        and permute_seed is not None
        and enqueue_params is None
    ):
        source_graph = graph_name[: graph_name.rfind("_permuted_")]
        source_name = _study_name_for(data_seed, graph=source_graph)
        enqueue_params = _top_params_from(source_name, permuted_top_k)
        if enqueue_params is not None:
            pruner = optuna.pruners.NopPruner()
        else:
            print(
                f"[permuted top-k] brak ukończonych triali w studium źródłowym "
                f"{source_name!r}; pełny tuning dla {graph_name}"
            )

    ts = copy.deepcopy(technical_settings)
    ts["logger_kwargs"]["experiment_name"] = experiment_name
    ts["keep_on_gpu"] = graph_keep_on_gpu
    ts["tags"] = {
        "graph_name": graph_name,
        "model_cls": model_cls.__name__,
        "conv_layer": conv_layer,
        "n_trials": n_trials,
        "sampler": type(sampler).__name__,
        **(extra_tags or {}),
    }

    # Fresh progress bar per study so each graph/model gets its own bar/total.
    callbacks = (
        [MetricsProgressCallback(desc=study_name, total=n_trials)]
        if show_metrics_progress
        else None
    )

    study = run_optuna_study_for_gnn(
        model_cls=model_cls,
        cv_folds=cv_folds,
        study_name=study_name,
        storage_url=study_storage,
        search_space=search_space,
        n_trials=n_trials,
        technical_settings=ts,
        base_params=base_params,
        direction=direction,
        suggest_params_func=suggest_params_func,
        sampler=sampler,
        pruner=pruner,
        enqueue_params=enqueue_params,
        pruning_mode=pruning_mode,
        callbacks=callbacks,
    )

    best = study.best_trial
    return {
        "model_cls": model_cls.__name__,
        "graph_name": graph_name,
        "best_value": study.best_value,
        **best.user_attrs.get("mean_metrics", {}),
    }


def run_config(
    graphs,
    edges_info,
    splits,
    *,
    base_tags,
    experiment_name,
    run_xgb,
    run_mlp,
    model_classes,
    conv_layer,
    search_space,
    technical_settings,
    direction,
    n_trials,
    sampler_seed,
    data_seed,
    optuna_storage,
    standardize=True,
    keep_on_gpu=True,
    n_startup_trials=10,
    top_k=None,
    main_data_seed=None,
    preprocessing_pipeline=None,
    xgb_search_space=None,
    cv_splits=1,
):
    """Train graph-dependent GNNs over `graphs`, plus optional graph-independent baselines."""
    X_train, X_valid, X_test, y_train, y_valid, y_test = splits
    X_all = pd.concat([X_train, X_valid, X_test])
    y_all = pd.concat([y_train, y_valid, y_test])
    if cv_splits and cv_splits > 1:
        X_trainval = pd.concat([X_train, X_valid])
        y_trainval = pd.concat([y_train, y_valid])
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=data_seed)
        folds = [
            {
                "train": X_trainval.index[train_idx],
                "valid": X_trainval.index[valid_idx],
                "test": X_test.index,
            }
            for train_idx, valid_idx in skf.split(X_trainval, y_trainval)
        ]
    else:
        folds = [{"train": X_train.index, "valid": X_valid.index, "test": X_test.index}]

    results = []

    if run_xgb:
        results.append(
            tune_xgboost(
                X_train=X_train,
                X_valid=X_valid,
                X_test=X_test,
                y_train=y_train,
                y_valid=y_valid,
                y_test=y_test,
                n_trials=n_trials,
                seed=data_seed,
                optuna_storage=optuna_storage,
                n_startup_trials=n_startup_trials,
                experiment_name=experiment_name,
                extra_tags=base_tags,
                top_k=top_k,
                main_data_seed=main_data_seed,
                preprocessing_pipeline=preprocessing_pipeline,
                xgb_search_space=xgb_search_space,
            )
        )

    if run_mlp and MyMLP in model_classes:
        empty_graph = nx.empty_graph(next(iter(graphs.values())))
        results.append(
            run_gnn_tuning(
                graph=empty_graph,
                graph_name="empty",
                model_cls=MyMLP,
                suggest_params_func=suggest_from_search_space,
                X_all=X_all,
                y_all=y_all,
                folds=folds,
                optuna_storage=optuna_storage,
                standardize=standardize,
                keep_on_gpu=keep_on_gpu,
                n_startup_trials=n_startup_trials,
                conv_layer=None,
                search_space=search_space,
                technical_settings=technical_settings,
                direction=direction,
                n_trials=n_trials,
                sampler_seed=sampler_seed,
                experiment_name=experiment_name,
                extra_tags={**base_tags, "n_features": empty_graph.number_of_nodes()},
                top_k=top_k,
                main_data_seed=main_data_seed,
                preprocessing_pipeline=preprocessing_pipeline,
            )
        )

    gnn_models = [m for m in model_classes if m is not MyMLP]
    for model_cls in gnn_models:
        for graph_name, graph_train in graphs.items():
            stats = edges_info.get(graph_name, {})
            graph_permute_seed = _extract_graph_permute_seed(graph_name)
            extra_tags = {
                **base_tags,
                "n_features": graph_train.number_of_nodes(),
                "is_permuted": graph_permute_seed is not None,
                "graph_permute_seed": graph_permute_seed,
                "graph_n_edges": graph_train.number_of_edges(),
                "graph_precision": stats.get("precision"),
                "graph_recall": stats.get("recall"),
                "graph_true_edges_found": stats.get("true_edges_in_set"),
                "graph_true_edges_total": stats.get("true_edges_len"),
            }
            results.append(
                run_gnn_tuning(
                    graph=graph_train,
                    graph_name=graph_name,
                    model_cls=model_cls,
                    suggest_params_func=suggest_from_search_space,
                    X_all=X_all,
                    y_all=y_all,
                    folds=folds,
                    optuna_storage=optuna_storage,
                    standardize=standardize,
                    keep_on_gpu=keep_on_gpu,
                    n_startup_trials=n_startup_trials,
                    conv_layer=conv_layer,
                    search_space=search_space,
                    technical_settings=technical_settings,
                    direction=direction,
                    n_trials=n_trials,
                    sampler_seed=sampler_seed,
                    experiment_name=experiment_name,
                    extra_tags=extra_tags,
                    top_k=top_k,
                    main_data_seed=main_data_seed,
                    preprocessing_pipeline=preprocessing_pipeline,
                )
            )

    return results


def run_fixed_folds_sweep(
    *,
    X,
    y,
    folds,
    graphs_by_name,
    model_classes=None,
    conv_layer=None,
    search_space=None,
    model_configs=None,
    technical_settings,
    direction,
    n_trials,
    sampler_seed,
    optuna_storage,
    experiment_name,
    base_tags,
    run_mlp=True,
    mlp_reference_graph="empty",
    standardize=True,
    keep_on_gpu=True,
    n_startup_trials=10,
    pruning_mode="epoch",
    top_k=None,
    main_data_seed=None,
    permuted_top_k=None,
    preprocessing_pipeline=None,
    pruning_warmup_steps=None,
):
    """Run tuning on a fixed set of precomputed folds and per-fold graphs."""
    results = []

    if model_configs is not None:
        expanded_configs = []
        for cfg in model_configs:
            convs = cfg.get("convs", [None])
            for conv in convs:
                expanded_configs.append(
                    {
                        "model_cls": cfg["model_cls"],
                        "search_space": cfg["search_space"],
                        "conv_layer": conv,
                    }
                )
    else:
        if model_classes is None or search_space is None:
            raise ValueError(
                "Provide either model_configs or (model_classes and search_space)"
            )
        expanded_configs = [
            {
                "model_cls": model_cls,
                "search_space": search_space,
                "conv_layer": None if model_cls is MyMLP else conv_layer,
            }
            for model_cls in model_classes
        ]

    mlp_configs = [cfg for cfg in expanded_configs if cfg["model_cls"] is MyMLP]
    gnn_configs = [cfg for cfg in expanded_configs if cfg["model_cls"] is not MyMLP]
    for cfg in gnn_configs:
        model_cls = cfg["model_cls"]
        for graph_name, graph_folds in graphs_by_name.items():
            first_graph = (
                graph_folds[0]
                if isinstance(graph_folds, (list, tuple))
                else graph_folds
            )
            graph_permute_seed = _extract_graph_permute_seed(graph_name)
            results.append(
                run_gnn_tuning(
                    graph=graph_folds,
                    graph_name=graph_name,
                    model_cls=model_cls,
                    suggest_params_func=suggest_from_search_space,
                    X_all=X,
                    y_all=y,
                    folds=folds,
                    optuna_storage=optuna_storage,
                    search_space=cfg["search_space"],
                    technical_settings=technical_settings,
                    direction=direction,
                    n_trials=n_trials,
                    sampler_seed=sampler_seed,
                    standardize=standardize,
                    keep_on_gpu=keep_on_gpu,
                    n_startup_trials=n_startup_trials,
                    pruning_mode=pruning_mode,
                    conv_layer=cfg["conv_layer"],
                    experiment_name=experiment_name,
                    extra_tags={
                        **base_tags,
                        "n_features": first_graph.number_of_nodes(),
                        "graph_n_edges": first_graph.number_of_edges(),
                        "is_permuted": graph_permute_seed is not None,
                        "graph_permute_seed": graph_permute_seed,
                    },
                    top_k=top_k,
                    main_data_seed=main_data_seed,
                    permuted_top_k=permuted_top_k,
                    preprocessing_pipeline=preprocessing_pipeline,
                    pruning_warmup_steps=pruning_warmup_steps,
                )
            )

    # MLP after the GNNs so it trains last in the sweep.
    if run_mlp and mlp_configs:
        if mlp_reference_graph not in graphs_by_name:
            raise KeyError(
                f"mlp_reference_graph={mlp_reference_graph!r} not found in graphs_by_name"
            )
        ref_graphs = graphs_by_name[mlp_reference_graph]
        ref_graph = (
            ref_graphs[0] if isinstance(ref_graphs, (list, tuple)) else ref_graphs
        )
        mlp_cfg = mlp_configs[0]
        results.append(
            run_gnn_tuning(
                graph=ref_graphs,
                graph_name=mlp_reference_graph,
                model_cls=MyMLP,
                suggest_params_func=suggest_from_search_space,
                X_all=X,
                y_all=y,
                folds=folds,
                optuna_storage=optuna_storage,
                search_space=mlp_cfg["search_space"],
                technical_settings=technical_settings,
                direction=direction,
                n_trials=n_trials,
                sampler_seed=sampler_seed,
                standardize=standardize,
                keep_on_gpu=keep_on_gpu,
                n_startup_trials=n_startup_trials,
                pruning_mode=pruning_mode,
                conv_layer=mlp_cfg["conv_layer"],
                experiment_name=experiment_name,
                extra_tags={
                    **base_tags,
                    "n_features": ref_graph.number_of_nodes(),
                },
                top_k=top_k,
                main_data_seed=main_data_seed,
                preprocessing_pipeline=preprocessing_pipeline,
                pruning_warmup_steps=pruning_warmup_steps,
            )
        )

    return results


__all__ = [
    "run_config",
    "run_fixed_folds_sweep",
    "run_gnn_tuning",
]
