"""Grid-sweep driver for synthetic/real generator experiments.

This module now focuses on the actual sweep loop over data configs
(noise/cov/train_size/data_seed). Model-level training orchestration lives in
:mod:`my_project.training.gnn_training` and :mod:`my_project.training.xgboost_training`.
"""

import copy

from sklearn import config_context

from my_project.data.utils import prepare_data_fixed_test
from my_project.graphs import (
    build_base_graphs,
    build_ii_graphs,
    build_node_permutation,
    build_true_graph,
    graph_stats,
)
from my_project.information_theory import compute_info
from my_project.training import gnn_training, xgboost_training

run_config = gnn_training.run_config
run_fixed_folds_sweep = gnn_training.run_fixed_folds_sweep
run_gnn_tuning = gnn_training.run_gnn_tuning
binary_metrics = xgboost_training.binary_metrics
run_xgboost_fixed_folds_study = xgboost_training.run_xgboost_fixed_folds_study
tune_xgboost = xgboost_training.tune_xgboost


def build_generator_name(generator_kind, *, generator_kwargs=None, f_function_id=None):
    """Return the canonical generator/experiment name used across sweeps.

    Most generators use their kind as-is. Friedman generators use the
    ``f{function_id}`` convention, where ``function_id`` can be passed explicitly
    or inferred from ``generator_kwargs['function_id']``.
    """
    if generator_kind == "f":
        if f_function_id is None and isinstance(generator_kwargs, dict):
            f_function_id = generator_kwargs.get("function_id")
        if f_function_id is None:
            f_function_id = 0
        return f"f{f_function_id}"
    return str(generator_kind)


def run_sweep(
    *,
    generate,
    permute_seeds,
    noise_grid,
    cov_grid,
    train_size_grid,
    data_seeds,
    main_data_seed,
    edge_mode,
    threshold_grid,
    n_edges_grid,
    model_configs=None,
    model_classes=None,
    conv_layer=None,
    search_space=None,
    technical_settings,
    direction,
    n_trials,
    n_bins,
    test_size,
    valid_size,
    top_k,
    experiment_name,
    generator_name,
    config_hash,
    generator_kwargs,
    optuna_storage,
    standardize=True,
    keep_on_gpu=True,
    n_startup_trials=10,
    interactions_names=None,
    feature_selection=None,
    preprocessing_pipeline=None,
    graph_preprocessing_pipeline=None,
    xgb_search_space=None,
    cv_splits=1,
    test_seed=99991,
    permuted_top_k=None,
):
    """Full grid sweep over (noise, cov, train_size, data_seed).

    The main seed is run first for every (noise, cov, train_size) cell so its
    full study exists before the other seeds reuse its top-k combinations.

    ``test_seed`` is the base for held-out test seeds. Each ``data_seed`` gets
    its own test seed, while all train sizes for that data seed reuse the same
    test set. Keep the base nonzero so test and train draws use different seeds
    (see ``prepare_data_fixed_test``).
    """
    results = []
    # `None` means "no noise sweep": run one pass with zero noise.
    noise_values = [0.0] if noise_grid is None else list(noise_grid)
    sweep_seeds = list(dict.fromkeys([main_data_seed, *data_seeds]))
    for noise_level in noise_values:
        for cov in cov_grid:
            for train_size in train_size_grid:
                for data_seed in sweep_seeds:
                    n_samples = train_size + test_size + valid_size
                    # Each data seed has its own test set, reused across train sizes.
                    # Train/valid samples remain independent for every train size.
                    true_edges_s, splits = prepare_data_fixed_test(
                        generate,
                        train_size=train_size,
                        data_seed=data_seed,
                        test_size=test_size,
                        valid_size=valid_size,
                        noise_level=noise_level,
                        feature_selection=feature_selection,
                        test_seed=test_seed + data_seed,
                        cov=cov,
                    )
                    X_train_s, y_train_s = splits[0], splits[3]
                    positive_rate = float(y_train_s.mean())
                    # Datasets with mixed/categorical features need discrete
                    # inputs for the interaction matrix: fit a graph preprocessing
                    # pipeline (e.g. discretization_pipeline) on train only and
                    # feed its output to compute_info. Otherwise bin via n_bins.
                    if graph_preprocessing_pipeline is not None:
                        graph_pipe = copy.deepcopy(graph_preprocessing_pipeline)
                        with config_context(transform_output="pandas"):
                            X_graph_s = graph_pipe.fit_transform(X_train_s, y_train_s)
                        ii_s = compute_info(X_graph_s, y_train_s, n_bins=None)
                    else:
                        ii_s = compute_info(X_train_s, y_train_s, n_bins=n_bins)
                    reference_columns = list(ii_s.columns)
                    permutation_mappings = {
                        seed: build_node_permutation(reference_columns, seed)
                        for seed in permute_seeds
                    }
                    graph_true_s = build_true_graph(ii_s, true_edges_s)

                    common_tags = {
                        "generator_name": generator_name,
                        "config_hash": config_hash,
                        "data_type": "synthetic",
                        "n_samples": n_samples,
                        "train_size": train_size,
                        "valid_size": valid_size,
                        "test_size": test_size,
                        "noise_level": noise_level,
                        "data_seed": data_seed,
                        "positive_rate": positive_rate,
                        "n_bins": n_bins,
                        "cov": cov,
                        # All generator params (interactions logged as readable names).
                        **generator_kwargs,
                    }
                    # `interactions` is only meaningful for generators that define a
                    # ground-truth interaction set (e.g. pairwise); real datasets like
                    # HIGGS/Madelon have none, so don't log a stale INTERACTIONS_NAMES dict.
                    if (
                        "interactions" in generator_kwargs
                        and interactions_names is not None
                    ):
                        common_tags["interactions"] = str(interactions_names)

                    # Threshold-independent graphs (oracle/empty/fully_connected) plus the
                    # graph-independent baselines (MLP, XGBoost): trained once per dataset.
                    base_graphs = build_base_graphs(graph_true_s)
                    base_info = graph_stats(base_graphs, true_edges_s)
                    print(
                        f"\\n=== data_seed={data_seed} | n_samples={n_samples} "
                        f"| cov={cov} | noise={noise_level} | base ==="
                    )

                    results.extend(
                        run_config(
                            base_graphs,
                            base_info,
                            splits,
                            base_tags={**common_tags, "threshold": None},
                            experiment_name=experiment_name,
                            run_xgb=True,
                            run_mlp=True,
                            model_configs=model_configs,
                            model_classes=model_classes,
                            conv_layer=conv_layer,
                            search_space=search_space,
                            technical_settings=technical_settings,
                            direction=direction,
                            n_trials=n_trials,
                            sampler_seed=data_seed,
                            data_seed=data_seed,
                            optuna_storage=optuna_storage,
                            standardize=standardize,
                            keep_on_gpu=keep_on_gpu,
                            n_startup_trials=n_startup_trials,
                            top_k=top_k,
                            main_data_seed=main_data_seed,
                            permuted_top_k=permuted_top_k,
                            preprocessing_pipeline=preprocessing_pipeline,
                            xgb_search_space=xgb_search_space,
                            cv_splits=cv_splits,
                        )
                    )

                    edge_grid = (
                        threshold_grid if edge_mode == "threshold" else n_edges_grid
                    )
                    for edge_value in edge_grid:
                        if edge_mode == "threshold":
                            ii_graphs = build_ii_graphs(
                                ii_s,
                                permute_seeds,
                                threshold=edge_value,
                                n_bins=n_bins,
                                permutation_mappings=permutation_mappings,
                            )
                            edge_tags = {"threshold": edge_value}
                        else:
                            ii_graphs = build_ii_graphs(
                                ii_s,
                                permute_seeds,
                                n_edges=edge_value,
                                permutation_mappings=permutation_mappings,
                            )
                            edge_tags = {"n_edges": edge_value}
                        ii_info = graph_stats(ii_graphs, true_edges_s)
                        print(
                            f"\\n=== data_seed={data_seed} | n_samples={n_samples} "
                            f"| cov={cov} | noise={noise_level} | {edge_mode}={edge_value} ==="
                        )
                        results.extend(
                            run_config(
                                ii_graphs,
                                ii_info,
                                splits,
                                base_tags={**common_tags, **edge_tags},
                                experiment_name=experiment_name,
                                run_xgb=False,
                                run_mlp=False,
                                model_configs=model_configs,
                                model_classes=model_classes,
                                conv_layer=conv_layer,
                                search_space=search_space,
                                technical_settings=technical_settings,
                                direction=direction,
                                n_trials=n_trials,
                                sampler_seed=data_seed,
                                data_seed=data_seed,
                                optuna_storage=optuna_storage,
                                standardize=standardize,
                                keep_on_gpu=keep_on_gpu,
                                n_startup_trials=n_startup_trials,
                                top_k=top_k,
                                main_data_seed=main_data_seed,
                                permuted_top_k=permuted_top_k,
                                preprocessing_pipeline=preprocessing_pipeline,
                                cv_splits=cv_splits,
                            )
                        )
    return results


__all__ = [
    "binary_metrics",
    "build_generator_name",
    "run_config",
    "run_fixed_folds_sweep",
    "run_gnn_tuning",
    "run_sweep",
    "run_xgboost_fixed_folds_study",
    "tune_xgboost",
]
