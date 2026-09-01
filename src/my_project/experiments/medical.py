"""Pipelines and shared Optuna search spaces for MEDICAL tuning.

Used by ``notebooks/gnn/tuning.ipynb``. Medical preprocessing stays local to this
module, while search spaces come from ``my_project.experiments.search_space`` and
are shared with the synthetic experiment.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from my_project.data.pipelines import (
    discretizer_transformer,
    encoder_transformer,
    high_missing_n_bins,
    high_missing_threshold,
    preprocessor_transformer,
)
from my_project.data.preprocessing import CategoryColumnMerger, HighMissingDiscretizer
from my_project.experiments.search_space import (
    GNN_SEARCH_SPACE,
    MLP_SEARCH_SPACE,
    XGB_SEARCH_SPACE,
)

troponin_category_merger = CategoryColumnMerger(
    first_column="tro",
    second_column="trot",
    output_column="troponina",
    first_prefix="TnI",
    second_prefix="TnT",
    include_both=False,
)

medical_preprocessing_pipeline = Pipeline(
    [
        (
            "high_missing_discretizer",
            HighMissingDiscretizer(
                threshold=high_missing_threshold, n_bins=high_missing_n_bins
            ),
        ),
        ("troponin_category_merger", troponin_category_merger),
        ("preprocessing", preprocessor_transformer),
    ]
)

medical_preprocessing_pipeline_nn = Pipeline(
    [
        (
            "high_missing_discretizer",
            HighMissingDiscretizer(
                threshold=high_missing_threshold, n_bins=high_missing_n_bins
            ),
        ),
        ("troponin_category_merger", troponin_category_merger),
        ("preprocessing", preprocessor_transformer),
        ("encoder", encoder_transformer),
    ]
)

medical_discretization_pipeline = Pipeline(
    [
        (
            "high_missing_discretizer",
            HighMissingDiscretizer(
                threshold=high_missing_threshold, n_bins=high_missing_n_bins
            ),
        ),
        ("troponin_category_merger", troponin_category_merger),
        ("preprocessing", preprocessor_transformer),
        ("discretizer", discretizer_transformer),
    ]
)

# --- Pipelines ---------------------------------------------------------------
# NN input (impute/encode -> ordinal codes for embeddings), XGBoost input
# (impute/encode, no ordinal shift needed) and graph input (discretize numerics
# so the interaction matrix sees a discrete representation).
NN_PREPROCESSING_PIPELINE = medical_preprocessing_pipeline_nn
XGBOOST_PREPROCESSING_PIPELINE = medical_preprocessing_pipeline
GRAPH_PREPROCESSING_PIPELINE = medical_discretization_pipeline

__all__ = [
    "NN_PREPROCESSING_PIPELINE",
    "XGBOOST_PREPROCESSING_PIPELINE",
    "GRAPH_PREPROCESSING_PIPELINE",
    "GNN_SEARCH_SPACE",
    "MLP_SEARCH_SPACE",
    "XGB_SEARCH_SPACE",
]
