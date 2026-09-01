"""Pipelines and shared Optuna search spaces for the SYNTHETIC sweep notebook.

Used by ``notebooks/gnn/synthetic_data_clean.ipynb`` (higgs / pairwise / xor / f
/ madelon / breast_cancer). Search spaces come from
``my_project.experiments.search_space`` and are shared with the medical experiment.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

from my_project.data.pipelines import shift_to_non_negative
from my_project.experiments.search_space import (
    GNN_SEARCH_SPACE,
    MLP_SEARCH_SPACE,
    XGB_SEARCH_SPACE,
)


def _to_ordered_category(data):
    # ordered so downstream feature_n_classes (Series.max on the codes) doesn't
    # raise on an unordered Categorical.
    return data.apply(lambda column: column.astype("category").cat.as_ordered())


_HIGGS_SKEWED_VARS = [
    "lepton_pt",
    "missing_energy_magnitude",
    "jet1_pt",
    "jet2_pt",
    "jet3_pt",
    "jet4_pt",
]
_HIGGS_CATEGORICAL_VARS = [
    "jet1_btag",
    "jet2_btag",
    "jet3_btag",
    "jet4_btag",
]

higgs_pipeline = Pipeline(
    [
        (
            "selector",
            ColumnTransformer(
                transformers=[
                    (
                        "skewed",
                        PowerTransformer(method="yeo-johnson", standardize=True),
                        _HIGGS_SKEWED_VARS,
                    ),
                    (
                        "categorical",
                        Pipeline(
                            [
                                (
                                    "ordinal_encoder",
                                    OrdinalEncoder(
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1,
                                    ),
                                ),
                                (
                                    "shift_to_non_negative",
                                    FunctionTransformer(shift_to_non_negative),
                                ),
                                (
                                    "to_category_dtype",
                                    FunctionTransformer(_to_ordered_category),
                                ),
                            ]
                        ),
                        _HIGGS_CATEGORICAL_VARS,
                    ),
                ],
                # Non-skewed columns get standardized; keep original feature names so
                # graph nodes / true edges stay aligned with the model's columns.
                remainder=StandardScaler(),
                verbose_feature_names_out=False,
            ),
        ),
    ]
).set_output(transform="pandas")

higgs_pipeline_discretized = Pipeline(
    [
        ("higgs_pipeline", higgs_pipeline),
        # Bin only the continuous features (ordinal, not the default one-hot). The
        # b-tag flags are already discrete (3 native levels), so pass them through
        # instead of re-binning them into fewer quantile bins.
        (
            "discretizer",
            ColumnTransformer(
                [
                    (
                        "num",
                        KBinsDiscretizer(
                            n_bins=5,
                            encode="ordinal",
                            strategy="quantile",
                            quantile_method="averaged_inverted_cdf",
                        ),
                        make_column_selector(dtype_include="number"),
                    )
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ),
        ),
    ]
).set_output(transform="pandas")

__all__ = [
    "GNN_SEARCH_SPACE",
    "MLP_SEARCH_SPACE",
    "XGB_SEARCH_SPACE",
    "higgs_pipeline",
    "higgs_pipeline_discretized",
]
