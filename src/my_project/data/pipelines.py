from my_project.data.preprocessing import (
    BooleanMissingEncoder,
    HighMissingDiscretizer,
    RareCategoryTransformer,
    ThresholdImputer,
)
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OrdinalEncoder,
    StandardScaler,
)


def shift_to_non_negative(x):
    return x + 1


high_missing_threshold = 0.10
high_missing_n_bins = 3

dtypes_dict = {
    "number": ["number"],
    "categorical": ["object", "category", "string"],
    "boolean": ["bool", "boolean"],
}

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        (
            "threshold_imputer",
            ThresholdImputer(threshold=0.02, strategy="most_frequent"),
        ),
        (
            "rare_category",
            RareCategoryTransformer(
                min_frequency=0.01, replace_with="OTHER", include_nan=True
            ),
        ),
    ]
)

boolean_pipeline = Pipeline(
    [
        # Low-missing booleans: impute with the mode (keeps the nullable
        # ``boolean`` dtype, so downstream dtype routing still treats them as
        # categorical instead of continuous numbers).
        (
            "threshold_imputer",
            ThresholdImputer(
                threshold=high_missing_threshold, strategy="most_frequent"
            ),
        ),
        # High-missing booleans: keep missingness as an explicit "Missing" level
        # instead of imputing, mirroring the numeric HighMissingDiscretizer.
        (
            "high_missing_encoder",
            BooleanMissingEncoder(threshold=high_missing_threshold),
        ),
    ]
)

preprocessor_transformer = ColumnTransformer(
    [
        (
            "cat",
            categorical_pipeline,
            selector(dtype_include=dtypes_dict["categorical"]),
        ),
        ("bool", boolean_pipeline, selector(dtype_include=dtypes_dict["boolean"])),
        ("num", numeric_pipeline, selector(dtype_include=dtypes_dict["number"])),
    ],
    verbose_feature_names_out=False,
)

ordinal_start_from_1_encoder = Pipeline(
    [
        # shift categories: missing/unknown (-1) -> 0, known classes 0..K-1 -> 1..K.
        (
            "ordinalencoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                encoded_missing_value=-1,
                unknown_value=-1,
            ),
        ),
        (
            "shift_to_non_negative",
            FunctionTransformer(shift_to_non_negative, feature_names_out="one-to-one"),
        ),
    ],
)

encoder_transformer = ColumnTransformer(
    [
        (
            "cat",
            ordinal_start_from_1_encoder,
            selector(dtype_include=dtypes_dict["categorical"]),
        ),
        (
            "bool",
            ordinal_start_from_1_encoder,
            selector(dtype_include=dtypes_dict["boolean"]),
        ),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)


preprocessing_pipeline = Pipeline(
    [
        (
            "high_missing_discretizer",
            HighMissingDiscretizer(
                threshold=high_missing_threshold, n_bins=high_missing_n_bins
            ),
        ),
        ("preprocessing", preprocessor_transformer),
    ]
)

preprocessing_pipeline_nn = Pipeline(
    [
        (
            "high_missing_discretizer",
            HighMissingDiscretizer(
                threshold=high_missing_threshold, n_bins=high_missing_n_bins
            ),
        ),
        ("preprocessing", preprocessor_transformer),
        ("encoder", encoder_transformer),
    ]
)

discretizer_transformer = ColumnTransformer(
    [
        (
            "num",
            KBinsDiscretizer(
                n_bins=3,
                encode="ordinal",
                strategy="quantile",
                quantile_method="averaged_inverted_cdf",
            ),
            selector(dtype_include=dtypes_dict["number"]),
        ),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

discretization_pipeline = Pipeline(
    [
        (
            "high_missing_discretizer",
            HighMissingDiscretizer(
                threshold=high_missing_threshold, n_bins=high_missing_n_bins
            ),
        ),
        ("preprocessing", preprocessor_transformer),
        ("discretizer", discretizer_transformer),
    ]
)
