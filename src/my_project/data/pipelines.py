from my_project.data.preprocessing import RareCategoryTransformer, ThresholdImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler,
)


def shift_to_non_negative(x):
    return x + 1


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
        (
            "threshold_imputer",
            ThresholdImputer(threshold=0.02, strategy="most_frequent"),
        ),
    ]
)

preprocessor_transformer = ColumnTransformer(
    [
        ("num", numeric_pipeline, selector(dtype_include=dtypes_dict["number"])),
        (
            "cat",
            categorical_pipeline,
            selector(dtype_include=dtypes_dict["categorical"]),
        ),
        ("bool", boolean_pipeline, selector(dtype_include=dtypes_dict["boolean"])),
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
        ("preprocessing", preprocessor_transformer),
    ]
)

preprocessing_pipeline_nn = Pipeline(
    [
        ("preprocessing", preprocessor_transformer),
        ("encoder", encoder_transformer),
    ]
)
