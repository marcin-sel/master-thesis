import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer


class NaNPreservingKBinsDiscretizer(BaseEstimator, TransformerMixin):
    """
    Wrapper around KBinsDiscretizer that preserves NaN values and opcjonalnie mapuje biny na etykiety.
    """

    LABELS_DICT_DEFAULT = {
        2: {0: "Low", 1: "High"},
        3: {0: "Low", 1: "Medium", 2: "High"},
        4: {0: "Very Low", 1: "Low", 2: "High", 3: "Very High"},
        5: {0: "Very Low", 1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    }

    def __init__(
        self,
        n_bins=5,
        encode="ordinal",
        strategy="quantile",
        quantile_method="averaged_inverted_cdf",
        dtype=None,
        subsample=200_000,
        random_state=None,
        columns=None,
        output_dtype="Int8",
        labels_dict=None,
    ):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state
        self.columns = columns
        self.output_dtype = output_dtype
        self.labels_dict = labels_dict

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = (
            X.columns.tolist() if self.columns is None else list(self.columns)
        )

        self.discretizers_ = {}
        for col in self.columns_:
            non_missing = X[col].dropna()
            if non_missing.empty:
                self.discretizers_[col] = None
                continue

            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode=self.encode,
                strategy=self.strategy,
                quantile_method=self.quantile_method,
                dtype=self.dtype,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            discretizer.fit(non_missing.to_frame())
            self.discretizers_[col] = discretizer

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        labels_dict = (
            self.labels_dict
            if self.labels_dict is not None
            else self.LABELS_DICT_DEFAULT
        )

        for col in self.columns_:
            discretizer = self.discretizers_[col]
            if discretizer is None:
                continue

            mask = X[col].notna()
            transformed = np.asarray(discretizer.transform(X.loc[mask, [col]])).ravel()

            n_bins = (
                int(discretizer.n_bins_[0])
                if hasattr(discretizer, "n_bins_")
                else int(self.n_bins)
            )

            transformed = pd.Series(transformed, index=X.index[mask])
            label_map = labels_dict.get(n_bins)
            if label_map is not None:
                transformed = transformed.map(label_map)

            transformed = transformed.astype(object)
            X[col] = X[col].astype(object)
            X.loc[mask, col] = transformed

        return X


class RareCategoryTransformer(BaseEstimator, TransformerMixin):
    """Replace infrequent categories with a specified value (default: NaN).

    Parameters
    ----------
    min_frequency : float or None
        Minimum relative frequency to keep a category (e.g. 0.01 = 1%).
    min_count : int or None
        Minimum absolute count to keep a category. Used when min_frequency is None.
    replace_with : scalar
        Value to substitute for rare categories (default: NaN).
    include_nan : bool
        Whether to include NaN when computing frequencies.
    """

    def __init__(
        self, min_frequency=0.01, min_count=None, replace_with=np.nan, include_nan=True
    ):
        self.min_frequency = min_frequency
        self.min_count = min_count
        self.replace_with = replace_with
        self.include_nan = include_nan

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.frequent_categories_ = {}

        for col in X.columns:
            counts = X[col].value_counts(dropna=not self.include_nan)

            if self.min_frequency is not None:
                freq = counts / counts.sum()
                mask = freq >= self.min_frequency
            else:
                mask = counts >= self.min_count

            self.frequent_categories_[col] = set(counts[mask].index)

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in X.columns:
            X[col] = X[col].astype(object)
            mask = ~X[col].isin(self.frequent_categories_[col])
            X.loc[mask, col] = self.replace_with

        return X


class ThresholdImputer(BaseEstimator, TransformerMixin):
    """Impute missing values only in columns whose missing rate is below a given threshold.

    Parameters
    ----------
    threshold : float
        Maximum fraction of missing values allowed for imputation (e.g. 0.05 = 5%).
    strategy : str
        Imputation strategy passed to SimpleImputer: 'mean', 'median', 'most_frequent', or 'constant'.
    fill_value : scalar, optional
        Value used when strategy='constant'.
    """

    def __init__(
        self, threshold: float = 0.05, strategy: str = "mean", fill_value=None
    ):
        self.threshold = threshold
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        missing_rate = X.isna().mean()
        self.cols_to_impute_ = missing_rate[
            missing_rate < self.threshold
        ].index.tolist()
        self.cols_skipped_ = missing_rate[missing_rate >= self.threshold].index.tolist()

        if self.cols_to_impute_:
            self.imputer_ = SimpleImputer(
                strategy=self.strategy,
                fill_value=self.fill_value,
            )
            subset = X[self.cols_to_impute_].replace({pd.NA: np.nan})
            self.imputer_.fit(subset)

        self.cols_to_impute_dtypes_ = X[self.cols_to_impute_].dtypes.to_dict()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        if self.cols_to_impute_:
            X[self.cols_to_impute_] = self.imputer_.transform(
                X[self.cols_to_impute_].replace({pd.NA: np.nan})
            )

        for col, dtype in self.cols_to_impute_dtypes_.items():
            X[col] = X[col].astype(dtype)

        return X


class HighMissingDiscretizer(BaseEstimator, TransformerMixin):
    """Quantile-bin numeric columns with high missingness, keeping "missing" as a category.

    Numeric columns whose train-set missing fraction exceeds ``threshold`` are
    discretized into ``n_bins`` quantile bins (NaN preserved during binning);
    their missing values are then replaced by ``missing_label``, so the column
    becomes a categorical feature with levels ``{missing_label, <bin labels>}``.
    This encodes missingness as an explicit, informative category without adding
    separate indicator columns, so every downstream model consumes the same
    representation. Columns at or below the threshold pass through unchanged.

    The high-missing columns are selected from the data seen in :meth:`fit`
    (train only), so no information leaks from validation/test rows.

    Parameters
    ----------
    threshold : float
        Minimum missing fraction (e.g. 0.10 = 10%) for a numeric column to be
        discretized.
    n_bins : int
        Number of quantile bins for the non-missing values.
    strategy, quantile_method :
        Forwarded to the underlying KBinsDiscretizer.
    missing_label : str
        Category assigned to missing values.
    labels_dict : dict or None
        Optional mapping ``{n_bins: {bin_index: label}}`` for bin labels.
    """

    def __init__(
        self,
        threshold: float = 0.10,
        n_bins: int = 5,
        strategy: str = "quantile",
        quantile_method: str = "averaged_inverted_cdf",
        missing_label: str = "Missing",
        labels_dict=None,
    ):
        self.threshold = threshold
        self.n_bins = n_bins
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.missing_label = missing_label
        self.labels_dict = labels_dict

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        numeric_cols = X.select_dtypes(include=["number"]).columns
        missing_rate = X[numeric_cols].isna().mean()
        self.columns_ = missing_rate[missing_rate > self.threshold].index.tolist()

        if self.columns_:
            self.discretizer_ = NaNPreservingKBinsDiscretizer(
                n_bins=self.n_bins,
                strategy=self.strategy,
                quantile_method=self.quantile_method,
                columns=self.columns_,
                labels_dict=self.labels_dict,
            ).fit(X)
        else:
            self.discretizer_ = None

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        if not self.columns_:
            return X

        X = self.discretizer_.transform(X)
        for col in self.columns_:
            filled = X[col].where(X[col].notna(), self.missing_label)
            X[col] = filled.astype("category")

        return X


class BooleanMissingEncoder(BaseEstimator, TransformerMixin):
    """Encode missing values in boolean columns as an explicit "Missing" level.

    Boolean columns whose train-set missing fraction exceeds ``threshold`` are
    turned into a 3-level categorical ``{False, True, missing_label}`` instead of
    being imputed, so a high rate of missingness is preserved as an informative
    category (mirroring :class:`HighMissingDiscretizer` for numeric columns).
    Columns at or below the threshold are left untouched (they are expected to be
    imputed by a preceding step), so fully observed booleans pass through
    unchanged.

    The high-missing columns are selected from the data seen in :meth:`fit`
    (train only), so no information leaks from validation/test rows.

    Parameters
    ----------
    threshold : float
        Minimum missing fraction (e.g. 0.10 = 10%) for a boolean column to be
        encoded with an explicit missing level instead of being imputed.
    missing_label : str
        Category assigned to missing values.
    """

    def __init__(self, threshold: float = 0.10, missing_label: str = "Missing"):
        self.threshold = threshold
        self.missing_label = missing_label

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        missing_rate = X.isna().mean()
        self.columns_ = missing_rate[missing_rate > self.threshold].index.tolist()
        # Pin a fixed, complete category set so the output dtype is stable across
        # folds/splits: a validation slice that happens to lack a level (e.g. no
        # missing, or only one boolean value) still gets the same categories
        # instead of an inconsistently-typed column.
        self.dtype_ = pd.CategoricalDtype(
            categories=["False", "True", self.missing_label]
        )
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in self.columns_:
            # Cast to str so the levels ("True"/"False"/"Missing") are mutually
            # sortable; a category mixing bool and str breaks the downstream
            # OrdinalEncoder. The fixed dtype keeps categories consistent.
            filled = X[col].astype(object).where(X[col].notna(), self.missing_label)
            X[col] = filled.astype(str).astype(self.dtype_)

        return X
