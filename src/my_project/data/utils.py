import numpy as np


def columns_info_func(df, sort_by_column="missing_pct", ascending=False):
    columns_info = df.isna().mean().rename("missing_pct").to_frame()
    columns_info["dtype"] = df.dtypes.values
    columns_info["n_unique"] = df.nunique().values

    columns_info = columns_info.sort_values(by=sort_by_column, ascending=ascending)
    columns_info["cumsum_pct"] = (np.arange(len(columns_info)) + 1) / len(columns_info)

    min_fraction = df.apply(
        lambda x: x.value_counts(normalize=True, dropna=False)
        .sort_values()
        .reset_index()
        .iloc[0]
        .rename({x.name: "min_var", "proportion": "min_proportion"}),
        axis=0,
    ).T
    max_fraction = df.apply(
        lambda x: x.value_counts(normalize=True, dropna=False)
        .sort_values()
        .reset_index()
        .iloc[-1]
        .rename({x.name: "max_var", "proportion": "max_proportion"}),
        axis=0,
    ).T
    columns_info = columns_info.merge(
        min_fraction, left_index=True, right_index=True, how="left"
    )
    columns_info = columns_info.merge(
        max_fraction, left_index=True, right_index=True, how="left"
    )
    columns_info["is_categorical"] = columns_info.dtype.astype(str).isin(
        ["object", "category"]
    )

    return columns_info
