from collections.abc import Iterable

import optuna
import pandas as pd

OPTUNA_TRIAL_ATTRS = (
    "number",
    "value",
    "state",
    "datetime_start",
    "datetime_complete",
    "duration",
    "params",
    "user_attrs",
)


def list_study_names(storage_url: str) -> list[str]:
    study_summaries = optuna.get_all_study_summaries(storage=storage_url)
    return sorted(summary.study_name for summary in study_summaries)


def load_trials_dataframe(
    study_name: str,
    storage_url: str,
    attrs: Iterable[str] = OPTUNA_TRIAL_ATTRS,
) -> pd.DataFrame:
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    df_trials = study.trials_dataframe(attrs=attrs)
    return df_trials.reset_index(drop=True)


def load_all_trials_dataframe(
    storage_url: str,
    study_names: Iterable[str] | None = None,
    attrs: Iterable[str] = OPTUNA_TRIAL_ATTRS,
) -> pd.DataFrame:
    if study_names is None:
        study_names = list_study_names(storage_url)

    trials_by_study = {
        study_name: load_trials_dataframe(
            study_name=study_name,
            storage_url=storage_url,
            attrs=attrs,
        )
        for study_name in study_names
    }

    if not trials_by_study:
        return pd.DataFrame()

    return (
        pd.concat(trials_by_study)
        .reset_index(level=0)
        .rename(columns={"level_0": "study_name"})
        .reset_index(drop=True)
    )
