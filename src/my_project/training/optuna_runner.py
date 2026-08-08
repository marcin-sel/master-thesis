from __future__ import annotations

from typing import Callable, List, Optional

import optuna
from tqdm.auto import tqdm


def build_tpe_sampler(seed: int, n_startup_trials: int) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        n_startup_trials=n_startup_trials,
    )


def build_pruner(n_startup_trials: int) -> optuna.pruners.PercentilePruner:
    return optuna.pruners.PercentilePruner(
        percentile=50,
        n_startup_trials=n_startup_trials,
        n_warmup_steps=15,
    )


def create_or_resume_study(
    *,
    study_name: str,
    storage_url: str,
    direction: str,
    seed: int,
    n_startup_trials: int,
    load_if_exists: bool = True,
) -> optuna.Study:
    sampler = build_tpe_sampler(seed, n_startup_trials)
    pruner = build_pruner(n_startup_trials)

    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )


def run_optuna_study(
    *,
    study_name: str,
    storage_url: str,
    direction: str,
    n_trials: int,
    objective: Callable,
    seed: int,
    n_startup_trials: int,
    callbacks: Optional[List] = None,
    show_progress_bar: bool = False,
    load_if_exists: bool = True,
) -> optuna.Study:
    study = create_or_resume_study(
        study_name=study_name,
        storage_url=storage_url,
        direction=direction,
        seed=seed,
        n_startup_trials=n_startup_trials,
        load_if_exists=load_if_exists,
    )

    finished = sum(
        1
        for t in study.trials
        if t.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    )
    remaining = max(0, n_trials - finished)

    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            callbacks=callbacks or [],
            show_progress_bar=show_progress_bar,
        )

    return study


class MetricsProgressCallback:
    """Optuna callback that shows trial-level mean metrics on a tqdm bar.

    After each completed trial it reads the aggregated fold metrics stored in
    ``trial.user_attrs["mean_metrics"]`` and surfaces a chosen subset plus the
    running best value in the bar's postfix.
    """

    def __init__(
        self,
        metrics=("val/loss", "val/auc", "val/avg_precision"),
        desc=None,
        total=None,
    ):
        self.metrics = metrics
        self.desc = desc
        self.total = total
        self._bar = None

    def __call__(self, study, trial):
        if self._bar is None:
            self._bar = tqdm(
                desc=self.desc or study.study_name, total=self.total, leave=True
            )

        # Sync the bar to the number of finished trials so it reflects true
        # progress even when resuming a partially completed study.
        n_done = sum(
            1
            for t in study.trials
            if t.state
            in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        )
        self._bar.n = n_done
        self._bar.refresh()

        postfix = {}
        try:
            postfix["best"] = f"{study.best_value:.4f}"
        except ValueError:
            pass

        mean_metrics = trial.user_attrs.get("mean_metrics", {})
        for name in self.metrics:
            if name in mean_metrics:
                postfix[name] = f"{mean_metrics[name]:.4f}"

        self._bar.set_postfix(postfix)

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None
