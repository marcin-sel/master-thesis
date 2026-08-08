"""Per-experiment pipelines and Optuna search spaces.

Each experiment (``medical``, ``synthetic``) owns its preprocessing pipelines and
hyperparameter grids so the medical tuning notebook and the synthetic sweep can
diverge independently instead of sharing one global config.
"""
