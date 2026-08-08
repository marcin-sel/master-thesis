"""Shared helpers for training orchestration modules."""

import os
import re


def resolve_artifact_location(tracking_uri):
    """Central MLflow artifact dir next to a sqlite backend, as a file:// URI.

    With a sqlite tracking backend and no explicit artifact root, MLflow writes
    artifacts to ``./mlruns/<id>`` relative to the process CWD (e.g. a notebook
    dir). Anchoring them beside the DB keeps every run's artifacts in one place.
    Returns ``None`` for non-sqlite backends so MLflow keeps its own default.
    """
    if not tracking_uri or not tracking_uri.startswith("sqlite:"):
        return None
    db_path = tracking_uri.split("sqlite://", 1)[-1].lstrip("/")
    db_path = "/" + db_path
    root = os.path.join(os.path.dirname(db_path), "mlartifacts")
    os.makedirs(root, exist_ok=True)
    return "file://" + root


def _build_study_name(*parts):
    """Build an Optuna study name from optional pieces."""
    return "__".join(str(part) for part in parts if part is not None)


def _extract_graph_permute_seed(graph_name):
    """Return permutation seed parsed from graph name, e.g. '*_permuted_7'."""
    if not isinstance(graph_name, str):
        return None
    match = re.search(r"_permuted_(\d+)$", graph_name)
    return int(match.group(1)) if match else None
