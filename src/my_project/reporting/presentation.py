"""Shared presentation conventions for analysis notebooks."""

from typing import Any

import seaborn as sns

CATEGORICAL_PALETTE = "husl"
SEQUENTIAL_PALETTE = "crest"

PLOT_LABELS_PL = {
    "model_cls": "model",
    "graph_name": "graf",
    "graph_name_main": "graf",
    "n_edges": "liczba krawędzi",
    "n_samples": "liczba próbek",
    "data_seed": "ziarno danych",
    "corr": "współczynnik korelacji",
    "cov": "współczynnik korelacji",
    "train_size": "rozmiar zbioru treningowego",
    "val": "walidacja",
    "test": "test",
    "auc": "AUC",
    "avg_precision": "PR AUC",
    "pr_auc": "PR AUC",
    "f1": "F1",
    "precision": "precyzja",
    "recall": "czułość",
    "specificity": "swoistość",
    "accuracy": "dokładność",
    "medical": "dane medyczne",
    "pairwise": "interakcje parami",
    "synthetic": "dane syntetyczne",
    "feature": "zmienna",
    "first_observed_month": "pierwszy miesiąc obserwacji",
    "estimated_use_start": "szacowany początek stosowania",
    "availability_before": "dostępność przed zmianą",
    "availability_after": "dostępność po zmianie",
    "availability_gain": "przyrost dostępności",
    "mean": "średnia",
    "median": "mediana",
    "std": "odch. stand.",
    "min": "min.",
    "max": "maks.",
    "skewness": "wsp. skośności",
    "missing_pct": "odsetek braków",
    "most_frequent": "odsetek najczęściej występującej wartości",
    "least_frequent": "odsetek najrzadszej wartości",
    "n_classes": "liczba kategorii",
    "mutual_information": "informacja wzajemna",
    "ii_max": "maksymalna informacja interakcyjna",
    "ii_mean": "średnia informacja interakcyjna",
    "ii_sum": "suma informacji interakcyjnej",
    "degree": "stopień wierzchołka",
    "meaning_pl": "znaczenie",
    "inferred_unit": "jednostka",
    "confidence": "pewność",
    "notes": "uwagi",
}

CATEGORY_LABELS_PL = {
    True: "Tak",
    False: "Nie",
    "Missing": "Brak danych",
    "Low": "Niski",
    "Medium": "Średni",
    "High": "Wysoki",
    "TnI_Low": "TnI_niska",
    "TnI_Medium": "TnI_średnia",
    "TnI_High": "TnI_wysoka",
    "TnT_Low": "TnT_niska",
    "TnT_Medium": "TnT_średnia",
    "TnT_High": "TnT_wysoka",
}

GRAPH_LABELS = {
    "empty": "empty",
    "fully_connected": "full",
    "oracle": "oracle",
    "ii_permuted": "ii_perm",
}


def configure_notebook_plots() -> None:
    """Apply the plotting theme shared by the analysis notebooks."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={"axes.grid.axis": "y"},
    )


def plot_label(value: Any) -> Any:
    """Return the Polish report label, preserving unknown values."""
    return PLOT_LABELS_PL.get(value, value)


def graph_display_name(graph_name: Any) -> str:
    """Normalize graph identifiers for plot and table labels."""
    graph_name = str(graph_name)
    suffix = "_permuted" if graph_name.endswith("_permuted") else ""
    base_name = graph_name.removesuffix(suffix)
    if base_name.startswith("ii_graph_"):
        threshold = base_name.rsplit("_", 1)[-1]
        if threshold.replace(".", "", 1).isdigit():
            base_name = "ii"
    return GRAPH_LABELS.get(
        graph_name,
        base_name + ("_perm" if suffix else ""),
    )


def uses_edge_count(graph_name: Any) -> bool:
    """Return whether a graph label should include its edge count."""
    graph_name = str(graph_name)
    return graph_name in {
        "ii",
        "ii_permuted",
        "empty",
        "fully_connected",
        "oracle",
    } or graph_name.startswith("ii_graph_")
