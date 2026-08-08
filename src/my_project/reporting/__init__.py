"""Reporting helpers (LaTeX export of figures and tables, plots)."""

from my_project.reporting.latex_export import (
    format_latex_bold_best,
    latex_escape,
    results_latex_dir,
    save_figure,
    save_latex_table,
    save_table,
    slugify,
)
from my_project.reporting.plotting import (
    edge_jaccard_matrix,
    pair_table_from_matrix,
    pairwise_ii_mi_table,
    plot_class_balance,
    plot_feature_histograms,
    plot_ii_by_edges,
    plot_ii_graph,
    top_k_edges,
)

__all__ = [
    "edge_jaccard_matrix",
    "format_latex_bold_best",
    "latex_escape",
    "pair_table_from_matrix",
    "pairwise_ii_mi_table",
    "plot_class_balance",
    "plot_feature_histograms",
    "plot_ii_by_edges",
    "plot_ii_graph",
    "results_latex_dir",
    "save_figure",
    "save_latex_table",
    "save_table",
    "slugify",
    "top_k_edges",
]
