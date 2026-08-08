"""Plotting and edge-analysis helpers for interaction-information graphs."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def pair_table_from_matrix(
    ii_matrix: pd.DataFrame, value_name: str = "interaction_information"
) -> pd.DataFrame:
    """Flatten a symmetric measure matrix to a table of pairs sorted descending.

    Only the upper triangle is used (each unordered pair once, diagonal
    excluded), giving columns ``feature_i``, ``feature_j`` and ``value_name``.
    """
    mask = np.triu(np.ones(ii_matrix.shape), k=1).astype(bool)
    return (
        ii_matrix.where(mask)
        .stack()
        .rename(value_name)
        .reset_index()
        .rename(columns={"level_0": "feature_i", "level_1": "feature_j"})
        .sort_values(value_name, ascending=False)
        .reset_index(drop=True)
    )


def pairwise_ii_mi_table(
    ii_matrix: pd.DataFrame,
    mi_matrix: pd.DataFrame,
    *,
    ii_name: str = "interaction_information",
    mi_name: str = "mutual_information",
) -> pd.DataFrame:
    """Pair table of interaction information plus feature-feature MI.

    Flattens ``ii_matrix`` with :func:`pair_table_from_matrix` (sorted descending
    by II) and appends a ``mi_name`` column read from ``mi_matrix`` for the same
    pairs.
    """
    table = pair_table_from_matrix(ii_matrix, value_name=ii_name)
    table[mi_name] = [
        mi_matrix.loc[i, j] for i, j in zip(table["feature_i"], table["feature_j"])
    ]
    return table


def top_k_edges(pair_table: pd.DataFrame, k: int) -> set[frozenset]:
    """Return the top-``k`` pairs of ``pair_table`` as a set of frozensets."""
    return {
        frozenset((row.feature_i, row.feature_j))
        for row in pair_table.head(k).itertuples()
    }


def edge_jaccard_matrix(pair_tables: Sequence[pd.DataFrame], k: int) -> pd.DataFrame:
    """Pairwise Jaccard overlap of the top-``k`` edges of each pair table.

    Useful for measuring how reproducible the selected graph is across CV folds:
    a value near 1 means the same top edges are picked, near 0 means the
    selection is unstable (dominated by estimation noise).
    """
    tops = [top_k_edges(pt, k) for pt in pair_tables]
    n = len(tops)
    jac = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            union = len(tops[i] | tops[j])
            jac[i, j] = jac[j, i] = len(tops[i] & tops[j]) / union if union else 0.0
    labels = [f"fold{i}" for i in range(n)]
    return pd.DataFrame(jac, index=labels, columns=labels)


def plot_ii_graph(
    pair_table: pd.DataFrame,
    nodes: Sequence,
    *,
    k: int = 40,
    ax: plt.Axes | None = None,
    title: str | None = None,
    seed: int = 42,
    add_colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> nx.Graph:
    """Draw the graph of the top-``k`` pairs of ``pair_table``.

    Edge width/colour and node size encode interaction strength (edge weight and
    degree). Pass an existing ``ax`` to compose a grid (e.g. one panel per fold);
    ``add_colorbar=False`` skips the per-axes colour bar in that case. ``vmin`` /
    ``vmax`` fix the edge colour scale so several panels can share one colour bar.
    Returns the built :class:`networkx.Graph`.
    """
    top_k = pair_table.head(k)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for fi, fj, w in zip(
        top_k["feature_i"], top_k["feature_j"], top_k["interaction_information"]
    ):
        graph.add_edge(fi, fj, weight=float(w))

    degrees = dict(graph.degree())
    node_sizes = [60 + 120 * degrees[node] for node in graph.nodes()]
    node_colors = [
        "#c9cbd1" if degrees[node] == 0 else "#4c72b0" for node in graph.nodes()
    ]
    edges = list(graph.edges())
    weights = np.array([graph[u][v]["weight"] for u, v in edges])
    edge_widths = (
        1.0 + 4.0 * (weights - weights.min()) / (np.ptp(weights) + 1e-12)
        if len(weights)
        else []
    )
    pos = nx.spring_layout(graph, weight="weight", k=1.4, iterations=300, seed=seed)

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 11))
    fig = ax.figure

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edgelist=edges,
        width=edge_widths,
        edge_color=weights,
        edge_cmap=plt.cm.viridis,
        edge_vmin=vmin,
        edge_vmax=vmax,
        alpha=0.85,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=6, font_color="black")

    deg = np.array(list(degrees.values()))
    nonzero = deg[deg > 0]
    nonzero_mean = nonzero.mean() if len(nonzero) else 0.0
    ax.text(
        0.01,
        0.01,
        f"śr. stopień: {deg.mean():.2f} | śr.≠0: {nonzero_mean:.2f} | "
        f"maks: {int(deg.max()) if len(deg) else 0}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8),
    )

    if add_colorbar and len(weights):
        norm = plt.Normalize(
            vmin=weights.min() if vmin is None else vmin,
            vmax=weights.max() if vmax is None else vmax,
        )
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        # cbar.set_label("II((Xi, Xj), y)")
        cbar.set_label("II")
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.margins(0.08)
    return graph


def plot_class_balance(
    y,
    *,
    ax: plt.Axes | None = None,
    xlabel: str = "klasa y",
    ylabel: str = "liczba przykładów",
    title: str | None = None,
) -> plt.Axes:
    """Bar plot of class counts for a binary/categorical target ``y``."""
    counts = pd.Series(y).value_counts().sort_index()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    counts.plot.bar(ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax


def plot_feature_histograms(
    X: pd.DataFrame,
    y,
    *,
    columns: Sequence | None = None,
    n_cols: int = 7,
    bins: int = 40,
    legend_title: str = "y",
    legend_bbox_y: float = -0.05,
) -> plt.Figure:
    """Grid of per-feature step histograms coloured by class ``y``.

    One panel per column of ``X`` (or ``columns`` if given), each overlaying a
    density histogram per class. Unused panels are hidden and a single shared
    legend is placed below the grid.
    """
    cols = list(X.columns if columns is None else columns)
    n_features = len(cols)
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        for cls in sorted(pd.Series(y).unique()):
            ax.hist(
                X.loc[y == cls, col].dropna(),
                bins=bins,
                histtype="step",
                density=True,
                label=f"y={cls}",
            )
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[n_features:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title=legend_title,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_bbox_y),
        ncol=len(labels),
    )
    fig.tight_layout()
    return fig


def plot_ii_by_edges(
    pair_table: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    value_col: str = "interaction_information",
    xlabel: str = "rank\n(%)",
    ylabel: str = "ii",
    n_ticks: int = 11,
) -> plt.Axes:
    """Line plot of edge values sorted descending, with rank/percent x-ticks.

    ``pair_table`` is expected sorted descending by ``value_col`` (as returned by
    :func:`pair_table_from_matrix`). The x-axis shows the edge rank together with
    its percentile of all edges.
    """
    values = pair_table[value_col].to_numpy()
    total = len(values)
    edge_count = np.arange(1, total + 1)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    ax.plot(edge_count, values, markersize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xticks = np.linspace(0, total, n_ticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(round(t))}\n({t / total * 100:.0f}%)" for t in xticks])
    return ax


__all__ = [
    "pair_table_from_matrix",
    "pairwise_ii_mi_table",
    "top_k_edges",
    "edge_jaccard_matrix",
    "plot_ii_graph",
    "plot_class_balance",
    "plot_feature_histograms",
    "plot_ii_by_edges",
]
