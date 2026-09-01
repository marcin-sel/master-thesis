"""Plotting and edge-analysis helpers for interaction-information graphs."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter


def _format_decimal_float(value: float, _position: int) -> str:
    label = np.format_float_positional(value, precision=8, trim="-")
    return label if "." in label else f"{label}.0"


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
    node_size: str | int = "degree",
    layout: str = "spring",
    curved_edges: bool = True,
    label_fontsize: int = 8,
    legend_label: str = "II",
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
    if node_size == "degree":
        node_sizes = [60 + 120 * degrees[node] for node in graph.nodes()]
    else:
        node_sizes = [node_size for _ in graph.nodes()]

    node_colors = ["#aec7e8" for _ in graph.nodes()]
    edges = list(graph.edges())
    weights = np.array([graph[u][v]["weight"] for u, v in edges])
    edge_widths = (
        1.0 + 4.0 * (weights - weights.min()) / (np.ptp(weights) + 1e-12)
        if len(weights)
        else []
    )
    # Node layout. "circular": evenly on a circle (readable for dense graphs,
    # labels don't overlap). "spring": force-directed only on nodes with edges,
    # while isolated ones are placed separately in a row at the top.
    if layout == "circular":
        pos = nx.circular_layout(graph)
    else:
        connected = [node for node, d in degrees.items() if d > 0]
        isolated = [node for node, d in degrees.items() if d == 0]
        layout_graph = graph.subgraph(connected) if connected else graph
        pos = nx.spring_layout(
            layout_graph,
            weight="weight",
            k=4.0 / np.sqrt(max(len(connected), 1)),
            iterations=600,
            seed=seed,
        )
        if isolated:
            xs = np.linspace(-1.0, 1.0, len(isolated)) if len(isolated) > 1 else [0.0]
            for xi, node in zip(xs, isolated):
                pos[node] = np.array([xi, 1.25])

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
        arrows=True,  # FancyArrowPatch -> supports curved edges
        arrowstyle="-",  # no arrowheads (undirected graph)
        # arc (rad>0) separates overlapping edges; rad=0 -> straight segments
        connectionstyle=f"arc3,rad={0.12 if curved_edges else 0.0}",
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
    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        font_size=label_fontsize,
        font_color="black",
    )

    deg = np.array(list(degrees.values()))
    nonzero = deg[deg > 0]
    nonzero_mean = nonzero.mean() if len(nonzero) else 0.0
    ax.text(
        0.01,
        0.01,
        f"średni stopień: {deg.mean():.2f} \n"
        f"średni niezerowy stopień: {nonzero_mean:.2f} \n"
        f"maksymalny stopień: {int(deg.max()) if len(deg) else 0}",
        transform=ax.transAxes,
        fontsize=label_fontsize,
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
        cbar.set_label(legend_label, fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 2)
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
    y=None,
    *,
    columns: Sequence | None = None,
    n_cols: int = 7,
    bins: int = 40,
    legend_title: str = "y",
    legend_bbox_y: float = -0.05,
    by_columns: bool = False,
    trim_quantiles: tuple[float, float] | None = None,
    tail_iqr_multiplier: float = 3.0,
) -> plt.Figure:
    """Grid of per-feature step histograms coloured by class ``y``.

    One panel per column of ``X`` (or ``columns`` if given), each overlaying a
    density histogram per class. Unused panels are hidden and a single shared
    legend is placed below the grid. With ``by_columns=True`` panels are filled
    column-first (top-to-bottom) instead of the default row-first order. If
    ``y`` is ``None`` a single unlabelled histogram is drawn per feature and the
    legend is omitted. ``trim_quantiles`` can limit only tails extending beyond
    ``tail_iqr_multiplier`` times the IQR without modifying the input data.
    """
    if trim_quantiles is not None:
        lower_q, upper_q = trim_quantiles
        if not 0 <= lower_q < upper_q <= 1:
            raise ValueError("trim_quantiles must satisfy 0 <= lower < upper <= 1.")

    cols = list(X.columns if columns is None else columns)
    n_features = len(cols)
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.6 * n_rows))
    axes = np.atleast_1d(axes)
    axes = axes.ravel(order="F") if by_columns else axes.ravel()

    for ax, col in zip(axes, cols):
        x = X[col].dropna()
        plotted_x = x
        if trim_quantiles is not None and not x.empty:
            q1, q3 = x.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound, upper_bound = x.min(), x.max()
            if iqr > 0:
                if q1 - x.min() > tail_iqr_multiplier * iqr:
                    lower_bound = x.quantile(lower_q)
                if x.max() - q3 > tail_iqr_multiplier * iqr:
                    upper_bound = x.quantile(upper_q)
            plotted_x = x[x.between(lower_bound, upper_bound)]

        if y is not None:
            for cls in sorted(pd.Series(y).unique()):
                ax.hist(
                    plotted_x.loc[pd.Series(y, index=X.index) == cls],
                    bins=bins,
                    histtype="step",
                    density=True,
                    label=f"y={cls}",
                )
        else:
            ax.hist(plotted_x, bins=bins, histtype="step", density=True)
        # ax.set_xlabel(col, fontsize=8)
        title = col
        ax.set_title(title, fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(_format_decimal_float))
        ax.tick_params(labelsize=7)
    for ax in axes[n_features:]:
        ax.axis("off")
    if y is not None:
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


def plot_category_barplots(
    X: pd.DataFrame,
    y=None,
    *,
    columns: Sequence | None = None,
    n_cols: int = 7,
    normalize: bool = True,
    max_categories: int | None = None,
    min_frequency: float | None = None,
    other_label: str = "Inne",
    category_order: Sequence | None = None,
    legend_title: str = "y",
    legend_bbox_y: float = -0.05,
    by_columns: bool = False,
    show_values: bool = True,
) -> plt.Figure:
    """Grid of per-feature bar plots of category frequencies coloured by class.

    Categorical counterpart of :func:`plot_feature_histograms`: one panel per
    column of ``X`` (or ``columns`` if given). With ``y`` the bars are grouped
    per class; without it a single bar per category is drawn. ``normalize=True``
    plots proportions (share within each class), ``False`` raw counts.
    ``max_categories`` keeps only the most frequent categories per feature.
    ``min_frequency`` groups categories whose overall non-missing share is below
    the threshold into ``other_label`` before plotting. The grouping is learned
    from all rows, so with ``y`` the meaning of the aggregate category remains
    identical across classes.
    ``category_order`` fixes the x-axis order (case-insensitive, e.g.
    ``["low", "medium", "high", "missing"]``); categories not listed are appended
    in descending frequency. Unused panels are hidden and a single shared legend
    is placed below the grid. ``by_columns=True`` fills panels column-first.
    ``show_values=True`` labels non-zero bars as percentages when normalized and
    as integer counts otherwise.
    """
    if min_frequency is not None and not 0 <= min_frequency <= 1:
        raise ValueError("min_frequency must be between 0 and 1.")

    cols = list(X.columns if columns is None else columns)
    n_features = len(cols)
    n_rows = int(np.ceil(n_features / n_cols))

    order_rank = (
        {str(c).lower(): i for i, c in enumerate(category_order)}
        if category_order is not None
        else None
    )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.6 * n_rows))
    axes = np.atleast_1d(axes)
    axes = axes.ravel(order="F") if by_columns else axes.ravel()

    ylabel = "%" if normalize else "liczność"
    for ax, col in zip(axes, cols):
        values = X[col].astype(object)
        if min_frequency is not None:
            frequencies = values.value_counts(normalize=True)
            rare_categories = frequencies[frequencies < min_frequency].index
            values = values.where(~values.isin(rare_categories), other_label)

        order = values.value_counts().index
        if max_categories is not None:
            order = order[:max_categories]
        if order_rank is not None:
            # Stable sort keeps unlisted categories in descending-frequency order.
            order = sorted(
                order, key=lambda c: order_rank.get(str(c).lower(), len(order_rank))
            )
        labels = [str(c) for c in order]
        if y is not None:
            classes = sorted(pd.Series(y).unique())
            width = 0.8 / len(classes)
            positions = np.arange(len(order))
            for i, cls in enumerate(classes):
                heights = (
                    values.loc[pd.Series(y, index=X.index) == cls]
                    .value_counts(normalize=normalize)
                    .reindex(order, fill_value=0)
                    .to_numpy()
                )
                bars = ax.bar(
                    positions + i * width, heights, width=width, label=f"y={cls}"
                )
                if show_values:
                    labels_above = [
                        (f"{value:.1%}" if normalize else f"{value:.0f}")
                        if value > 0
                        else ""
                        for value in heights
                    ]
                    ax.bar_label(bars, labels=labels_above, padding=2, fontsize=7)
            ax.set_xticks(positions + width * (len(classes) - 1) / 2)
            ax.set_xticklabels(labels, rotation=45, ha="right")
        else:
            heights = values.value_counts(normalize=normalize).reindex(order).to_numpy()
            bars = ax.bar(labels, heights)
            if show_values:
                labels_above = [
                    (f"{value:.1%}" if normalize else f"{value:.0f}")
                    if value > 0
                    else ""
                    for value in heights
                ]
                ax.bar_label(bars, labels=labels_above, padding=2, fontsize=7)
            ax.tick_params(axis="x", rotation=45)
        ax.set_title(col, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        if normalize:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.tick_params(labelsize=7)
        if show_values:
            ax.margins(y=0.15)
    for ax in axes[n_features:]:
        ax.axis("off")
    if y is not None:
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


__all__ = [
    "pair_table_from_matrix",
    "pairwise_ii_mi_table",
    "top_k_edges",
    "edge_jaccard_matrix",
    "plot_ii_graph",
    "plot_class_balance",
    "plot_feature_histograms",
    "plot_category_barplots",
    "plot_ii_by_edges",
]
