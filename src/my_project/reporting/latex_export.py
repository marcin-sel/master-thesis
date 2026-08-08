"""Spójny eksport rycin i tabel do LaTeX-a (na potrzeby pracy).

Wszystkie notebooki używają tej samej konwencji:

- ryciny -> ``$RESULTS_LATEX_DIR/plots/<slug>.pdf``,
- tabele -> ``$RESULTS_LATEX_DIR/tables/<slug>.tex`` (sam ``tabular``, bez float
  ``table``/caption/label — plik wstawiasz przez ``\\input`` we własnym
  środowisku ``table``),

gdzie ``<slug>`` powstaje z nazwy przez :func:`slugify`. Katalog bazowy czytany
jest z zmiennej środowiskowej ``RESULTS_LATEX_DIR`` (ustaw w ``.env`` i wywołaj
``dotenv.load_dotenv()`` w notebooku); można go nadpisać argumentem ``base_dir``.
Gdy katalog nie jest ustawiony, funkcje wypisują ostrzeżenie i nic nie zapisują.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "results_latex_dir",
    "slugify",
    "latex_escape",
    "save_figure",
    "save_table",
    "format_latex_bold_best",
    "save_latex_table",
]


# LaTeX special characters -> safe escapes (for table cell / label strings that
# are NOT passed through pandas' own ``escape=True``).
_LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\u2014": "---",  # em dash
    "\u2013": "--",  # en dash
}


def results_latex_dir(base_dir: str | os.PathLike | None = None) -> Path | None:
    """Resolve the export base directory.

    Precedence: explicit ``base_dir`` argument, then the ``RESULTS_LATEX_DIR``
    environment variable, otherwise ``None`` (export disabled).
    """
    base = base_dir if base_dir is not None else os.getenv("RESULTS_LATEX_DIR")
    return Path(base) if base else None


def slugify(name: str) -> str:
    """Safe file stem: spaces/slashes -> underscores, drop odd characters."""
    slug = str(name).replace("/", "-")
    slug = re.sub(r"\s+", "_", slug.strip())
    slug = re.sub(r"[^0-9A-Za-z._-]", "", slug)
    return slug or "output"


def latex_escape(value: Any) -> Any:
    """Escape LaTeX-special characters in a string (non-strings returned as-is)."""
    if not isinstance(value, str):
        return value
    for old, new in _LATEX_REPLACEMENTS.items():
        value = value.replace(old, new)
    return value


def save_figure(
    fig,
    name: str,
    *,
    base_dir: str | os.PathLike | None = None,
    subdir: str = "plots",
    ext: str = "pdf",
    **savefig_kwargs,
) -> Path | None:
    """Save a matplotlib figure under ``<base>/<subdir>/<slug>.<ext>`` (PDF)."""
    base = results_latex_dir(base_dir)
    if base is None:
        print("RESULTS_LATEX_DIR is not set; skipping figure export.")
        return None
    out_dir = base / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(name)}.{ext}"
    savefig_kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(out_path, **savefig_kwargs)
    print(f"Saved {out_path}")
    return out_path


def save_table(
    df,
    name: str,
    *,
    base_dir: str | os.PathLike | None = None,
    subdir: str = "tables",
    escape: bool = True,
    index: bool = True,
    float_format: str | None = "%g",
    **to_latex_kwargs,
) -> Path | None:
    """Save a DataFrame as a LaTeX ``tabular`` under ``<base>/<subdir>/<slug>.tex``.

    Defaults are the "safe" ones: ``escape=True`` (escapes ``_``, ``&`` etc. in
    labels such as feature names ``jet1_btag``) and ``float_format="%g"`` (trims
    trailing zeros, respecting any prior ``.round(...)``). For tables whose cells
    are ALREADY LaTeX (e.g. bolded ``\\textbf{...}`` or hand-escaped labels), pass
    ``escape=False`` (and usually ``index=False``).
    """
    base = results_latex_dir(base_dir)
    if base is None:
        print("RESULTS_LATEX_DIR is not set; skipping table export.")
        return None
    out_dir = base / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(name)}.tex"
    df.to_latex(
        out_path,
        escape=escape,
        index=index,
        float_format=float_format,
        **to_latex_kwargs,
    )
    print(f"Saved {out_path}")
    return out_path


def format_latex_bold_best(summary_table, *, rank_col=("mean rank", "", "")):
    """Format a (mean/std/rank) summary table into LaTeX-ready strings.

    Within each config column the best ``mean`` is bolded (``\\textbf``); per-config
    ranks are integers, the overall ``rank_col`` shows 2 decimals, and all
    index/column labels are LaTeX-escaped (e.g. ``_`` in ``ii_permuted``). Expects
    the column MultiIndex whose last level is the stat name (``mean``/``std``/``rank``).
    """


def format_latex_bold_best(summary_table, *, rank_col=("mean rank", "", "")):
    """Format a (mean/std/rank) summary table into LaTeX-ready strings.

    Within each config the best row (highest ``mean``) is bolded across *all* its
    stats (``mean``, ``std`` and ``rank``); per-config ranks are integers, the
    overall ``rank_col`` shows 2 decimals, and all index/column labels are
    LaTeX-escaped (e.g. ``_`` in ``ii_permuted``). Expects the column MultiIndex
    whose last level is the stat name (``mean``/``std``/``rank``).
    """
    # Best row per config (identified by the mean column) so we can bold the
    # whole winning row, not just its mean cell.
    best_idx_by_config = {}
    for col in summary_table.columns:
        if col == rank_col or col[-1] != "mean":
            continue
        s = summary_table[col]
        if s.notna().any():
            best_idx_by_config[col[:-1]] = s.idxmax()

    def _bold(text, idx, best_idx):
        return f"\\textbf{{{text}}}" if idx == best_idx else text

    formatted = summary_table.astype(object).copy()
    for col in summary_table.columns:
        s = summary_table[col]
        if col == rank_col:
            formatted[col] = s.map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
            continue
        fmt = "{:.0f}" if col[-1] == "rank" else "{:.4f}"
        best_idx = best_idx_by_config.get(col[:-1])
        formatted[col] = [
            "" if pd.isna(v) else _bold(fmt.format(v), idx, best_idx)
            for idx, v in s.items()
        ]

    formatted.index = pd.MultiIndex.from_tuples(
        [tuple(latex_escape(part) for part in tup) for tup in formatted.index],
        names=[latex_escape(name) for name in formatted.index.names],
    )
    formatted.columns = pd.MultiIndex.from_tuples(
        [tuple(latex_escape(part) for part in col) for col in formatted.columns],
        names=[latex_escape(name) for name in formatted.columns.names],
    )
    return formatted


def _drop_constant_config_levels(summary_table, *, rank_col):
    """Drop column levels (e.g. ``cov``) that carry a single value everywhere.

    The stat level (last) is always kept; a *config* level is removed only when
    every real config shares the same value (e.g. higgs logs a single
    ``cov = 0``), because it then adds nothing but an empty header row. The
    overall ``rank_col`` label is re-homed onto the first surviving level so it
    is not lost together with a dropped leading level.

    Returns ``(trimmed_table, new_rank_col)``.
    """
    columns = summary_table.columns
    if not isinstance(columns, pd.MultiIndex) or columns.nlevels <= 2:
        return summary_table, rank_col

    stat_level = columns.nlevels - 1
    config_levels = list(range(stat_level))
    kept = [
        lvl
        for lvl in config_levels
        if len({col[lvl] for col in columns if col != rank_col}) > 1
    ]
    # Nothing constant, or everything constant (keep as-is to avoid a degenerate
    # single-level table that would strip the rank label's home).
    if not kept or len(kept) == len(config_levels):
        return summary_table, rank_col

    keep_levels = kept + [stat_level]
    new_rank_col = tuple(rank_col[lvl] for lvl in keep_levels)
    if all(part == "" for part in new_rank_col):
        label = next((part for part in rank_col if part != ""), "mean rank")
        new_rank_col = (label, *("" for _ in keep_levels[1:]))

    new_tuples = [
        new_rank_col if col == rank_col else tuple(col[lvl] for lvl in keep_levels)
        for col in columns
    ]
    trimmed = summary_table.copy()
    trimmed.columns = pd.MultiIndex.from_tuples(
        new_tuples, names=[columns.names[lvl] for lvl in keep_levels]
    )
    return trimmed, new_rank_col


def _grouped_column_format(summary_table, *, rank_col, index_ncols):
    """LaTeX ``column_format`` that puts a vertical rule between config groups.

    Index columns, each ``(cov, train_size)`` group of stat columns, and the
    trailing overall-rank column are separated by ``|`` so the wide
    mean/std/rank blocks stay visually distinct (``lll|lll|...|l``).
    """
    columns = summary_table.columns
    if not isinstance(columns, pd.MultiIndex):
        return None

    group_sizes: list[int] = []
    prev_key: object = object()
    for col in columns:
        if col == rank_col:
            continue
        key = col[:-1]
        if key != prev_key:
            group_sizes.append(0)
            prev_key = key
        group_sizes[-1] += 1

    parts = ["l" * index_ncols] if index_ncols else []
    parts += ["l" * n for n in group_sizes]
    if any(col == rank_col for col in columns):
        parts.append("l")
    return "|".join(parts)


def _pretty_level_label(name: Any) -> str:
    """Human-readable header for a column level (drops ``tags.``/``params.``)."""
    text = str(name)
    for prefix in ("tags.", "params."):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _labelled_header_frame(formatted, *, rank_col, level_labels):
    """Move the index names onto the bottom header row and label config levels.

    ``reset_index`` alone leaves the ``cov`` / ``train_size`` header rows without
    any indication of what they mean. Here the model/graph/n_edges names are
    placed on the last (``stat``) header row, and the config level names
    (``cov``, ``train_size``) become left-hand labels on their respective header
    rows, e.g.::

        cov        &   &   & \\multicolumn{3}{c}{0}    & ... &
        train_size &   &   & \\multicolumn{3}{c}{1000} & ... &
        model & graph & n_edges & mean & std & rank & ... & mean rank
    """
    if not isinstance(formatted.columns, pd.MultiIndex):
        return formatted.reset_index()

    nlevels = formatted.columns.nlevels
    index_names = list(formatted.index.names)
    reset = formatted.reset_index()

    rank_label = next((part for part in rank_col if part != ""), "")
    pad = ("",) * (nlevels - 1)

    new_columns = []
    for pos, col in enumerate(reset.columns):
        if pos < len(index_names):
            name = index_names[pos]
            # Only the first index column carries the config-level labels; the
            # index names themselves live on the last (stat) header row.
            head = tuple(level_labels) if pos == 0 else pad
            new_columns.append((*head, name))
        elif col == rank_col:
            new_columns.append((*pad, rank_label))
        else:
            new_columns.append(col)

    out = reset.copy()
    out.columns = pd.MultiIndex.from_tuples(new_columns)
    return out


def save_latex_table(
    summary_table,
    name: str,
    *,
    base_dir: str | os.PathLike | None = None,
    rank_col=("mean rank", "", ""),
    group_rules: bool = True,
    **save_kwargs,
) -> Path | None:
    """Format (bold best per config) and save a summary table as LaTeX ``tabular``.

    Cells are pre-formatted / hand-escaped, so this always writes with
    ``escape=False, index=False``.

    Column levels that only ever hold one value (e.g. ``cov = 0`` for higgs) are
    dropped so they don't add an empty header row, with ``group_rules`` a
    vertical rule is placed between each ``(cov, train_size)`` config block, and
    the ``cov`` / ``train_size`` header rows are labelled on the left so it is
    clear what the grouped values mean.
    """
    trimmed, rank_col = _drop_constant_config_levels(summary_table, rank_col=rank_col)

    if group_rules and "column_format" not in save_kwargs:
        column_format = _grouped_column_format(
            trimmed, rank_col=rank_col, index_ncols=trimmed.index.nlevels
        )
        if column_format is not None:
            save_kwargs["column_format"] = column_format
    save_kwargs.setdefault("multicolumn_format", "c")

    # Display labels for the config levels (everything but the trailing stat
    # level), e.g. ``tags.cov`` -> ``cov``.
    level_labels = [
        latex_escape(_pretty_level_label(name)) for name in trimmed.columns.names[:-1]
    ]

    formatted = format_latex_bold_best(trimmed, rank_col=rank_col)
    export = _labelled_header_frame(
        formatted, rank_col=rank_col, level_labels=level_labels
    )
    return save_table(
        export, name, base_dir=base_dir, escape=False, index=False, **save_kwargs
    )
