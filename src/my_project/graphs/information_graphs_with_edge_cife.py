"""Edge-level information-theoretic feature-pair selection.

A small selector that lifts the classic single-feature information criteria
(CIFE / JMI / MRMR / MIFS) from features to feature-pair *edges*.  Every feature
stays a node; each candidate edge ``(X_k, X_l)`` becomes a single discrete
*meta-feature* (the joint code of the pair), and the greedy criterion selection
is delegated to the ready-made :mod:`skfeature` ``LCSI`` core — so this module
only prepares the meta-features and assembles the resulting graph.

The criterion controls the redundancy term exactly as in Brown et al. (2012):

==========  ====================================  =================
criterion   redundancy term                       aggregation
==========  ====================================  =================
``mifs``    ``I(A;B)``                             ``beta`` (const.)
``mrmr``    ``I(A;B)``                             mean (``1/|S|``)
``cife``    ``I(A;B) - I(A;B|y)``                  sum
``jmi``     ``I(A;B) - I(A;B|y)``                  mean (``1/|S|``)
==========  ====================================  =================
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from skfeature.function.information_theoretical_based import LCSI

from my_project.graphs.base import GraphBuilder
from my_project.information_theory.estimators import _mi_pair, encode_features
from my_project.information_theory.matrices import compute_information_matrices

Edge = tuple[str, str]
# Greedy, redundancy-aware criteria delegated to skfeature's LCSI.
LCSI_CRITERIA = ("cife", "jmi", "mrmr", "mifs")
# Plain top-k rankings of a single pairwise measure (no redundancy term).
RANKING_CRITERIA = ("joint_mi", "interaction")
EDGE_CRITERIA = LCSI_CRITERIA + RANKING_CRITERIA


def _lcsi_kwargs(criterion: str, beta: float | None) -> dict[str, object]:
    """Map an edge criterion to the ``LCSI.lcsi`` parameters (Brown et al. 2012)."""
    if criterion == "cife":
        return {"beta": 1, "gamma": 1}
    if criterion == "mifs":
        return {"beta": 0.5 if beta is None else beta, "gamma": 0}
    if criterion == "mrmr":
        return {"gamma": 0, "function_name": "MRMR"}
    if criterion == "jmi":
        return {"function_name": "JMI"}
    raise ValueError(f"criterion must be one of {EDGE_CRITERIA}, got {criterion!r}.")


def _edge_meta_feature(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Encode a feature pair as a single dense discrete meta-feature."""
    cardinality = int(right.max()) + 1
    combined = left.astype(np.int64) * cardinality + right.astype(np.int64)
    return pd.factorize(combined)[0]


class EdgeInteractionGraphBuilder(GraphBuilder):
    """Edge selection by an information-theoretic criterion.

    Every feature is a node.  Candidate edges (feature pairs) are scored and
    selected by ``criterion``, which comes in two flavours:

    * **Greedy, redundancy-aware** (``"cife"``, ``"jmi"``, ``"mrmr"``,
      ``"mifs"``): each edge becomes a discrete meta-feature and the greedy
      selection is delegated to :mod:`skfeature`'s ``LCSI``.  The relevance term
      is the pair's joint mutual information ``I((X_k, X_l); y)`` and the
      criterion fixes how the edge-edge redundancy is formed and aggregated.
    * **Plain top-k rankings** (no redundancy term): ``"joint_mi"`` keeps the
      ``n_edges`` pairs with the highest joint mutual information
      ``I((X_k, X_l); y)``, and ``"interaction"`` keeps the highest interaction
      information ``II(X_k; X_l; y)``.

    Parameters
    ----------
    n_edges:
        Maximum number of edges to select.
    criterion:
        One of ``"cife"`` (default), ``"jmi"``, ``"mrmr"``, ``"mifs"``,
        ``"joint_mi"`` or ``"interaction"``.
    beta:
        Redundancy weight for ``"mifs"`` only (``skfeature`` default ``0.5``).
        Ignored by the other criteria, which derive their weight internally.
    positive_only:
        Restrict the candidate pool to edges with positive interaction
        information (synergistic pairs under the pyitlib convention).
    encode, n_bins, estimator, n_jobs:
        Discrete-data preparation / estimator forwarded to
        :func:`compute_information_matrices`.
    interaction_matrix:
        Optional precomputed interaction-information matrix (same feature names
        as ``X``) reused instead of recomputing it.

    Fitted attributes
    -----------------
    interaction_matrix_:
        Symmetric interaction-information matrix.
    joint_mi_matrix_:
        Symmetric joint target mutual-information matrix ``I((X_k, X_l); y)``.
    node_target_mi_:
        ``I(X_k; y)`` per feature, attached to graph nodes as ``target_mi``.
    selected_edges_:
        Selected edges in greedy order.
    selection_history_:
        ``DataFrame`` with columns ``step``, ``source``, ``target``,
        ``interaction_information``, ``relevance`` (``I(pair; y)``) and
        ``objective`` (the criterion value at selection time).
    """

    def __init__(
        self,
        n_edges: int = 20,
        *,
        criterion: str = "cife",
        beta: float | None = None,
        positive_only: bool = True,
        encode: bool = True,
        n_bins: int | None = None,
        estimator: str = "ML",
        interaction_matrix: pd.DataFrame | None = None,
        n_jobs: int | None = None,
    ):
        if n_edges < 0:
            raise ValueError("n_edges must be non-negative.")
        if criterion not in EDGE_CRITERIA:
            raise ValueError(
                f"criterion must be one of {EDGE_CRITERIA}, got {criterion!r}."
            )
        self.n_edges = n_edges
        self.criterion = criterion
        self.beta = beta
        self.positive_only = positive_only
        self.encode = encode
        self.n_bins = n_bins
        self.estimator = estimator
        self.interaction_matrix = interaction_matrix
        self.n_jobs = n_jobs

    def _discretize(self, X: pd.DataFrame) -> pd.DataFrame:
        """Bin and ordinal-encode features into integer codes."""
        prepared = X
        if self.n_bins:
            prepared = prepared.apply(pd.qcut, q=self.n_bins, axis=0, labels=False)
        if self.encode:
            prepared = encode_features(prepared)
        if prepared.isna().any().any():
            raise ValueError("Discretized features still contain missing values.")
        return prepared.astype(int)

    def _candidate_edges(self) -> list[Edge]:
        columns = self.feature_names_
        candidates: list[Edge] = []
        for i, left in enumerate(columns):
            for right in columns[i + 1 :]:
                score = float(self.interaction_matrix_.loc[left, right])
                if not np.isfinite(score):
                    continue
                if self.positive_only and score <= 0.0:
                    continue
                candidates.append((left, right))
        return candidates

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> EdgeInteractionGraphBuilder:
        """Discretize, rank edges with the chosen criterion and store the order."""
        if X is None or y is None:
            raise ValueError("Edge selection requires both X and y.")
        if X.shape[1] < 2:
            raise ValueError("Edge selection requires at least two feature columns.")
        if not X.columns.is_unique:
            raise ValueError("X must have unique column names.")

        self.feature_names_ = list(X.columns)
        discrete = self._discretize(X)
        y_codes = np.asarray(pd.factorize(y)[0])
        arrays = {col: discrete[col].to_numpy() for col in self.feature_names_}

        # Only compute the matrices that are actually needed: the joint MI is
        # used exclusively by the ranking criteria, and the interaction matrix
        # is skipped entirely when a precomputed one is supplied -- so passing
        # ``interaction_matrix`` for a greedy criterion recomputes nothing.
        needs_joint = self.criterion in RANKING_CRITERIA
        needs_interaction = self.interaction_matrix is None
        measures = tuple(
            measure
            for measure, needed in (
                ("joint_target_mi", needs_joint),
                ("interaction_information", needs_interaction),
            )
            if needed
        )
        matrices = (
            compute_information_matrices(
                discrete,
                pd.Series(y_codes, index=discrete.index),
                measures=measures,
                encode=False,
                estimator=self.estimator,
                n_jobs=self.n_jobs,
            )
            if measures
            else {}
        )
        self.joint_mi_matrix_ = matrices.get("joint_target_mi")
        if self.interaction_matrix is None:
            self.interaction_matrix_ = matrices["interaction_information"]
        else:
            self.interaction_matrix_ = self.interaction_matrix.loc[
                self.feature_names_, self.feature_names_
            ].astype(float)

        self.node_target_mi_ = pd.Series(
            {
                col: _mi_pair(arrays[col], y_codes, estimator=self.estimator)
                for col in self.feature_names_
            },
            name="target_mi",
            dtype=float,
        )

        candidates = self._candidate_edges()
        n_select = min(self.n_edges, len(candidates))
        history: list[dict[str, object]] = []

        if n_select > 0 and self.criterion in RANKING_CRITERIA:
            score_matrix = (
                self.joint_mi_matrix_
                if self.criterion == "joint_mi"
                else self.interaction_matrix_
            )
            ranked = sorted(
                candidates,
                key=lambda edge: -float(score_matrix.loc[edge[0], edge[1]]),
            )[:n_select]
            for step, (left, right) in enumerate(ranked, start=1):
                joint = float(self.joint_mi_matrix_.loc[left, right])
                interaction = float(self.interaction_matrix_.loc[left, right])
                history.append(
                    {
                        "step": step,
                        "source": left,
                        "target": right,
                        "interaction_information": interaction,
                        "relevance": joint,
                        "objective": joint
                        if self.criterion == "joint_mi"
                        else interaction,
                    }
                )
        elif n_select > 0:
            meta = np.column_stack(
                [
                    _edge_meta_feature(arrays[left], arrays[right])
                    for left, right in candidates
                ]
            )
            order, objective, relevance = LCSI.lcsi(
                meta,
                y_codes,
                n_selected_features=n_select,
                **_lcsi_kwargs(self.criterion, self.beta),
            )
            for step, idx in enumerate(order, start=1):
                left, right = candidates[int(idx)]
                history.append(
                    {
                        "step": step,
                        "source": left,
                        "target": right,
                        "interaction_information": float(
                            self.interaction_matrix_.loc[left, right]
                        ),
                        "relevance": float(relevance[step - 1]),
                        "objective": float(objective[step - 1]),
                    }
                )

        self.selected_edges_ = [(row["source"], row["target"]) for row in history]
        self.selection_history_ = pd.DataFrame(
            history,
            columns=[
                "step",
                "source",
                "target",
                "interaction_information",
                "relevance",
                "objective",
            ],
        )
        return self

    def graph_at(self, n_edges: int | None = None) -> nx.Graph:
        """Build a graph from the first ``n_edges`` of the greedy order."""
        if not hasattr(self, "selected_edges_"):
            raise RuntimeError("Call fit before graph_at.")
        if n_edges is None:
            n_edges = len(self.selected_edges_)
        if n_edges < 0 or n_edges > len(self.selected_edges_):
            raise ValueError(
                f"n_edges must be in [0, {len(self.selected_edges_)}], got {n_edges}."
            )

        graph = nx.Graph()
        graph.add_nodes_from(
            (node, {"target_mi": float(self.node_target_mi_.loc[node])})
            for node in self.feature_names_
        )

        for row in self.selection_history_.iloc[:n_edges].itertuples(index=False):
            graph.add_edge(
                row.source,
                row.target,
                weight=float(row.interaction_information),
                interaction_information=float(row.interaction_information),
                relevance=float(row.relevance),
                objective=float(row.objective),
                selection_step=int(row.step),
            )
        return graph

    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        n_edges: int | None = None,
    ) -> nx.Graph:
        """Fit the selector and return the selected feature graph."""
        self.fit(X, y)
        return self.graph_at(n_edges)
