import torch
from torch_geometric.data import Data


def build_graph_dataset(X, y, edges_tensor):
    y_arr = y.astype(int).values
    node_ids = torch.arange(X.shape[1], dtype=torch.float)
    graph_ids = X.index.to_numpy()

    return [
        Data(
            x=torch.stack(
                [
                    torch.tensor(X.iloc[i].values, dtype=torch.float),
                    node_ids.clone(),
                ],
                dim=1,
            ),
            edge_index=edges_tensor.clone(),
            y=torch.tensor(y_arr[i], dtype=torch.long),
            graph_id=torch.tensor([int(graph_ids[i])], dtype=torch.long),
        )
        for i in range(len(X))
    ]


from typing import Optional

import networkx as nx
import torch
from torch_geometric.utils import add_self_loops, to_undirected


def prepare_graph(
    graph,
    n_nodes: Optional[int] = None,
    undirected: bool = True,
    self_loops: bool = True,
):
    if n_nodes is None:
        n_nodes = len(graph.nodes())

    nodes_index_dict = dict(zip(graph.nodes(), range(len(graph.nodes()))))
    index_nodes_dict = {idx: node for node, idx in nodes_index_dict.items()}
    index_nodes_dict = dict(sorted(index_nodes_dict.items(), key=lambda item: item[0]))

    edges = [
        [nodes_index_dict[u], nodes_index_dict[v]]
        for u, v in graph.edges()
        if u in nodes_index_dict and v in nodes_index_dict
    ]

    edges_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()

    if len(edges_tensor) > 0:
        if undirected:
            edges_tensor = to_undirected(edges_tensor, num_nodes=n_nodes)

    if self_loops:
        edges_tensor, _ = add_self_loops(edges_tensor, num_nodes=n_nodes)

    return edges_tensor, nodes_index_dict, index_nodes_dict


def remove_node_with_descendents(var, G):
    G2 = G.copy()
    nodes_to_remove = nx.descendants(G2, var) | {var}
    G2.remove_nodes_from(nodes_to_remove)
    return G2


def remove_nodes(nodes_to_remove, G):
    G2 = G.copy()
    G2.remove_nodes_from(nodes_to_remove)
    return G2


def categorical_features_setup(
    X,
    categorical_features=[],
):
    categorical_features_indexes = [
        idx
        for idx, node_name in enumerate(X.columns)
        if node_name in categorical_features
    ]

    categorical_features_n_classes = dict(
        zip(
            categorical_features_indexes,
            X.iloc[:, categorical_features_indexes].max().astype(int),
        )
    ).copy()

    return categorical_features_indexes, categorical_features_n_classes


def x_to_graph(graph, row):
    graph_i = graph.copy()
    graph_i_simple = nx.Graph()
    graph_i_simple.add_nodes_from(graph_i.nodes())
    graph_i_simple.add_edges_from(graph_i.edges())

    for node in graph_i_simple.nodes:
        graph_i_simple.nodes[node]["value"] = row[node]

    return graph_i_simple
