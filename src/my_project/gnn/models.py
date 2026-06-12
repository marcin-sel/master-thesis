from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import (
    FAConv,
    GATConv,
    GraphConv,
    SAGEConv,
    TransformerConv,
    global_max_pool,
    global_mean_pool,
)

TRANSFORMERS_DICT = {
    "GraphConv": GraphConv,
    "SAGEConv": SAGEConv,
    "GATConv": GATConv,
    "FAConv": FAConv,
    "TransformerConv": TransformerConv,
}


class NumericEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=None, output_dim=8):
        super().__init__()
        dims = [input_dim]
        if hidden_dim is not None:
            dims = dims + [hidden_dim]
        dims = dims + [output_dim]

        self.mlp = nn.Sequential()
        for in_channels, out_channels in zip(dims[:-1], dims[1:]):
            self.mlp.append(nn.Linear(in_channels, out_channels))
            self.mlp.append(nn.ReLU())

    def forward(self, x):
        return self.mlp(x)


class EncodeX(nn.Module):
    def __init__(
        self,
        n_nodes,
        emb_dim=8,
        num_emb_hidden=None,
        numeric_features_indexes=None,
        categorical_features_index_n_classes_map=None,
    ):
        super().__init__()

        self.n_nodes = n_nodes
        self.emb_dim = emb_dim
        self.categorical_features_index_n_classes_map = (
            categorical_features_index_n_classes_map or {}
        )
        self.categorical_features_indexes = list(
            self.categorical_features_index_n_classes_map.keys()
        )
        if numeric_features_indexes is not None:
            self.numeric_features_indexes = numeric_features_indexes
        else:
            self.numeric_features_indexes = list(
                set(range(n_nodes)) - set(self.categorical_features_indexes)
            )

        self.categorical_features_indexes_map = {
            idx: i for i, idx in enumerate(self.categorical_features_indexes)
        }
        self.numeric_features_indexes_map = {
            idx: i for i, idx in enumerate(self.numeric_features_indexes)
        }

        self.value_embeddings = nn.ModuleDict(
            {
                str(idx): nn.Embedding(
                    self.categorical_features_index_n_classes_map[idx] + 1, emb_dim
                )
                for idx in self.categorical_features_indexes
            }
        )

        self.num_embeddings = nn.ModuleDict(
            {
                str(idx): NumericEncoder(
                    input_dim=1, hidden_dim=num_emb_hidden, output_dim=emb_dim
                )
                for idx in self.numeric_features_indexes
            }
        )

    def forward(self, x):
        x_raw = x

        raw_value = x_raw[:, 0].float()

        if x_raw.is_cuda and torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
        else:
            target_dtype = torch.float32

        x = torch.zeros(
            (x_raw.size(0), self.emb_dim),
            device=x_raw.device,
            dtype=target_dtype,
        )

        batch_size = x_raw.size(0) // self.n_nodes

        raw_value_2d = raw_value.view(batch_size, self.n_nodes)
        x_2d = x.view(batch_size, self.n_nodes, self.emb_dim)

        for node_idx in self.categorical_features_indexes:
            vocab_size = self.value_embeddings[str(node_idx)].num_embeddings
            node_values = raw_value_2d[:, node_idx].long().clamp(0, vocab_size - 1)
            x_2d[:, node_idx, :] = self.value_embeddings[str(node_idx)](node_values).to(
                dtype=x.dtype
            )

        for node_idx in self.numeric_features_indexes:
            node_values = raw_value_2d[:, node_idx].unsqueeze(1)
            x_2d[:, node_idx, :] = self.num_embeddings[str(node_idx)](node_values).to(
                dtype=x.dtype
            )

        return x


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        dropout=0.3,
        activation="relu",
        negative_slope=0.01,
        batch_norm=True,
    ):
        super().__init__()

        if activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "leaky_relu":
            activation_layer = lambda: nn.LeakyReLU(negative_slope=negative_slope)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        for i, (in_features, out_features) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_features, out_features))
            if i < len(dims) - 2:
                layers.append(activation_layer())
                layers.append(nn.Dropout(dropout))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(out_features))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class SingleConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_layer=SAGEConv,
        dropout=0.3,
        add_skip=False,
        batch_norm=True,
        heads=None,
    ):
        super().__init__()

        self.heads = heads

        if conv_layer == GATConv and heads is not None:
            self.conv = conv_layer(in_channels, out_channels, heads=heads)
        else:
            self.conv = conv_layer(in_channels, out_channels)

        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.add_skip = add_skip

    def forward(self, x, edge_index):
        if self.add_skip:
            x_skip = x
        x = self.conv(x, edge_index)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        if self.add_skip:
            x = x + x_skip
        return x


class GNN(nn.Module):
    def __init__(
        self,
        n_nodes,
        emb_dim=8,
        hidden_dim=[8],
        dropout=0.3,
        heads=1,
        categorical_features_index_n_classes_map=dict(),
        add_skip=False,
        batch_norm=True,
        conv_layer: Literal[
            "GraphConv", "SAGEConv", "GATConv", "FAConv", "TransformerConv"
        ] = "GraphConv",
    ):
        super().__init__()

        if add_skip:
            for d in hidden_dim:
                if d != emb_dim:
                    raise ValueError(
                        "For add_skip=True, all hidden_dim values must be equal to emb_dim"
                    )

        self.heads = heads if conv_layer == "GATConv" else None

        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.categorical_features_index_n_classes_map = (
            categorical_features_index_n_classes_map
        )

        if conv_layer in TRANSFORMERS_DICT:
            conv_layer = TRANSFORMERS_DICT[conv_layer]
        else:
            raise ValueError(f"Unsupported conv_layer: {conv_layer}")

        hidden_dims = [emb_dim] + hidden_dim
        self.conv_layers = nn.Sequential()
        for in_channels, out_channels in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.conv_layers.append(
                SingleConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv_layer=conv_layer,
                    dropout=dropout,
                    add_skip=add_skip,
                    batch_norm=batch_norm,
                    heads=self.heads,
                )
            )

    def forward(self, x, edge_index=None, batch=None):
        for conv in self.conv_layers:
            x = conv(x, edge_index)

        return x


class MyGNN(nn.Module):
    def __init__(
        self,
        n_nodes,
        n_classes=2,
        mlp_hidden_dim=[16],
        emb_dim=8,
        conv_hidden_dim=[8],
        dropout=0.3,
        numeric_features_indexes=None,
        categorical_features_index_n_classes_map=dict(),
        num_emb_hidden=None,
        add_skip=False,
        batch_norm=True,
        heads=None,
        conv_layer: Literal["GraphConv", "SAGEConv", "GATConv"] = "GraphConv",
        pooling_type: Literal["mean", "max"] = "mean",
    ):
        super().__init__()

        self.heads = heads if conv_layer == "GATConv" else None

        self.encode_x = EncodeX(
            n_nodes=n_nodes,
            emb_dim=emb_dim,
            numeric_features_indexes=numeric_features_indexes,
            categorical_features_index_n_classes_map=categorical_features_index_n_classes_map,
            num_emb_hidden=num_emb_hidden,
        )

        self.pre_conv_dropout = nn.Dropout(dropout)

        self.GNN = GNN(
            n_nodes=n_nodes,
            emb_dim=emb_dim,
            hidden_dim=conv_hidden_dim,
            dropout=dropout,
            add_skip=add_skip,
            batch_norm=batch_norm,
            conv_layer=conv_layer,
            heads=self.heads,
        )

        self.conv_hidden_dim = conv_hidden_dim
        self.n_nodes = n_nodes

        mlp_hidden_dims = [conv_hidden_dim[-1]] + mlp_hidden_dim + [n_classes]
        self.mlp = MLP(mlp_hidden_dims, dropout)

        if pooling_type == "mean":
            self.pooling = global_mean_pool
        elif pooling_type == "max":
            self.pooling = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")

    def forward(self, x, edge_index=None, batch=None):
        if batch is not None:
            # batch_size = int(batch.max().item()) + 1
            batch_size = batch.max().item() + 1
        else:
            batch_size = x.size(0) // self.n_nodes

        x = self.encode_x(x)
        x = self.pre_conv_dropout(x)
        x = self.GNN(x, edge_index, batch)

        if batch is None:
            batch = torch.arange(batch_size, device=x.device).repeat_interleave(
                self.n_nodes
            )

        x = self.pooling(x, batch)

        return self.mlp(x)


class MyMLP(nn.Module):
    def __init__(
        self,
        n_nodes,
        n_classes=2,
        mlp_hidden_dim=[16],
        emb_dim=8,
        dropout=0.3,
        numeric_features_indexes=None,
        categorical_features_index_n_classes_map=None,
        num_emb_hidden=None,
    ):
        super().__init__()

        if categorical_features_index_n_classes_map is None:
            categorical_features_index_n_classes_map = dict()

        self.encode_x = EncodeX(
            n_nodes=n_nodes,
            emb_dim=emb_dim,
            numeric_features_indexes=numeric_features_indexes,
            categorical_features_index_n_classes_map=categorical_features_index_n_classes_map,
            num_emb_hidden=num_emb_hidden,
        )

        self.emb_dim = emb_dim
        self.n_nodes = n_nodes

        mlp_hidden_dims = [n_nodes * emb_dim] + mlp_hidden_dim + [n_classes]
        self.mlp = MLP(mlp_hidden_dims, dropout)

    def forward(self, x, edge_index=None, batch=None):
        if batch is not None:
            batch_size = int(batch.max().item()) + 1
        else:
            batch_size = x.size(0) // self.n_nodes

        x = self.encode_x(x)

        x = x.reshape(batch_size, self.n_nodes * self.emb_dim)

        return self.mlp(x)
