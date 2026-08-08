import inspect
import math
import warnings
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


def _as_hidden_dims(hidden_dim, n_layers=None):
    """Normalize a hidden-dim spec to a list of per-layer sizes.

    ``hidden_dim`` may be either an explicit list of per-layer sizes or a single
    int repeated ``n_layers`` times (useful for grid search over a width + depth
    pair instead of an explicit list).
    """
    if isinstance(hidden_dim, int):
        if n_layers is None:
            raise ValueError("n_layers must be provided when hidden_dim is an int")
        return [hidden_dim] * n_layers
    return list(hidden_dim)


class NumericEncoder(nn.Module):
    def __init__(
        self, input_dim=1, hidden_dims=None, dropout=None, batch_norm=True, output_dim=8
    ):
        super().__init__()
        dims = [input_dim]

        if hidden_dims is not None:
            if isinstance(hidden_dims, int):
                hidden_dims = [hidden_dims]
            dims = dims + hidden_dims
        dims = dims + [output_dim]

        self.mlp = MLP(dims, dropout=dropout, activation="relu", batch_norm=batch_norm)

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
        num_dropout=None,
        num_batch_norm=True,
        post_activation: Literal[None, "relu"] = None,
        post_batch_norm=False,
    ):
        super().__init__()

        if post_activation not in (None, "relu"):
            raise ValueError(f"Unsupported post_activation: {post_activation}")
        self.post_activation = post_activation

        # emb_dim=None -> passthrough: forward the raw column-0 value per node
        # (effective output dim is 1), skipping all learned embeddings.
        self.passthrough = emb_dim is None
        self.n_nodes = n_nodes
        self.emb_dim = 1 if self.passthrough else emb_dim
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

        if self.passthrough:
            self.value_embeddings = nn.ModuleDict()
            self.num_embeddings = nn.ModuleDict()
            return

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
                    input_dim=1,
                    hidden_dims=num_emb_hidden,
                    dropout=num_dropout,
                    batch_norm=num_batch_norm,
                    output_dim=emb_dim,
                )
                for idx in self.numeric_features_indexes
            }
        )

        if post_activation not in (None, "relu"):
            raise ValueError(f"Unsupported post_activation: {post_activation}")
        self.post_batch_norm = post_batch_norm
        self.post_activation = post_activation

    def forward(self, x):
        x_raw = x

        raw_value = x_raw[:, 0].float()

        if x_raw.is_cuda and torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
        else:
            target_dtype = torch.float32

        if self.passthrough:
            return raw_value.unsqueeze(1).to(target_dtype)

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

        if self.post_activation == "relu":
            x_2d = torch.relu(x_2d)
        if self.post_batch_norm:
            b, n, d = x_2d.shape
            x_2d = nn.BatchNorm1d(d)(x_2d.reshape(b * n, d)).reshape(b, n, d)
        return x_2d.reshape(batch_size * self.n_nodes, self.emb_dim)


class EncodeXVectorized(nn.Module):
    """Vectorized, drop-in equivalent of :class:`EncodeX`.

    Same constructor and ``forward`` contract as ``EncodeX`` (input
    ``[B * n_nodes, F]`` with the value in column 0, output
    ``[B * n_nodes, emb_dim]``), but the per-feature Python loops are replaced by
    batched ops, so cost no longer scales with the number of features:

    * **categorical** features share a single ``nn.Embedding`` addressed through
      per-feature offsets, keeping the same ``n_classes + 1`` vocab and the same
      clamp-based out-of-range / missing handling as ``EncodeX``;
    * **numeric** features are encoded by *grouped* per-feature linear layers
      (``einsum`` over a ``[n_num, in, out]`` weight), optionally stacked with
      per-feature BatchNorm / ReLU / dropout to mirror ``NumericEncoder``.

    """

    def __init__(
        self,
        n_nodes,
        emb_dim=8,
        num_emb_hidden=None,
        numeric_features_indexes=None,
        categorical_features_index_n_classes_map=None,
        num_dropout=None,
        num_batch_norm=True,
        post_activation: Literal[None, "relu"] = None,
        post_batch_norm=False,
    ):
        super().__init__()

        if post_activation not in (None, "relu"):
            raise ValueError(f"Unsupported post_activation: {post_activation}")
        self.post_activation = post_activation
        self.post_batch_norm = post_batch_norm
        # emb_dim=None -> passthrough: forward the raw column-0 value per node
        # (effective output dim is 1), skipping all learned embeddings.
        self.passthrough = emb_dim is None
        self.n_nodes = n_nodes
        self.emb_dim = 1 if self.passthrough else emb_dim
        self.categorical_features_index_n_classes_map = (
            categorical_features_index_n_classes_map or {}
        )
        self.categorical_features_indexes = list(
            self.categorical_features_index_n_classes_map.keys()
        )
        if numeric_features_indexes is not None:
            self.numeric_features_indexes = list(numeric_features_indexes)
        else:
            self.numeric_features_indexes = list(
                set(range(n_nodes)) - set(self.categorical_features_indexes)
            )

        if self.passthrough:
            if self.categorical_features_indexes:
                warnings.warn(
                    "EncodeXVectorized passthrough (emb_dim=None): forwarding raw "
                    "column-0 values for all nodes, categorical features included "
                    "(their integer codes are passed through unchanged).",
                    stacklevel=2,
                )
            self.has_categorical = False
            self.has_numeric = False
            return

        # --- Categorical: one shared table + per-feature offsets. ------------
        self.has_categorical = len(self.categorical_features_indexes) > 0
        if self.has_categorical:
            # Vocab per feature matches EncodeX (n_classes + 1 for the OOV slot).
            vocab = [
                self.categorical_features_index_n_classes_map[idx] + 1
                for idx in self.categorical_features_indexes
            ]
            offsets = torch.tensor(
                [0] + list(torch.cumsum(torch.tensor(vocab[:-1]), dim=0)),
                dtype=torch.long,
            )
            self.cat_embedding = nn.Embedding(sum(vocab), emb_dim)
            self.register_buffer("cat_offsets", offsets)
            self.register_buffer("cat_vocab", torch.tensor(vocab, dtype=torch.long))
            self.register_buffer(
                "cat_cols",
                torch.tensor(self.categorical_features_indexes, dtype=torch.long),
            )

        # --- Numeric: grouped per-feature MLP (Linear(1 -> emb) by default). --
        self.has_numeric = len(self.numeric_features_indexes) > 0
        self.num_dropout_p = num_dropout
        if self.has_numeric:
            n_num = len(self.numeric_features_indexes)
            if num_emb_hidden is None:
                hidden = []
            elif isinstance(num_emb_hidden, int):
                hidden = [num_emb_hidden]
            else:
                hidden = list(num_emb_hidden)
            dims = [1] + hidden + [emb_dim]

            self.num_weights = nn.ParameterList()
            self.num_biases = nn.ParameterList()
            self.num_norms = nn.ModuleList()
            for i in range(len(dims) - 1):
                in_d, out_d = dims[i], dims[i + 1]
                weight = nn.Parameter(torch.empty(n_num, in_d, out_d))
                bias = nn.Parameter(torch.empty(n_num, out_d))
                # Match nn.Linear's default init: U(-1/sqrt(fan_in), ...).
                bound = 1.0 / math.sqrt(in_d)
                nn.init.uniform_(weight, -bound, bound)
                nn.init.uniform_(bias, -bound, bound)
                self.num_weights.append(weight)
                self.num_biases.append(bias)
                # BatchNorm / activation / dropout only between hidden layers,
                # never after the final projection (mirrors NumericEncoder/MLP).
                is_hidden = i < len(dims) - 2
                if is_hidden and num_batch_norm:
                    self.num_norms.append(nn.BatchNorm1d(n_num * out_d))
                else:
                    self.num_norms.append(nn.Identity())

            self.num_dropout = (
                nn.Dropout(num_dropout) if num_dropout is not None else None
            )
            self.register_buffer(
                "num_cols",
                torch.tensor(self.numeric_features_indexes, dtype=torch.long),
            )

        # Kept for interface parity with EncodeX; not applied (as in EncodeX,
        # whose forward leaves the LayerNorm commented out).

    def _encode_numeric(self, h):
        """Grouped per-feature MLP. ``h``: [B, n_num, 1] -> [B, n_num, emb]."""
        n_layers = len(self.num_weights)
        for i in range(n_layers):
            # Per-feature affine: each feature f uses its own [in, out] weight.
            h = (
                torch.einsum("bfi,fio->bfo", h, self.num_weights[i])
                + self.num_biases[i]
            )
            if i < n_layers - 1:
                b, f, o = h.shape
                # BatchNorm1d(n_num * out) == n_num independent BatchNorm1d(out).
                h = self.num_norms[i](h.reshape(b, f * o)).reshape(b, f, o)
                h = torch.relu(h)
                if self.num_dropout is not None:
                    h = self.num_dropout(h)
        return h

    def forward(self, x):
        raw_value = x[:, 0].float()

        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
        else:
            target_dtype = torch.float32

        if self.passthrough:
            return raw_value.unsqueeze(1).to(target_dtype)

        batch_size = raw_value.size(0) // self.n_nodes
        raw_2d = raw_value.view(batch_size, self.n_nodes)

        out = torch.zeros(
            batch_size,
            self.n_nodes,
            self.emb_dim,
            device=x.device,
            dtype=target_dtype,
        )

        if self.has_categorical:
            cat_vals = raw_2d[:, self.cat_cols].long()
            cat_vals = cat_vals.clamp(min=0)
            cat_vals = torch.minimum(cat_vals, self.cat_vocab - 1)
            cat_emb = self.cat_embedding(cat_vals + self.cat_offsets)
            out[:, self.cat_cols, :] = cat_emb.to(target_dtype)

        if self.has_numeric:
            num_vals = raw_2d[:, self.num_cols].unsqueeze(-1)
            num_emb = self._encode_numeric(num_vals)
            out[:, self.num_cols, :] = num_emb.to(target_dtype)

        if self.post_activation == "relu":
            out = torch.relu(out)
        if self.post_batch_norm:
            b, n, d = out.shape
            out = nn.BatchNorm1d(d)(out.reshape(b * n, d)).reshape(b, n, d)

        return out.reshape(batch_size * self.n_nodes, self.emb_dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        dropout=0.3,
        activation="relu",
        batch_norm=True,
    ):
        super().__init__()

        if activation == "relu":
            activation_layer = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        for i, (in_features, out_features) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_features, out_features))
            if i < len(dims) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(out_features))
                layers.append(activation_layer())
                if dropout is not None:
                    layers.append(nn.Dropout(dropout))

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
        self.add_skip = add_skip

        if conv_layer == GATConv and heads is not None:
            self.conv = conv_layer(
                in_channels,
                out_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
            )
            conv_out_channels = out_channels
        else:
            self.conv = conv_layer(in_channels, out_channels)
            conv_out_channels = out_channels

        self.batch_norm = nn.BatchNorm1d(conv_out_channels) if batch_norm else None

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.add_skip and in_channels != conv_out_channels:
            self.skip_proj = nn.Linear(in_channels, conv_out_channels)
        else:
            self.skip_proj = None

    def forward(self, x, edge_index):
        x_skip = x

        x = self.conv(x, edge_index)

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.relu(x)
        x = self.dropout(x)

        if self.add_skip:
            if self.skip_proj is not None:
                x_skip = self.skip_proj(x_skip)
            x = x + x_skip

        return x


class GNN(nn.Module):
    def __init__(
        self,
        n_nodes,
        emb_dim=8,
        hidden_dim=[8],
        n_layers=None,
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

        hidden_dim = _as_hidden_dims(hidden_dim, n_layers)

        if add_skip:
            for d in hidden_dim:
                if d != emb_dim:
                    raise ValueError(
                        "For add_skip=True, all hidden_dim values must be equal to emb_dim"
                    )

        self.heads = heads if conv_layer == "GATConv" and heads is not None else 1

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
        emb_dim=8,
        hidden_dim=8,
        n_layers=1,
        mlp_hidden_dim=None,
        n_mlp_layers=1,
        dropout=0.3,
        numeric_features_indexes=None,
        categorical_features_index_n_classes_map=dict(),
        num_emb_hidden=None,
        add_skip=False,
        batch_norm=True,
        heads=None,
        conv_layer: Literal["GraphConv", "SAGEConv", "GATConv"] = "GraphConv",
        pooling_type: Literal["mean", "max", "concat"] = "mean",
        vectorized_encoder=True,
        encoder_post_activation: Literal[None, "relu"] = None,
    ):
        super().__init__()

        hidden_dim = _as_hidden_dims(hidden_dim, n_layers)

        if mlp_hidden_dim is None:
            mlp_hidden_dim = []
        mlp_hidden_dim = _as_hidden_dims(mlp_hidden_dim, n_mlp_layers)

        # emb_dim=None -> passthrough encoder (raw values); GNN sees 1 input dim.
        gnn_in_dim = 1 if emb_dim is None else emb_dim

        self.heads = heads if conv_layer == "GATConv" and heads is not None else 1

        encoder_cls = EncodeXVectorized if vectorized_encoder else EncodeX
        self.encode_x = encoder_cls(
            n_nodes=n_nodes,
            emb_dim=emb_dim,
            numeric_features_indexes=numeric_features_indexes,
            categorical_features_index_n_classes_map=categorical_features_index_n_classes_map,
            num_emb_hidden=num_emb_hidden,
            post_activation=encoder_post_activation,
        )

        self.GNN = GNN(
            n_nodes=n_nodes,
            emb_dim=gnn_in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            add_skip=add_skip,
            batch_norm=batch_norm,
            conv_layer=conv_layer,
            heads=self.heads,
        )

        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes

        if pooling_type == "mean":
            self.pooling = global_mean_pool
        elif pooling_type == "max":
            self.pooling = global_max_pool
        elif pooling_type == "concat":
            self.pooling = None
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")

        if self.pooling is None:
            mlp_hidden_dims = [n_nodes * hidden_dim[-1]] + mlp_hidden_dim + [n_classes]
        else:
            mlp_hidden_dims = [hidden_dim[-1]] + mlp_hidden_dim + [n_classes]
        self.mlp = MLP(mlp_hidden_dims, dropout)

    def forward(self, x, edge_index=None, batch=None):
        if batch is not None:
            # batch_size = int(batch.max().item()) + 1
            batch_size = batch.max().item() + 1
        else:
            batch_size = x.size(0) // self.n_nodes

        x = self.encode_x(x)
        x = self.GNN(x, edge_index, batch)

        if batch is None:
            batch = torch.arange(batch_size, device=x.device).repeat_interleave(
                self.n_nodes
            )

        if self.pooling is None:
            x = x.view(batch_size, self.n_nodes, -1)
            x = x.reshape(batch_size, self.n_nodes * x.size(-1))
        else:
            x = self.pooling(x, batch)

        return self.mlp(x)


class MyGNNConcat(MyGNN):
    """`MyGNN` with concat pooling fixed as a model variant.

    Pins ``pooling_type="concat"`` so pooling is chosen by picking a class
    (a distinct ``__name__`` for study naming / tagging) instead of being an
    Optuna hyperparameter.
    """

    def __init__(self, *args, **kwargs):
        kwargs["pooling_type"] = "concat"
        super().__init__(*args, **kwargs)

    # Expose MyGNN's real parameters so callers that filter kwargs by
    # ``inspect.signature(model_cls.__init__)`` (e.g. training.build_*) still
    # see n_nodes and friends instead of just (*args, **kwargs).
    __init__.__signature__ = inspect.signature(MyGNN.__init__)


class MyMLP(nn.Module):
    def __init__(
        self,
        n_nodes,
        n_classes=2,
        hidden_dim=16,
        n_layers=1,
        emb_dim=None,
        dropout=0.3,
        numeric_features_indexes=None,
        categorical_features_index_n_classes_map=None,
        num_emb_hidden=None,
        vectorized_encoder=True,
        encoder_post_activation: Literal[None, "relu"] = None,
    ):
        super().__init__()

        if categorical_features_index_n_classes_map is None:
            categorical_features_index_n_classes_map = dict()

        hidden_dim = _as_hidden_dims(hidden_dim, n_layers)

        encoder_cls = EncodeXVectorized if vectorized_encoder else EncodeX
        self.encode_x = encoder_cls(
            n_nodes=n_nodes,
            emb_dim=emb_dim,
            numeric_features_indexes=numeric_features_indexes,
            categorical_features_index_n_classes_map=categorical_features_index_n_classes_map,
            num_emb_hidden=num_emb_hidden,
            post_activation=encoder_post_activation,
        )

        # emb_dim=None -> passthrough encoder (raw values); effective dim is 1.
        self.emb_dim = 1 if emb_dim is None else emb_dim
        self.n_nodes = n_nodes

        mlp_hidden_dims = [n_nodes * self.emb_dim] + hidden_dim + [n_classes]
        self.mlp = MLP(mlp_hidden_dims, dropout)

    def forward(self, x, edge_index=None, batch=None):
        if batch is not None:
            batch_size = int(batch.max().item()) + 1
        else:
            batch_size = x.size(0) // self.n_nodes

        x = self.encode_x(x)

        x = x.reshape(batch_size, self.n_nodes * self.emb_dim)

        return self.mlp(x)
