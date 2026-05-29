from __future__ import annotations

from typing import Sequence

import lightning as L
from my_project.gnn.utils import build_graph_dataset, prepare_graph
from torch_geometric.loader import DataLoader


class GNNDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        graph,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test=None,
        y_test=None,
        batch_size: int = 32,
        num_workers: int = 0,
        categorical_features: Sequence[str] | None = None,
    ):
        super().__init__()
        self.graph = graph
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = (None,)

    def setup(self, stage: str | None = None) -> None:
        columns = list(self.graph.nodes)

        self.X_train = self.X_train[columns]
        self.X_valid = self.X_valid[columns]
        if self.X_test is not None:
            self.X_test = self.X_test[columns]

        edge_index, _, _ = prepare_graph(self.graph)

        self.train_dataset = build_graph_dataset(self.X_train, self.y_train, edge_index)
        self.valid_dataset = build_graph_dataset(self.X_valid, self.y_valid, edge_index)
        if self.X_test is not None and self.y_test is not None:
            self.test_dataset = build_graph_dataset(
                self.X_test, self.y_test, edge_index
            )
        else:
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
