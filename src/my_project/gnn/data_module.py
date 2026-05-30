from __future__ import annotations

import lightning as L
from my_project.gnn.utils import build_graph_dataset
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
        batch_size: int = 256,
        num_workers: int = 0,
        persistent_workers: bool | None = None,
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
        self.test_dataset = None

        if persistent_workers is None:
            self.persistent_workers = num_workers > 0
        else:
            self.persistent_workers = persistent_workers

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = build_graph_dataset(self.X_train, self.y_train, self.graph)
        self.valid_dataset = build_graph_dataset(self.X_valid, self.y_valid, self.graph)
        if self.X_test is not None and self.y_test is not None:
            self.test_dataset = build_graph_dataset(
                self.X_test, self.y_test, self.graph
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
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
