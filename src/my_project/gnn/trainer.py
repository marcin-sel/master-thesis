import copy
from typing import Optional

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)


class GNNLightningModule(L.LightningModule):
    """
    PyTorch Lightning wrapper for neural network models.

    Parameters
    ----------
    model : Optional[torch.nn.Module]
        The neural network model to be trained. If None, a new instance of ``model_cls`` will be created using ``model_kwargs``.
    model_cls : type
        Class of the model to be trained (e.g. MyGNN).
    model_kwargs : dict
        Dictionary passed directly to ``model(**model_kwargs)``.
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    class_weights : Optional[torch.Tensor]
        Class weights for CrossEntropyLoss (for imbalanced datasets).
    threshold : float
        Binary classification threshold (default equals positive class frequency).
    """

    def __init__(
        self,
        model=None,
        model_cls=None,
        model_kwargs: Optional[dict] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        scheduler_monitor: Optional[str] = "val/loss",
        scheduler_monitor_mode: Optional[str] = "min",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "model"], logger=False)

        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        if model is None:
            model = model_cls(**model_kwargs)
        else:
            model_cls = type(model)
            model_cls_param_names = set(model_cls.__init__.__code__.co_varnames)
            model_kwargs = {
                k: v
                for k, v in model_kwargs.items()
                if k in model_cls_param_names and k != "self"
            }

        self.model = copy.deepcopy(model)
        self.lr = lr
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_monitor_mode = scheduler_monitor_mode

        self.criterion = (
            nn.CrossEntropyLoss(weight=class_weights)
            if class_weights is not None
            else nn.CrossEntropyLoss()
        )

        metric_collection = MetricCollection(
            {
                "accuracy": BinaryAccuracy(threshold=self.threshold),
                "auc": BinaryAUROC(),
                "avg_precision": BinaryAveragePrecision(),
                "f1": BinaryF1Score(threshold=self.threshold),
                "precision": BinaryPrecision(threshold=self.threshold, zero_division=0),
                "recall": BinaryRecall(threshold=self.threshold, zero_division=0),
                "specificity": BinarySpecificity(
                    threshold=self.threshold, zero_division=0
                ),
            }
        )
        self.train_metrics = metric_collection.clone(prefix="train/")
        self.val_metrics = metric_collection.clone(prefix="val/")
        self.test_metrics = metric_collection.clone(prefix="test/")

    def forward(self, x, edge_index, batch):
        return self.model(x, edge_index, batch)

    def _shared_step(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        out = self(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(out, batch.y.long())
        probs = torch.softmax(out, dim=1)[:, 1]
        batch_size = int(batch.y.size(0))
        return loss, probs, batch.y, batch_size

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, probs, labels, batch_size = self._shared_step(batch)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.train_metrics.update(probs.detach(), labels.detach().long())
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, probs, labels, batch_size = self._shared_step(batch)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.val_metrics.update(probs.detach(), labels.detach().long())

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        metrics["val/balanced_acc"] = (
            metrics["val/recall"] + metrics["val/specificity"]
        ) / 2
        metrics.pop("val/specificity")
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
        )
        self.log("global_step", self.global_step, on_step=False, on_epoch=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx: int) -> None:
        _, probs, labels, _ = self._shared_step(batch)
        self.test_metrics.update(probs.detach(), labels.detach().long())

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        metrics["test/balanced_acc"] = (
            metrics["test/recall"] + metrics["test/specificity"]
        ) / 2
        metrics.pop("test/specificity")
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
        )
        self.test_metrics.reset()

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        metrics["train/balanced_acc"] = (
            metrics["train/recall"] + metrics["train/specificity"]
        ) / 2
        metrics.pop("train/specificity")

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            # step=self.current_epoch,
        )
        self.train_metrics.reset()

    def predict_step(self, batch, batch_idx: int) -> dict:
        _, probs, labels, _ = self._shared_step(batch)
        preds = (probs >= self.threshold).long()
        return {
            "graph_ids": batch.graph_id.cpu(),
            "y_true": labels.cpu(),
            "y_pred": preds.cpu(),
            "y_score": probs.cpu(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.scheduler_monitor is None:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.scheduler_monitor_mode,
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.scheduler_monitor},
        }
