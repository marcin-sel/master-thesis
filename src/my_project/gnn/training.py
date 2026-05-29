import copy
from inspect import signature

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from my_project.gnn.trainer import GNNLightningModule
from sklearn import set_config

set_config(transform_output="pandas")


def train_gnn(
    params,
    model_cls,
    data,
    checkpoint_dir=None,
    monitor_metric="val/loss",
    monitor_mode="min",
    max_epochs=100,
    enable_progress_bar=True,
    enable_model_summary=True,
    early_stopping_verbose=True,
    early_stopping_patience=15,
    log_every_n_steps=None,
    precision=None,
):
    batch_size = params.get("batch_size", None)
    if batch_size is not None:
        data = copy.deepcopy(data)
        data.batch_size = batch_size

    model_params_names = list(signature(model_cls.__init__).parameters.keys())
    model_params = {k: v for k, v in params.items() if k in model_params_names}

    li_params_names = list(signature(GNNLightningModule.__init__).parameters.keys())
    li_params = {k: v for k, v in params.items() if k in li_params_names}

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename="best",
    )

    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        mode=monitor_mode,
        patience=early_stopping_patience,
        verbose=early_stopping_verbose,
    )

    callbacks = [
        early_stopping_callback,
        checkpoint_callback,
    ]

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
        logger=False,
        precision=precision,
    )

    lightning_module = GNNLightningModule(
        model_cls=model_cls,
        model_kwargs=model_params,
        **li_params,
    )

    trainer.fit(lightning_module, datamodule=data)

    return trainer
