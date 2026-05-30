import copy
from inspect import signature

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from my_project.gnn.trainer import GNNLightningModule


def train_gnn(
    params,
    model_cls,
    data,
    trial_id=None,
    fold_id=None,
    trainer_kwargs=None,
    monitor_kwargs=None,
    early_stopping_kwargs=None,
):
    if trainer_kwargs is None:
        trainer_kwargs = {}

    if monitor_kwargs is None:
        monitor_kwargs = {
            "monitor": "val/loss",
            "mode": "min",
        }

    checkpoint_dir = trainer_kwargs.pop("checkpoint_dir", None)
    if checkpoint_dir:
        if trial_id is not None:
            checkpoint_dir = (
                f"{checkpoint_dir}/trial_{trainer_kwargs.get('trial_number', 0)}"
            )

        if fold_id:
            checkpoint_dir = f"{checkpoint_dir}/fold_{fold_id}"
        else:
            checkpoint_dir = None

    batch_size = params.get("batch_size", None)
    if batch_size is not None:
        data = copy.deepcopy(data)
        data.batch_size = batch_size

    model_params_names = list(signature(model_cls.__init__).parameters.keys())
    model_params = {k: v for k, v in params.items() if k in model_params_names}

    li_params_names = list(signature(GNNLightningModule.__init__).parameters.keys())
    li_params = {k: v for k, v in params.items() if k in li_params_names}

    checkpoint_callback = ModelCheckpoint(
        **monitor_kwargs,
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename="best",
    )

    early_stopping_callback = EarlyStopping(
        **early_stopping_kwargs,
    )

    callbacks = [
        early_stopping_callback,
        checkpoint_callback,
    ]

    trainer = L.Trainer(
        callbacks=callbacks,
        logger=False,
        **trainer_kwargs,
    )

    lightning_module = GNNLightningModule(
        model_cls=model_cls,
        model_kwargs=model_params,
        **li_params,
    )

    trainer.fit(lightning_module, datamodule=data)

    return trainer
