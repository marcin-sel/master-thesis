import copy
import json
from inspect import signature

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
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
    logger_kwargs=None,
):
    if trainer_kwargs is None:
        trainer_kwargs = {}
    else:
        trainer_kwargs = trainer_kwargs.copy()

    if monitor_kwargs is None:
        monitor_kwargs = {
            "monitor": "val/loss",
            "mode": "min",
        }
    else:
        monitor_kwargs = monitor_kwargs.copy()

    if logger_kwargs is None:
        logger_kwargs = {}
    else:
        logger_kwargs = logger_kwargs.copy()

    experiment_name = logger_kwargs.pop("experiment_name", None)

    if experiment_name is not None:
        mlflow_run_name = experiment_name
        if trial_id is not None:
            mlflow_run_name = f"{mlflow_run_name}_trial_{trial_id}"
        if fold_id is not None:
            mlflow_run_name = f"{mlflow_run_name}_fold_{fold_id}"

        client = MlflowClient(tracking_uri=logger_kwargs.get("tracking_uri", None))
        exp = client.get_experiment_by_name(experiment_name)

        if exp is None:
            try:
                client.create_experiment(experiment_name)
            except MlflowException as e:
                pass

        logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=mlflow_run_name,
            **logger_kwargs,
        )
        logger.log_hyperparams(
            {k: v for k, v in params.items() if not isinstance(v, (dict))}
        )
        logger.log_hyperparams(
            {k: json.dumps(v) for k, v in params.items() if isinstance(v, (dict))}
        )

        if trial_id is not None:
            logger.experiment.set_tag(
                run_id=logger.run_id,
                key="trial_id",
                value=str(trial_id),
            )
            logger.log_hyperparams({"trial_id": int(trial_id)})
        if fold_id is not None:
            logger.experiment.set_tag(
                run_id=logger.run_id,
                key="fold_id",
                value=str(fold_id),
            )
            logger.log_hyperparams({"fold_id": int(fold_id)})

    else:
        logger = False

    checkpoint_dir = trainer_kwargs.pop("checkpoint_dir", None)
    if checkpoint_dir:
        if trial_id is not None:
            checkpoint_dir = f"{checkpoint_dir}/trial_{trial_id}"

        if fold_id is not None:
            checkpoint_dir = f"{checkpoint_dir}/fold_{fold_id}"

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
        logger=logger,
        **trainer_kwargs,
    )

    lightning_module = GNNLightningModule(
        model_cls=model_cls,
        model_kwargs=model_params,
        **li_params,
    )

    trainer.fit(lightning_module, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    if logger and best_model_path:
        logger.experiment.log_artifact(
            run_id=logger.run_id,
            local_path=best_model_path,
            artifact_path="checkpoints",
        )

    return trainer
