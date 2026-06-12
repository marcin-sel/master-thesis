import copy
import json
from inspect import signature
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from my_project.gnn.trainer import GNNLightningModule


class EpochMLFlowLogger(MLFlowLogger):
    def log_metrics(self, metrics, step=None):
        trainer = self._trainer
        epoch = trainer.current_epoch
        super().log_metrics(metrics, step=epoch)


def train_gnn(
    params,
    model_cls,
    data,
    trial_id: Optional[int] = None,
    fold_id: Optional[int] = None,
    extra_params: Optional[dict] = None,
    tags: Optional[dict] = None,
    extra_metrics: Optional[dict] = None,
    trainer_kwargs: Optional[dict] = None,
    monitor_kwargs: Optional[dict] = None,
    early_stopping_kwargs: Optional[dict] = None,
    logger_kwargs: Optional[dict] = None,
    parent_run_id: Optional[str] = None,
):
    extra_params = {} if extra_params is None else extra_params.copy()
    tags = {} if tags is None else tags.copy()
    extra_metrics = {} if extra_metrics is None else extra_metrics.copy()

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
    experiment_tags = logger_kwargs.pop("experiment_tags", None)

    # Collect run-level params, tags and metrics from the explicit dicts.
    run_params = {}
    for key, value in (("trial_id", trial_id), ("fold_id", fold_id)):
        if value is not None:
            run_params[key] = value

    collision = set(params) & set(extra_params)
    if collision:
        raise ValueError(f"extra_params keys collide with params: {sorted(collision)}")
    run_params.update(extra_params)

    run_tags = dict(tags)
    for key, value in (("trial_id", trial_id), ("fold_id", fold_id)):
        if value is not None:
            run_tags.setdefault(key, value)

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
                experiment_id = client.create_experiment(experiment_name)
                exp = client.get_experiment(experiment_id)
            except MlflowException as e:
                exp = client.get_experiment_by_name(experiment_name)

        if experiment_tags and exp is not None:
            for key, value in experiment_tags.items():
                try:
                    client.set_experiment_tag(exp.experiment_id, key, str(value))
                except MlflowException as e:
                    pass

        # Nest this run under a parent run (MLflow nested runs).
        if parent_run_id is not None:
            logger_tags = dict(logger_kwargs.pop("tags", None) or {})
            logger_tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
            logger_kwargs["tags"] = logger_tags

        logger = EpochMLFlowLogger(
            experiment_name=experiment_name,
            run_name=mlflow_run_name,
            **logger_kwargs,
        )

        # Ensure the parent link is set even if the logger backend ignores tags.
        if parent_run_id is not None:
            try:
                logger.experiment.set_tag(
                    run_id=logger.run_id,
                    key=MLFLOW_PARENT_RUN_ID,
                    value=parent_run_id,
                )
            except MlflowException:
                pass

        # Params (immutable): hyperparameters + run-level params.
        all_params = {**params, **run_params}
        logger.log_hyperparams(
            {k: v for k, v in all_params.items() if not isinstance(v, dict)}
        )
        logger.log_hyperparams(
            {k: json.dumps(v) for k, v in all_params.items() if isinstance(v, dict)}
        )

        # Tags (filterable labels).
        for k, v in run_tags.items():
            logger.experiment.set_tag(
                run_id=logger.run_id,
                key=k,
                value=str(v),
            )

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
        logger=None,
        **trainer_kwargs,
    )

    if logger:
        if logger.__class__ == EpochMLFlowLogger:
            logger._trainer = trainer
        trainer.logger = logger

    lightning_module = GNNLightningModule(
        model_cls=model_cls,
        model_kwargs=model_params,
        **li_params,
    )

    if logger:
        n_params = sum(p.numel() for p in lightning_module.model.parameters())
        n_trainable_params = sum(
            p.numel() for p in lightning_module.model.parameters() if p.requires_grad
        )
        logger.log_metrics(
            {
                "model/n_params": n_params,
                "model/n_trainable_params": n_trainable_params,
                **extra_metrics,
            }
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
