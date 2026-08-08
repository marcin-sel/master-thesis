import copy
import json
from inspect import signature
from typing import Optional

import lightning as L
import optuna
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from my_project.gnn.trainer import GNNLightningModule
from my_project.gnn.training_helpers import resolve_artifact_location


class EpochMLFlowLogger(MLFlowLogger):
    def log_metrics(self, metrics, step=None):
        trainer = self._trainer
        epoch = trainer.current_epoch
        super().log_metrics(metrics, step=epoch)


class OptunaPruningCallback(L.Callback):
    """Report the monitored metric to an Optuna trial after every validation
    epoch and abort (prune) unpromising trials early.

    Kept in-repo instead of ``optuna.integration.PyTorchLightningPruningCallback``
    because that integration lives in the separate ``optuna-integration`` package
    (not installed here) and tends to lag behind Lightning releases. The trial's
    ``step`` is the epoch index, so the pruner's ``n_warmup_steps`` is measured
    in epochs.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str):
        super().__init__()
        self._trial = trial
        self._monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        # Skip the pre-training sanity-check pass so it does not report a
        # meaningless value at epoch 0.
        if trainer.sanity_checking:
            return
        # Only act during fit(): the post-fit best-checkpoint validate() passes
        # also trigger this hook, and must not report or prune.
        if trainer.state.fn != "fit":
            return
        current = trainer.callback_metrics.get(self._monitor)
        if current is None:
            return
        epoch = trainer.current_epoch
        self._trial.report(current.item(), step=epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial {self._trial.number} pruned at epoch {epoch} "
                f"({self._monitor}={current.item():.5f})."
            )


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
    log_params: bool = True,
    run_name: Optional[str] = None,
    pruning_trial: Optional[optuna.trial.Trial] = None,
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
    # `model_cls` is logged as a param (its name) so it can be used as a chart
    # axis; `conv_layer` is additionally surfaced as a tag for filtering.
    run_params = {"model_cls": model_cls.__name__}
    for key, value in (("trial_id", trial_id), ("fold_id", fold_id)):
        if value is not None:
            run_params[key] = value

    if log_params:
        collision = set(params) & set(extra_params)
        if collision:
            raise ValueError(
                f"extra_params keys collide with params: {sorted(collision)}"
            )
    run_params.update(extra_params)

    run_tags = dict(tags)
    if "conv_layer" in params:
        run_tags.setdefault("conv_layer", params["conv_layer"])
    for key, value in (("trial_id", trial_id), ("fold_id", fold_id)):
        if value is not None:
            run_tags.setdefault(key, value)

    if experiment_name is not None:
        if run_name is not None:
            mlflow_run_name = run_name
        else:
            mlflow_run_name = experiment_name
            if trial_id is not None:
                mlflow_run_name = f"{mlflow_run_name}_trial_{trial_id}"
            if fold_id is not None:
                mlflow_run_name = f"{mlflow_run_name}_fold_{fold_id}"

        client = MlflowClient(tracking_uri=logger_kwargs.get("tracking_uri", None))
        exp = client.get_experiment_by_name(experiment_name)

        if exp is None:
            try:
                experiment_id = client.create_experiment(
                    experiment_name,
                    artifact_location=resolve_artifact_location(
                        logger_kwargs.get("tracking_uri")
                    ),
                )
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
        # When log_params is False (e.g. nested CV folds), the shared
        # hyperparameters live on the parent run, so only log run-level params.
        all_params = {**params, **run_params} if log_params else dict(run_params)
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

    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")

    callbacks = [
        early_stopping_callback,
        checkpoint_callback,
        lr_monitor_callback,
    ]

    # Optional per-epoch pruning: report the monitored metric to the trial each
    # validation epoch and let the study's pruner stop hopeless trials early.
    if pruning_trial is not None:
        callbacks.append(
            OptunaPruningCallback(pruning_trial, monitor_kwargs["monitor"])
        )

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
        run_metrics = {
            "model/n_params": n_params,
            "model/n_trainable_params": n_trainable_params,
            **extra_metrics,
        }
        logger.log_metrics(run_metrics)
        # Expose run-level metrics so callers (e.g. CV objective) can aggregate
        # them onto a parent run without re-deriving the names here.
        trainer.logged_run_metrics = {k: float(v) for k, v in run_metrics.items()}

    try:
        trainer.fit(lightning_module, datamodule=data)
    except optuna.TrialPruned:
        # Lightning finalizes the logger with a FAILED status on ANY exception
        # raised during fit, including our per-epoch pruning signal. That would
        # make pruned trials show up as errors in MLflow. Re-mark the run as
        # KILLED with trial_status=pruned (MLflow has no PRUNED status), matching
        # how the CV parent run is tagged, then re-raise so Optuna still records
        # the prune.
        if logger:
            try:
                logger.experiment.set_tag(logger.run_id, "trial_status", "pruned")
                logger.experiment.set_terminated(logger.run_id, status="KILLED")
            except MlflowException:
                pass
        raise

    best_model_path = checkpoint_callback.best_model_path
    if logger and best_model_path:
        logger.experiment.log_artifact(
            run_id=logger.run_id,
            local_path=best_model_path,
            artifact_path="checkpoints",
        )

    # Log the estimated feature graph as a JSON edge list so each run carries
    # the exact structure the GNN was trained on. Skipped for the fully
    # connected graph (structure is trivial) and for graphless setups whose
    # datamodule holds an empty graph (e.g. MLP), keyed on the run's
    # `graph_name` tag.
    graph = getattr(data, "graph", None)
    graph_name = tags.get("graph_name")
    if (
        logger
        and graph is not None
        and graph_name not in (None, "empty", "fully_connected")
    ):
        try:
            graph_payload = {
                "graph_name": graph_name,
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "nodes": [str(node) for node in graph.nodes()],
                "edges": [[str(u), str(v)] for u, v in graph.edges()],
            }
            logger.experiment.log_dict(
                logger.run_id, graph_payload, "graph/estimated_graph.json"
            )
        except MlflowException:
            pass

    return trainer
