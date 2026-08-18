"""Training loops, schedulers and checkpoint handling.

The modules here are deliberately free of orchestration: each owns one concern
and :mod:`src.train` composes them into a run.
"""

from src.training.checkpoints import (
    CheckpointError,
    CheckpointSpecification,
    ResumeState,
    load_checkpoint,
    save_checkpoint,
)
from src.training.early_stopping import EarlyStopping, EarlyStoppingSpecification
from src.training.engine import evaluate, train_one_epoch
from src.training.metrics import EpochMetrics, EpochOutcome, write_history
from src.training.optim import (
    OptimizerSpecification,
    SchedulerSpecification,
    build_optimizer,
    build_scheduler,
)
from src.training.precision import PrecisionSpecification, build_grad_scaler

__all__ = [
    "CheckpointError",
    "CheckpointSpecification",
    "EarlyStopping",
    "EarlyStoppingSpecification",
    "EpochMetrics",
    "EpochOutcome",
    "OptimizerSpecification",
    "PrecisionSpecification",
    "ResumeState",
    "SchedulerSpecification",
    "build_grad_scaler",
    "build_optimizer",
    "build_scheduler",
    "evaluate",
    "load_checkpoint",
    "save_checkpoint",
    "train_one_epoch",
    "write_history",
]
