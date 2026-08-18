"""Command line entry point for fine-tuning the DINOv2 classifier.

Running ``python -m src.train`` audits the dataset, builds the dataloaders and
the model, and trains for the configured number of epochs. The dataset audit is
a hard gate: a dataset carrying any error-severity issue aborts the run before
the backbone is loaded.

Only the class count is derived at runtime; every other value comes from the
configuration file.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src import reporting
from src.cli import bootstrap, build_parser
from src.config import Config, ConfigError, load_config, with_overrides
from src.datasets.loaders import DataBundle, DataLoaderSpecification, build_dataloaders
from src.datasets.transforms import TransformSpecification
from src.datasets.validation import (
    DatasetAudit,
    DatasetSpecification,
    DatasetValidationError,
    audit_dataset,
)
from src.logger import configure_console_encoding, get_logger
from src.model import DinoV2Classifier, ModelBuildError, build_model
from src.training.checkpoints import (
    CheckpointError,
    CheckpointSpecification,
    load_checkpoint,
    save_checkpoint,
)
from src.training.early_stopping import MODE_MAX, EarlyStopping, EarlyStoppingSpecification
from src.training.engine import (
    evaluate,
    log_epoch_start,
    peak_gpu_memory_mib,
    reset_gpu_memory_statistics,
    train_one_epoch,
)
from src.training.metrics import EpochMetrics, write_history
from src.training.optim import (
    OptimizerSpecification,
    SchedulerSpecification,
    build_optimizer,
    build_scheduler,
    current_learning_rate,
)
from src.training.precision import PrecisionSpecification, build_grad_scaler
from src.utils import Timer, format_duration, write_json
from src.visualization.plots import PlotSpecification, write_curves

#: Configuration key holding the number of epochs to run.
EPOCHS_KEY: Final[str] = "training.epochs"

#: Configuration key holding the gradient clipping threshold.
CLIP_NORM_KEY: Final[str] = "training.gradient_clip_norm"

_TITLE: Final[str] = "MILESTONE 4 — TRAINING"

_LOGGER: Final = get_logger("train")


class TrainingError(RuntimeError):
    """Raised when a training run cannot start or cannot continue."""


@dataclass(frozen=True)
class TrainingArtifacts:
    """Files produced by a completed run."""

    history: Path
    best_checkpoint: Path
    last_checkpoint: Path
    curves: dict[str, Path]

    def as_dict(self) -> dict[str, str]:
        """Return the artifact paths as a serialisable mapping."""
        return {
            "history": str(self.history),
            "best_checkpoint": str(self.best_checkpoint),
            "last_checkpoint": str(self.last_checkpoint),
            **{name: str(path) for name, path in self.curves.items()},
        }


@dataclass(frozen=True)
class TrainingOutcome:
    """Summary of a completed run."""

    history: tuple[EpochMetrics, ...]
    best_metrics: EpochMetrics
    monitor: str
    stopped_early: bool
    total_seconds: float
    num_classes: int
    precision_label: str
    artifacts: TrainingArtifacts


def run_training(config_path: str | Path, *, resume: Path | None = None) -> TrainingOutcome:
    """Audit the dataset, build everything and run the training loop.

    Raises:
        DatasetValidationError: If the dataset on disk is unusable.
        TrainingError: If the configuration describes no runnable schedule.
        CheckpointError: If ``resume`` cannot be reconciled with this run.
        ConfigError: If the configuration is incomplete or malformed.
        ModelBuildError: If the backbone cannot be assembled.
    """
    boot = bootstrap(config_path, log_filename=_log_filename(config_path))
    config = boot.config
    device = boot.device_info.device
    results_dir = Path(boot.paths.results)

    _LOGGER.info(
        "Training run starting on %s (%s).", device, boot.device_info.name
    )

    dataset_specification = DatasetSpecification.from_config(config)
    _gate_on_dataset_audit(dataset_specification, results_dir)

    bundle = build_dataloaders(
        dataset_specification,
        TransformSpecification.from_config(config),
        DataLoaderSpecification.from_config(config),
        seed=config.get("project.seed"),
        device=device,
    )
    model = _build_classifier(config, bundle)

    epochs = config.get(EPOCHS_KEY)
    if not isinstance(epochs, int) or epochs <= 0:
        raise TrainingError(f"{EPOCHS_KEY} must be a positive integer, got {epochs!r}.")

    precision = PrecisionSpecification.resolve(config, device)
    optimizer = build_optimizer(model, OptimizerSpecification.from_config(config))
    scheduler = build_scheduler(
        optimizer, SchedulerSpecification.from_config(config), epochs=epochs
    )
    scaler = build_grad_scaler(precision)
    criterion = nn.CrossEntropyLoss()

    stopping_specification = EarlyStoppingSpecification.from_config(config)
    checkpoint_specification = CheckpointSpecification.from_config(config)
    checkpoints_dir = Path(boot.paths.checkpoints)

    start_epoch = 1
    history: list[EpochMetrics] = []
    tracker = EarlyStopping(stopping_specification)

    if resume is not None:
        state = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            expected_class_to_idx=bundle.class_to_idx,
        )
        start_epoch = state.start_epoch
        history = list(state.history)
        tracker = EarlyStopping(
            stopping_specification,
            best_value=state.best_value,
            epochs_without_improvement=state.epochs_without_improvement,
        )
        if start_epoch > epochs:
            raise TrainingError(
                f"Checkpoint {resume} already completed {start_epoch - 1} epoch(s), "
                f"which meets {EPOCHS_KEY}={epochs}. Raise the epoch count to continue."
            )

    _log_run_header(bundle, model, epochs, precision, start_epoch)

    stopped_early = False
    with Timer() as timer:
        for epoch in range(start_epoch, epochs + 1):
            metrics = _run_epoch(
                epoch=epoch,
                epochs=epochs,
                model=model,
                bundle=bundle,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                precision=precision,
                device=device,
                gradient_clip_norm=float(config.get(CLIP_NORM_KEY)),
            )
            history.append(metrics)
            improved = tracker.update(metrics)

            _save_checkpoints(
                directory=checkpoints_dir,
                specification=checkpoint_specification,
                improved=improved,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                history=history,
                tracker=tracker,
                config=config,
                bundle=bundle,
            )
            _log_epoch_summary(metrics, epochs, history, improved=improved)

            if tracker.should_stop:
                stopped_early = True
                _LOGGER.info(
                    "Early stopping at epoch %d: %s did not improve for %d epoch(s).",
                    epoch,
                    tracker.monitor,
                    tracker.epochs_without_improvement,
                )
                break

    artifacts = _write_artifacts(
        config=config,
        history=history,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
        checkpoint_specification=checkpoint_specification,
    )
    best_metrics = _best_epoch(history, tracker, stopping_specification)

    _LOGGER.info(
        "Training complete in %s: best %s=%.4f at epoch %d.",
        format_duration(timer.elapsed),
        tracker.monitor,
        best_metrics.value_of(tracker.monitor),
        best_metrics.epoch,
    )
    return TrainingOutcome(
        history=tuple(history),
        best_metrics=best_metrics,
        monitor=tracker.monitor,
        stopped_early=stopped_early,
        total_seconds=timer.elapsed,
        num_classes=bundle.num_classes,
        precision_label=precision.label,
        artifacts=artifacts,
    )


def render_summary(outcome: TrainingOutcome) -> str:
    """Render the console report shown at the end of a run."""
    best = outcome.best_metrics
    lines = reporting.banner(_TITLE)
    lines.extend(
        reporting.entries(
            [
                ("Classes", str(outcome.num_classes)),
                ("Epochs Completed", str(len(outcome.history))),
                ("Precision", outcome.precision_label),
                ("Stopped Early", "Yes" if outcome.stopped_early else "No"),
                ("Total Time", format_duration(outcome.total_seconds)),
                ("Best Epoch", str(best.epoch)),
                ("Monitored Metric", f"{outcome.monitor} = {best.value_of(outcome.monitor):.4f}"),
                ("Best Train Accuracy", f"{best.train_accuracy:.4f}"),
                ("Best Validation Accuracy", f"{best.val_accuracy:.4f}"),
                ("Best Validation Loss", f"{best.val_loss:.4f}"),
            ]
        )
    )
    lines.extend(reporting.rule())
    lines.extend(reporting.entries(sorted(outcome.artifacts.as_dict().items())))
    lines.extend(reporting.closing("TRAINING STATUS : PASS"))
    return reporting.render(lines)


def build_training_parser() -> argparse.ArgumentParser:
    """Build the argument parser, adding ``--resume`` to the shared options."""
    parser = build_parser(
        prog="python -m src.train",
        description="Fine-tune the configured DINOv2 classifier on the dataset in data/.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Checkpoint to continue from, e.g. checkpoints/last_model.pt. "
            "Restores model, optimizer, scheduler, scaler, epoch and history."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the training command line interface and return a process exit code."""
    configure_console_encoding()
    arguments = build_training_parser().parse_args(argv)

    try:
        outcome = run_training(arguments.config, resume=arguments.resume)
    except (
        DatasetValidationError,
        TrainingError,
        CheckpointError,
        ModelBuildError,
        ConfigError,
        KeyError,
        OSError,
        ValueError,
    ) as error:
        _LOGGER.error("Training failed: %s", error)
        print(reporting.render([*reporting.banner(_TITLE), f"ERROR: {error}", ""]))
        print(reporting.render(reporting.closing("TRAINING STATUS : FAIL")))
        return 1

    print(render_summary(outcome))
    return 0


def _log_filename(config_path: str | Path) -> str:
    """Read the training log file name.

    The name is needed to configure logging, which happens inside the bootstrap,
    so the configuration is read once here before the bootstrap reads it again.
    """
    return load_config(config_path).get("training.log_filename")


def _gate_on_dataset_audit(specification: DatasetSpecification, results_dir: Path) -> None:
    """Audit the dataset and abort the run when it carries any error.

    Raises:
        DatasetValidationError: If the audit reports at least one error.
    """
    audit = audit_dataset(specification)
    report_path = write_json(results_dir / specification.audit_filename, audit.as_dict())
    _LOGGER.info("Dataset audit report written to %s.", report_path)

    for issue in audit.warnings:
        _LOGGER.warning("%s at %s: %s", issue.category, issue.location, issue.detail)

    if audit.passed:
        _LOGGER.info(
            "Dataset audit passed: %d classes, %d images.", audit.class_count, audit.total_images
        )
        return

    raise DatasetValidationError(
        f"Dataset audit failed with {len(audit.errors)} error(s); training aborted. "
        f"First error: {_describe_first_error(audit)}. Full report: {report_path}."
    )


def _describe_first_error(audit: DatasetAudit) -> str:
    """Summarise the first error of ``audit`` for an exception message."""
    issue = audit.errors[0]
    return f"{issue.category} at {issue.location} ({issue.detail})"


def _build_classifier(config: Config, bundle: DataBundle) -> DinoV2Classifier:
    """Build the model with the class count discovered on disk."""
    model_config = with_overrides(config, model={"num_classes": bundle.num_classes})
    return build_model(model_config)


def _run_epoch(
    *,
    epoch: int,
    epochs: int,
    model: DinoV2Classifier,
    bundle: DataBundle,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    precision: PrecisionSpecification,
    device: torch.device,
    gradient_clip_norm: float,
) -> EpochMetrics:
    """Run one training epoch followed by validation and return its metrics."""
    learning_rate = current_learning_rate(optimizer)
    log_epoch_start(epoch, epochs, learning_rate)
    reset_gpu_memory_statistics(device)

    with Timer() as timer:
        train_outcome = train_one_epoch(
            model,
            bundle.train_loader,
            criterion,
            optimizer,
            scaler,
            precision,
            device,
            gradient_clip_norm=gradient_clip_norm,
            description=f"Epoch {epoch}/{epochs} [train]",
        )
        val_outcome = evaluate(
            model,
            bundle.val_loader,
            criterion,
            precision,
            device,
            description=f"Epoch {epoch}/{epochs} [val]",
        )
    scheduler.step()

    return EpochMetrics(
        epoch=epoch,
        train_loss=train_outcome.loss,
        train_accuracy=train_outcome.accuracy,
        val_loss=val_outcome.loss,
        val_accuracy=val_outcome.accuracy,
        learning_rate=learning_rate,
        epoch_seconds=timer.elapsed,
        gpu_peak_mib=peak_gpu_memory_mib(device),
    )


def _save_checkpoints(
    *,
    directory: Path,
    specification: CheckpointSpecification,
    improved: bool,
    model: DinoV2Classifier,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    history: Sequence[EpochMetrics],
    tracker: EarlyStopping,
    config: Config,
    bundle: DataBundle,
) -> None:
    """Write the last checkpoint, and the best one when the epoch improved."""
    common = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "epoch": epoch,
        "history": history,
        "best_value": tracker.best_value,
        "epochs_without_improvement": tracker.epochs_without_improvement,
        "config": config,
        "class_to_idx": bundle.class_to_idx,
    }

    last_path = save_checkpoint(directory / specification.last_filename, **common)
    _LOGGER.info("Checkpoint saved: last -> %s", last_path)

    if improved:
        best_path = save_checkpoint(directory / specification.best_filename, **common)
        _LOGGER.info(
            "Checkpoint saved: best -> %s (%s=%.4f).",
            best_path,
            tracker.monitor,
            tracker.best_value,
        )


def _write_artifacts(
    *,
    config: Config,
    history: Sequence[EpochMetrics],
    results_dir: Path,
    checkpoints_dir: Path,
    checkpoint_specification: CheckpointSpecification,
) -> TrainingArtifacts:
    """Write the history file and the training curves."""
    history_path = write_history(
        results_dir / config.get("training.history_filename"), history
    )
    _LOGGER.info("History written: %s", history_path)

    curves = write_curves(history, results_dir, PlotSpecification.from_config(config))
    return TrainingArtifacts(
        history=history_path,
        best_checkpoint=checkpoints_dir / checkpoint_specification.best_filename,
        last_checkpoint=checkpoints_dir / checkpoint_specification.last_filename,
        curves=curves,
    )


def _best_epoch(
    history: Sequence[EpochMetrics],
    tracker: EarlyStopping,
    specification: EarlyStoppingSpecification,
) -> EpochMetrics:
    """Return the epoch whose monitored metric is the best in ``history``."""
    key = specification.monitor
    if specification.mode == MODE_MAX:
        return max(history, key=lambda metrics: metrics.value_of(key))
    return min(history, key=lambda metrics: metrics.value_of(key))


def _log_run_header(
    bundle: DataBundle,
    model: DinoV2Classifier,
    epochs: int,
    precision: PrecisionSpecification,
    start_epoch: int,
) -> None:
    """Log the dataset and model summary that opens every run."""
    _LOGGER.info(
        "Dataset: %d classes, %s.",
        bundle.num_classes,
        ", ".join(f"{split}={size}" for split, size in bundle.split_sizes.items()),
    )
    _LOGGER.info(
        "Model: %s + %s head, %d trainable of %d parameters.",
        model.specification.display_name,
        model.classifier_specification.display_name,
        model.count_trainable_parameters(),
        model.count_parameters(),
    )
    _LOGGER.info(
        "Schedule: epochs %d..%d of %d, precision %s.",
        start_epoch,
        epochs,
        epochs,
        precision.label,
    )


def _log_epoch_summary(
    metrics: EpochMetrics,
    epochs: int,
    history: Sequence[EpochMetrics],
    *,
    improved: bool,
) -> None:
    """Log the result of one epoch together with an estimate of the time left."""
    remaining = epochs - metrics.epoch
    mean_seconds = sum(entry.epoch_seconds for entry in history) / len(history)
    eta = format_duration(mean_seconds * remaining) if remaining > 0 else "done"

    _LOGGER.info(
        "Epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | "
        "lr %.3e | %s | gpu peak %.1f MiB | eta %s%s",
        metrics.epoch,
        epochs,
        metrics.train_loss,
        metrics.train_accuracy,
        metrics.val_loss,
        metrics.val_accuracy,
        metrics.learning_rate,
        format_duration(metrics.epoch_seconds),
        metrics.gpu_peak_mib,
        eta,
        " | improved" if improved else "",
    )


if __name__ == "__main__":
    raise SystemExit(main())
