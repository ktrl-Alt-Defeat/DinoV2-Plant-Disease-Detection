"""End-to-end verification of the training pipeline.

Running ``python -m src.verify_pipeline`` exercises one batch through every
stage a real run depends on — data loading, augmentation, forward, loss,
backward, optimizer step, checkpointing, resume, metrics and plots — and reports
each stage separately.

Artifacts are written into a temporary directory, so a verification run never
touches ``checkpoints/`` or overwrites a real training history.
"""

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import nn
from torch.utils.data import DataLoader

from src import reporting
from src.cli import bootstrap, build_parser
from src.config import Config, ConfigError, with_overrides
from src.datasets.loaders import DataBundle, DataLoaderSpecification, build_dataloaders
from src.datasets.transforms import TransformSpecification
from src.datasets.validation import DatasetSpecification, DatasetValidationError
from src.logger import configure_console_encoding, get_logger
from src.model import DinoV2Classifier, ModelBuildError, build_model
from src.training.checkpoints import CheckpointError, load_checkpoint, save_checkpoint
from src.training.early_stopping import EarlyStopping, EarlyStoppingSpecification
from src.training.engine import peak_gpu_memory_mib
from src.training.metrics import EpochMetrics, write_history
from src.training.optim import (
    OptimizerSpecification,
    SchedulerSpecification,
    build_optimizer,
    build_scheduler,
    clip_gradients,
    current_learning_rate,
)
from src.training.precision import PrecisionSpecification, build_grad_scaler
from src.verification import FAILED, PASSED, VerificationCheck, VerificationReport
from src.visualization.plots import PlotSpecification, write_curves

#: Log file this entry point writes to.
VERIFY_LOG_FILENAME: Final[str] = "verify_pipeline.log"

#: Epochs of synthetic history used to exercise the metric and plotting stages.
_SYNTHETIC_EPOCHS: Final[int] = 2

#: Epoch count the throwaway scheduler is built with.
_SCHEDULER_EPOCHS: Final[int] = 1

#: Rank of a batch tensor: [batch, channel, height, width].
_EXPECTED_BATCH_RANK: Final[int] = 4

#: Channel count each decoded image is expected to carry.
_IMAGE_CHANNELS: Final[int] = 3

_TITLE: Final[str] = "MILESTONE 4 — PIPELINE VERIFICATION"

_LOGGER: Final = get_logger("verify_pipeline")


@dataclass(frozen=True)
class _Fixture:
    """Everything one verification run needs, built once and reused by each stage."""

    config: Config
    device: torch.device
    bundle: DataBundle
    model: DinoV2Classifier
    precision: PrecisionSpecification
    criterion: nn.Module
    images: torch.Tensor
    targets: torch.Tensor


def verify_pipeline(config: Config, device: torch.device, workspace: Path) -> VerificationReport:
    """Run every pipeline stage against one real batch and collect the results."""
    checks: list[VerificationCheck] = []

    bundle, load_check = _check_dataset_loads(config, device)
    checks.append(load_check)

    images, targets, batch_checks = _check_batch(bundle.train_loader, config, bundle)
    checks.extend(batch_checks)

    model = _build_model(config, bundle)
    fixture = _Fixture(
        config=config,
        device=device,
        bundle=bundle,
        model=model,
        precision=PrecisionSpecification.resolve(config, device),
        criterion=nn.CrossEntropyLoss(),
        images=images.to(device),
        targets=targets.to(device),
    )

    checks.extend(_check_optimisation(fixture))
    checks.extend(_check_checkpointing(fixture, workspace))
    checks.extend(_check_reporting(fixture, workspace))

    return VerificationReport(model_summary=model.describe(), checks=tuple(checks))


def render_report(report: VerificationReport) -> str:
    """Render the console report shown at the end of a verification run."""
    lines = reporting.banner(_TITLE)
    for index, check in enumerate(report.checks, start=1):
        lines.extend(
            reporting.entry(
                f"Verification {index}",
                f"{check.name} ... {check.status}",
                check.details,
            )
        )
    lines.extend(reporting.closing(f"PIPELINE STATUS : {report.status}"))
    return reporting.render(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the training pipeline and return a process exit code."""
    configure_console_encoding()
    parser = build_parser(
        prog="python -m src.verify_pipeline",
        description="Exercise the dataset, model, optimisation and reporting stages once.",
    )
    arguments = parser.parse_args(argv)

    try:
        boot = bootstrap(arguments.config, log_filename=VERIFY_LOG_FILENAME)
        with tempfile.TemporaryDirectory() as workspace:
            report = verify_pipeline(boot.config, boot.device_info.device, Path(workspace))
    except (
        DatasetValidationError,
        CheckpointError,
        ModelBuildError,
        ConfigError,
        KeyError,
        OSError,
        ValueError,
    ) as error:
        _LOGGER.error("Pipeline verification failed: %s", error)
        print(reporting.render([*reporting.banner(_TITLE), f"ERROR: {error}", ""]))
        print(reporting.render(reporting.closing(f"PIPELINE STATUS : {FAILED}")))
        return 1

    print(render_report(report))
    return 0 if report.passed else 1


def _check_dataset_loads(
    config: Config, device: torch.device
) -> tuple[DataBundle, VerificationCheck]:
    """Build the dataloaders and report the discovered dataset."""
    bundle = build_dataloaders(
        DatasetSpecification.from_config(config),
        TransformSpecification.from_config(config),
        DataLoaderSpecification.from_config(config),
        seed=config.get("project.seed"),
        device=device,
    )
    sizes = ", ".join(f"{split}={size}" for split, size in bundle.split_sizes.items())
    return bundle, VerificationCheck(
        name="Dataset Loads",
        status=PASSED,
        details=f"{bundle.num_classes} classes discovered, {sizes}",
    )


def _check_batch(
    loader: DataLoader,
    config: Config,
    bundle: DataBundle,
) -> tuple[torch.Tensor, torch.Tensor, list[VerificationCheck]]:
    """Draw one augmented batch and validate its shape, dtype and labels."""
    images, targets = next(iter(loader))
    image_size = config.get("model.image_size")
    expected_image_shape = (_IMAGE_CHANNELS, image_size, image_size)

    shape_ok = (
        images.ndim == _EXPECTED_BATCH_RANK
        and tuple(images.shape[1:]) == expected_image_shape
    )
    labels_ok = bool(
        targets.dtype == torch.int64
        and targets.numel() == images.shape[0]
        and int(targets.min()) >= 0
        and int(targets.max()) < bundle.num_classes
    )
    finite_ok = bool(torch.isfinite(images).all())

    return (
        images,
        targets,
        [
            VerificationCheck(
                name="Transforms Applied",
                status=PASSED if finite_ok and images.dtype == torch.float32 else FAILED,
                details=(
                    f"dtype={images.dtype}, finite={finite_ok}, "
                    f"range=[{float(images.min()):.3f}, {float(images.max()):.3f}]"
                ),
            ),
            VerificationCheck(
                name="Batch Shape",
                status=PASSED if shape_ok else FAILED,
                details=f"got {list(images.shape)}, expected [N, 3, {image_size}, {image_size}]",
            ),
            VerificationCheck(
                name="Label Validity",
                status=PASSED if labels_ok else FAILED,
                details=(
                    f"{targets.numel()} labels of dtype {targets.dtype} within "
                    f"[0, {bundle.num_classes - 1}]"
                ),
            ),
        ],
    )


def _build_model(config: Config, bundle: DataBundle) -> DinoV2Classifier:
    """Build the classifier with the discovered class count."""
    return build_model(with_overrides(config, model={"num_classes": bundle.num_classes}))


def _check_optimisation(fixture: _Fixture) -> list[VerificationCheck]:
    """Run forward, loss, backward and one optimizer step on the sampled batch."""
    model = fixture.model
    model.train()

    optimizer = build_optimizer(model, OptimizerSpecification.from_config(fixture.config))
    scaler = build_grad_scaler(fixture.precision)

    optimizer.zero_grad(set_to_none=True)
    with fixture.precision.autocast():
        logits = model(fixture.images)
        loss = fixture.criterion(logits, fixture.targets)

    forward_ok = tuple(logits.shape) == (fixture.images.shape[0], fixture.bundle.num_classes)
    loss_ok = bool(torch.isfinite(loss) and loss.item() > 0.0)

    scaler.scale(loss).backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    backward_ok = bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients)

    scaler.unscale_(optimizer)
    clip_gradients(model, float(fixture.config.get("training.gradient_clip_norm")))
    projection = _head_projection(model)
    before = projection.weight.detach().clone()
    scaler.step(optimizer)
    scaler.update()
    stepped = not torch.equal(before, projection.weight.detach())

    return [
        VerificationCheck(
            name="Forward Pass",
            status=PASSED if forward_ok else FAILED,
            details=(
                f"logits {list(logits.shape)} on {logits.device.type} "
                f"using {fixture.precision.label}"
            ),
        ),
        VerificationCheck(
            name="Loss Computation",
            status=PASSED if loss_ok else FAILED,
            details=f"CrossEntropyLoss = {float(loss):.4f}, finite={bool(torch.isfinite(loss))}",
        ),
        VerificationCheck(
            name="Backward Pass",
            status=PASSED if backward_ok else FAILED,
            details=f"{len(gradients)} gradient tensor(s) populated, all finite={backward_ok}",
        ),
        VerificationCheck(
            name="Optimizer Step",
            status=PASSED if stepped else FAILED,
            details=(
                f"classifier weights changed={stepped}, "
                f"lr={current_learning_rate(optimizer):.3e}"
            ),
        ),
    ]


def _head_projection(model: DinoV2Classifier) -> nn.Linear:
    """Return the linear projection of the head, whether or not dropout wraps it.

    Raises:
        ValueError: If the head exposes no linear layer to observe.
    """
    projections = [module for module in model.classifier.modules() if isinstance(module, nn.Linear)]
    if not projections:
        raise ValueError("Classification head exposes no nn.Linear layer to verify.")
    return projections[-1]


def _check_checkpointing(fixture: _Fixture, workspace: Path) -> list[VerificationCheck]:
    """Save a checkpoint, load it back and confirm the resume state round-trips."""
    optimizer = build_optimizer(fixture.model, OptimizerSpecification.from_config(fixture.config))
    scheduler = build_scheduler(
        optimizer,
        SchedulerSpecification.from_config(fixture.config),
        epochs=_SCHEDULER_EPOCHS,
    )
    scaler = build_grad_scaler(fixture.precision)
    history = _synthetic_history(fixture.device)
    tracker = EarlyStopping(EarlyStoppingSpecification.from_config(fixture.config))
    for metrics in history:
        tracker.update(metrics)

    path = save_checkpoint(
        workspace / "verification_checkpoint.pt",
        model=fixture.model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=history[-1].epoch,
        history=history,
        best_value=tracker.best_value,
        epochs_without_improvement=tracker.epochs_without_improvement,
        config=fixture.config,
        class_to_idx=fixture.bundle.class_to_idx,
    )
    saved = VerificationCheck(
        name="Checkpoint Save",
        status=PASSED if path.is_file() else FAILED,
        details=f"{path.name} written, {path.stat().st_size / 1024**2:.1f} MiB",
    )

    state = load_checkpoint(
        path,
        model=fixture.model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=fixture.device,
        expected_class_to_idx=fixture.bundle.class_to_idx,
    )
    loaded_ok = len(state.history) == len(history)
    resume_ok = state.start_epoch == history[-1].epoch + 1

    return [
        saved,
        VerificationCheck(
            name="Checkpoint Load",
            status=PASSED if loaded_ok else FAILED,
            details=f"{len(state.history)} history row(s) restored, best={state.best_value:.4f}",
        ),
        VerificationCheck(
            name="Resume State",
            status=PASSED if resume_ok else FAILED,
            details=(
                f"resumes at epoch {state.start_epoch} after {history[-1].epoch} completed; "
                f"class vocabulary reconciled ({len(fixture.bundle.class_to_idx)} classes)"
            ),
        ),
    ]


def _check_reporting(fixture: _Fixture, workspace: Path) -> list[VerificationCheck]:
    """Write a history file and both curves from synthetic metrics."""
    history = _synthetic_history(fixture.device)

    history_path = write_history(workspace / "verification_history.csv", history)
    curves = write_curves(history, workspace, PlotSpecification.from_config(fixture.config))
    curves_ok = all(path.is_file() and path.stat().st_size > 0 for path in curves.values())

    return [
        VerificationCheck(
            name="Metrics Saved",
            status=PASSED if history_path.is_file() else FAILED,
            details=f"{history_path.name} with {len(history)} row(s)",
        ),
        VerificationCheck(
            name="Plots Generated",
            status=PASSED if curves_ok else FAILED,
            details=", ".join(sorted(path.name for path in curves.values())),
        ),
    ]


def _synthetic_history(device: torch.device) -> list[EpochMetrics]:
    """Build a short, monotonically improving history for the reporting stages."""
    return [
        EpochMetrics(
            epoch=epoch,
            train_loss=1.0 / epoch,
            train_accuracy=0.5 + 0.1 * epoch,
            val_loss=1.1 / epoch,
            val_accuracy=0.4 + 0.1 * epoch,
            learning_rate=1e-4,
            epoch_seconds=1.0,
            gpu_peak_mib=peak_gpu_memory_mib(device),
        )
        for epoch in range(1, _SYNTHETIC_EPOCHS + 1)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
