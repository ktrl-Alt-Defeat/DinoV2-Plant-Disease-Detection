"""Checkpoint writing and restoration.

A checkpoint carries everything needed to continue a run exactly where it
stopped: the model, the optimizer, the scheduler, the loss scaler, the epoch
counter, the full metric history and the configuration and class vocabulary it
was produced with. Only primitives and tensors are stored, so checkpoints load
under ``weights_only=True``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.config import Config, TypeSpec, validate_keys
from src.logger import get_logger
from src.paths import ensure_directory, resolve
from src.training.metrics import EpochMetrics

KEY_MODEL: Final[str] = "model"
KEY_OPTIMIZER: Final[str] = "optimizer"
KEY_SCHEDULER: Final[str] = "scheduler"
KEY_SCALER: Final[str] = "scaler"
KEY_EPOCH: Final[str] = "epoch"
KEY_METRICS: Final[str] = "metrics"
KEY_CONFIG: Final[str] = "config"
KEY_CLASS_TO_IDX: Final[str] = "class_to_idx"
KEY_BEST_VALUE: Final[str] = "best_value"
KEY_STALE_EPOCHS: Final[str] = "epochs_without_improvement"

#: Keys every checkpoint written by this project contains.
CHECKPOINT_KEYS: Final[tuple[str, ...]] = (
    KEY_MODEL,
    KEY_OPTIMIZER,
    KEY_SCHEDULER,
    KEY_SCALER,
    KEY_EPOCH,
    KEY_METRICS,
    KEY_CONFIG,
    KEY_CLASS_TO_IDX,
    KEY_BEST_VALUE,
    KEY_STALE_EPOCHS,
)

#: Configuration contract of the ``training.checkpoints`` section.
CHECKPOINT_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("training.checkpoints.best_filename", str),
    ("training.checkpoints.last_filename", str),
)

_LOGGER: Final = get_logger("training.checkpoints")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be written, read or reconciled."""


@dataclass(frozen=True)
class CheckpointSpecification:
    """Checkpoint file names resolved from the configuration."""

    best_filename: str
    last_filename: str

    @classmethod
    def from_config(cls, config: Config) -> "CheckpointSpecification":
        """Read and validate the ``training.checkpoints`` section of ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
        """
        validate_keys(config, CHECKPOINT_REQUIRED_KEYS, context="training.checkpoints section")
        return cls(
            best_filename=config.get("training.checkpoints.best_filename"),
            last_filename=config.get("training.checkpoints.last_filename"),
        )


@dataclass(frozen=True)
class ResumeState:
    """Training state restored from a checkpoint."""

    start_epoch: int
    history: tuple[EpochMetrics, ...]
    best_value: float
    epochs_without_improvement: int


@dataclass(frozen=True)
class CheckpointMetadata:
    """What a checkpoint records about the run that produced it."""

    path: Path
    epoch: int
    best_value: float
    history: tuple[EpochMetrics, ...]
    class_to_idx: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        """Return the metadata as a serialisable mapping."""
        return {
            "path": str(self.path),
            "epoch": self.epoch,
            "best_value": self.best_value,
            "epochs_recorded": len(self.history),
            "num_classes": len(self.class_to_idx),
        }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    history: Sequence[EpochMetrics],
    best_value: float,
    epochs_without_improvement: int,
    config: Config,
    class_to_idx: Mapping[str, int],
) -> Path:
    """Write a complete checkpoint to ``path``.

    Returns:
        The absolute path that was written.

    Raises:
        CheckpointError: If the file cannot be written.
    """
    target = resolve(path)
    ensure_directory(target.parent)

    payload: dict[str, Any] = {
        KEY_MODEL: model.state_dict(),
        KEY_OPTIMIZER: optimizer.state_dict(),
        KEY_SCHEDULER: scheduler.state_dict(),
        KEY_SCALER: scaler.state_dict(),
        KEY_EPOCH: int(epoch),
        KEY_METRICS: [metrics.as_row() for metrics in history],
        KEY_CONFIG: config.as_dict(),
        KEY_CLASS_TO_IDX: dict(class_to_idx),
        KEY_BEST_VALUE: float(best_value),
        KEY_STALE_EPOCHS: int(epochs_without_improvement),
    }

    try:
        torch.save(payload, target)
    except OSError as error:
        raise CheckpointError(f"Unable to write checkpoint {target}: {error}") from error
    return target


@dataclass(frozen=True)
class CheckpointContents:
    """A checkpoint decoded once, for consumers that build from it.

    An inference process has no dataset to derive the class vocabulary from, so
    the checkpoint is the authority: it carries the configuration it was trained
    with, the class mapping and the weights.
    """

    path: Path
    epoch: int
    best_value: float
    class_to_idx: dict[str, int]
    config: dict[str, Any]
    model_state: dict[str, Any]

    @property
    def num_classes(self) -> int:
        """Number of classes the stored head predicts."""
        return len(self.class_to_idx)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Class names ordered by their index."""
        inverse = {index: name for name, index in self.class_to_idx.items()}
        return tuple(inverse[index] for index in range(self.num_classes))


def read_checkpoint(path: str | Path, device: torch.device) -> CheckpointContents:
    """Decode a checkpoint without needing a dataset to compare it against.

    Raises:
        CheckpointError: If the file is missing, unreadable or incomplete.
    """
    source, payload = _read_payload(path, device, expected_class_to_idx=None)
    return CheckpointContents(
        path=source,
        epoch=int(payload[KEY_EPOCH]),
        best_value=float(payload[KEY_BEST_VALUE]),
        class_to_idx=dict(payload[KEY_CLASS_TO_IDX]),
        config=dict(payload[KEY_CONFIG]),
        model_state=payload[KEY_MODEL],
    )


def _read_payload(
    path: str | Path,
    device: torch.device,
    expected_class_to_idx: Mapping[str, int] | None,
) -> tuple[Path, dict[str, Any]]:
    """Read a checkpoint and validate its structure and class vocabulary.

    ``expected_class_to_idx`` of ``None`` skips the vocabulary comparison, for
    callers that have no dataset and treat the checkpoint as authoritative.

    Raises:
        CheckpointError: If the file is missing, unreadable, incomplete or was
            produced with a different class vocabulary.
    """
    source = resolve(path)
    if not source.is_file():
        raise CheckpointError(f"Checkpoint not found: {source}.")

    try:
        payload = torch.load(source, map_location=device, weights_only=True)
    except (OSError, RuntimeError) as error:
        raise CheckpointError(f"Unable to read checkpoint {source}: {error}") from error

    missing = [key for key in CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise CheckpointError(
            f"Checkpoint {source} is missing entries: {', '.join(missing)}. "
            f"Expected: {', '.join(CHECKPOINT_KEYS)}."
        )

    stored_classes = payload[KEY_CLASS_TO_IDX]
    if expected_class_to_idx is not None and stored_classes != dict(expected_class_to_idx):
        raise CheckpointError(
            f"Checkpoint {source} was trained on a different class vocabulary "
            f"({len(stored_classes)} classes) than the dataset on disk "
            f"({len(expected_class_to_idx)} classes). Re-run the dataset audit "
            "or start a fresh run."
        )
    return source, payload


def load_model_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    device: torch.device,
    expected_class_to_idx: Mapping[str, int],
) -> CheckpointMetadata:
    """Restore only the model weights, for evaluation and inference.

    No optimizer, scheduler or scaler is touched, so a read-only consumer does
    not have to build training state it will never use.

    Raises:
        CheckpointError: If the checkpoint cannot be read or does not match the
            current model.
    """
    source, payload = _read_payload(path, device, expected_class_to_idx)

    try:
        model.load_state_dict(payload[KEY_MODEL])
    except (RuntimeError, ValueError, KeyError) as error:
        raise CheckpointError(
            f"Checkpoint {source} does not match the current model: {error}"
        ) from error

    metadata = CheckpointMetadata(
        path=source,
        epoch=int(payload[KEY_EPOCH]),
        best_value=float(payload[KEY_BEST_VALUE]),
        history=tuple(EpochMetrics.from_row(row) for row in payload[KEY_METRICS]),
        class_to_idx=dict(payload[KEY_CLASS_TO_IDX]),
    )
    _LOGGER.info(
        "Loaded weights from %s (epoch %d, best %.4f).",
        source,
        metadata.epoch,
        metadata.best_value,
    )
    return metadata


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    expected_class_to_idx: Mapping[str, int],
) -> ResumeState:
    """Restore model and training state from the checkpoint at ``path``.

    The stored class vocabulary is compared with the one discovered on disk, so
    resuming against a changed dataset fails instead of training a head whose
    outputs no longer mean what they did.

    Raises:
        CheckpointError: If the file is missing, unreadable, incomplete or was
            produced with a different class vocabulary.
    """
    source, payload = _read_payload(path, device, expected_class_to_idx)

    try:
        model.load_state_dict(payload[KEY_MODEL])
        optimizer.load_state_dict(payload[KEY_OPTIMIZER])
        scheduler.load_state_dict(payload[KEY_SCHEDULER])
        scaler.load_state_dict(payload[KEY_SCALER])
    except (RuntimeError, ValueError, KeyError) as error:
        raise CheckpointError(
            f"Checkpoint {source} does not match the current model or optimizer: {error}"
        ) from error

    history = tuple(EpochMetrics.from_row(row) for row in payload[KEY_METRICS])
    completed_epoch = int(payload[KEY_EPOCH])

    _LOGGER.info(
        "Resumed from %s: %d epoch(s) completed, best %s.",
        source,
        completed_epoch,
        f"{float(payload[KEY_BEST_VALUE]):.4f}",
    )
    return ResumeState(
        start_epoch=completed_epoch + 1,
        history=history,
        best_value=float(payload[KEY_BEST_VALUE]),
        epochs_without_improvement=int(payload[KEY_STALE_EPOCHS]),
    )
