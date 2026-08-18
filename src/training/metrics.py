"""Per-epoch metric records and the training history file."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from src.utils import write_csv

#: Column order of ``results/history.csv``.
HISTORY_FIELDNAMES: Final[tuple[str, ...]] = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "learning_rate",
    "epoch_seconds",
    "gpu_peak_mib",
)

#: Metrics that may be named by ``training.early_stopping.monitor``.
MONITORABLE_METRICS: Final[frozenset[str]] = frozenset(
    {"train_loss", "train_accuracy", "val_loss", "val_accuracy"}
)


@dataclass(frozen=True)
class EpochOutcome:
    """Aggregate loss and accuracy of one pass over a split."""

    loss: float
    accuracy: float


@dataclass(frozen=True)
class EpochMetrics:
    """Everything recorded about a single completed epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    epoch_seconds: float
    gpu_peak_mib: float

    @classmethod
    def from_row(cls, row: Mapping[str, float]) -> "EpochMetrics":
        """Rebuild a record from a history row, as stored inside a checkpoint.

        Raises:
            KeyError: If the row does not carry every history column.
        """
        return cls(
            epoch=int(row["epoch"]),
            train_loss=float(row["train_loss"]),
            train_accuracy=float(row["train_accuracy"]),
            val_loss=float(row["val_loss"]),
            val_accuracy=float(row["val_accuracy"]),
            learning_rate=float(row["learning_rate"]),
            epoch_seconds=float(row["epoch_seconds"]),
            gpu_peak_mib=float(row["gpu_peak_mib"]),
        )

    def as_row(self) -> dict[str, float]:
        """Return the record as a history row."""
        return asdict(self)

    def value_of(self, metric: str) -> float:
        """Return the value of ``metric``.

        Raises:
            ValueError: If ``metric`` is not one of :data:`MONITORABLE_METRICS`.
        """
        if metric not in MONITORABLE_METRICS:
            supported = ", ".join(sorted(MONITORABLE_METRICS))
            raise ValueError(f"Unknown metric '{metric}'. Supported metrics: {supported}.")
        return float(getattr(self, metric))


def write_history(path: str | Path, history: Sequence[EpochMetrics]) -> Path:
    """Write the epoch history as CSV.

    Returns:
        The absolute path that was written.

    Raises:
        ValueError: If ``history`` is empty.
    """
    if not history:
        raise ValueError("Cannot write an empty training history.")
    return write_csv(path, [metrics.as_row() for metrics in history], fieldnames=HISTORY_FIELDNAMES)
