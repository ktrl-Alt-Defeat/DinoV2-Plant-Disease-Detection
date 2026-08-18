"""Best-metric tracking and early stopping.

The tracker records the best epoch regardless of whether early stopping is
enabled, because the best checkpoint is selected from the same signal. Disabling
early stopping therefore only removes the stop condition, never the selection.
"""

from dataclasses import dataclass
from typing import Final

from src.config import Config, TypeSpec, validate_keys
from src.logger import get_logger
from src.training.metrics import MONITORABLE_METRICS, EpochMetrics

#: Improvement means a smaller value.
MODE_MIN: Final[str] = "min"

#: Improvement means a larger value.
MODE_MAX: Final[str] = "max"

SUPPORTED_MODES: Final[frozenset[str]] = frozenset({MODE_MIN, MODE_MAX})

#: Configuration contract of the ``training.early_stopping`` section.
EARLY_STOPPING_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("training.early_stopping.enabled", bool),
    ("training.early_stopping.monitor", str),
    ("training.early_stopping.mode", str),
    ("training.early_stopping.patience", int),
    ("training.early_stopping.min_delta", (int, float)),
)

_LOGGER: Final = get_logger("training.early_stopping")


@dataclass(frozen=True)
class EarlyStoppingSpecification:
    """Early stopping settings resolved from the configuration."""

    enabled: bool
    monitor: str
    mode: str
    patience: int
    min_delta: float

    @classmethod
    def from_config(cls, config: Config) -> "EarlyStoppingSpecification":
        """Read and validate the ``training.early_stopping`` section of ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If the monitored metric, mode, patience or delta is invalid.
        """
        validate_keys(
            config, EARLY_STOPPING_REQUIRED_KEYS, context="training.early_stopping section"
        )

        monitor = config.get("training.early_stopping.monitor").strip().lower()
        if monitor not in MONITORABLE_METRICS:
            supported = ", ".join(sorted(MONITORABLE_METRICS))
            raise ValueError(
                f"Unsupported training.early_stopping.monitor '{monitor}'. "
                f"Supported metrics: {supported}."
            )

        mode = config.get("training.early_stopping.mode").strip().lower()
        if mode not in SUPPORTED_MODES:
            supported = ", ".join(sorted(SUPPORTED_MODES))
            raise ValueError(
                f"Unsupported training.early_stopping.mode '{mode}'. Supported: {supported}."
            )

        patience = config.get("training.early_stopping.patience")
        if patience <= 0:
            raise ValueError(
                f"training.early_stopping.patience must be positive, got {patience}."
            )

        min_delta = float(config.get("training.early_stopping.min_delta"))
        if min_delta < 0.0:
            raise ValueError(
                f"training.early_stopping.min_delta must be non-negative, got {min_delta}."
            )

        return cls(
            enabled=config.get("training.early_stopping.enabled"),
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
        )

    @property
    def initial_best(self) -> float:
        """The value every candidate improves on."""
        return float("-inf") if self.mode == MODE_MAX else float("inf")

    def improves_on(self, candidate: float, best: float) -> bool:
        """Whether ``candidate`` beats ``best`` by at least ``min_delta``."""
        if self.mode == MODE_MAX:
            return candidate > best + self.min_delta
        return candidate < best - self.min_delta


class EarlyStopping:
    """Tracks the best monitored value and how long it has stood."""

    def __init__(
        self,
        specification: EarlyStoppingSpecification,
        *,
        best_value: float | None = None,
        epochs_without_improvement: int = 0,
    ) -> None:
        """Create a tracker, optionally restoring the state of an interrupted run."""
        self._specification = specification
        self._best_value = specification.initial_best if best_value is None else best_value
        self._epochs_without_improvement = epochs_without_improvement

    def update(self, metrics: EpochMetrics) -> bool:
        """Record ``metrics`` and report whether they improved on the best so far.

        Raises:
            ValueError: If the monitored metric is unknown.
        """
        candidate = metrics.value_of(self._specification.monitor)
        if self._specification.improves_on(candidate, self._best_value):
            self._best_value = candidate
            self._epochs_without_improvement = 0
            return True

        self._epochs_without_improvement += 1
        _LOGGER.info(
            "No improvement in %s for %d epoch(s); best remains %.4f.",
            self._specification.monitor,
            self._epochs_without_improvement,
            self._best_value,
        )
        return False

    @property
    def should_stop(self) -> bool:
        """Whether patience is exhausted and training should end early."""
        if not self._specification.enabled:
            return False
        return self._epochs_without_improvement >= self._specification.patience

    @property
    def best_value(self) -> float:
        """Best monitored value seen so far."""
        return self._best_value

    @property
    def epochs_without_improvement(self) -> int:
        """Consecutive epochs since the last improvement."""
        return self._epochs_without_improvement

    @property
    def monitor(self) -> str:
        """Name of the monitored metric."""
        return self._specification.monitor
