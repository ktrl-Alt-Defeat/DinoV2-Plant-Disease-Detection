"""Training curve rendering.

Both curves are produced by one plotting primitive, so the loss and accuracy
figures stay visually identical apart from the series they draw.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

# The Agg backend renders to file without a display server, which is what a
# training run needs. It has to be selected before pyplot is imported.
matplotlib.use("Agg")

from matplotlib import pyplot as plt

from src.config import Config, TypeSpec, validate_keys
from src.logger import get_logger
from src.paths import ensure_directory, resolve
from src.training.metrics import EpochMetrics

#: Configuration contract of the ``visualization`` section.
VISUALIZATION_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("visualization.loss_curve_filename", str),
    ("visualization.accuracy_curve_filename", str),
    ("visualization.figure_width", (int, float)),
    ("visualization.figure_height", (int, float)),
    ("visualization.dpi", int),
)

_LOSS_TITLE: Final[str] = "Training and validation loss"
_ACCURACY_TITLE: Final[str] = "Training and validation accuracy"
_X_LABEL: Final[str] = "Epoch"

_LOGGER: Final = get_logger("visualization.plots")


@dataclass(frozen=True)
class PlotSpecification:
    """Figure settings resolved from the ``visualization`` section."""

    loss_filename: str
    accuracy_filename: str
    figure_size: tuple[float, float]
    dpi: int

    @classmethod
    def from_config(cls, config: Config) -> "PlotSpecification":
        """Read and validate the ``visualization`` section of ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If a figure dimension or the resolution is not positive.
        """
        validate_keys(config, VISUALIZATION_REQUIRED_KEYS, context="visualization section")

        width = float(config.get("visualization.figure_width"))
        height = float(config.get("visualization.figure_height"))
        dpi = config.get("visualization.dpi")

        for key, value in (
            ("visualization.figure_width", width),
            ("visualization.figure_height", height),
            ("visualization.dpi", dpi),
        ):
            if value <= 0:
                raise ValueError(f"{key} must be positive, got {value}.")

        return cls(
            loss_filename=config.get("visualization.loss_curve_filename"),
            accuracy_filename=config.get("visualization.accuracy_curve_filename"),
            figure_size=(width, height),
            dpi=dpi,
        )


def write_curves(
    history: Sequence[EpochMetrics],
    results_dir: str | Path,
    specification: PlotSpecification,
) -> dict[str, Path]:
    """Render the loss and accuracy curves into ``results_dir``.

    Returns:
        The absolute path of each figure, keyed by curve name.

    Raises:
        ValueError: If ``history`` is empty.
    """
    if not history:
        raise ValueError("Cannot plot an empty training history.")

    directory = ensure_directory(results_dir)
    epochs = [metrics.epoch for metrics in history]

    paths = {
        "loss_curve": _line_plot(
            directory / specification.loss_filename,
            title=_LOSS_TITLE,
            y_label="Loss",
            epochs=epochs,
            series={
                "Train": [metrics.train_loss for metrics in history],
                "Validation": [metrics.val_loss for metrics in history],
            },
            specification=specification,
        ),
        "accuracy_curve": _line_plot(
            directory / specification.accuracy_filename,
            title=_ACCURACY_TITLE,
            y_label="Accuracy",
            epochs=epochs,
            series={
                "Train": [metrics.train_accuracy for metrics in history],
                "Validation": [metrics.val_accuracy for metrics in history],
            },
            specification=specification,
        ),
    }
    for name, path in paths.items():
        _LOGGER.info("Figure written: %s -> %s", name, path)
    return paths


def _line_plot(
    path: Path,
    *,
    title: str,
    y_label: str,
    epochs: Sequence[int],
    series: Mapping[str, Sequence[float]],
    specification: PlotSpecification,
) -> Path:
    """Draw one figure with a line per series and save it.

    Returns:
        The absolute path that was written.
    """
    target = resolve(path)
    figure, axes = plt.subplots(figsize=specification.figure_size)
    try:
        for label, values in series.items():
            axes.plot(epochs, values, marker="o", label=label)

        axes.set_title(title)
        axes.set_xlabel(_X_LABEL)
        axes.set_ylabel(y_label)
        axes.grid(visible=True, alpha=0.3)
        axes.legend()
        figure.tight_layout()
        figure.savefig(target, dpi=specification.dpi)
    finally:
        plt.close(figure)
    return target
