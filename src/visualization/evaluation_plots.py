"""Evaluation figures: confusion matrix, ROC, precision-recall and calibration.

These live beside the training curves so every figure the project produces is
rendered by the same backend and the same figure settings.

With many classes a legend entry per curve is unreadable, so the ROC and PR
figures draw the per-class curves as thin translucent lines and highlight the
micro-average, whose value is quoted in the title alongside the macro average.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np

# The Agg backend renders to file without a display server. It has to be
# selected before pyplot is imported.
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from src.config import Config, TypeSpec, validate_keys
from src.evaluation.metrics import EvaluationMetrics
from src.logger import get_logger
from src.paths import ensure_directory
from src.visualization.plots import PlotSpecification

#: Configuration contract of the evaluation figure settings.
FIGURE_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("evaluation.filenames.confusion_matrix_plot", str),
    ("evaluation.filenames.roc_curves_plot", str),
    ("evaluation.filenames.pr_curves_plot", str),
    ("evaluation.filenames.calibration_plot", str),
    ("evaluation.confusion_matrix_figure_size", (int, float)),
)

#: Above this class count the per-curve legend is dropped as unreadable.
_MAX_LEGEND_CLASSES: Final[int] = 12

#: Styling of the thin per-class curves.
_CURVE_ALPHA: Final[float] = 0.35
_CURVE_WIDTH: Final[float] = 0.8

#: Colour map of the confusion matrix.
_MATRIX_COLORMAP: Final[str] = "viridis"

#: Font size of the confusion matrix tick labels.
_MATRIX_TICK_FONTSIZE: Final[float] = 6.0

_LOGGER: Final = get_logger("visualization.evaluation_plots")


@dataclass(frozen=True)
class EvaluationPlotSpecification:
    """Figure settings for the evaluation artifacts."""

    confusion_matrix_filename: str
    roc_curves_filename: str
    pr_curves_filename: str
    calibration_filename: str
    figure_size: tuple[float, float]
    confusion_matrix_size: float
    dpi: int

    @classmethod
    def from_config(cls, config: Config) -> "EvaluationPlotSpecification":
        """Read the evaluation figure settings from ``config``.

        The shared figure width, height and resolution are reused from the
        ``visualization`` section so every figure matches.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If the confusion matrix edge is not positive.
        """
        validate_keys(config, FIGURE_REQUIRED_KEYS, context="evaluation figure settings")
        shared = PlotSpecification.from_config(config)

        matrix_size = float(config.get("evaluation.confusion_matrix_figure_size"))
        if matrix_size <= 0:
            raise ValueError(
                f"evaluation.confusion_matrix_figure_size must be positive, got {matrix_size}."
            )

        return cls(
            confusion_matrix_filename=config.get("evaluation.filenames.confusion_matrix_plot"),
            roc_curves_filename=config.get("evaluation.filenames.roc_curves_plot"),
            pr_curves_filename=config.get("evaluation.filenames.pr_curves_plot"),
            calibration_filename=config.get("evaluation.filenames.calibration_plot"),
            figure_size=shared.figure_size,
            confusion_matrix_size=matrix_size,
            dpi=shared.dpi,
        )


def write_evaluation_figures(
    metrics: EvaluationMetrics,
    probabilities: np.ndarray,
    targets: np.ndarray,
    results_dir: str | Path,
    specification: EvaluationPlotSpecification,
) -> dict[str, Path]:
    """Render every evaluation figure into ``results_dir``.

    Returns:
        The absolute path of each figure, keyed by figure name.
    """
    directory = ensure_directory(results_dir)
    paths = {
        "confusion_matrix_plot": _plot_confusion_matrix(
            directory / specification.confusion_matrix_filename, metrics, specification
        ),
        "roc_curves_plot": _plot_roc_curves(
            directory / specification.roc_curves_filename,
            metrics,
            probabilities,
            targets,
            specification,
        ),
        "pr_curves_plot": _plot_pr_curves(
            directory / specification.pr_curves_filename,
            metrics,
            probabilities,
            targets,
            specification,
        ),
        "calibration_plot": _plot_calibration(
            directory / specification.calibration_filename, metrics, specification
        ),
    }
    for name, path in paths.items():
        _LOGGER.info("Figure written: %s -> %s", name, path)
    return paths


def _plot_confusion_matrix(
    path: Path,
    metrics: EvaluationMetrics,
    specification: EvaluationPlotSpecification,
) -> Path:
    """Draw the row-normalised confusion matrix.

    Normalising by true-class support turns each row into the recall profile of
    that class, which stays readable when class sizes differ by an order of
    magnitude. The raw counts are exported to CSV alongside.
    """
    matrix = metrics.confusion_matrix.astype(np.float64)
    support = matrix.sum(axis=1, keepdims=True)
    # A class with no test sample would divide by zero; leave its row at zero.
    normalised = np.divide(matrix, support, out=np.zeros_like(matrix), where=support > 0)

    edge = specification.confusion_matrix_size
    figure, axes = plt.subplots(figsize=(edge, edge))
    try:
        image = axes.imshow(normalised, cmap=_MATRIX_COLORMAP, vmin=0.0, vmax=1.0)
        figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label="Fraction of true class")

        positions = np.arange(len(metrics.class_names))
        axes.set_xticks(positions)
        axes.set_yticks(positions)
        axes.set_xticklabels(metrics.class_names, rotation=90, fontsize=_MATRIX_TICK_FONTSIZE)
        axes.set_yticklabels(metrics.class_names, fontsize=_MATRIX_TICK_FONTSIZE)
        axes.set_xlabel("Predicted class")
        axes.set_ylabel("True class")
        axes.set_title(
            f"Confusion matrix, row-normalised — "
            f"top-1 accuracy {metrics.overall.top1_accuracy:.4f}"
        )
        figure.tight_layout()
        figure.savefig(path, dpi=specification.dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


def _plot_roc_curves(
    path: Path,
    metrics: EvaluationMetrics,
    probabilities: np.ndarray,
    targets: np.ndarray,
    specification: EvaluationPlotSpecification,
) -> Path:
    """Draw one-vs-rest ROC curves plus the micro-average."""
    figure, axes = plt.subplots(figsize=specification.figure_size)
    try:
        for index, entry in enumerate(metrics.per_class):
            if entry.roc_auc is None:
                continue
            binary = (targets == index).astype(np.int8)
            false_positive, true_positive, _ = roc_curve(binary, probabilities[:, index])
            axes.plot(
                false_positive,
                true_positive,
                alpha=_CURVE_ALPHA,
                linewidth=_CURVE_WIDTH,
                label=f"{entry.name} ({entry.roc_auc:.3f})",
            )

        micro_auc = _plot_micro_roc(axes, probabilities, targets, len(metrics.class_names))
        axes.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="grey", linewidth=1.0)
        axes.set_xlabel("False positive rate")
        axes.set_ylabel("True positive rate")
        axes.set_title(_curve_title("ROC", metrics.overall.macro_roc_auc, micro_auc))
        _apply_curve_legend(axes, len(metrics.class_names))
        axes.grid(visible=True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(path, dpi=specification.dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


def _plot_pr_curves(
    path: Path,
    metrics: EvaluationMetrics,
    probabilities: np.ndarray,
    targets: np.ndarray,
    specification: EvaluationPlotSpecification,
) -> Path:
    """Draw one-vs-rest precision-recall curves plus the micro-average."""
    figure, axes = plt.subplots(figsize=specification.figure_size)
    try:
        for index, entry in enumerate(metrics.per_class):
            if entry.pr_auc is None:
                continue
            binary = (targets == index).astype(np.int8)
            precision, recall, _ = precision_recall_curve(binary, probabilities[:, index])
            axes.plot(
                recall,
                precision,
                alpha=_CURVE_ALPHA,
                linewidth=_CURVE_WIDTH,
                label=f"{entry.name} ({entry.pr_auc:.3f})",
            )

        one_hot = _one_hot(targets, len(metrics.class_names))
        micro_ap = float(average_precision_score(one_hot.ravel(), probabilities.ravel()))
        precision, recall, _ = precision_recall_curve(one_hot.ravel(), probabilities.ravel())
        axes.plot(recall, precision, color="black", linewidth=2.0, label=f"micro ({micro_ap:.4f})")

        axes.set_xlabel("Recall")
        axes.set_ylabel("Precision")
        axes.set_title(_curve_title("Precision-recall", metrics.overall.macro_pr_auc, micro_ap))
        _apply_curve_legend(axes, len(metrics.class_names))
        axes.grid(visible=True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(path, dpi=specification.dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


def _plot_calibration(
    path: Path,
    metrics: EvaluationMetrics,
    specification: EvaluationPlotSpecification,
) -> Path:
    """Draw the reliability diagram above a histogram of confidences."""
    calibration = metrics.calibration
    populated = [bucket for bucket in calibration.bins if bucket.count > 0]
    centres = [(bucket.lower + bucket.upper) / 2.0 for bucket in calibration.bins]
    width = (calibration.bins[0].upper - calibration.bins[0].lower) * 0.9

    width_inches, height_inches = specification.figure_size
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(width_inches, height_inches * 1.4),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    try:
        top.bar(
            [(bucket.lower + bucket.upper) / 2.0 for bucket in populated],
            [bucket.accuracy for bucket in populated],
            width=width,
            edgecolor="black",
            label="Accuracy",
        )
        top.plot(
            [bucket.mean_confidence for bucket in populated],
            [bucket.accuracy for bucket in populated],
            marker="o",
            color="darkorange",
            label="Accuracy vs mean confidence",
        )
        top.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="grey", label="Perfect calibration")
        top.set_ylabel("Accuracy")
        top.set_ylim(0.0, 1.0)
        top.set_title(
            f"Reliability diagram — ECE {calibration.expected_calibration_error:.4f}, "
            f"MCE {calibration.maximum_calibration_error:.4f}, "
            f"mean confidence {calibration.mean_confidence:.4f} vs "
            f"accuracy {calibration.accuracy:.4f}"
        )
        top.legend(loc="upper left")
        top.grid(visible=True, alpha=0.3)

        bottom.bar(
            centres,
            [bucket.count for bucket in calibration.bins],
            width=width,
            edgecolor="black",
            color="steelblue",
        )
        bottom.set_xlabel("Predicted confidence")
        bottom.set_ylabel("Samples")
        bottom.set_yscale("symlog")
        bottom.grid(visible=True, alpha=0.3)

        figure.tight_layout()
        figure.savefig(path, dpi=specification.dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


def _plot_micro_roc(
    axes: plt.Axes,
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> float:
    """Draw the micro-average ROC curve and return its area."""
    one_hot = _one_hot(targets, num_classes)
    false_positive, true_positive, _ = roc_curve(one_hot.ravel(), probabilities.ravel())
    area = float(auc(false_positive, true_positive))
    axes.plot(
        false_positive, true_positive, color="black", linewidth=2.0, label=f"micro ({area:.4f})"
    )
    return area


def _one_hot(targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Return the one-hot encoding of ``targets``."""
    encoded = np.zeros((targets.size, num_classes), dtype=np.int8)
    encoded[np.arange(targets.size), targets] = 1
    return encoded


def _curve_title(kind: str, macro: float | None, micro: float) -> str:
    """Build a curve figure title quoting both averages."""
    macro_text = "undefined" if macro is None else f"{macro:.4f}"
    return f"{kind} curves, one-vs-rest — macro {macro_text}, micro {micro:.4f}"


def _apply_curve_legend(axes: plt.Axes, num_classes: int) -> None:
    """Show a per-class legend only when it would stay readable."""
    if num_classes <= _MAX_LEGEND_CLASSES:
        axes.legend(loc="lower right", fontsize="small")
        return
    handles, labels = axes.get_legend_handles_labels()
    micro = [(handle, label) for handle, label in zip(handles, labels, strict=True)
             if label.startswith("micro")]
    if micro:
        axes.legend([micro[0][0]], [micro[0][1]], loc="lower right")
