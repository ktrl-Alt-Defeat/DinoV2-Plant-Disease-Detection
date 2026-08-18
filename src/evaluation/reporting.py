"""Evaluation artifact writers.

Every file the evaluation produces is written from here, so the set of exported
artifacts is visible in one place. The writers reuse the project-wide
serialisation helpers rather than formatting JSON or CSV by hand.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.config import Config, TypeSpec, validate_keys
from src.evaluation.metrics import PER_CLASS_FIELDNAMES, EvaluationMetrics
from src.logger import get_logger
from src.utils import write_csv, write_json, write_text

#: Header of the first column of the confusion matrix CSV.
CONFUSION_INDEX_COLUMN: Final[str] = "true_class"

#: Configuration contract of the ``evaluation.filenames`` section.
FILENAME_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("evaluation.filenames.report", str),
    ("evaluation.filenames.per_class", str),
    ("evaluation.filenames.classification_report", str),
    ("evaluation.filenames.confusion_matrix_csv", str),
    ("evaluation.filenames.benchmark", str),
)

_LOGGER: Final = get_logger("evaluation.reporting")


@dataclass(frozen=True)
class ReportFilenames:
    """Names of the non-figure artifacts, resolved from the configuration."""

    report: str
    per_class: str
    classification_report: str
    confusion_matrix_csv: str
    benchmark: str

    @classmethod
    def from_config(cls, config: Config) -> "ReportFilenames":
        """Read and validate the ``evaluation.filenames`` entries used here.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
        """
        validate_keys(config, FILENAME_REQUIRED_KEYS, context="evaluation.filenames section")
        return cls(
            report=config.get("evaluation.filenames.report"),
            per_class=config.get("evaluation.filenames.per_class"),
            classification_report=config.get("evaluation.filenames.classification_report"),
            confusion_matrix_csv=config.get("evaluation.filenames.confusion_matrix_csv"),
            benchmark=config.get("evaluation.filenames.benchmark"),
        )


def write_reports(
    metrics: EvaluationMetrics,
    payload: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    results_dir: str | Path,
    filenames: ReportFilenames,
) -> dict[str, Path]:
    """Write every non-figure artifact into ``results_dir``.

    Returns:
        The absolute path of each artifact, keyed by artifact name.
    """
    directory = Path(results_dir)
    paths = {
        "evaluation_report": write_json(directory / filenames.report, dict(payload)),
        "per_class_metrics": write_csv(
            directory / filenames.per_class,
            [entry.as_row() for entry in metrics.per_class],
            fieldnames=PER_CLASS_FIELDNAMES,
        ),
        "classification_report": write_text(
            directory / filenames.classification_report,
            _classification_report_text(metrics),
        ),
        "confusion_matrix_csv": write_csv(
            directory / filenames.confusion_matrix_csv,
            _confusion_rows(metrics.confusion_matrix, metrics.class_names),
            fieldnames=(CONFUSION_INDEX_COLUMN, *metrics.class_names),
        ),
        "inference_benchmark": write_json(directory / filenames.benchmark, dict(benchmark)),
    }
    for name, path in paths.items():
        _LOGGER.info("Artifact written: %s -> %s", name, path)
    return paths


def _classification_report_text(metrics: EvaluationMetrics) -> str:
    """Render the classification report with its provenance header."""
    limitations = metrics.limitations
    lines = [
        "Classification report",
        f"Samples: {metrics.sample_count:,}",
        f"Classes: {len(metrics.class_names)}",
        "",
        metrics.classification_report,
        "",
        "Notes",
    ]
    lines.extend(f"  - {note}" for note in limitations.notes)
    return "\n".join(lines)


def _confusion_rows(
    matrix: np.ndarray,
    class_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Turn the confusion matrix into CSV rows of raw counts.

    Rows are true classes and columns are predicted classes, matching the
    convention used by the figure.
    """
    return [
        {
            CONFUSION_INDEX_COLUMN: name,
            **{
                predicted: int(matrix[row_index, column_index])
                for column_index, predicted in enumerate(class_names)
            },
        }
        for row_index, name in enumerate(class_names)
    ]
