"""Metrics, evaluation loops and reporting.

:mod:`~src.evaluation.inference` runs the held-out split once,
:mod:`~src.evaluation.metrics` turns that single pass into every reported
number, :mod:`~src.evaluation.integrity` proves the run was read-only, and
:mod:`~src.evaluation.reporting` exports the artifacts. :mod:`src.evaluate`
composes them into a run.
"""

from src.evaluation.inference import (
    BenchmarkResult,
    BenchmarkSpecification,
    InferenceOutputs,
    benchmark_inference,
    run_inference,
    softmax_probabilities,
)
from src.evaluation.integrity import (
    Fingerprint,
    IntegrityCheck,
    IntegrityError,
    fingerprint_file,
    parameter_digest,
)
from src.evaluation.metrics import (
    CalibrationSummary,
    ClassMetrics,
    EvaluationMetrics,
    MetricLimitations,
    OverallMetrics,
    compute_metrics,
)
from src.evaluation.reporting import ReportFilenames, write_reports

__all__ = [
    "BenchmarkResult",
    "BenchmarkSpecification",
    "CalibrationSummary",
    "ClassMetrics",
    "EvaluationMetrics",
    "Fingerprint",
    "InferenceOutputs",
    "IntegrityCheck",
    "IntegrityError",
    "MetricLimitations",
    "OverallMetrics",
    "ReportFilenames",
    "benchmark_inference",
    "compute_metrics",
    "fingerprint_file",
    "parameter_digest",
    "run_inference",
    "softmax_probabilities",
    "write_reports",
]
