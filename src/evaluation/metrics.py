"""Test-set metric computation.

Every metric is derived from a single set of probabilities and targets, so the
report cannot describe two different passes. Metrics that are mathematically
undefined for a given class — a class with no test samples has no recall, a
class that is never negative has no ROC curve — are recorded as ``None`` and
listed in :class:`MetricLimitations` rather than silently coerced to zero.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Final

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.logger import get_logger

#: Column order of ``results/per_class_metrics.csv``.
PER_CLASS_FIELDNAMES: Final[tuple[str, ...]] = (
    "index",
    "name",
    "support",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
)

#: Value written to CSV where a metric is undefined for a class.
UNDEFINED_CSV_VALUE: Final[str] = ""

_LOGGER: Final = get_logger("evaluation.metrics")


@dataclass(frozen=True)
class ClassMetrics:
    """Metrics of one class, computed one-vs-rest."""

    index: int
    name: str
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None

    def as_row(self) -> dict[str, Any]:
        """Return the metrics as a CSV row, blanking undefined values."""
        return {
            "index": self.index,
            "name": self.name,
            "support": self.support,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "roc_auc": _round_or_blank(self.roc_auc),
            "pr_auc": _round_or_blank(self.pr_auc),
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the metrics as a serialisable mapping."""
        row = self.as_row()
        row["roc_auc"] = self.roc_auc
        row["pr_auc"] = self.pr_auc
        return row


@dataclass(frozen=True)
class CalibrationBin:
    """One confidence bucket of the reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    def as_dict(self) -> dict[str, Any]:
        """Return the bucket as a serialisable mapping."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True)
class CalibrationSummary:
    """Reliability of the predicted confidences."""

    bins: tuple[CalibrationBin, ...]
    expected_calibration_error: float
    maximum_calibration_error: float
    mean_confidence: float
    accuracy: float

    @property
    def overconfidence(self) -> float:
        """Mean confidence minus accuracy; positive means overconfident."""
        return self.mean_confidence - self.accuracy

    def as_dict(self) -> dict[str, Any]:
        """Return the summary as a serialisable mapping."""
        return {
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
            "overconfidence": self.overconfidence,
            "bins": [bucket.as_dict() for bucket in self.bins],
        }


@dataclass(frozen=True)
class OverallMetrics:
    """Aggregate metrics across the whole evaluated split."""

    loss: float
    top1_accuracy: float
    topk_accuracy: float
    top_k: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    macro_roc_auc: float | None
    weighted_roc_auc: float | None
    macro_pr_auc: float | None
    weighted_pr_auc: float | None
    expected_calibration_error: float

    def as_dict(self) -> dict[str, Any]:
        """Return the metrics as a serialisable mapping."""
        return {
            "loss": self.loss,
            "top1_accuracy": self.top1_accuracy,
            f"top{self.top_k}_accuracy": self.topk_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "macro_roc_auc_ovr": self.macro_roc_auc,
            "weighted_roc_auc_ovr": self.weighted_roc_auc,
            "macro_pr_auc": self.macro_pr_auc,
            "weighted_pr_auc": self.weighted_pr_auc,
            "expected_calibration_error": self.expected_calibration_error,
        }


@dataclass(frozen=True)
class MetricLimitations:
    """Metrics that could not be computed, and the reason why."""

    classes_without_support: tuple[str, ...]
    classes_without_roc_auc: tuple[str, ...]
    classes_without_pr_auc: tuple[str, ...]
    requested_top_k: int
    effective_top_k: int
    notes: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        """Whether every requested metric was computable."""
        return not (
            self.classes_without_support
            or self.classes_without_roc_auc
            or self.classes_without_pr_auc
            or self.requested_top_k != self.effective_top_k
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the limitations as a serialisable mapping."""
        return {
            "complete": self.is_complete,
            "classes_without_support": list(self.classes_without_support),
            "classes_without_roc_auc": list(self.classes_without_roc_auc),
            "classes_without_pr_auc": list(self.classes_without_pr_auc),
            "requested_top_k": self.requested_top_k,
            "effective_top_k": self.effective_top_k,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EvaluationMetrics:
    """Everything computed from one evaluation pass."""

    overall: OverallMetrics
    per_class: tuple[ClassMetrics, ...]
    confusion_matrix: np.ndarray
    class_names: tuple[str, ...]
    calibration: CalibrationSummary
    limitations: MetricLimitations
    classification_report: str
    sample_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return the metrics as a serialisable mapping, matrix excluded."""
        return {
            "sample_count": self.sample_count,
            "num_classes": len(self.class_names),
            "overall": self.overall.as_dict(),
            "calibration": self.calibration.as_dict(),
            "limitations": self.limitations.as_dict(),
            "per_class": [metrics.as_dict() for metrics in self.per_class],
        }


def compute_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str],
    *,
    loss: float,
    top_k: int,
    calibration_bins: int,
) -> EvaluationMetrics:
    """Derive every reported metric from one set of probabilities and targets.

    Args:
        probabilities: ``[samples, classes]`` softmax output.
        targets: ``[samples]`` ground-truth class indices.
        class_names: Class name per index, ordered by index.
        loss: Mean cross-entropy already accumulated over the split.
        top_k: Requested k for Top-k accuracy.
        calibration_bins: Number of equal-width confidence buckets.

    Raises:
        ValueError: If the arrays disagree in shape or the settings are invalid.
    """
    _validate_inputs(probabilities, targets, class_names, top_k, calibration_bins)

    num_classes = len(class_names)
    labels = np.arange(num_classes)
    predictions = probabilities.argmax(axis=1)

    matrix = confusion_matrix(targets, predictions, labels=labels)
    support = matrix.sum(axis=1)
    true_positives = matrix.diagonal()
    false_positives = matrix.sum(axis=0) - true_positives
    false_negatives = support - true_positives

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    roc_auc, pr_auc, without_roc, without_pr = _per_class_curve_scores(
        probabilities, targets, labels, class_names
    )

    per_class = tuple(
        ClassMetrics(
            index=int(index),
            name=class_names[index],
            support=int(support[index]),
            true_positives=int(true_positives[index]),
            false_positives=int(false_positives[index]),
            false_negatives=int(false_negatives[index]),
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            roc_auc=roc_auc[index],
            pr_auc=pr_auc[index],
        )
        for index in labels
    )

    effective_k = min(top_k, num_classes)
    calibration = _calibration(probabilities, targets, predictions, calibration_bins)
    limitations = _limitations(
        class_names, support, without_roc, without_pr, top_k, effective_k, num_classes
    )

    overall = OverallMetrics(
        loss=loss,
        top1_accuracy=float((predictions == targets).mean()),
        topk_accuracy=_top_k_accuracy(probabilities, targets, effective_k),
        top_k=effective_k,
        macro_precision=_mean(precision),
        macro_recall=_mean(recall),
        macro_f1=_mean(f1),
        weighted_precision=_weighted(precision, support),
        weighted_recall=_weighted(recall, support),
        weighted_f1=_weighted(f1, support),
        macro_roc_auc=_mean_defined(roc_auc),
        weighted_roc_auc=_weighted_defined(roc_auc, support),
        macro_pr_auc=_mean_defined(pr_auc),
        weighted_pr_auc=_weighted_defined(pr_auc, support),
        expected_calibration_error=calibration.expected_calibration_error,
    )

    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )

    _LOGGER.info(
        "Metrics computed over %d samples: top-1 %.4f, macro-F1 %.4f, ECE %.4f.",
        targets.size,
        overall.top1_accuracy,
        overall.macro_f1,
        overall.expected_calibration_error,
    )
    return EvaluationMetrics(
        overall=overall,
        per_class=per_class,
        confusion_matrix=matrix,
        class_names=tuple(class_names),
        calibration=calibration,
        limitations=limitations,
        classification_report=report,
        sample_count=int(targets.size),
    )


def _validate_inputs(
    probabilities: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str],
    top_k: int,
    calibration_bins: int,
) -> None:
    """Check the arrays and settings agree before any metric is computed.

    Raises:
        ValueError: If a shape, index or setting is unusable.
    """
    if probabilities.ndim != 2:
        raise ValueError(f"Probabilities must be 2-D, got shape {probabilities.shape}.")
    if targets.ndim != 1:
        raise ValueError(f"Targets must be 1-D, got shape {targets.shape}.")
    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Probabilities describe {probabilities.shape[0]} samples but there are "
            f"{targets.shape[0]} targets."
        )
    if probabilities.shape[1] != len(class_names):
        raise ValueError(
            f"Probabilities have {probabilities.shape[1]} columns but "
            f"{len(class_names)} class names were supplied."
        )
    if targets.size == 0:
        raise ValueError("Cannot compute metrics over an empty split.")
    if int(targets.min()) < 0 or int(targets.max()) >= len(class_names):
        raise ValueError(
            f"Targets fall outside [0, {len(class_names) - 1}]: "
            f"observed [{int(targets.min())}, {int(targets.max())}]."
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    if calibration_bins <= 0:
        raise ValueError(f"calibration_bins must be positive, got {calibration_bins}.")


def _per_class_curve_scores(
    probabilities: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> tuple[list[float | None], list[float | None], list[str], list[str]]:
    """Compute one-vs-rest ROC-AUC and PR-AUC for every class.

    Both curves need at least one positive and one negative sample. A class that
    has neither is reported as ``None`` and named in the returned lists instead
    of contributing a misleading score to the averages.
    """
    roc_scores: list[float | None] = []
    pr_scores: list[float | None] = []
    without_roc: list[str] = []
    without_pr: list[str] = []

    for index in labels:
        binary = (targets == index).astype(np.int8)
        positives = int(binary.sum())
        scores = probabilities[:, index]

        if 0 < positives < binary.size:
            roc_scores.append(float(roc_auc_score(binary, scores)))
            pr_scores.append(float(average_precision_score(binary, scores)))
            continue

        roc_scores.append(None)
        pr_scores.append(None)
        without_roc.append(class_names[index])
        without_pr.append(class_names[index])

    return roc_scores, pr_scores, without_roc, without_pr


def _calibration(
    probabilities: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    bin_count: int,
) -> CalibrationSummary:
    """Bucket predictions by confidence and measure calibration error.

    ECE is the support-weighted mean gap between confidence and accuracy across
    buckets; MCE is the largest single gap.
    """
    confidence = probabilities.max(axis=1)
    correct = (predictions == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bin_count + 1)

    buckets: list[CalibrationBin] = []
    total_error = 0.0
    maximum_error = 0.0

    for lower, upper in pairwise(edges):
        in_bin = _bucket_mask(confidence, float(lower), float(upper))
        count = int(in_bin.sum())
        if count == 0:
            buckets.append(CalibrationBin(float(lower), float(upper), 0, 0.0, 0.0))
            continue

        mean_confidence = float(confidence[in_bin].mean())
        accuracy = float(correct[in_bin].mean())
        gap = abs(accuracy - mean_confidence)
        total_error += (count / confidence.size) * gap
        maximum_error = max(maximum_error, gap)
        buckets.append(
            CalibrationBin(float(lower), float(upper), count, mean_confidence, accuracy)
        )

    return CalibrationSummary(
        bins=tuple(buckets),
        expected_calibration_error=total_error,
        maximum_calibration_error=maximum_error,
        mean_confidence=float(confidence.mean()),
        accuracy=float(correct.mean()),
    )


def _bucket_mask(confidence: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Select the confidences falling in one bucket.

    Buckets are half-open on the left so no sample is counted twice; the first
    bucket also owns its lower edge so a confidence of exactly zero is kept.
    """
    if lower == 0.0:
        return (confidence >= lower) & (confidence <= upper)
    return (confidence > lower) & (confidence <= upper)


def _limitations(
    class_names: Sequence[str],
    support: np.ndarray,
    without_roc: Sequence[str],
    without_pr: Sequence[str],
    requested_k: int,
    effective_k: int,
    num_classes: int,
) -> MetricLimitations:
    """Collect everything the report has to disclose about missing metrics."""
    missing = tuple(
        class_names[index] for index in range(len(class_names)) if support[index] == 0
    )
    notes: list[str] = []

    if missing:
        notes.append(
            f"{len(missing)} class(es) have no sample in the evaluated split; their "
            "precision, recall and F1 are reported as 0 and they contribute nothing "
            "to the weighted averages."
        )
    if without_roc:
        notes.append(
            f"ROC-AUC and PR-AUC are undefined for {len(without_roc)} class(es) that "
            "have no positive or no negative sample; they are excluded from the "
            "macro and weighted curve averages."
        )
    if requested_k != effective_k:
        notes.append(
            f"Top-{requested_k} accuracy was requested but the dataset has only "
            f"{num_classes} classes, so Top-{effective_k} is reported instead."
        )
    if not notes:
        notes.append("Every requested metric was computable for every class.")

    return MetricLimitations(
        classes_without_support=missing,
        classes_without_roc_auc=tuple(without_roc),
        classes_without_pr_auc=tuple(without_pr),
        requested_top_k=requested_k,
        effective_top_k=effective_k,
        notes=tuple(notes),
    )


def _top_k_accuracy(probabilities: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Fraction of samples whose true class is among the k highest scores."""
    if k >= probabilities.shape[1]:
        return 1.0
    top = np.argpartition(-probabilities, kth=k - 1, axis=1)[:, :k]
    return float((top == targets[:, None]).any(axis=1).mean())


def _mean(values: np.ndarray) -> float:
    """Unweighted mean of a per-class metric."""
    return float(np.mean(values))


def _weighted(values: np.ndarray, support: np.ndarray) -> float:
    """Support-weighted mean of a per-class metric."""
    if support.sum() == 0:
        return 0.0
    return float(np.average(values, weights=support))


def _mean_defined(values: Sequence[float | None]) -> float | None:
    """Unweighted mean over the classes where the metric is defined."""
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def _weighted_defined(values: Sequence[float | None], support: np.ndarray) -> float | None:
    """Support-weighted mean over the classes where the metric is defined."""
    pairs = [(value, support[index]) for index, value in enumerate(values) if value is not None]
    total = sum(weight for _, weight in pairs)
    if not pairs or total == 0:
        return None
    return float(sum(value * weight for value, weight in pairs) / total)


def _round_or_blank(value: float | None) -> Any:
    """Round a metric for CSV, or blank it when it is undefined."""
    return UNDEFINED_CSV_VALUE if value is None else round(value, 6)
