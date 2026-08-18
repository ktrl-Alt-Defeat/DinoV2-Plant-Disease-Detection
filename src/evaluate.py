"""Command line entry point for evaluating a trained checkpoint.

Running ``python -m src.evaluate`` restores the best checkpoint, runs it once
over the held-out split and exports the full evaluation report. The run is
read-only by construction: no optimizer, scheduler or scaler is built, the pass
executes under ``model.eval()`` and ``torch.no_grad()``, no augmentation is
applied and no test-time augmentation is performed. The checkpoint file and the
model weights are fingerprinted before and after the pass, and a mismatch aborts
the run rather than producing a report.
"""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import reporting
from src.cli import bootstrap, build_parser
from src.config import Config, ConfigError, TypeSpec, load_config, validate_keys, with_overrides
from src.datasets.loaders import DataBundle, DataLoaderSpecification, build_dataloaders
from src.datasets.transforms import TransformSpecification
from src.datasets.validation import (
    DatasetSpecification,
    DatasetValidationError,
    audit_dataset,
)
from src.evaluation import integrity
from src.evaluation.inference import (
    EVALUATION_PRECISION,
    BenchmarkResult,
    BenchmarkSpecification,
    benchmark_inference,
    peak_memory_mib,
    run_inference,
    softmax_probabilities,
)
from src.evaluation.integrity import IntegrityError
from src.evaluation.metrics import EvaluationMetrics, compute_metrics
from src.evaluation.reporting import ReportFilenames, write_reports
from src.logger import configure_console_encoding, get_logger
from src.model import DinoV2Classifier, ModelBuildError, build_model
from src.training.checkpoints import CheckpointError, CheckpointMetadata, load_model_checkpoint
from src.utils import format_duration
from src.visualization.evaluation_plots import (
    EvaluationPlotSpecification,
    write_evaluation_figures,
)

#: Configuration key naming the split that is evaluated.
SPLIT_KEY: Final[str] = "evaluation.split"

#: Configuration contract of the scalar ``evaluation`` settings.
EVALUATION_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("evaluation.checkpoint_filename", str),
    ("evaluation.split", str),
    ("evaluation.log_filename", str),
    ("evaluation.top_k", int),
    ("evaluation.calibration_bins", int),
    ("evaluation.probability_tolerance", (int, float)),
)

_TITLE: Final[str] = "MILESTONE 5 — TEST SET EVALUATION"

_LOGGER: Final = get_logger("evaluate")


class EvaluationError(RuntimeError):
    """Raised when an evaluation run cannot be completed."""


@dataclass(frozen=True)
class EvaluationSettings:
    """Evaluation settings resolved from the ``evaluation`` section."""

    checkpoint_filename: str
    split: str
    log_filename: str
    top_k: int
    calibration_bins: int
    probability_tolerance: float

    @classmethod
    def from_config(cls, config: Config) -> "EvaluationSettings":
        """Read and validate the scalar ``evaluation`` settings.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If a setting is out of range.
        """
        validate_keys(config, EVALUATION_REQUIRED_KEYS, context="evaluation section")

        top_k = config.get("evaluation.top_k")
        bins = config.get("evaluation.calibration_bins")
        tolerance = float(config.get("evaluation.probability_tolerance"))

        if top_k <= 0:
            raise ValueError(f"evaluation.top_k must be positive, got {top_k}.")
        if bins <= 0:
            raise ValueError(f"evaluation.calibration_bins must be positive, got {bins}.")
        if tolerance <= 0.0:
            raise ValueError(
                f"evaluation.probability_tolerance must be positive, got {tolerance}."
            )

        return cls(
            checkpoint_filename=config.get("evaluation.checkpoint_filename"),
            split=config.get(SPLIT_KEY),
            log_filename=config.get("evaluation.log_filename"),
            top_k=top_k,
            calibration_bins=bins,
            probability_tolerance=tolerance,
        )


@dataclass(frozen=True)
class EvaluationOutcome:
    """Everything a completed evaluation produced."""

    split: str
    metrics: EvaluationMetrics
    benchmark: BenchmarkResult
    checkpoint: CheckpointMetadata
    integrity_checks: tuple[integrity.IntegrityCheck, ...]
    total_parameters: int
    trainable_parameters: int
    end_to_end_throughput: float
    peak_gpu_memory_mib: float
    artifacts: dict[str, Path]


def run_evaluation(config_path: str | Path, *, checkpoint: Path | None = None) -> EvaluationOutcome:
    """Evaluate the configured checkpoint on the held-out split.

    Raises:
        DatasetValidationError: If the dataset on disk is unusable.
        CheckpointError: If the checkpoint cannot be read or does not match.
        IntegrityError: If a read-only guarantee was violated.
        EvaluationError: If the configured split cannot be evaluated.
    """
    settings = EvaluationSettings.from_config(load_config(config_path))
    boot = bootstrap(config_path, log_filename=settings.log_filename)
    config = boot.config
    device = boot.device_info.device
    results_dir = Path(boot.paths.results)

    _LOGGER.info(
        "Evaluation starting on %s (%s), split '%s', precision %s.",
        device,
        boot.device_info.name,
        settings.split,
        EVALUATION_PRECISION,
    )

    bundle = _build_bundle(config, device)
    loader = _select_loader(bundle, settings.split)
    checkpoint_path = _resolve_checkpoint(boot.paths.checkpoints, settings, checkpoint)

    fingerprint_before = integrity.fingerprint_file(checkpoint_path)
    _LOGGER.info(
        "Checkpoint fingerprint before evaluation: sha256 %s (%s bytes).",
        fingerprint_before.sha256[:16],
        f"{fingerprint_before.size_bytes:,}",
    )

    model = _build_model(config, bundle)
    metadata = load_model_checkpoint(
        checkpoint_path,
        model=model,
        device=device,
        expected_class_to_idx=bundle.class_to_idx,
    )
    model.eval()
    weights_before = integrity.parameter_digest(model)

    outputs = run_inference(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        description=f"Evaluating [{settings.split}]",
    )
    evaluation_peak_mib = peak_memory_mib(device)

    weights_after = integrity.parameter_digest(model)
    fingerprint_after = integrity.fingerprint_file(checkpoint_path)

    logits = outputs.logits.numpy()
    probabilities = softmax_probabilities(outputs.logits)
    targets = outputs.targets.numpy()

    checks = (
        integrity.check_checkpoint_unchanged(fingerprint_before, fingerprint_after),
        integrity.check_weights_unchanged(weights_before, weights_after),
        integrity.check_evaluation_mode(model),
        integrity.check_finite("Logits Finite", logits),
        integrity.check_finite("Probabilities Finite", probabilities),
        integrity.check_probabilities(
            probabilities, tolerance=settings.probability_tolerance
        ),
        integrity.check_split_coverage(
            bundle.split_sizes[settings.split], outputs.sample_count, settings.split
        ),
    )
    integrity.enforce(checks)

    metrics = compute_metrics(
        probabilities,
        targets,
        _class_names(bundle),
        loss=outputs.loss,
        top_k=settings.top_k,
        calibration_bins=settings.calibration_bins,
    )
    benchmark = benchmark_inference(
        model,
        loader,
        device,
        BenchmarkSpecification.from_config(config),
        description="Benchmarking forward pass",
    )

    payload = _report_payload(
        settings=settings,
        metrics=metrics,
        benchmark=benchmark,
        metadata=metadata,
        checks=checks,
        fingerprint=fingerprint_before,
        model=model,
        device=device,
        bundle=bundle,
        outputs_seconds=outputs.seconds,
        throughput=outputs.throughput,
        peak_mib=evaluation_peak_mib,
    )
    artifacts = _write_artifacts(
        config=config,
        metrics=metrics,
        benchmark=benchmark,
        payload=payload,
        probabilities=probabilities,
        targets=targets,
        results_dir=results_dir,
    )

    _log_headline(metrics, settings.split)
    return EvaluationOutcome(
        split=settings.split,
        metrics=metrics,
        benchmark=benchmark,
        checkpoint=metadata,
        integrity_checks=checks,
        total_parameters=model.count_parameters(),
        trainable_parameters=model.count_trainable_parameters(),
        end_to_end_throughput=outputs.throughput,
        peak_gpu_memory_mib=evaluation_peak_mib,
        artifacts=artifacts,
    )


def render_summary(outcome: EvaluationOutcome) -> str:
    """Render the console report shown at the end of an evaluation."""
    overall = outcome.metrics.overall
    benchmark = outcome.benchmark
    lines = reporting.banner(_TITLE)
    lines.extend(
        reporting.entries(
            [
                ("Split", f"{outcome.split} ({outcome.metrics.sample_count:,} images)"),
                (
                    "Checkpoint",
                    f"{outcome.checkpoint.path.name} (epoch {outcome.checkpoint.epoch})",
                ),
                ("Precision", EVALUATION_PRECISION),
                ("Test Loss", f"{overall.loss:.6f}"),
                ("Top-1 Accuracy", f"{overall.top1_accuracy:.6f}"),
                (f"Top-{overall.top_k} Accuracy", f"{overall.topk_accuracy:.6f}"),
                (
                    "Macro P / R / F1",
                    f"{overall.macro_precision:.6f} / {overall.macro_recall:.6f} / "
                    f"{overall.macro_f1:.6f}",
                ),
                (
                    "Weighted P / R / F1",
                    f"{overall.weighted_precision:.6f} / {overall.weighted_recall:.6f} / "
                    f"{overall.weighted_f1:.6f}",
                ),
                ("ROC-AUC (OvR)", _optional(overall.macro_roc_auc, overall.weighted_roc_auc)),
                ("PR-AUC", _optional(overall.macro_pr_auc, overall.weighted_pr_auc)),
                (
                    "Calibration",
                    f"ECE {overall.expected_calibration_error:.6f}, "
                    f"MCE {outcome.metrics.calibration.maximum_calibration_error:.6f}",
                ),
                (
                    "Latency",
                    f"{benchmark.mean_batch_ms:.2f} ms/batch, "
                    f"{benchmark.mean_image_ms:.3f} ms/image "
                    f"(p95 {benchmark.percentile_batch_ms[95]:.2f} ms)",
                ),
                (
                    "Throughput",
                    f"{benchmark.throughput_images_per_second:.1f} images/s forward-only, "
                    f"{outcome.end_to_end_throughput:.1f} images/s end-to-end",
                ),
                ("Peak GPU Memory", f"{outcome.peak_gpu_memory_mib:.1f} MiB"),
                (
                    "Parameters",
                    f"{outcome.total_parameters:,} total, "
                    f"{outcome.trainable_parameters:,} trainable",
                ),
            ]
        )
    )
    lines.extend(reporting.rule())

    for check in outcome.integrity_checks:
        lines.extend(reporting.entry(f"Integrity: {check.name}", check.status, check.details))

    lines.extend(reporting.rule())
    for note in outcome.metrics.limitations.notes:
        lines.extend(reporting.entry("Note", note))

    lines.extend(reporting.rule())
    lines.extend(
        reporting.entries(
            sorted((name, str(path)) for name, path in outcome.artifacts.items())
        )
    )
    lines.extend(reporting.closing("EVALUATION STATUS : PASS"))
    return reporting.render(lines)


def build_evaluation_parser() -> argparse.ArgumentParser:
    """Build the argument parser, adding ``--checkpoint`` to the shared options."""
    parser = build_parser(
        prog="python -m src.evaluate",
        description="Evaluate a trained checkpoint on the held-out split.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint to evaluate. Overrides evaluation.checkpoint_filename; "
            "relative paths resolve against the repository root."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation command line interface and return a process exit code."""
    configure_console_encoding()
    arguments = build_evaluation_parser().parse_args(argv)

    try:
        outcome = run_evaluation(arguments.config, checkpoint=arguments.checkpoint)
    except (
        DatasetValidationError,
        CheckpointError,
        IntegrityError,
        EvaluationError,
        ModelBuildError,
        ConfigError,
        KeyError,
        OSError,
        ValueError,
    ) as error:
        _LOGGER.error("Evaluation failed: %s", error)
        print(reporting.render([*reporting.banner(_TITLE), f"ERROR: {error}", ""]))
        print(reporting.render(reporting.closing("EVALUATION STATUS : FAIL")))
        return 1

    print(render_summary(outcome))
    return 0


def _build_bundle(config: Config, device: torch.device) -> DataBundle:
    """Audit the dataset and build the dataloaders.

    Raises:
        DatasetValidationError: If the dataset carries any error-severity issue.
    """
    specification = DatasetSpecification.from_config(config)
    audit = audit_dataset(specification)
    if not audit.passed:
        first = audit.errors[0]
        raise DatasetValidationError(
            f"Dataset audit failed with {len(audit.errors)} error(s); evaluation aborted. "
            f"First error: {first.category} at {first.location} ({first.detail})."
        )
    for issue in audit.warnings:
        _LOGGER.warning("%s at %s: %s", issue.category, issue.location, issue.detail)

    return build_dataloaders(
        specification,
        TransformSpecification.from_config(config),
        DataLoaderSpecification.from_config(config),
        seed=config.get("project.seed"),
        device=device,
    )


def _select_loader(bundle: DataBundle, split: str) -> DataLoader:
    """Return the dataloader of the evaluated split.

    Raises:
        EvaluationError: If the configured split is unknown or empty.
    """
    try:
        loader = bundle.loader_for(split)
    except KeyError as error:
        raise EvaluationError(f"Cannot evaluate split '{split}': {error}") from error

    if bundle.split_sizes[split] == 0:
        raise EvaluationError(f"Split '{split}' holds no image; nothing to evaluate.")
    return loader


def _resolve_checkpoint(
    checkpoints_dir: Path,
    settings: EvaluationSettings,
    override: Path | None,
) -> Path:
    """Resolve which checkpoint file to evaluate."""
    if override is not None:
        return override
    return checkpoints_dir / settings.checkpoint_filename


def _build_model(config: Config, bundle: DataBundle) -> DinoV2Classifier:
    """Build the model with the class count discovered on disk.

    The backbone is instantiated without pretrained download because the
    checkpoint supplies every weight immediately afterwards.
    """
    model_config = with_overrides(
        config, model={"num_classes": bundle.num_classes, "pretrained": False}
    )
    return build_model(model_config)


def _class_names(bundle: DataBundle) -> tuple[str, ...]:
    """Return the class names ordered by their index."""
    inverse = bundle.idx_to_class
    return tuple(inverse[index] for index in range(bundle.num_classes))


def _report_payload(
    *,
    settings: EvaluationSettings,
    metrics: EvaluationMetrics,
    benchmark: BenchmarkResult,
    metadata: CheckpointMetadata,
    checks: Sequence[integrity.IntegrityCheck],
    fingerprint: integrity.Fingerprint,
    model: DinoV2Classifier,
    device: torch.device,
    bundle: DataBundle,
    outputs_seconds: float,
    throughput: float,
    peak_mib: float,
) -> dict[str, Any]:
    """Assemble the complete ``evaluation.json`` payload."""
    payload = metrics.as_dict()
    payload.update(
        {
            "split": settings.split,
            "split_sizes": dict(bundle.split_sizes),
            "precision": EVALUATION_PRECISION,
            "test_time_augmentation": False,
            "device": str(device),
            "checkpoint": {**metadata.as_dict(), **fingerprint.as_dict()},
            "model": {
                "backbone": model.name,
                "backbone_display": model.specification.display_name,
                "feature_dim": model.feature_dim,
                "classifier": model.classifier_specification.display_name,
                "total_parameters": model.count_parameters(),
                "trainable_parameters": model.count_trainable_parameters(),
                "model_size_mb": round(model.model_size_mb(), 2),
            },
            "runtime": {
                "evaluation_seconds": outputs_seconds,
                "evaluation_duration": format_duration(outputs_seconds),
                "end_to_end_images_per_second": throughput,
                "peak_gpu_memory_mib": peak_mib,
            },
            "benchmark": benchmark.as_dict(),
            "integrity": integrity.summarise(checks),
            "class_to_idx": dict(bundle.class_to_idx),
        }
    )
    return payload


def _write_artifacts(
    *,
    config: Config,
    metrics: EvaluationMetrics,
    benchmark: BenchmarkResult,
    payload: Mapping[str, Any],
    probabilities: np.ndarray,
    targets: np.ndarray,
    results_dir: Path,
) -> dict[str, Path]:
    """Write every report and figure, returning their paths."""
    paths = write_reports(
        metrics,
        payload,
        benchmark.as_dict(),
        results_dir,
        ReportFilenames.from_config(config),
    )
    paths.update(
        write_evaluation_figures(
            metrics,
            probabilities,
            targets,
            results_dir,
            EvaluationPlotSpecification.from_config(config),
        )
    )
    return paths


def _log_headline(metrics: EvaluationMetrics, split: str) -> None:
    """Log the headline numbers of a completed evaluation."""
    overall = metrics.overall
    _LOGGER.info(
        "Evaluation on '%s': loss %.6f, top-1 %.6f, top-%d %.6f, macro-F1 %.6f, ECE %.6f.",
        split,
        overall.loss,
        overall.top1_accuracy,
        overall.top_k,
        overall.topk_accuracy,
        overall.macro_f1,
        overall.expected_calibration_error,
    )
    if not metrics.limitations.is_complete:
        for note in metrics.limitations.notes:
            _LOGGER.warning("Metric limitation: %s", note)


def _optional(macro: float | None, weighted: float | None) -> str:
    """Render a macro/weighted metric pair that may be undefined."""
    macro_text = "undefined" if macro is None else f"{macro:.6f}"
    weighted_text = "undefined" if weighted is None else f"{weighted:.6f}"
    return f"macro {macro_text}, weighted {weighted_text}"


if __name__ == "__main__":
    raise SystemExit(main())
