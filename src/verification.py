"""Structural verification of the integrated DINOv2 classifier.

Every check runs on synthetic tensors under ``torch.inference_mode()``: no
dataset is read, no optimizer or scheduler is created and no gradient is ever
computed. This module is the entry point behind ``python -m src.model``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from src import reporting
from src.cli import bootstrap, build_parser
from src.config import ConfigError
from src.logger import configure_console_encoding, get_logger
from src.model import DinoV2Classifier, ModelBuildError, build_model
from src.utils import write_json, write_text

#: Batch size of the synthetic verification tensor.
VERIFICATION_BATCH_SIZE: Final[int] = 2

#: Channel count of the synthetic verification tensor.
INPUT_CHANNELS: Final[int] = 3

MODEL_SUMMARY_FILENAME: Final[str] = "model_summary.txt"
MODEL_VERIFICATION_FILENAME: Final[str] = "model_verification.json"

PASSED: Final[str] = "PASS"
FAILED: Final[str] = "FAIL"
SKIPPED: Final[str] = "SKIP"

_TITLE: Final[str] = "MILESTONE 3 — MODEL INTEGRATION VERIFICATION"

#: Labels of the architecture facts printed above the verification list, paired
#: with the :meth:`DinoV2Classifier.describe` key holding the value.
_SUMMARY_LABELS: Final[tuple[tuple[str, str], ...]] = (
    ("Backbone", "backbone_display"),
    ("Pretrained", "pretrained"),
    ("Feature Dimension", "feature_dim"),
    ("Classifier", "classifier"),
    ("Classes", "num_classes"),
    ("Frozen Backbone", "frozen_backbone"),
    ("Total Parameters", "total_parameters"),
    ("Trainable Parameters", "trainable_parameters"),
    ("Approximate Model Size", "model_size_mb"),
)

_LOGGER: Final = get_logger("verification")


@dataclass(frozen=True)
class VerificationCheck:
    """Outcome of a single structural check."""

    name: str
    status: str
    details: str

    def as_dict(self) -> dict[str, str]:
        """Return the check as a serialisable mapping."""
        return {"name": self.name, "status": self.status, "details": self.details}


@dataclass(frozen=True)
class VerificationReport:
    """Model description plus the outcome of every structural check."""

    model_summary: dict[str, Any]
    checks: tuple[VerificationCheck, ...]

    @property
    def passed(self) -> bool:
        """Whether no check failed; skipped checks do not cause a failure."""
        return all(check.status != FAILED for check in self.checks)

    @property
    def status(self) -> str:
        """Overall verdict, ``"PASS"`` or ``"FAIL"``."""
        return PASSED if self.passed else FAILED

    def as_dict(self) -> dict[str, Any]:
        """Return the whole report as a serialisable mapping."""
        return {
            "model": self.model_summary,
            "checks": [check.as_dict() for check in self.checks],
            "status": self.status,
        }


@dataclass(frozen=True)
class ForwardOutputs:
    """Embeddings and logits produced by one synthetic pass on a given device."""

    features: torch.Tensor
    logits: torch.Tensor


def synthetic_batch(batch_size: int, image_size: int, device: torch.device) -> torch.Tensor:
    """Return a random image batch of shape ``[batch_size, 3, image_size, image_size]``."""
    return torch.randn(batch_size, INPUT_CHANNELS, image_size, image_size, device=device)


def forward_on(
    model: DinoV2Classifier,
    device: torch.device,
    batch_size: int = VERIFICATION_BATCH_SIZE,
) -> ForwardOutputs:
    """Move ``model`` to ``device`` and run one synthetic pass without gradients."""
    model.to(device)
    inputs = synthetic_batch(batch_size, model.image_size, device)
    with torch.inference_mode():
        return ForwardOutputs(features=model.forward_features(inputs), logits=model(inputs))


def verify_model(
    model: DinoV2Classifier,
    *,
    batch_size: int = VERIFICATION_BATCH_SIZE,
) -> VerificationReport:
    """Run every structural check against ``model`` and collect the results.

    The model is temporarily moved between devices and restored to its original
    device before the report is built.
    """
    original_device = model.device
    features_shape = (batch_size, model.feature_dim)
    logits_shape = (batch_size, model.num_classes)
    checks: list[VerificationCheck] = []

    try:
        outputs = forward_on(model, original_device, batch_size)
        checks.append(_shape_check("Feature Extraction", outputs.features, features_shape))
        checks.append(_shape_check("Classifier Output", outputs.logits, logits_shape))

        cpu_outputs = forward_on(model, torch.device("cpu"), batch_size)
        checks.append(_shape_check("CPU Forward", cpu_outputs.logits, logits_shape))

        checks.append(_cuda_check(model, batch_size, logits_shape))
        checks.append(_finite_check("NaN Check", outputs.logits, torch.isnan, "NaN"))
        checks.append(_finite_check("Inf Check", outputs.logits, torch.isinf, "infinite"))
    finally:
        model.to(original_device)

    return VerificationReport(model_summary=model.describe(), checks=tuple(checks))


def write_artifacts(
    model: DinoV2Classifier,
    report: VerificationReport,
    results_dir: str | Path,
) -> dict[str, Path]:
    """Write the model summary and the verification report into ``results_dir``.

    Returns:
        The absolute path of each artifact, keyed by artifact name.
    """
    directory = Path(results_dir)
    return {
        "model_summary": write_text(
            directory / MODEL_SUMMARY_FILENAME, render_model_summary(model, report)
        ),
        "model_verification": write_json(
            directory / MODEL_VERIFICATION_FILENAME, report.as_dict()
        ),
    }


def render_report(report: VerificationReport) -> str:
    """Render the console report shown at the end of a verification run."""
    lines = reporting.banner(_TITLE)
    lines.extend(reporting.entries(_summary_rows(report.model_summary)))
    lines.extend(reporting.rule())

    for index, check in enumerate(report.checks, start=1):
        lines.extend(
            reporting.entry(
                f"Verification {index}",
                f"{check.name} ... {check.status}",
                check.details,
            )
        )

    lines.extend(reporting.rule())
    lines.extend([f"MODEL STATUS : {report.status}", ""])
    lines.append(reporting.MAJOR_RULE)
    return reporting.render(lines)


def render_model_summary(model: DinoV2Classifier, report: VerificationReport) -> str:
    """Render the human readable ``model_summary.txt`` artifact."""
    lines = reporting.banner(_TITLE)
    lines.extend(reporting.entries(_summary_rows(report.model_summary)))
    lines.extend(reporting.rule())
    lines.extend(["Module tree", "", repr(model), ""])
    return reporting.render(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the configured model, verify it, write the artifacts and report."""
    configure_console_encoding()
    parser = build_parser(
        prog="python -m src.model",
        description="Build the configured DINOv2 classifier and verify it on synthetic tensors.",
    )
    arguments = parser.parse_args(argv)

    try:
        boot = bootstrap(arguments.config)
        model = build_model(boot.config)
        report = verify_model(model)
        artifacts = write_artifacts(model, report, boot.paths.results)
    except (ModelBuildError, ConfigError, KeyError, OSError, ValueError) as error:
        _LOGGER.error("Model verification failed: %s", error)
        print(reporting.render([*reporting.banner(_TITLE), f"ERROR: {error}", ""]))
        print(reporting.render(reporting.closing(f"MODEL STATUS : {FAILED}")))
        return 1

    for name, path in artifacts.items():
        _LOGGER.info("Artifact written: %s -> %s", name, path)

    print(render_report(report))
    return 0 if report.passed else 1


def _summary_rows(model_summary: dict[str, Any]) -> list[tuple[str, str]]:
    """Pair each console label with its formatted value from the model summary."""
    return [(label, _format_value(key, model_summary[key])) for label, key in _SUMMARY_LABELS]


def _format_value(key: str, value: Any) -> str:
    """Format a summary value for display: thousands separators, units, plain text."""
    if key.endswith("_parameters"):
        return f"{value:,}"
    if key == "model_size_mb":
        return f"{value:.2f} MiB"
    return str(value)


def _shape_check(
    name: str,
    tensor: torch.Tensor,
    expected: tuple[int, ...],
) -> VerificationCheck:
    """Check that ``tensor`` has exactly the expected shape."""
    actual = tuple(tensor.shape)
    return VerificationCheck(
        name=name,
        status=PASSED if actual == expected else FAILED,
        details=(
            f"output {list(actual)} on {tensor.device.type}, expected {list(expected)}"
        ),
    )


def _cuda_check(
    model: DinoV2Classifier,
    batch_size: int,
    logits_shape: tuple[int, ...],
) -> VerificationCheck:
    """Run the forward pass on CUDA, or record a skip when no GPU is visible."""
    if not torch.cuda.is_available():
        return VerificationCheck(
            name="CUDA Forward",
            status=SKIPPED,
            details="CUDA is not available on this machine; CPU execution verified instead",
        )
    outputs = forward_on(model, torch.device("cuda"), batch_size)
    return _shape_check("CUDA Forward", outputs.logits, logits_shape)


def _finite_check(
    name: str,
    tensor: torch.Tensor,
    predicate: Callable[[torch.Tensor], torch.Tensor],
    label: str,
) -> VerificationCheck:
    """Check that ``predicate`` matches no element of ``tensor``."""
    offending = int(predicate(tensor).sum().item())
    return VerificationCheck(
        name=name,
        status=PASSED if offending == 0 else FAILED,
        details=f"{offending} {label} value(s) across {tensor.numel()} elements",
    )
