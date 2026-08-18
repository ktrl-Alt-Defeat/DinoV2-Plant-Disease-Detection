"""Integrity guarantees around an evaluation run.

Evaluation is a read-only operation, and this module is what makes that claim
checkable rather than assumed. The checkpoint file is hashed before and after
the pass, the model parameters are hashed before and after, and the outputs are
inspected for non-finite values and for probabilities that do not form a
distribution. Any failure is reported as a failed check rather than silently
tolerated, so a report can never describe a run whose inputs moved underneath it.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from src.logger import get_logger
from src.paths import resolve

#: Bytes read per chunk when hashing a checkpoint file.
_HASH_CHUNK_BYTES: Final[int] = 1024 * 1024

PASSED: Final[str] = "PASS"
FAILED: Final[str] = "FAIL"

_LOGGER: Final = get_logger("evaluation.integrity")


class IntegrityError(RuntimeError):
    """Raised when an integrity guarantee of the evaluation run is violated."""


@dataclass(frozen=True)
class IntegrityCheck:
    """Outcome of a single integrity guarantee."""

    name: str
    passed: bool
    details: str

    @property
    def status(self) -> str:
        """``"PASS"`` or ``"FAIL"``."""
        return PASSED if self.passed else FAILED

    def as_dict(self) -> dict[str, Any]:
        """Return the check as a serialisable mapping."""
        return {"name": self.name, "status": self.status, "details": self.details}


@dataclass(frozen=True)
class Fingerprint:
    """Content digest of the checkpoint file on disk."""

    path: Path
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        """Return the fingerprint as a serialisable mapping."""
        return {"path": str(self.path), "sha256": self.sha256, "size_bytes": self.size_bytes}


def fingerprint_file(path: str | Path) -> Fingerprint:
    """Hash a file on disk.

    Raises:
        IntegrityError: If the file cannot be read.
    """
    target = resolve(path)
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        size = target.stat().st_size
    except OSError as error:
        raise IntegrityError(f"Unable to fingerprint {target}: {error}") from error
    return Fingerprint(path=target, sha256=digest.hexdigest(), size_bytes=size)


def parameter_digest(model: nn.Module) -> str:
    """Hash every parameter and buffer of ``model``.

    Two identical digests before and after a pass prove no weight was updated.
    """
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().to("cpu", torch.float32).numpy().tobytes())
    return digest.hexdigest()


def check_checkpoint_unchanged(before: Fingerprint, after: Fingerprint) -> IntegrityCheck:
    """Confirm the checkpoint file was not rewritten by the evaluation."""
    unchanged = before.sha256 == after.sha256 and before.size_bytes == after.size_bytes
    return IntegrityCheck(
        name="Checkpoint File Unchanged",
        passed=unchanged,
        details=(
            f"sha256 {before.sha256[:16]} before and after, {before.size_bytes:,} bytes"
            if unchanged
            else f"file changed: {before.sha256[:16]} -> {after.sha256[:16]}"
        ),
    )


def check_weights_unchanged(before: str, after: str) -> IntegrityCheck:
    """Confirm no parameter moved during the evaluation pass."""
    unchanged = before == after
    return IntegrityCheck(
        name="Model Weights Unchanged",
        passed=unchanged,
        details=(
            f"state_dict digest {before[:16]} before and after the pass"
            if unchanged
            else f"weights changed: {before[:16]} -> {after[:16]}"
        ),
    )


def check_evaluation_mode(model: nn.Module) -> IntegrityCheck:
    """Confirm the model is in evaluation mode and holds no gradient."""
    training = model.training
    with_grad = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    return IntegrityCheck(
        name="Evaluation Mode",
        passed=not training and not with_grad,
        details=(
            f"model.training={training}, {len(with_grad)} parameter(s) carry a gradient"
        ),
    )


def check_finite(name: str, values: np.ndarray) -> IntegrityCheck:
    """Confirm an array carries no NaN and no infinity."""
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    return IntegrityCheck(
        name=name,
        passed=nan_count == 0 and inf_count == 0,
        details=(
            f"{nan_count} NaN and {inf_count} infinite value(s) "
            f"across {values.size:,} elements"
        ),
    )


def check_probabilities(probabilities: np.ndarray, *, tolerance: float) -> IntegrityCheck:
    """Confirm every row of ``probabilities`` is a probability distribution.

    Each row must sum to one within ``tolerance`` and every entry must fall in
    ``[0, 1]``; otherwise the calibration and curve metrics would be meaningless.
    """
    row_sums = probabilities.sum(axis=1)
    deviation = float(np.abs(row_sums - 1.0).max())
    minimum = float(probabilities.min())
    maximum = float(probabilities.max())

    normalised = deviation <= tolerance
    bounded = minimum >= 0.0 and maximum <= 1.0
    return IntegrityCheck(
        name="Probability Normalization",
        passed=normalised and bounded,
        details=(
            f"max |rowsum - 1| = {deviation:.3e} (tolerance {tolerance:.1e}), "
            f"values within [{minimum:.6f}, {maximum:.6f}]"
        ),
    )


def check_split_coverage(expected: int, observed: int, split: str) -> IntegrityCheck:
    """Confirm the pass covered every sample of the split."""
    return IntegrityCheck(
        name="Split Coverage",
        passed=expected == observed,
        details=(
            f"{observed:,} of {expected:,} '{split}' samples evaluated"
            + ("" if expected == observed else "; some samples were skipped")
        ),
    )


def summarise(checks: Sequence[IntegrityCheck]) -> dict[str, Any]:
    """Return a serialisable summary of every integrity check."""
    failed = [check.name for check in checks if not check.passed]
    return {
        "status": FAILED if failed else PASSED,
        "failed": failed,
        "checks": [check.as_dict() for check in checks],
    }


def enforce(checks: Sequence[IntegrityCheck]) -> None:
    """Abort the run when any integrity guarantee was violated.

    Raises:
        IntegrityError: If at least one check failed.
    """
    failed = [check for check in checks if not check.passed]
    for check in checks:
        _LOGGER.info("Integrity %s: %s (%s)", check.status, check.name, check.details)
    if not failed:
        return
    detail = "; ".join(f"{check.name}: {check.details}" for check in failed)
    raise IntegrityError(f"{len(failed)} integrity check(s) failed: {detail}")
