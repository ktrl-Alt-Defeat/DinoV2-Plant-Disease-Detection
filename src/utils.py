"""Generic, reusable helpers.

Everything here is deliberately model- and dataset-agnostic: serialisation,
formatting, timing and parameter accounting that any milestone can rely on.
"""

import csv
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import torch

from src.paths import ensure_directory, resolve

_BYTES_PER_MIB: Final[int] = 1024**2
_BYTE_UNITS: Final[tuple[str, ...]] = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
_SECONDS_PER_MINUTE: Final[int] = 60
_SECONDS_PER_HOUR: Final[int] = 3600


@dataclass(frozen=True)
class ParameterCounts:
    """Parameter accounting for a module."""

    total: int
    trainable: int

    @property
    def frozen(self) -> int:
        """Number of parameters excluded from optimisation."""
        return self.total - self.trainable


class Timer:
    """Context manager measuring wall-clock duration with a monotonic clock.

    Example:
        >>> with Timer() as timer:
        ...     pass
        >>> timer.elapsed >= 0.0
        True
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._end: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Seconds elapsed, measured live while running and frozen once exited.

        Raises:
            RuntimeError: If the timer has never been started.
        """
        if self._start is None:
            raise RuntimeError("Timer has not been started; use it as a context manager.")
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start


def count_parameters(module: torch.nn.Module) -> ParameterCounts:
    """Count the total and trainable parameters of ``module``."""
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return ParameterCounts(total=total, trainable=trainable)


def model_size_mb(module: torch.nn.Module) -> float:
    """Return the in-memory size of ``module``'s parameters and buffers, in MiB."""
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in module.parameters()
    )
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in module.buffers())
    return (parameter_bytes + buffer_bytes) / _BYTES_PER_MIB


def read_json(path: str | Path) -> Any:
    """Read and deserialise the UTF-8 JSON document at ``path``."""
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, content: str) -> Path:
    """Write ``content`` as UTF-8 text, creating parent directories as needed.

    Returns:
        The absolute path that was written.
    """
    target = resolve(path)
    ensure_directory(target.parent)
    target.write_text(content, encoding="utf-8")
    return target


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Serialise ``payload`` as UTF-8 JSON, creating parent directories as needed.

    Returns:
        The absolute path that was written.
    """
    return write_text(path, json.dumps(payload, indent=indent, ensure_ascii=False, sort_keys=True))


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write ``rows`` as a UTF-8 CSV file with a header line.

    Args:
        path: Destination file; parent directories are created as needed.
        rows: Records to write, each mapping column names to values.
        fieldnames: Column order. Defaults to the keys of the first row.

    Returns:
        The absolute path that was written.

    Raises:
        ValueError: If ``rows`` is empty and no ``fieldnames`` were supplied.
    """
    columns = list(fieldnames) if fieldnames is not None else _infer_fieldnames(rows)

    target = resolve(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return target


def format_bytes(num_bytes: int) -> str:
    """Render a byte count using binary units, e.g. ``"8.00 GiB"``.

    Raises:
        ValueError: If ``num_bytes`` is negative.
    """
    if num_bytes < 0:
        raise ValueError(f"Byte count must be non-negative, got {num_bytes}.")

    size = float(num_bytes)
    for unit in _BYTE_UNITS[:-1]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} {_BYTE_UNITS[-1]}"


def format_duration(seconds: float) -> str:
    """Render a duration compactly, e.g. ``"850 ms"``, ``"12.30 s"``, ``"1h 05m 03s"``.

    Raises:
        ValueError: If ``seconds`` is negative.
    """
    if seconds < 0:
        raise ValueError(f"Duration must be non-negative, got {seconds}.")

    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.2f} s"

    hours, remainder = divmod(int(seconds), _SECONDS_PER_HOUR)
    minutes, whole_seconds = divmod(remainder, _SECONDS_PER_MINUTE)
    if hours:
        return f"{hours}h {minutes:02d}m {whole_seconds:02d}s"
    return f"{minutes}m {whole_seconds:02d}s"


def _infer_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Derive CSV column names from the first row.

    Raises:
        ValueError: If ``rows`` is empty, since the header cannot be inferred.
    """
    if not rows:
        raise ValueError("Cannot infer CSV field names from an empty row sequence.")
    return list(rows[0])
