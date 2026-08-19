"""Compute device selection and hardware introspection."""

import platform
from dataclasses import dataclass
from typing import Final

import torch

from src.logger import get_logger

#: Accepted values for the ``device.preferred`` configuration entry.
SUPPORTED_PREFERENCES: Final[frozenset[str]] = frozenset({"auto", "cuda", "cpu"})

_BYTES_PER_MIB: Final[int] = 1024**2

_LOGGER: Final = get_logger("device")


@dataclass(frozen=True)
class DeviceInfo:
    """Hardware description of the device a run executes on.

    Attributes:
        device: The selected :class:`torch.device`.
        name: Human readable device name (GPU model or CPU identifier).
        cuda_version: CUDA toolkit version PyTorch was built against, if any.
        cudnn_version: cuDNN version reported by PyTorch, if available.
        total_memory_bytes: Total VRAM of the selected GPU, ``None`` on CPU.
    """

    device: torch.device
    name: str
    cuda_version: str | None
    cudnn_version: int | None
    total_memory_bytes: int | None


def get_device(preferred: str = "auto") -> torch.device:
    """Resolve the device to run on.

    ``"auto"`` picks CUDA when it is usable and falls back to CPU otherwise.
    ``"cuda"`` expresses a preference rather than a hard requirement: when no
    GPU is visible the call warns and falls back to CPU, so the same
    configuration stays runnable on a laptop.

    Raises:
        ValueError: If ``preferred`` is not one of :data:`SUPPORTED_PREFERENCES`.
    """
    choice = preferred.strip().lower()
    if choice not in SUPPORTED_PREFERENCES:
        supported = ", ".join(sorted(SUPPORTED_PREFERENCES))
        raise ValueError(
            f"Unsupported device preference '{preferred}'. Supported values: {supported}."
        )

    if choice == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())

    if choice == "cuda":
        _LOGGER.warning("CUDA was requested but is not available; falling back to CPU.")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> DeviceInfo:
    """Collect the hardware details of ``device`` for reporting and logging."""
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()

    if device.type != "cuda":
        return DeviceInfo(
            device=device,
            name=_cpu_name(),
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            total_memory_bytes=None,
        )

    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return DeviceInfo(
        device=device,
        name=properties.name,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        total_memory_bytes=properties.total_memory,
    )


def synchronize(device: torch.device) -> None:
    """Wait for outstanding CUDA work so timings measure completed compute.

    A no-op on CPU, where execution is already synchronous.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    """Reset the CUDA peak-allocation counter so it measures one phase.

    A no-op on CPU, where the statistic does not exist.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mib(device: torch.device) -> float:
    """Return the peak CUDA allocation since the last reset, in MiB.

    Returns ``0.0`` on CPU, where the statistic does not exist.
    """
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / _BYTES_PER_MIB


def _cpu_name() -> str:
    """Return the most descriptive CPU identifier the platform exposes."""
    return platform.processor() or platform.machine() or "Unknown CPU"
