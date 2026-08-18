"""Mixed precision resolution.

Automatic mixed precision is requested through the configuration but granted by
the hardware. This module resolves the two into a single specification and logs
every downgrade, so the precision a run actually used is never a guess.
"""

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Final

import torch

from src.config import Config, TypeSpec, validate_keys
from src.logger import get_logger

#: Configured dtype name to the torch dtype it selects.
AMP_DTYPES: Final[Mapping[str, torch.dtype]] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

#: Dtype name that picks the widest precision the GPU supports natively.
AUTO_DTYPE: Final[str] = "auto"

#: Device type mixed precision is supported on in this project.
CUDA_DEVICE_TYPE: Final[str] = "cuda"

#: Configuration contract of the ``training.amp`` section.
PRECISION_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("training.amp.enabled", bool),
    ("training.amp.dtype", str),
)

_LOGGER: Final = get_logger("training.precision")


@dataclass(frozen=True)
class PrecisionSpecification:
    """The autocast configuration a run will actually execute with."""

    enabled: bool
    device_type: str
    dtype: torch.dtype | None

    @classmethod
    def resolve(cls, config: Config, device: torch.device) -> "PrecisionSpecification":
        """Combine the configured request with what ``device`` supports.

        Mixed precision is only applied on CUDA. ``"auto"`` selects bfloat16 when
        the GPU supports it and float16 otherwise; an explicit bfloat16 request
        on a GPU without support is downgraded to float16 rather than failing,
        and the downgrade is logged.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If the configured dtype name is unknown.
        """
        validate_keys(config, PRECISION_REQUIRED_KEYS, context="training.amp section")

        requested = config.get("training.amp.dtype").strip().lower()
        if requested != AUTO_DTYPE and requested not in AMP_DTYPES:
            supported = ", ".join([AUTO_DTYPE, *sorted(AMP_DTYPES)])
            raise ValueError(
                f"Unsupported training.amp.dtype '{requested}'. Supported values: {supported}."
            )

        if not config.get("training.amp.enabled"):
            _LOGGER.info("Mixed precision disabled by configuration; running in float32.")
            return cls(enabled=False, device_type=device.type, dtype=None)

        if device.type != CUDA_DEVICE_TYPE:
            _LOGGER.info(
                "Mixed precision requested but device is %s; running in float32.", device.type
            )
            return cls(enabled=False, device_type=device.type, dtype=None)

        dtype = cls._select_dtype(requested)
        _LOGGER.info(
            "Mixed precision enabled: %s autocast on %s.", _dtype_label(dtype), device.type
        )
        return cls(enabled=True, device_type=device.type, dtype=dtype)

    @staticmethod
    def _select_dtype(requested: str) -> torch.dtype:
        """Pick the autocast dtype, downgrading bfloat16 when it is unsupported."""
        supports_bf16 = torch.cuda.is_bf16_supported()
        if requested == AUTO_DTYPE:
            return torch.bfloat16 if supports_bf16 else torch.float16

        dtype = AMP_DTYPES[requested]
        if dtype is torch.bfloat16 and not supports_bf16:
            _LOGGER.warning(
                "bfloat16 requested but unsupported on this GPU; falling back to float16."
            )
            return torch.float16
        return dtype

    @property
    def uses_grad_scaler(self) -> bool:
        """Whether loss scaling is required.

        Only float16 needs it: bfloat16 carries the exponent range of float32,
        so gradients do not underflow.
        """
        return self.enabled and self.dtype is torch.float16

    @property
    def label(self) -> str:
        """Readable description of the precision in use."""
        return _dtype_label(self.dtype) if self.enabled else "fp32"

    def autocast(self) -> AbstractContextManager[None]:
        """Return the autocast context for a forward pass, or a no-op when disabled."""
        if not self.enabled or self.dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)


def build_grad_scaler(specification: PrecisionSpecification) -> torch.amp.GradScaler:
    """Build the gradient scaler for ``specification``.

    A disabled scaler is a transparent pass-through, so the training step needs
    no branch on whether scaling is active.
    """
    return torch.amp.GradScaler(
        specification.device_type,
        enabled=specification.uses_grad_scaler,
    )


def _dtype_label(dtype: torch.dtype | None) -> str:
    """Return the configured name of ``dtype``."""
    for name, candidate in AMP_DTYPES.items():
        if candidate is dtype:
            return name
    return "fp32"
