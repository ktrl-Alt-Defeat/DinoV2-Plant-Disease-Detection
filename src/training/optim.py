"""Optimizer and learning rate scheduler construction."""

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from src.config import Config, TypeSpec, validate_keys
from src.logger import get_logger

#: Optimizers accepted by ``training.optimizer.name``.
SUPPORTED_OPTIMIZERS: Final[frozenset[str]] = frozenset({"adamw"})

#: Schedulers accepted by ``training.scheduler.name``.
SUPPORTED_SCHEDULERS: Final[frozenset[str]] = frozenset({"cosine"})

#: Number of coefficients ``AdamW`` expects in ``betas``.
_BETA_COUNT: Final[int] = 2

#: Configuration contract of the ``training.optimizer`` section.
OPTIMIZER_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("training.optimizer.name", str),
    ("training.optimizer.learning_rate", (int, float)),
    ("training.optimizer.weight_decay", (int, float)),
    ("training.optimizer.betas", list),
)

#: Configuration contract of the ``training.scheduler`` section.
SCHEDULER_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("training.scheduler.name", str),
    ("training.scheduler.min_learning_rate", (int, float)),
)

_LOGGER: Final = get_logger("training.optim")


@dataclass(frozen=True)
class OptimizerSpecification:
    """Optimizer settings resolved from the ``training.optimizer`` section."""

    name: str
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]

    @classmethod
    def from_config(cls, config: Config) -> "OptimizerSpecification":
        """Read and validate the ``training.optimizer`` section of ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If the optimizer is unsupported or a coefficient is invalid.
        """
        validate_keys(config, OPTIMIZER_REQUIRED_KEYS, context="training.optimizer section")

        name = config.get("training.optimizer.name").strip().lower()
        if name not in SUPPORTED_OPTIMIZERS:
            supported = ", ".join(sorted(SUPPORTED_OPTIMIZERS))
            raise ValueError(
                f"Unsupported training.optimizer.name '{name}'. Supported: {supported}."
            )

        learning_rate = float(config.get("training.optimizer.learning_rate"))
        if learning_rate <= 0.0:
            raise ValueError(
                f"training.optimizer.learning_rate must be positive, got {learning_rate}."
            )

        weight_decay = float(config.get("training.optimizer.weight_decay"))
        if weight_decay < 0.0:
            raise ValueError(
                f"training.optimizer.weight_decay must be non-negative, got {weight_decay}."
            )

        return cls(
            name=name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=_read_betas(config.get("training.optimizer.betas")),
        )


@dataclass(frozen=True)
class SchedulerSpecification:
    """Scheduler settings resolved from the ``training.scheduler`` section."""

    name: str
    min_learning_rate: float

    @classmethod
    def from_config(cls, config: Config) -> "SchedulerSpecification":
        """Read and validate the ``training.scheduler`` section of ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If the scheduler is unsupported or the floor is negative.
        """
        validate_keys(config, SCHEDULER_REQUIRED_KEYS, context="training.scheduler section")

        name = config.get("training.scheduler.name").strip().lower()
        if name not in SUPPORTED_SCHEDULERS:
            supported = ", ".join(sorted(SUPPORTED_SCHEDULERS))
            raise ValueError(
                f"Unsupported training.scheduler.name '{name}'. Supported: {supported}."
            )

        minimum = float(config.get("training.scheduler.min_learning_rate"))
        if minimum < 0.0:
            raise ValueError(
                f"training.scheduler.min_learning_rate must be non-negative, got {minimum}."
            )
        return cls(name=name, min_learning_rate=minimum)


def build_optimizer(model: nn.Module, specification: OptimizerSpecification) -> Optimizer:
    """Build the configured optimizer over the trainable parameters of ``model``.

    Frozen parameters are excluded, so freezing the backbone shrinks the
    optimizer state instead of carrying unused moments.

    Raises:
        ValueError: If the model exposes no trainable parameter.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(
            "Model has no trainable parameter; check model.freeze_backbone and the head."
        )

    optimizer = AdamW(
        parameters,
        lr=specification.learning_rate,
        weight_decay=specification.weight_decay,
        betas=specification.betas,
    )
    _LOGGER.info(
        "Optimizer: %s(lr=%.2e, weight_decay=%.4f, betas=%s) over %d tensors.",
        specification.name,
        specification.learning_rate,
        specification.weight_decay,
        specification.betas,
        len(parameters),
    )
    return optimizer


def build_scheduler(
    optimizer: Optimizer,
    specification: SchedulerSpecification,
    *,
    epochs: int,
) -> LRScheduler:
    """Build the configured scheduler, annealing across ``epochs`` steps.

    Raises:
        ValueError: If ``epochs`` is not positive.
    """
    if epochs <= 0:
        raise ValueError(f"Scheduler needs a positive epoch count, got {epochs}.")

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=specification.min_learning_rate,
    )
    _LOGGER.info(
        "Scheduler: %s(T_max=%d, eta_min=%.2e).",
        specification.name,
        epochs,
        specification.min_learning_rate,
    )
    return scheduler


def current_learning_rate(optimizer: Optimizer) -> float:
    """Return the learning rate of the first parameter group."""
    return float(optimizer.param_groups[0]["lr"])


def _read_betas(values: list[object]) -> tuple[float, float]:
    """Read the ``AdamW`` beta coefficients.

    Raises:
        ValueError: If the pair is malformed or falls outside ``[0, 1)``.
    """
    if len(values) != _BETA_COUNT:
        raise ValueError(
            f"training.optimizer.betas must list exactly {_BETA_COUNT} values, got {len(values)}."
        )

    betas = tuple(float(value) for value in values)
    if any(not 0.0 <= beta < 1.0 for beta in betas):
        raise ValueError(f"training.optimizer.betas must be within [0, 1), got {list(betas)}.")
    return betas[0], betas[1]


def clip_gradients(model: nn.Module, max_norm: float) -> None:
    """Clip the global gradient norm of ``model`` in place.

    A non-positive ``max_norm`` disables clipping.
    """
    if max_norm <= 0.0:
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
