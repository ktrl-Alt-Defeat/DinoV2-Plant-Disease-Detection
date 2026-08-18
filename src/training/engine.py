"""The train and evaluate passes over a single epoch.

Both passes share one accumulator so loss and accuracy are computed identically
in training and evaluation; only gradient handling differs between them.
"""

from typing import Final

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logger import get_logger
from src.training.metrics import EpochOutcome
from src.training.optim import clip_gradients
from src.training.precision import PrecisionSpecification

#: Unit shown by the progress bars.
_PROGRESS_UNIT: Final[str] = "batch"

_LOGGER: Final = get_logger("training.engine")


class _OutcomeAccumulator:
    """Running loss and accuracy over the batches of one epoch."""

    def __init__(self) -> None:
        self._loss_sum = 0.0
        self._correct = 0
        self._seen = 0

    def update(self, loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Fold one batch into the running totals."""
        batch_size = targets.size(0)
        self._loss_sum += float(loss.item()) * batch_size
        self._correct += int(logits.argmax(dim=1).eq(targets).sum().item())
        self._seen += batch_size

    @property
    def seen(self) -> int:
        """Number of samples folded in so far."""
        return self._seen

    def outcome(self) -> EpochOutcome:
        """Return the averaged loss and accuracy.

        Raises:
            ValueError: If no sample was seen, which means an empty dataloader.
        """
        if self._seen == 0:
            raise ValueError("Epoch produced no sample; the dataloader is empty.")
        return EpochOutcome(
            loss=self._loss_sum / self._seen,
            accuracy=self._correct / self._seen,
        )

    def running_loss(self) -> float:
        """Mean loss so far, for live progress reporting."""
        return self._loss_sum / self._seen if self._seen else 0.0

    def running_accuracy(self) -> float:
        """Mean accuracy so far, for live progress reporting."""
        return self._correct / self._seen if self._seen else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    precision: PrecisionSpecification,
    device: torch.device,
    *,
    gradient_clip_norm: float,
    description: str,
) -> EpochOutcome:
    """Run one optimisation pass over ``loader``.

    Gradients are unscaled before clipping so the clip threshold applies to true
    gradient magnitudes rather than scaled ones.

    Raises:
        ValueError: If the dataloader yields no sample.
    """
    model.train()
    accumulator = _OutcomeAccumulator()
    progress = tqdm(loader, desc=description, unit=_PROGRESS_UNIT, leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with precision.autocast():
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if gradient_clip_norm > 0.0:
            scaler.unscale_(optimizer)
            clip_gradients(model, gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        accumulator.update(loss.detach(), logits.detach(), targets)
        progress.set_postfix(
            loss=f"{accumulator.running_loss():.4f}",
            acc=f"{accumulator.running_accuracy():.4f}",
        )

    progress.close()
    return accumulator.outcome()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    precision: PrecisionSpecification,
    device: torch.device,
    *,
    description: str,
) -> EpochOutcome:
    """Run one gradient-free pass over ``loader``.

    Raises:
        ValueError: If the dataloader yields no sample.
    """
    model.eval()
    accumulator = _OutcomeAccumulator()
    progress = tqdm(loader, desc=description, unit=_PROGRESS_UNIT, leave=False)

    with torch.no_grad():
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with precision.autocast():
                logits = model(images)
                loss = criterion(logits, targets)

            accumulator.update(loss, logits, targets)
            progress.set_postfix(
                loss=f"{accumulator.running_loss():.4f}",
                acc=f"{accumulator.running_accuracy():.4f}",
            )

    progress.close()
    return accumulator.outcome()


def peak_gpu_memory_mib(device: torch.device) -> float:
    """Return the peak CUDA allocation since the last reset, in MiB.

    Returns ``0.0`` on CPU, where the statistic does not exist.
    """
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**2


def reset_gpu_memory_statistics(device: torch.device) -> None:
    """Reset the CUDA peak-allocation counter so it measures one epoch."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def log_epoch_start(epoch: int, epochs: int, learning_rate: float) -> None:
    """Log the header line of an epoch."""
    _LOGGER.info("Epoch %d/%d starting (lr=%.3e).", epoch, epochs, learning_rate)
