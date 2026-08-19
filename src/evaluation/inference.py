"""The evaluation forward pass and the inference benchmark.

Both run under ``model.eval()`` and ``torch.no_grad()``, in full float32 and
without autocast. Mixed precision is deliberately not used here: expected
calibration error, ROC-AUC and PR-AUC read the probability values themselves,
and a one-off test pass has no throughput pressure that would justify trading
numerical fidelity for speed. The benchmark measures that same float32 path, so
its latency describes the configuration the metrics were produced with.

No augmentation is applied — the evaluation transform is the deterministic
resize/centre-crop pipeline — and nothing here ever calls ``backward`` or steps
an optimizer.
"""

import time
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, TypeSpec, validate_keys
from src.device import peak_memory_mib, reset_peak_memory, synchronize
from src.logger import get_logger

#: Percentiles reported for per-batch latency.
LATENCY_PERCENTILES: Final[tuple[int, ...]] = (50, 90, 95, 99)

#: Precision the evaluation pass and the benchmark both run in.
EVALUATION_PRECISION: Final[str] = "fp32"

#: Configuration contract of the ``evaluation.benchmark`` section.
BENCHMARK_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("evaluation.benchmark.warmup_batches", int),
    ("evaluation.benchmark.measured_batches", int),
)

_MILLISECONDS_PER_SECOND: Final[float] = 1000.0

_LOGGER: Final = get_logger("evaluation.inference")


@dataclass(frozen=True)
class InferenceOutputs:
    """Raw outputs of one full pass over an evaluated split."""

    logits: torch.Tensor
    targets: torch.Tensor
    loss: float
    seconds: float
    batches: int

    @property
    def sample_count(self) -> int:
        """Number of samples the pass covered."""
        return int(self.targets.numel())

    @property
    def throughput(self) -> float:
        """End-to-end images per second, data loading included."""
        return self.sample_count / self.seconds if self.seconds > 0 else 0.0


@dataclass(frozen=True)
class BenchmarkSpecification:
    """Benchmark settings resolved from the ``evaluation.benchmark`` section."""

    warmup_batches: int
    measured_batches: int

    @classmethod
    def from_config(cls, config: Config) -> "BenchmarkSpecification":
        """Read and validate the ``evaluation.benchmark`` section of ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If a batch count is out of range.
        """
        validate_keys(config, BENCHMARK_REQUIRED_KEYS, context="evaluation.benchmark section")

        warmup = config.get("evaluation.benchmark.warmup_batches")
        measured = config.get("evaluation.benchmark.measured_batches")
        if warmup < 0:
            raise ValueError(
                f"evaluation.benchmark.warmup_batches must be non-negative, got {warmup}."
            )
        if measured <= 0:
            raise ValueError(
                f"evaluation.benchmark.measured_batches must be positive, got {measured}."
            )
        return cls(warmup_batches=warmup, measured_batches=measured)


@dataclass(frozen=True)
class BenchmarkResult:
    """Latency and throughput of the forward pass alone."""

    device: str
    precision: str
    batch_size: int
    warmup_batches: int
    measured_batches: int
    measured_images: int
    mean_batch_ms: float
    percentile_batch_ms: dict[int, float]
    mean_image_ms: float
    throughput_images_per_second: float
    peak_gpu_memory_mib: float

    def as_dict(self) -> dict[str, Any]:
        """Return the benchmark as a serialisable mapping."""
        return {
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "warmup_batches": self.warmup_batches,
            "measured_batches": self.measured_batches,
            "measured_images": self.measured_images,
            "mean_batch_ms": self.mean_batch_ms,
            "percentile_batch_ms": {
                f"p{percentile}": value
                for percentile, value in sorted(self.percentile_batch_ms.items())
            },
            "mean_image_ms": self.mean_image_ms,
            "throughput_images_per_second": self.throughput_images_per_second,
            "peak_gpu_memory_mib": self.peak_gpu_memory_mib,
            "note": (
                "Latency covers host-to-device transfer plus the forward pass, "
                "measured with CUDA synchronisation around each batch. Data "
                "loading and decoding are excluded; see the end-to-end "
                "throughput in evaluation.json for the figure that includes them."
            ),
        }


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    description: str,
) -> InferenceOutputs:
    """Run one gradient-free pass and collect every logit and target.

    The model is switched to evaluation mode, which disables dropout, and the
    whole pass runs inside ``torch.no_grad()`` so no graph is built and no
    parameter can accumulate a gradient.

    Raises:
        ValueError: If the dataloader yields no sample.
    """
    model.eval()
    reset_peak_memory(device)

    logit_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    loss_sum = 0.0
    seen = 0
    batches = 0

    progress = tqdm(loader, desc=description, unit="batch", leave=False)
    start = time.perf_counter()

    with torch.no_grad():
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = int(targets.size(0))
            loss_sum += float(loss.item()) * batch_size
            seen += batch_size
            batches += 1

            logit_batches.append(logits.detach().to("cpu", torch.float32))
            target_batches.append(targets.detach().to("cpu"))

    synchronize(device)
    elapsed = time.perf_counter() - start
    progress.close()

    if seen == 0:
        raise ValueError("Evaluation pass produced no sample; the dataloader is empty.")

    outputs = InferenceOutputs(
        logits=torch.cat(logit_batches),
        targets=torch.cat(target_batches),
        loss=loss_sum / seen,
        seconds=elapsed,
        batches=batches,
    )
    _LOGGER.info(
        "Inference pass complete: %d samples in %d batches, %.2f s, %.1f images/s.",
        outputs.sample_count,
        outputs.batches,
        outputs.seconds,
        outputs.throughput,
    )
    return outputs


def benchmark_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    specification: BenchmarkSpecification,
    *,
    description: str,
) -> BenchmarkResult:
    """Measure forward-pass latency over a bounded number of batches.

    Warm-up batches are executed and discarded first so lazy CUDA context
    creation and cuDNN algorithm selection do not land in the measurements.

    Raises:
        ValueError: If the loader supplies no batch to measure.
    """
    model.eval()
    reset_peak_memory(device)

    durations: list[float] = []
    images_measured = 0
    batch_size = 0
    limit = specification.warmup_batches + specification.measured_batches

    progress = tqdm(total=limit, desc=description, unit="batch", leave=False)
    with torch.no_grad():
        for index, (images, _) in enumerate(loader):
            if index >= limit:
                break

            images = images.to(device, non_blocking=True)
            synchronize(device)

            start = time.perf_counter()
            model(images)
            synchronize(device)
            elapsed = time.perf_counter() - start

            if index >= specification.warmup_batches:
                durations.append(elapsed)
                images_measured += int(images.size(0))
                batch_size = int(images.size(0))
            progress.update(1)
    progress.close()

    if not durations:
        raise ValueError(
            f"Benchmark measured no batch: the loader supplied fewer than "
            f"{specification.warmup_batches + 1} batches."
        )

    milliseconds = np.asarray(durations, dtype=np.float64) * _MILLISECONDS_PER_SECOND
    total_seconds = float(np.sum(durations))

    result = BenchmarkResult(
        device=str(device),
        precision=EVALUATION_PRECISION,
        batch_size=batch_size,
        warmup_batches=specification.warmup_batches,
        measured_batches=len(durations),
        measured_images=images_measured,
        mean_batch_ms=float(milliseconds.mean()),
        percentile_batch_ms={
            percentile: float(np.percentile(milliseconds, percentile))
            for percentile in LATENCY_PERCENTILES
        },
        mean_image_ms=float(milliseconds.sum() / images_measured),
        throughput_images_per_second=images_measured / total_seconds if total_seconds > 0 else 0.0,
        peak_gpu_memory_mib=peak_memory_mib(device),
    )
    _LOGGER.info(
        "Benchmark: %.2f ms/batch (p95 %.2f), %.3f ms/image, %.1f images/s.",
        result.mean_batch_ms,
        result.percentile_batch_ms[95],
        result.mean_image_ms,
        result.throughput_images_per_second,
    )
    return result


def softmax_probabilities(logits: torch.Tensor) -> np.ndarray:
    """Convert logits into float64 probabilities.

    The softmax is computed in float64 so the row sums used by the probability
    normalisation check are not limited by float32 rounding.
    """
    return torch.softmax(logits.to(torch.float64), dim=1).numpy()


