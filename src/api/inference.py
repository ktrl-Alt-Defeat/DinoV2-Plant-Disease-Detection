"""The served model and the inference it performs.

One :class:`InferenceEngine` is built during application startup and reused for
every request; nothing here reloads weights or rebuilds the transform per call.

The checkpoint is the sole authority for what is served: it carries the
configuration it was trained with and its own class vocabulary, so the service
runs without the dataset being present on disk.

Thread safety: FastAPI dispatches synchronous endpoints onto a worker thread
pool, so several requests can reach :meth:`InferenceEngine.predict` at once. The
forward pass is serialised behind a lock. That keeps a single CUDA stream from
being driven by several threads and makes the reported latency describe one
uncontended pass; batching through ``POST /predict/batch`` is the way to raise
throughput, not concurrent single-image calls.
"""

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Final

import torch
from PIL import Image, UnidentifiedImageError

from src.api.errors import InferenceFailedError, InvalidImageError
from src.config import Config
from src.datasets.transforms import TransformSpecification, build_eval_transform
from src.device import get_device
from src.evaluation.integrity import fingerprint_file
from src.logger import get_logger
from src.model import DinoV2Classifier, build_model
from src.training.checkpoints import CheckpointContents, read_checkpoint

#: Precision inference runs in. Matches the evaluation pass, so a served
#: prediction reproduces the reported test metrics exactly.
INFERENCE_PRECISION: Final[str] = "fp32"

#: Colour mode every decoded upload is converted to before preprocessing.
IMAGE_MODE: Final[str] = "RGB"

_MILLISECONDS_PER_SECOND: Final[float] = 1000.0
_BYTES_PER_MIB: Final[int] = 1024**2

_LOGGER: Final = get_logger("api.inference")


@dataclass(frozen=True)
class ScoredClass:
    """One ranked class and its probability."""

    label: str
    index: int
    confidence: float


@dataclass(frozen=True)
class ImagePrediction:
    """Ranked predictions for a single image."""

    filename: str
    ranked: tuple[ScoredClass, ...]

    @property
    def top(self) -> ScoredClass:
        """The highest scoring class."""
        return self.ranked[0]


@dataclass(frozen=True)
class BatchPrediction:
    """Predictions for a batch plus the time the forward pass took."""

    predictions: tuple[ImagePrediction, ...]
    inference_time_ms: float


@dataclass(frozen=True)
class CheckpointIdentity:
    """Identity of the checkpoint the engine serves."""

    filename: str
    sha256: str
    epoch: int
    best_value: float


class InferenceEngine:
    """A loaded model plus the preprocessing it expects.

    Build one with :meth:`load` at startup and share it across requests.
    """

    def __init__(
        self,
        *,
        model: DinoV2Classifier,
        class_names: Sequence[str],
        device: torch.device,
        transform: Any,
        top_k: int,
        version: str,
        checkpoint: CheckpointIdentity,
    ) -> None:
        """Assemble an engine from an already loaded model.

        Raises:
            ValueError: If ``top_k`` or the class list is unusable.
        """
        if not class_names:
            raise ValueError("Inference engine needs at least one class name.")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")

        self._model = model
        self._class_names = tuple(class_names)
        self._device = device
        self._transform = transform
        self._top_k = min(top_k, len(self._class_names))
        self._version = version
        self._checkpoint = checkpoint
        self._lock = threading.Lock()
        self._loaded_at = time.monotonic()

        self._model.eval()

    @classmethod
    def load(
        cls,
        config: Config,
        checkpoint_path: str | Path,
        *,
        top_k: int,
        version: str,
    ) -> "InferenceEngine":
        """Read the checkpoint, rebuild the model it describes and load its weights.

        Raises:
            CheckpointError: If the checkpoint is missing, unreadable or does not
                match the architecture it describes.
        """
        device = get_device(config.get("device.preferred"))
        contents = read_checkpoint(checkpoint_path, device)
        model = _rebuild_model(contents, device)

        fingerprint = fingerprint_file(contents.path)
        engine = cls(
            model=model,
            class_names=contents.class_names,
            device=device,
            transform=build_eval_transform(TransformSpecification.from_config(config)),
            top_k=top_k,
            version=version,
            checkpoint=CheckpointIdentity(
                filename=contents.path.name,
                sha256=fingerprint.sha256,
                epoch=contents.epoch,
                best_value=contents.best_value,
            ),
        )
        _LOGGER.info(
            "Model ready: %s, %d classes, %s parameters, on %s (%s).",
            contents.path.name,
            engine.num_classes,
            f"{model.count_parameters():,}",
            device,
            INFERENCE_PRECISION,
        )
        return engine

    def predict(self, images: Sequence[tuple[str, Image.Image]]) -> BatchPrediction:
        """Score decoded images in one forward pass.

        Args:
            images: ``(filename, image)`` pairs, already decoded and RGB.

        Raises:
            InferenceFailedError: If the forward pass produces non-finite scores.
            ValueError: If no image was supplied.
        """
        if not images:
            raise ValueError("Cannot run inference on an empty batch.")

        batch = torch.stack([self._transform(image) for _, image in images])

        with self._lock:
            batch = batch.to(self._device, non_blocking=True)
            start = time.perf_counter()
            with torch.inference_mode():
                logits = self._model(batch)
                probabilities = torch.softmax(logits.float(), dim=1)
            self._synchronize()
            elapsed = time.perf_counter() - start
            scores = probabilities.detach().to("cpu")

        if not bool(torch.isfinite(scores).all()):
            raise InferenceFailedError("Model produced non-finite probabilities.")

        confidences, indices = scores.topk(self._top_k, dim=1)
        predictions = tuple(
            ImagePrediction(
                filename=filename,
                ranked=tuple(
                    ScoredClass(
                        label=self._class_names[int(index)],
                        index=int(index),
                        confidence=float(confidence),
                    )
                    for confidence, index in zip(row_scores, row_indices, strict=True)
                ),
            )
            for (filename, _), row_scores, row_indices in zip(
                images, confidences, indices, strict=True
            )
        )
        return BatchPrediction(
            predictions=predictions,
            inference_time_ms=elapsed * _MILLISECONDS_PER_SECOND,
        )

    def describe(self) -> dict[str, Any]:
        """Return the static description of the served model."""
        return {
            "model_version": self._version,
            "backbone": self._model.name,
            "backbone_display": self._model.specification.display_name,
            "feature_dim": self._model.feature_dim,
            "image_size": self._model.image_size,
            "num_classes": self.num_classes,
            "classes": list(self._class_names),
            "class_to_idx": {name: index for index, name in enumerate(self._class_names)},
            "total_parameters": self._model.count_parameters(),
            "device": str(self._device),
            "precision": INFERENCE_PRECISION,
            "top_k": self._top_k,
        }

    def gpu_status(self) -> dict[str, Any] | None:
        """Return CUDA device detail, or ``None`` when running on CPU."""
        if self._device.type != "cuda":
            return None
        properties = torch.cuda.get_device_properties(self._device)
        return {
            "name": properties.name,
            "capability": f"sm_{properties.major}{properties.minor}",
            "total_memory_mib": properties.total_memory / _BYTES_PER_MIB,
            "allocated_memory_mib": torch.cuda.memory_allocated(self._device) / _BYTES_PER_MIB,
            "reserved_memory_mib": torch.cuda.memory_reserved(self._device) / _BYTES_PER_MIB,
        }

    @property
    def num_classes(self) -> int:
        """Number of classes the served head predicts."""
        return len(self._class_names)

    @property
    def version(self) -> str:
        """Version reported for the served model."""
        return self._version

    @property
    def device(self) -> torch.device:
        """Device the model runs on."""
        return self._device

    @property
    def checkpoint(self) -> CheckpointIdentity:
        """Identity of the served checkpoint."""
        return self._checkpoint

    @property
    def uptime_seconds(self) -> float:
        """Seconds since the model finished loading."""
        return time.monotonic() - self._loaded_at

    def _synchronize(self) -> None:
        """Wait for CUDA work so the measured latency covers completed compute."""
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)


def decode_image(filename: str, payload: bytes) -> Image.Image:
    """Decode an uploaded byte string into an RGB image.

    Raises:
        InvalidImageError: If the payload is empty or is not a readable image.
    """
    if not payload:
        raise InvalidImageError(f"Uploaded file '{filename}' is empty.")

    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            return image.convert(IMAGE_MODE)
    except UnidentifiedImageError as error:
        raise InvalidImageError(
            f"Uploaded file '{filename}' is not a recognisable image."
        ) from error
    except (OSError, ValueError) as error:
        raise InvalidImageError(
            f"Uploaded file '{filename}' could not be decoded: {error}"
        ) from error


def _rebuild_model(contents: CheckpointContents, device: torch.device) -> DinoV2Classifier:
    """Rebuild the architecture the checkpoint describes and load its weights.

    The backbone is built without pretrained download because every weight comes
    from the checkpoint a moment later.

    Raises:
        CheckpointError: If the stored weights do not fit the described model.
    """
    from src.config import with_overrides
    from src.training.checkpoints import CheckpointError

    stored = Config(contents.config)
    model_config = with_overrides(
        stored,
        model={"num_classes": contents.num_classes, "pretrained": False},
        device={"preferred": device.type},
    )
    model = build_model(model_config)

    try:
        model.load_state_dict(contents.model_state)
    except (RuntimeError, ValueError, KeyError) as error:
        raise CheckpointError(
            f"Checkpoint {contents.path} does not match the model it describes: {error}"
        ) from error

    model.to(device)
    model.eval()
    return model
