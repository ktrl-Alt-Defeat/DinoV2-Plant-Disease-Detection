"""Image preprocessing and augmentation pipelines.

Training and evaluation pipelines are built by separate functions from the same
specification, which is what keeps augmentation out of the validation and test
paths: the evaluation pipeline is deterministic by construction, not by a flag.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from torchvision import transforms

from src.config import Config, TypeSpec, validate_keys

#: Number of channels the normalisation statistics must describe.
NORMALIZATION_CHANNELS: Final[int] = 3

#: Configuration contract of the transform pipeline.
TRANSFORM_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("model.image_size", int),
    ("dataset.resize_size", int),
    ("dataset.normalization.mean", list),
    ("dataset.normalization.std", list),
    ("dataset.augmentation.random_resized_crop_scale", list),
    ("dataset.augmentation.random_resized_crop_ratio", list),
    ("dataset.augmentation.horizontal_flip_probability", (int, float)),
    ("dataset.augmentation.rotation_degrees", (int, float)),
    ("dataset.augmentation.color_jitter_brightness", (int, float)),
    ("dataset.augmentation.color_jitter_contrast", (int, float)),
    ("dataset.augmentation.color_jitter_saturation", (int, float)),
    ("dataset.augmentation.color_jitter_hue", (int, float)),
    ("dataset.augmentation.random_erasing_probability", (int, float)),
)

#: Widest hue shift ``ColorJitter`` accepts.
_MAX_HUE: Final[float] = 0.5


@dataclass(frozen=True)
class NormalizationSpecification:
    """Per-channel mean and standard deviation applied after tensor conversion."""

    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(frozen=True)
class AugmentationSpecification:
    """Randomised transforms applied to training images only."""

    crop_scale: tuple[float, float]
    crop_ratio: tuple[float, float]
    horizontal_flip_probability: float
    rotation_degrees: float
    brightness: float
    contrast: float
    saturation: float
    hue: float
    erasing_probability: float


@dataclass(frozen=True)
class TransformSpecification:
    """Everything needed to build the training and evaluation pipelines."""

    image_size: int
    resize_size: int
    normalization: NormalizationSpecification
    augmentation: AugmentationSpecification

    @classmethod
    def from_config(cls, config: Config) -> "TransformSpecification":
        """Read and validate every transform setting from ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If a size, probability or jitter magnitude is unusable.
        """
        validate_keys(config, TRANSFORM_REQUIRED_KEYS, context="transform settings")

        image_size = config.get("model.image_size")
        resize_size = config.get("dataset.resize_size")
        if image_size <= 0:
            raise ValueError(f"model.image_size must be positive, got {image_size}.")
        if resize_size < image_size:
            raise ValueError(
                f"dataset.resize_size ({resize_size}) must be at least "
                f"model.image_size ({image_size}); cropping cannot enlarge an image."
            )

        return cls(
            image_size=image_size,
            resize_size=resize_size,
            normalization=cls._read_normalization(config),
            augmentation=cls._read_augmentation(config),
        )

    @staticmethod
    def _read_normalization(config: Config) -> NormalizationSpecification:
        """Read and validate the normalisation statistics."""
        mean = _float_tuple(config.get("dataset.normalization.mean"), "mean")
        std = _float_tuple(config.get("dataset.normalization.std"), "std")

        for name, values in (("mean", mean), ("std", std)):
            if len(values) != NORMALIZATION_CHANNELS:
                raise ValueError(
                    f"dataset.normalization.{name} must list {NORMALIZATION_CHANNELS} "
                    f"values, got {len(values)}."
                )
        if any(value <= 0.0 for value in std):
            raise ValueError(f"dataset.normalization.std values must be positive, got {std}.")
        return NormalizationSpecification(mean=mean, std=std)

    @staticmethod
    def _read_augmentation(config: Config) -> AugmentationSpecification:
        """Read and validate the training augmentation settings."""
        crop_scale = _bounded_pair(
            config.get("dataset.augmentation.random_resized_crop_scale"),
            "dataset.augmentation.random_resized_crop_scale",
        )
        crop_ratio = _bounded_pair(
            config.get("dataset.augmentation.random_resized_crop_ratio"),
            "dataset.augmentation.random_resized_crop_ratio",
        )

        flip = _probability(config, "dataset.augmentation.horizontal_flip_probability")
        erasing = _probability(config, "dataset.augmentation.random_erasing_probability")

        rotation = float(config.get("dataset.augmentation.rotation_degrees"))
        if rotation < 0.0:
            raise ValueError(
                f"dataset.augmentation.rotation_degrees must be non-negative, got {rotation}."
            )

        jitter = {
            name: _non_negative(config, f"dataset.augmentation.color_jitter_{name}")
            for name in ("brightness", "contrast", "saturation")
        }
        hue = _non_negative(config, "dataset.augmentation.color_jitter_hue")
        if hue > _MAX_HUE:
            raise ValueError(
                f"dataset.augmentation.color_jitter_hue must be at most {_MAX_HUE}, got {hue}."
            )

        return AugmentationSpecification(
            crop_scale=crop_scale,
            crop_ratio=crop_ratio,
            horizontal_flip_probability=flip,
            rotation_degrees=rotation,
            erasing_probability=erasing,
            hue=hue,
            **jitter,
        )


def build_train_transform(specification: TransformSpecification) -> transforms.Compose:
    """Build the randomised training pipeline.

    ``RandomErasing`` operates on tensors, so it is applied after normalisation;
    every other augmentation acts on the decoded image.
    """
    augmentation = specification.augmentation
    return transforms.Compose(
        [
            transforms.Resize(specification.resize_size),
            transforms.RandomResizedCrop(
                specification.image_size,
                scale=augmentation.crop_scale,
                ratio=augmentation.crop_ratio,
            ),
            transforms.RandomHorizontalFlip(p=augmentation.horizontal_flip_probability),
            transforms.RandomRotation(degrees=augmentation.rotation_degrees),
            transforms.ColorJitter(
                brightness=augmentation.brightness,
                contrast=augmentation.contrast,
                saturation=augmentation.saturation,
                hue=augmentation.hue,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=specification.normalization.mean,
                std=specification.normalization.std,
            ),
            transforms.RandomErasing(p=augmentation.erasing_probability),
        ]
    )


def build_eval_transform(specification: TransformSpecification) -> transforms.Compose:
    """Build the deterministic pipeline used for validation and test data."""
    return transforms.Compose(
        [
            transforms.Resize(specification.resize_size),
            transforms.CenterCrop(specification.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=specification.normalization.mean,
                std=specification.normalization.std,
            ),
        ]
    )


def _float_tuple(values: Sequence[Any], name: str) -> tuple[float, ...]:
    """Convert a configured sequence into a tuple of floats.

    Raises:
        ValueError: If an entry is not numeric.
    """
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                f"dataset.normalization.{name} entries must be numbers, got {value!r}."
            )
        converted.append(float(value))
    return tuple(converted)


def _bounded_pair(values: Sequence[Any], key: str) -> tuple[float, float]:
    """Read an ascending ``[low, high]`` pair of positive numbers.

    Raises:
        ValueError: If the pair is malformed, non-positive or not ascending.
    """
    if len(values) != 2:
        raise ValueError(f"{key} must list exactly two values, got {len(values)}.")

    low, high = (float(value) for value in values)
    if low <= 0.0 or high <= 0.0:
        raise ValueError(f"{key} values must be positive, got [{low}, {high}].")
    if low > high:
        raise ValueError(f"{key} must be ascending, got [{low}, {high}].")
    return low, high


def _probability(config: Config, key: str) -> float:
    """Read a configured probability.

    Raises:
        ValueError: If the value falls outside ``[0, 1]``.
    """
    value = float(config.get(key))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be within [0, 1], got {value}.")
    return value


def _non_negative(config: Config, key: str) -> float:
    """Read a configured non-negative magnitude.

    Raises:
        ValueError: If the value is negative.
    """
    value = float(config.get(key))
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative, got {value}.")
    return value
