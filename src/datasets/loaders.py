"""Dataset and dataloader construction.

The three splits are discovered with :class:`~torchvision.datasets.ImageFolder`,
which derives the class vocabulary from the directory names. The mapping is
compared across splits so an inconsistent layout fails here rather than
producing silently mislabelled batches.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.config import Config, TypeSpec, validate_keys
from src.datasets.transforms import (
    TransformSpecification,
    build_eval_transform,
    build_train_transform,
)
from src.datasets.validation import (
    REFERENCE_SPLIT,
    SPLIT_NAMES,
    DatasetSpecification,
    DatasetValidationError,
)
from src.logger import get_logger

#: NumPy's legacy global generator only accepts seeds in ``[0, 2**32)``.
_NUMPY_SEED_MODULUS: Final[int] = 2**32

#: Configuration contract of the ``dataloader`` section.
DATALOADER_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("dataloader.batch_size", int),
    ("dataloader.num_workers", int),
    ("dataloader.pin_memory", bool),
    ("dataloader.persistent_workers", bool),
    ("dataloader.prefetch_factor", int),
    ("dataloader.drop_last", bool),
)

_LOGGER: Final = get_logger("dataset.loaders")


@dataclass(frozen=True)
class DataLoaderSpecification:
    """Batching and worker settings resolved from the ``dataloader`` section."""

    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    drop_last: bool

    @classmethod
    def from_config(cls, config: Config) -> "DataLoaderSpecification":
        """Read and validate the ``dataloader`` section of ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If a size or count is out of range.
        """
        validate_keys(config, DATALOADER_REQUIRED_KEYS, context="dataloader section")

        batch_size = config.get("dataloader.batch_size")
        num_workers = config.get("dataloader.num_workers")
        prefetch_factor = config.get("dataloader.prefetch_factor")

        if batch_size <= 0:
            raise ValueError(f"dataloader.batch_size must be positive, got {batch_size}.")
        if num_workers < 0:
            raise ValueError(f"dataloader.num_workers must be non-negative, got {num_workers}.")
        if prefetch_factor <= 0:
            raise ValueError(
                f"dataloader.prefetch_factor must be positive, got {prefetch_factor}."
            )

        return cls(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=config.get("dataloader.pin_memory"),
            persistent_workers=config.get("dataloader.persistent_workers"),
            prefetch_factor=prefetch_factor,
            drop_last=config.get("dataloader.drop_last"),
        )


@dataclass(frozen=True)
class DataBundle:
    """The three dataloaders plus the class vocabulary they share."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_to_idx: dict[str, int]
    split_sizes: dict[str, int]

    @property
    def num_classes(self) -> int:
        """Number of classes discovered on disk."""
        return len(self.class_to_idx)

    @property
    def idx_to_class(self) -> dict[int, str]:
        """Inverse of :attr:`class_to_idx`."""
        return {index: name for name, index in self.class_to_idx.items()}

    def loader_for(self, split: str) -> DataLoader:
        """Return the dataloader of ``split``.

        Raises:
            KeyError: If ``split`` is not one of the three known splits.
        """
        loaders = {
            "train": self.train_loader,
            "val": self.val_loader,
            "test": self.test_loader,
        }
        if split not in loaders:
            raise KeyError(
                f"Unknown split '{split}'. Expected one of: {', '.join(sorted(loaders))}."
            )
        return loaders[split]

    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary of the discovered dataset."""
        return {
            "num_classes": self.num_classes,
            "split_sizes": dict(self.split_sizes),
            "total_images": sum(self.split_sizes.values()),
        }


def build_dataloaders(
    dataset_specification: DatasetSpecification,
    transform_specification: TransformSpecification,
    loader_specification: DataLoaderSpecification,
    *,
    seed: int,
    device: torch.device,
) -> DataBundle:
    """Build the train, validation and test dataloaders.

    Only the training split is shuffled and augmented. Shuffling is driven by a
    seeded generator and each worker reseeds the Python and NumPy generators, so
    a run is reproducible for a given seed and worker count.

    Raises:
        DatasetValidationError: If a split cannot be read or the splits disagree
            about the class vocabulary.
    """
    train_transform = build_train_transform(transform_specification)
    eval_transform = build_eval_transform(transform_specification)
    is_valid_file = _extension_filter(dataset_specification.extensions)

    datasets = {
        split: _build_dataset(
            dataset_specification.split_directory(split),
            transform=train_transform if split == REFERENCE_SPLIT else eval_transform,
            is_valid_file=is_valid_file,
        )
        for split in SPLIT_NAMES
    }
    class_to_idx = _shared_class_to_idx(datasets)

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = loader_specification.pin_memory and device.type == "cuda"
    _log_effective_settings(loader_specification, pin_memory=pin_memory)

    loaders = {
        split: _build_loader(
            dataset,
            loader_specification,
            shuffle=split == REFERENCE_SPLIT,
            pin_memory=pin_memory,
            generator=generator,
        )
        for split, dataset in datasets.items()
    }

    bundle = DataBundle(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        class_to_idx=class_to_idx,
        split_sizes={split: len(dataset) for split, dataset in datasets.items()},
    )
    _LOGGER.info(
        "Dataloaders ready: %d classes, %s.",
        bundle.num_classes,
        ", ".join(f"{split}={size}" for split, size in bundle.split_sizes.items()),
    )
    return bundle


def _build_dataset(
    directory: Path,
    *,
    transform: Callable[[Any], torch.Tensor],
    is_valid_file: Callable[[str], bool],
) -> ImageFolder:
    """Build one :class:`ImageFolder` over ``directory``.

    Raises:
        DatasetValidationError: If the directory cannot be read or holds no image.
    """
    try:
        dataset = ImageFolder(
            root=str(directory),
            transform=transform,
            is_valid_file=is_valid_file,
        )
    except (FileNotFoundError, RuntimeError) as error:
        raise DatasetValidationError(
            f"Unable to read dataset split at {directory}: {error}"
        ) from error

    if len(dataset) == 0:
        raise DatasetValidationError(f"Dataset split at {directory} contains no usable image.")
    return dataset


def _extension_filter(extensions: frozenset[str]) -> Callable[[str], bool]:
    """Return a predicate accepting paths whose suffix is allowed, ignoring case."""

    def is_valid_file(path: str) -> bool:
        return Path(path).suffix.lower() in extensions

    return is_valid_file


def _shared_class_to_idx(datasets: dict[str, ImageFolder]) -> dict[str, int]:
    """Return the class mapping shared by every split.

    Raises:
        DatasetValidationError: If two splits disagree about the mapping, which
            would silently assign different labels to the same class name.
    """
    reference = datasets[REFERENCE_SPLIT].class_to_idx
    for split, dataset in datasets.items():
        if dataset.class_to_idx != reference:
            missing = sorted(set(reference) - set(dataset.class_to_idx))
            extra = sorted(set(dataset.class_to_idx) - set(reference))
            raise DatasetValidationError(
                f"Split '{split}' does not share the class vocabulary of "
                f"'{REFERENCE_SPLIT}'. Missing: {missing or 'none'}; unexpected: "
                f"{extra or 'none'}. Run the dataset audit for details."
            )
    return dict(reference)


def _build_loader(
    dataset: ImageFolder,
    specification: DataLoaderSpecification,
    *,
    shuffle: bool,
    pin_memory: bool,
    generator: torch.Generator,
) -> DataLoader:
    """Build one dataloader, applying worker options only when workers are used."""
    options: dict[str, Any] = {}
    if specification.num_workers > 0:
        options["persistent_workers"] = specification.persistent_workers
        options["prefetch_factor"] = specification.prefetch_factor

    return DataLoader(
        dataset,
        batch_size=specification.batch_size,
        shuffle=shuffle,
        num_workers=specification.num_workers,
        pin_memory=pin_memory,
        drop_last=specification.drop_last and shuffle,
        worker_init_fn=_seed_worker,
        generator=generator,
        **options,
    )


def _seed_worker(worker_id: int) -> None:
    """Reseed the Python and NumPy generators inside a dataloader worker.

    PyTorch seeds its own generator per worker; the other two are process-global
    and would otherwise repeat the parent's stream in every worker.
    """
    worker_seed = torch.initial_seed() % _NUMPY_SEED_MODULUS
    np.random.seed(worker_seed)
    random.seed(worker_seed + worker_id)


def _log_effective_settings(
    specification: DataLoaderSpecification,
    *,
    pin_memory: bool,
) -> None:
    """Report the settings actually applied, including any that were overridden."""
    _LOGGER.info(
        "Dataloader settings: batch_size=%d, num_workers=%d, pin_memory=%s, drop_last=%s.",
        specification.batch_size,
        specification.num_workers,
        pin_memory,
        specification.drop_last,
    )
    if specification.num_workers == 0:
        _LOGGER.info(
            "num_workers is 0: persistent_workers and prefetch_factor are not applied."
        )
    else:
        _LOGGER.info(
            "Worker options: persistent_workers=%s, prefetch_factor=%d.",
            specification.persistent_workers,
            specification.prefetch_factor,
        )
    if specification.pin_memory and not pin_memory:
        _LOGGER.info("pin_memory requested but disabled: it requires a CUDA device.")
