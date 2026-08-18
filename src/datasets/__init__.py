"""Dataset readers, transforms and dataloader factories.

:mod:`~src.datasets.validation` audits the dataset on disk, and the audit gates
everything else: :mod:`~src.datasets.loaders` is only reached once the layout is
known to be readable.
"""

from src.datasets.loaders import DataBundle, DataLoaderSpecification, build_dataloaders
from src.datasets.transforms import (
    TransformSpecification,
    build_eval_transform,
    build_train_transform,
)
from src.datasets.validation import (
    DatasetAudit,
    DatasetIssue,
    DatasetSpecification,
    DatasetValidationError,
    audit_dataset,
)

__all__ = [
    "DataBundle",
    "DataLoaderSpecification",
    "DatasetAudit",
    "DatasetIssue",
    "DatasetSpecification",
    "DatasetValidationError",
    "TransformSpecification",
    "audit_dataset",
    "build_dataloaders",
    "build_eval_transform",
    "build_train_transform",
]
