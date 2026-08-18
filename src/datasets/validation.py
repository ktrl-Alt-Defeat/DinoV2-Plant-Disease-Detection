"""Dataset discovery and integrity validation.

The audit walks every configured split, decodes every candidate image and
reports what it finds. It is the gate in front of training: an audit carrying
any error-severity issue aborts the run before a model is built, so a training
job never starts on a dataset that cannot be read end to end.

Class names are discovered from the directory layout and are never hardcoded.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from PIL import Image, UnidentifiedImageError

from src.config import Config, TypeSpec, validate_keys
from src.logger import get_logger
from src.paths import resolve

#: File suffix to the image formats PIL may legitimately report for it. Also the
#: set of suffixes ``dataset.extensions`` may be drawn from.
#:
#: One suffix can decode to more than one format: MPO is a JEITA/CIPA container
#: that packs several JPEG frames into a JPEG-compatible file, written by phone
#: cameras in burst, HDR and stereo modes. Such a file is a valid JPEG and PIL
#: returns its first frame, so it is accepted under the JPEG suffixes. Mapping a
#: suffix to a set keeps the check meaningful — a ``.png`` decoding as JPEG is
#: still reported — without rejecting a genuine JPEG-family image.
EXTENSION_FORMATS: Final[Mapping[str, frozenset[str]]] = {
    ".jpg": frozenset({"JPEG", "MPO"}),
    ".jpeg": frozenset({"JPEG", "MPO"}),
    ".png": frozenset({"PNG"}),
    ".bmp": frozenset({"BMP"}),
    ".webp": frozenset({"WEBP"}),
    ".tif": frozenset({"TIFF"}),
    ".tiff": frozenset({"TIFF"}),
}

#: Logical split names, in the order they are reported.
SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "val", "test")

#: Split whose class list defines the reference vocabulary of the dataset.
REFERENCE_SPLIT: Final[str] = "train"

SEVERITY_ERROR: Final[str] = "error"
SEVERITY_WARNING: Final[str] = "warning"

ISSUE_MISSING_ROOT: Final[str] = "missing_root"
ISSUE_MISSING_SPLIT: Final[str] = "missing_split"
ISSUE_NO_CLASSES: Final[str] = "no_classes"
ISSUE_MISSING_CLASS: Final[str] = "missing_class"
ISSUE_ORPHAN_CLASS: Final[str] = "orphan_class"
ISSUE_EMPTY_CLASS: Final[str] = "empty_class"
ISSUE_ZERO_BYTE: Final[str] = "zero_byte_file"
ISSUE_INVALID_EXTENSION: Final[str] = "invalid_extension"
ISSUE_UNSUPPORTED_FORMAT: Final[str] = "unsupported_format"
ISSUE_UNREADABLE: Final[str] = "unreadable_image"
ISSUE_CORRUPT: Final[str] = "corrupt_image"
ISSUE_DUPLICATE_NAME: Final[str] = "duplicate_filename"
ISSUE_IMBALANCE: Final[str] = "class_imbalance"

#: Configuration contract of the ``dataset`` section consumed by the audit.
DATASET_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("dataset.root", str),
    ("dataset.extensions", list),
    ("dataset.imbalance_ratio_threshold", (int, float)),
    ("dataset.audit_filename", str),
)

_LOGGER: Final = get_logger("dataset.validation")


class DatasetValidationError(RuntimeError):
    """Raised when the dataset on disk cannot be used for training."""


@dataclass(frozen=True)
class DatasetIssue:
    """A single problem found while auditing the dataset."""

    category: str
    severity: str
    location: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        """Return the issue as a serialisable mapping."""
        return {
            "category": self.category,
            "severity": self.severity,
            "location": self.location,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SplitStatistics:
    """Per-class image counts of one split."""

    name: str
    directory: Path
    class_counts: dict[str, int]

    @property
    def class_count(self) -> int:
        """Number of class directories discovered in the split."""
        return len(self.class_counts)

    @property
    def image_count(self) -> int:
        """Number of valid images across every class of the split."""
        return sum(self.class_counts.values())

    @property
    def imbalance_ratio(self) -> float:
        """Largest class size divided by the smallest, or ``0.0`` when empty."""
        if not self.class_counts:
            return 0.0
        smallest = min(self.class_counts.values())
        if smallest == 0:
            return float("inf")
        return max(self.class_counts.values()) / smallest

    def as_dict(self) -> dict[str, Any]:
        """Return the statistics as a serialisable mapping."""
        return {
            "name": self.name,
            "directory": str(self.directory),
            "class_count": self.class_count,
            "image_count": self.image_count,
            "imbalance_ratio": round(self.imbalance_ratio, 3),
            "class_counts": dict(sorted(self.class_counts.items())),
        }


@dataclass(frozen=True)
class DatasetAudit:
    """Outcome of a full dataset audit."""

    root: Path
    classes: tuple[str, ...]
    splits: tuple[SplitStatistics, ...]
    issues: tuple[DatasetIssue, ...]

    @property
    def errors(self) -> tuple[DatasetIssue, ...]:
        """Issues that make the dataset unusable."""
        return tuple(issue for issue in self.issues if issue.severity == SEVERITY_ERROR)

    @property
    def warnings(self) -> tuple[DatasetIssue, ...]:
        """Issues worth reporting that do not block training."""
        return tuple(issue for issue in self.issues if issue.severity == SEVERITY_WARNING)

    @property
    def passed(self) -> bool:
        """Whether the dataset carries no error-severity issue."""
        return not self.errors

    @property
    def status(self) -> str:
        """Overall verdict, ``"PASS"`` or ``"FAIL"``."""
        return "PASS" if self.passed else "FAIL"

    @property
    def class_count(self) -> int:
        """Number of distinct classes discovered across the dataset."""
        return len(self.classes)

    @property
    def total_images(self) -> int:
        """Number of valid images across every split."""
        return sum(split.image_count for split in self.splits)

    def as_dict(self) -> dict[str, Any]:
        """Return the whole audit as a serialisable mapping."""
        return {
            "root": str(self.root),
            "status": self.status,
            "class_count": self.class_count,
            "classes": list(self.classes),
            "total_images": self.total_images,
            "splits": [split.as_dict() for split in self.splits],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class DatasetSpecification:
    """Dataset settings resolved from the ``dataset`` configuration section."""

    root: Path
    split_directories: dict[str, str]
    extensions: frozenset[str]
    imbalance_ratio_threshold: float
    audit_filename: str

    @classmethod
    def from_config(cls, config: Config) -> "DatasetSpecification":
        """Read and validate the ``dataset`` section of ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If a split entry, extension or threshold is unusable.
        """
        validate_keys(config, DATASET_REQUIRED_KEYS, context="dataset section")

        split_directories = {name: config.get(f"dataset.splits.{name}") for name in SPLIT_NAMES}
        for name, directory in split_directories.items():
            if not isinstance(directory, str) or not directory:
                raise ValueError(
                    f"dataset.splits.{name} must be a non-empty string, got {directory!r}."
                )

        extensions = _normalise_extensions(config.get("dataset.extensions"))

        threshold = float(config.get("dataset.imbalance_ratio_threshold"))
        if threshold <= 1.0:
            raise ValueError(
                f"dataset.imbalance_ratio_threshold must be greater than 1, got {threshold}."
            )

        return cls(
            root=resolve(config.get("dataset.root")),
            split_directories=split_directories,
            extensions=extensions,
            imbalance_ratio_threshold=threshold,
            audit_filename=config.get("dataset.audit_filename"),
        )

    @property
    def allowed_formats(self) -> frozenset[str]:
        """Image formats PIL may report for the configured extensions.

        ``extensions`` is validated non-empty, so the union always has a term.
        """
        return frozenset().union(
            *(EXTENSION_FORMATS[extension] for extension in self.extensions)
        )

    def split_directory(self, split: str) -> Path:
        """Return the absolute directory of ``split``."""
        return self.root / self.split_directories[split]


def audit_dataset(specification: DatasetSpecification) -> DatasetAudit:
    """Walk every split and report the state of the dataset on disk.

    Every candidate file is fully decoded, so the audit is authoritative about
    whether the images can actually be read. It never raises for dataset
    problems; the caller inspects :attr:`DatasetAudit.passed` instead.
    """
    _LOGGER.info("Auditing dataset at %s.", specification.root)

    if not specification.root.is_dir():
        issue = DatasetIssue(
            category=ISSUE_MISSING_ROOT,
            severity=SEVERITY_ERROR,
            location=str(specification.root),
            detail="Dataset root directory does not exist.",
        )
        return DatasetAudit(root=specification.root, classes=(), splits=(), issues=(issue,))

    issues: list[DatasetIssue] = []
    statistics: list[SplitStatistics] = []
    discovered: dict[str, tuple[str, ...]] = {}

    for split in SPLIT_NAMES:
        directory = specification.split_directory(split)
        if not directory.is_dir():
            issues.append(
                DatasetIssue(
                    category=ISSUE_MISSING_SPLIT,
                    severity=SEVERITY_ERROR,
                    location=str(directory),
                    detail=f"Split '{split}' directory does not exist.",
                )
            )
            continue

        class_names = _discover_classes(directory)
        discovered[split] = class_names
        if not class_names:
            issues.append(
                DatasetIssue(
                    category=ISSUE_NO_CLASSES,
                    severity=SEVERITY_ERROR,
                    location=str(directory),
                    detail=f"Split '{split}' contains no class sub-directories.",
                )
            )
            continue

        counts, split_issues = _scan_split(split, directory, class_names, specification)
        issues.extend(split_issues)
        statistics.append(SplitStatistics(name=split, directory=directory, class_counts=counts))

    classes = tuple(sorted({name for names in discovered.values() for name in names}))
    issues.extend(_compare_class_vocabularies(discovered, specification))
    issues.extend(_detect_imbalance(statistics, specification.imbalance_ratio_threshold))

    audit = DatasetAudit(
        root=specification.root,
        classes=classes,
        splits=tuple(statistics),
        issues=tuple(issues),
    )
    _LOGGER.info(
        "Audit complete: %d classes, %d images, %d error(s), %d warning(s).",
        audit.class_count,
        audit.total_images,
        len(audit.errors),
        len(audit.warnings),
    )
    return audit


def _normalise_extensions(raw: Sequence[Any]) -> frozenset[str]:
    """Lower-case and validate the configured file extensions.

    Raises:
        ValueError: If the list is empty or names an unsupported extension.
    """
    if not raw:
        raise ValueError("dataset.extensions must list at least one file extension.")

    extensions: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str):
            raise ValueError(f"dataset.extensions entries must be strings, got {entry!r}.")
        extension = entry.strip().lower()
        if extension not in EXTENSION_FORMATS:
            supported = ", ".join(sorted(EXTENSION_FORMATS))
            raise ValueError(
                f"Unsupported dataset extension '{entry}'. Supported extensions: {supported}."
            )
        extensions.add(extension)
    return frozenset(extensions)


def _discover_classes(split_directory: Path) -> tuple[str, ...]:
    """Return the sorted class directory names inside ``split_directory``."""
    return tuple(sorted(entry.name for entry in split_directory.iterdir() if entry.is_dir()))


def _scan_split(
    split: str,
    directory: Path,
    class_names: Sequence[str],
    specification: DatasetSpecification,
) -> tuple[dict[str, int], list[DatasetIssue]]:
    """Inspect every file of every class and count the usable images."""
    counts: dict[str, int] = {}
    issues: list[DatasetIssue] = []

    for class_name in class_names:
        class_directory = directory / class_name
        files = sorted(entry for entry in class_directory.iterdir() if entry.is_file())

        valid = 0
        for path in files:
            issue = _inspect_image(path, specification)
            if issue is None:
                valid += 1
            else:
                issues.append(issue)
        counts[class_name] = valid

        issues.extend(_detect_duplicate_names(class_directory, files))
        if valid == 0:
            issues.append(
                DatasetIssue(
                    category=ISSUE_EMPTY_CLASS,
                    severity=SEVERITY_ERROR,
                    location=str(class_directory),
                    detail=(
                        f"Class '{class_name}' in split '{split}' holds no readable image "
                        f"({len(files)} file(s) present)."
                    ),
                )
            )

    return counts, issues


def _inspect_image(path: Path, specification: DatasetSpecification) -> DatasetIssue | None:
    """Return the problem found with ``path``, or ``None`` when it is a usable image."""
    extension = path.suffix.lower()
    if extension not in specification.extensions:
        allowed = ", ".join(sorted(specification.extensions))
        return DatasetIssue(
            category=ISSUE_INVALID_EXTENSION,
            severity=SEVERITY_ERROR,
            location=str(path),
            detail=f"Extension '{path.suffix}' is not one of the allowed extensions: {allowed}.",
        )

    if path.stat().st_size == 0:
        return DatasetIssue(
            category=ISSUE_ZERO_BYTE,
            severity=SEVERITY_ERROR,
            location=str(path),
            detail="File is zero bytes.",
        )

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            image_format = image.format
    except UnidentifiedImageError:
        return DatasetIssue(
            category=ISSUE_CORRUPT,
            severity=SEVERITY_ERROR,
            location=str(path),
            detail="File is not a recognisable image.",
        )
    except Image.DecompressionBombError as error:
        return DatasetIssue(
            category=ISSUE_CORRUPT,
            severity=SEVERITY_ERROR,
            location=str(path),
            detail=f"Image is implausibly large: {error}",
        )
    except (OSError, ValueError) as error:
        return DatasetIssue(
            category=ISSUE_UNREADABLE,
            severity=SEVERITY_ERROR,
            location=str(path),
            detail=f"Image could not be decoded: {error}",
        )

    if image_format not in specification.allowed_formats:
        allowed = ", ".join(sorted(specification.allowed_formats))
        return DatasetIssue(
            category=ISSUE_UNSUPPORTED_FORMAT,
            severity=SEVERITY_ERROR,
            location=str(path),
            detail=(
                f"Decoded format '{image_format}' is not supported; expected one of: {allowed}."
            ),
        )
    return None


def _detect_duplicate_names(class_directory: Path, files: Sequence[Path]) -> list[DatasetIssue]:
    """Report file stems that occur more than once inside one class directory.

    A collision does not imply duplicated data and never affects training:
    ``ImageFolder`` indexes by full filename, so colliding files are already two
    independent samples. It is reported as a warning because anything keyed by
    stem — a feature cache, a per-image prediction file — would silently
    overwrite one with the other.
    """
    stems = Counter(path.stem.lower() for path in files)
    return [
        DatasetIssue(
            category=ISSUE_DUPLICATE_NAME,
            severity=SEVERITY_WARNING,
            location=str(class_directory),
            detail=f"File name '{stem}' occurs {count} times with differing extensions.",
        )
        for stem, count in sorted(stems.items())
        if count > 1
    ]


def _compare_class_vocabularies(
    discovered: Mapping[str, Sequence[str]],
    specification: DatasetSpecification,
) -> list[DatasetIssue]:
    """Report classes missing from a split or present only outside the reference split."""
    reference = set(discovered.get(REFERENCE_SPLIT, ()))
    if not reference:
        return []

    issues: list[DatasetIssue] = []
    for split, class_names in discovered.items():
        if split == REFERENCE_SPLIT:
            continue
        present = set(class_names)
        directory = specification.split_directory(split)

        for class_name in sorted(reference - present):
            issues.append(
                DatasetIssue(
                    category=ISSUE_MISSING_CLASS,
                    severity=SEVERITY_ERROR,
                    location=str(directory / class_name),
                    detail=(
                        f"Class '{class_name}' exists in '{REFERENCE_SPLIT}' "
                        f"but is missing from '{split}'."
                    ),
                )
            )
        for class_name in sorted(present - reference):
            issues.append(
                DatasetIssue(
                    category=ISSUE_ORPHAN_CLASS,
                    severity=SEVERITY_ERROR,
                    location=str(directory / class_name),
                    detail=(
                        f"Class '{class_name}' exists in '{split}' "
                        f"but not in '{REFERENCE_SPLIT}'."
                    ),
                )
            )
    return issues


def _detect_imbalance(
    statistics: Sequence[SplitStatistics],
    threshold: float,
) -> list[DatasetIssue]:
    """Report splits whose largest class dwarfs the smallest."""
    issues: list[DatasetIssue] = []
    for split in statistics:
        ratio = split.imbalance_ratio
        if ratio <= threshold:
            continue
        largest = max(split.class_counts, key=lambda name: split.class_counts[name])
        smallest = min(split.class_counts, key=lambda name: split.class_counts[name])
        issues.append(
            DatasetIssue(
                category=ISSUE_IMBALANCE,
                severity=SEVERITY_WARNING,
                location=str(split.directory),
                detail=(
                    f"Split '{split.name}' is imbalanced: ratio {ratio:.2f} exceeds "
                    f"{threshold:.2f} ('{largest}'={split.class_counts[largest]}, "
                    f"'{smallest}'={split.class_counts[smallest]})."
                ),
            )
        )
    return issues
