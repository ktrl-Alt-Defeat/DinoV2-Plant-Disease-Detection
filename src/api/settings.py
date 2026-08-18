"""API settings resolved from the shared configuration file.

The service reads the same ``configs/config.yaml`` as every other entry point,
so the served checkpoint, the preprocessing and the class vocabulary cannot
drift from the ones training and evaluation used.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from src.config import Config, TypeSpec, validate_keys

#: Configuration contract of the ``api`` section.
API_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("api.title", str),
    ("api.description", str),
    ("api.checkpoint_filename", str),
    ("api.log_filename", str),
    ("api.top_k", int),
    ("api.max_batch_size", int),
    ("api.max_image_bytes", int),
    ("api.allowed_content_types", list),
)


@dataclass(frozen=True)
class ApiSettings:
    """Everything the service needs that is not the model itself."""

    title: str
    description: str
    version: str
    checkpoint_filename: str
    log_filename: str
    top_k: int
    max_batch_size: int
    max_image_bytes: int
    allowed_content_types: frozenset[str]

    @classmethod
    def from_config(cls, config: Config) -> "ApiSettings":
        """Read and validate the ``api`` section of ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If a limit is not positive or a content type is unusable.
        """
        validate_keys(config, API_REQUIRED_KEYS, context="api section")

        top_k = config.get("api.top_k")
        max_batch = config.get("api.max_batch_size")
        max_bytes = config.get("api.max_image_bytes")

        for key, value in (
            ("api.top_k", top_k),
            ("api.max_batch_size", max_batch),
            ("api.max_image_bytes", max_bytes),
        ):
            if value <= 0:
                raise ValueError(f"{key} must be positive, got {value}.")

        return cls(
            title=config.get("api.title"),
            description=config.get("api.description"),
            version=config.get("project.version"),
            checkpoint_filename=config.get("api.checkpoint_filename"),
            log_filename=config.get("api.log_filename"),
            top_k=top_k,
            max_batch_size=max_batch,
            max_image_bytes=max_bytes,
            allowed_content_types=_content_types(config.get("api.allowed_content_types")),
        )


def _content_types(values: Sequence[object]) -> frozenset[str]:
    """Normalise the accepted upload content types.

    Raises:
        ValueError: If the list is empty or holds a non-string entry.
    """
    if not values:
        raise ValueError("api.allowed_content_types must list at least one media type.")

    types: set[str] = set()
    for entry in values:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(
                f"api.allowed_content_types entries must be non-empty strings, got {entry!r}."
            )
        types.add(entry.strip().lower())
    return frozenset(types)
