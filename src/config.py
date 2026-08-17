"""YAML configuration loading and validation.

The configuration file is the single source of truth for every tunable value in
the project. This module reads it, checks that the keys required by the
infrastructure layer are present and well typed, and exposes the result through
a read-only :class:`Config` object with dotted-key lookups.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Final

import yaml

from src.paths import resolve

#: A required key accepts either a single type or any type from a tuple.
TypeSpec = type | tuple[type, ...]

#: Dotted keys the infrastructure layer requires, with the type each value has
#: to have. Subsystems declare their own key contract and check it with
#: :func:`validate_keys` when they consume the configuration.
REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("project.name", str),
    ("project.version", str),
    ("project.seed", int),
    ("paths.logs", str),
    ("paths.checkpoints", str),
    ("paths.results", str),
    ("device.preferred", str),
    ("logging.level", str),
    ("logging.save_file", bool),
    ("reproducibility.deterministic", bool),
    ("reproducibility.benchmark", bool),
)

_KEY_SEPARATOR: Final[str] = "."

# Sentinel telling ``Config.get`` that no default was supplied, so a missing key
# is an error rather than a ``None`` result.
_MISSING: Final[object] = object()


class ConfigError(ValueError):
    """Raised when a configuration file is unreadable, malformed or incomplete."""


class Config:
    """Read-only view over a nested configuration mapping.

    The mapping passed to the constructor is deep copied, and every accessor
    returns copies as well, so a loaded configuration cannot be mutated by
    accident from anywhere in the codebase.
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise ConfigError(
                f"Configuration root must be a mapping, got {type(data).__name__}."
            )
        self._data: dict[str, Any] = deepcopy(dict(data))

    def get(self, key: str, default: Any = _MISSING) -> Any:
        """Return the value stored at a dotted ``key`` such as ``"project.seed"``.

        Args:
            key: Dotted path into the configuration tree.
            default: Value returned when the key is absent. When omitted, a
                missing key raises :class:`ConfigError`.

        Raises:
            ConfigError: If the key is absent and no default was supplied, or if
                an intermediate segment is not a mapping.
        """
        node: Any = self._data
        walked: list[str] = []
        for segment in key.split(_KEY_SEPARATOR):
            if not isinstance(node, Mapping) or segment not in node:
                if default is not _MISSING:
                    return default
                raise ConfigError(self._missing_key_message(key, walked, node))
            node = node[segment]
            walked.append(segment)
        return deepcopy(node)

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level section as a plain dictionary.

        Raises:
            ConfigError: If the section is missing or is not a mapping.
        """
        value = self.get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(
                f"Configuration section '{name}' must be a mapping, "
                f"got {type(value).__name__}."
            )
        return dict(value)

    def as_dict(self) -> dict[str, Any]:
        """Return a deep copy of the whole configuration tree."""
        return deepcopy(self._data)

    def __contains__(self, key: str) -> bool:
        try:
            self.get(key)
        except ConfigError:
            return False
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sections={sorted(self._data)})"

    def _missing_key_message(self, key: str, walked: list[str], node: Any) -> str:
        """Build an error message that points at the exact failing segment."""
        location = _KEY_SEPARATOR.join(walked) if walked else "<root>"
        if isinstance(node, Mapping):
            available = ", ".join(sorted(str(name) for name in node)) or "<empty>"
            return (
                f"Missing configuration key '{key}'. "
                f"Section '{location}' contains: {available}."
            )
        return (
            f"Missing configuration key '{key}'. "
            f"Section '{location}' is a {type(node).__name__}, not a mapping."
        )


def load_config(path: str | Path) -> Config:
    """Load, parse and validate the YAML configuration at ``path``.

    Relative paths are resolved against the repository root so the command line
    behaves identically regardless of the working directory.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, does not contain
            a mapping, or is missing a required key.
    """
    config_path = resolve(path)
    if not config_path.is_file():
        raise ConfigError(
            f"Configuration file not found: {config_path}. "
            "Pass an existing file with --config."
        )

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {config_path}: {error}") from error
    except OSError as error:
        raise ConfigError(f"Unable to read {config_path}: {error}") from error

    if raw is None:
        raise ConfigError(f"Configuration file is empty: {config_path}")
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"Configuration root of {config_path} must be a mapping, "
            f"got {type(raw).__name__}."
        )

    config = Config(raw)
    validate_keys(config, REQUIRED_KEYS, context=str(config_path))
    return config


def validate_keys(
    config: Config,
    required_keys: Sequence[tuple[str, TypeSpec]],
    *,
    context: str,
) -> None:
    """Check that every required key is present and holds a value of the right type.

    Args:
        config: Configuration to inspect.
        required_keys: Pairs of dotted key and accepted value type(s).
        context: Short description of what is being validated, used in the error
            message (a file path, or the name of the section being consumed).

    Raises:
        ConfigError: Listing every problem found, not just the first one.
    """
    problems: list[str] = []
    for key, expected_type in required_keys:
        try:
            value = config.get(key)
        except ConfigError as error:
            problems.append(str(error))
            continue
        if not _has_type(value, expected_type):
            problems.append(
                f"Key '{key}' must be of type {_type_name(expected_type)}, "
                f"got {type(value).__name__} ({value!r})."
            )

    if problems:
        details = "\n  - ".join(problems)
        raise ConfigError(f"Invalid configuration ({context}):\n  - {details}")


def _has_type(value: Any, expected_type: TypeSpec) -> bool:
    """Return whether ``value`` matches one of the accepted types.

    ``bool`` is a subclass of ``int`` in Python, so booleans are handled apart
    from the numeric types to keep flags and numeric settings from being
    interchangeable.
    """
    candidates = expected_type if isinstance(expected_type, tuple) else (expected_type,)
    if isinstance(value, bool):
        return bool in candidates
    return isinstance(value, candidates)


def _type_name(expected_type: TypeSpec) -> str:
    """Render accepted type(s) for an error message."""
    if isinstance(expected_type, tuple):
        return " or ".join(candidate.__name__ for candidate in expected_type)
    return expected_type.__name__
