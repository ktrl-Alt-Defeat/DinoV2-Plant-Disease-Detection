"""Project filesystem layout.

Every path used by the project is derived from the repository root through
:mod:`pathlib`, which keeps the code portable across Windows, macOS and Linux.
Relative paths coming from the configuration are always interpreted relative to
the repository root so that behaviour does not depend on the current working
directory.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

#: Repository root, i.e. the directory that contains ``src`` and ``pyproject.toml``.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: Keys the ``paths`` configuration section must provide.
REQUIRED_PATH_KEYS: Final[tuple[str, ...]] = ("logs", "checkpoints", "results")


def project_root() -> Path:
    """Return the absolute path of the repository root."""
    return PROJECT_ROOT


def resolve(path: str | Path) -> Path:
    """Return ``path`` as an absolute path anchored at the repository root.

    Absolute inputs are returned unchanged, which lets a deployment override any
    configured location with a machine specific directory.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def ensure_directory(path: str | Path) -> Path:
    """Create ``path`` (and any missing parent) and return its absolute form."""
    directory = resolve(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@dataclass(frozen=True)
class ProjectPaths:
    """Absolute locations of the directories written to by the project."""

    root: Path
    logs: Path
    checkpoints: Path
    results: Path

    @classmethod
    def from_mapping(cls, paths_config: Mapping[str, str]) -> "ProjectPaths":
        """Build the layout from the ``paths`` section of the configuration.

        Raises:
            KeyError: If one of :data:`REQUIRED_PATH_KEYS` is absent.
        """
        missing = [key for key in REQUIRED_PATH_KEYS if key not in paths_config]
        if missing:
            raise KeyError(
                "Missing entries in the 'paths' configuration section: "
                f"{', '.join(missing)}. Expected: {', '.join(REQUIRED_PATH_KEYS)}."
            )
        return cls(
            root=PROJECT_ROOT,
            logs=resolve(paths_config["logs"]),
            checkpoints=resolve(paths_config["checkpoints"]),
            results=resolve(paths_config["results"]),
        )

    def create(self) -> "ProjectPaths":
        """Create every managed directory and return ``self`` for chaining."""
        for directory in self.as_dict().values():
            ensure_directory(directory)
        return self

    def as_dict(self) -> dict[str, Path]:
        """Return the managed directories keyed by their configuration name."""
        return {
            "logs": self.logs,
            "checkpoints": self.checkpoints,
            "results": self.results,
        }
