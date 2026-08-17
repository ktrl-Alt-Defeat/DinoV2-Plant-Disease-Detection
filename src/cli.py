"""Command line entry point for the project infrastructure.

Running ``python -m src.cli`` boots the engineering foundation end to end:
configuration, logging, reproducibility, directory layout and device detection.
It creates no model and touches no dataset.
"""

import argparse
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
import torchvision

from src import reporting
from src.config import Config, load_config
from src.device import DeviceInfo, get_device, get_device_info
from src.logger import configure_console_encoding, configure_logging, get_logger
from src.paths import ProjectPaths, resolve
from src.seed import set_seed
from src.utils import Timer, format_bytes, format_duration

#: Configuration used when ``--config`` is not supplied.
DEFAULT_CONFIG_PATH: Final[str] = "configs/config.yaml"

_TITLE: Final[str] = "MILESTONE 1 — PROJECT FOUNDATION"
_NOT_AVAILABLE: Final[str] = "Not available"

#: Bootstrap steps, in the order they are reported by the summary.
_SUMMARY_STEPS: Final[tuple[str, ...]] = (
    "Directories",
    "Configuration",
    "Logger",
    "Seed",
    "Device",
)

#: Exceptions that represent a recoverable, user-actionable bootstrap failure.
#: ``ConfigError`` derives from ``ValueError`` and is therefore already covered.
_BOOTSTRAP_ERRORS: Final[tuple[type[Exception], ...]] = (KeyError, OSError, ValueError)


@dataclass(frozen=True)
class BootstrapReport:
    """Outcome of a successful infrastructure bootstrap."""

    config: Config
    paths: ProjectPaths
    device_info: DeviceInfo
    completed_steps: tuple[str, ...]
    duration_seconds: float


def build_parser(
    *,
    prog: str = "python -m src.cli",
    description: str = "Initialise and verify the DINOv2-S project infrastructure.",
) -> argparse.ArgumentParser:
    """Build an argument parser accepting the shared ``--config`` option.

    Every command line tool of the project reads the same configuration file, so
    they share this parser and only override the program name and description.
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_PATH),
        help=(
            "Path to the YAML configuration file. Relative paths are resolved "
            f"against the repository root (default: {DEFAULT_CONFIG_PATH})."
        ),
    )
    return parser


def bootstrap(config_path: str | Path) -> BootstrapReport:
    """Initialise every infrastructure component and report what was set up.

    The steps are executed in a fixed order and the first failure aborts the
    bootstrap, so a returned report always describes fully completed steps.

    Raises:
        ConfigError: If the configuration is missing, malformed or incomplete.
        ValueError: If a configured value is not supported.
        KeyError: If the ``paths`` section lacks a required entry.
        OSError: If a required directory or log file cannot be created.
    """
    completed: list[str] = []

    with Timer() as timer:
        config = load_config(config_path)
        completed.append("Configuration")

        project_name = config.get("project.name")
        log_dir = resolve(config.get("paths.logs")) if config.get("logging.save_file") else None
        logger = configure_logging(
            level=config.get("logging.level"),
            log_dir=log_dir,
            filename=f"{project_name}.log",
        )
        completed.append("Logger")
        logger.info("Configuration loaded from %s.", resolve(config_path))

        seed = config.get("project.seed")
        set_seed(
            seed,
            deterministic=config.get("reproducibility.deterministic"),
            benchmark=config.get("reproducibility.benchmark"),
        )
        completed.append("Seed")
        logger.info("Random seed set to %d.", seed)

        project_paths = ProjectPaths.from_mapping(config.section("paths")).create()
        completed.append("Directories")
        for name, directory in project_paths.as_dict().items():
            logger.info("Directory ready: %s -> %s", name, directory)

        device = get_device(config.get("device.preferred"))
        device_info = get_device_info(device)
        completed.append("Device")
        logger.info("Selected device: %s (%s).", device, device_info.name)

    logger.info("Infrastructure bootstrap completed in %s.", format_duration(timer.elapsed))
    return BootstrapReport(
        config=config,
        paths=project_paths,
        device_info=device_info,
        completed_steps=tuple(completed),
        duration_seconds=timer.elapsed,
    )


def render_summary(report: BootstrapReport) -> str:
    """Render the human readable project summary shown at the end of a run."""
    config = report.config
    info = report.device_info

    entries: list[tuple[str, str]] = [
        ("Project", config.get("project.name")),
        ("Version", config.get("project.version")),
        ("Python", platform.python_version()),
        ("PyTorch", torch.__version__),
        ("TorchVision", torchvision.__version__),
        ("CUDA", info.cuda_version or _NOT_AVAILABLE),
        ("cuDNN", str(info.cudnn_version) if info.cudnn_version else _NOT_AVAILABLE),
        ("GPU", _format_gpu(info)),
    ]
    entries.extend(
        (step, "PASS" if step in report.completed_steps else "FAIL") for step in _SUMMARY_STEPS
    )

    passed = all(step in report.completed_steps for step in _SUMMARY_STEPS)
    entries.append(("Infrastructure", "READY" if passed else "INCOMPLETE"))

    return _render_block(entries, status="PASS" if passed else "FAIL")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return a process exit code."""
    configure_console_encoding()
    arguments = build_parser().parse_args(argv)

    try:
        report = bootstrap(arguments.config)
    except _BOOTSTRAP_ERRORS as error:
        get_logger("cli").error("Infrastructure bootstrap failed: %s", error)
        print(f"ERROR: {error}", file=sys.stderr)
        print(_render_block([("Infrastructure", "INCOMPLETE")], status="FAIL"))
        return 1

    print(render_summary(report))
    return 0


def _format_gpu(info: DeviceInfo) -> str:
    """Describe the GPU, or state that the run falls back to the CPU."""
    if info.device.type != "cuda":
        return f"{_NOT_AVAILABLE} (CPU: {info.name})"
    if info.total_memory_bytes is None:
        return info.name
    return f"{info.name} ({format_bytes(info.total_memory_bytes)})"


def _render_block(items: Sequence[tuple[str, str]], *, status: str) -> str:
    """Format the titled report block with indented ``label``/``value`` pairs."""
    lines = reporting.banner(_TITLE)
    lines.extend(reporting.entries(items))
    lines.extend(reporting.closing(f"STATUS: {status}"))
    return reporting.render(lines)


if __name__ == "__main__":
    raise SystemExit(main())
