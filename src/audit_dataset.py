"""Command line entry point for the dataset audit.

Running ``python -m src.audit_dataset`` walks the dataset, writes
``results/dataset_audit.json`` and exits non-zero when the dataset cannot be
used for training. :mod:`src.train` performs the same audit before every run, so
this command is the standalone way to inspect the dataset on its own.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from src import reporting
from src.cli import bootstrap, build_parser
from src.config import ConfigError
from src.datasets.validation import (
    SEVERITY_ERROR,
    DatasetAudit,
    DatasetSpecification,
    DatasetValidationError,
    audit_dataset,
)
from src.logger import configure_console_encoding, get_logger
from src.utils import write_json

#: Log file this entry point writes to.
AUDIT_LOG_FILENAME: Final[str] = "dataset_audit.log"

#: Maximum number of issues quoted in the console report; the JSON holds them all.
MAX_REPORTED_ISSUES: Final[int] = 20

_TITLE: Final[str] = "MILESTONE 4 — DATASET AUDIT"

_LOGGER: Final = get_logger("audit_dataset")


def render_audit(audit: DatasetAudit, report_path: Path) -> str:
    """Render the console report shown at the end of an audit."""
    lines = reporting.banner(_TITLE)
    lines.extend(
        reporting.entries(
            [
                ("Dataset Root", str(audit.root)),
                ("Classes", str(audit.class_count)),
                ("Total Images", f"{audit.total_images:,}"),
            ]
        )
    )
    lines.extend(reporting.rule())

    for split in audit.splits:
        lines.extend(
            reporting.entry(
                f"Split '{split.name}'",
                f"{split.image_count:,} images across {split.class_count} classes",
                f"imbalance ratio {split.imbalance_ratio:.2f}",
            )
        )

    lines.extend(reporting.rule())
    lines.extend(_issue_lines(audit))
    lines.extend(reporting.entry("Report", str(report_path)))
    lines.extend(reporting.closing(f"DATASET STATUS : {audit.status}"))
    return reporting.render(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Audit the configured dataset, write the report and return an exit code."""
    configure_console_encoding()
    parser = build_parser(
        prog="python -m src.audit_dataset",
        description="Validate the dataset on disk and write results/dataset_audit.json.",
    )
    arguments = parser.parse_args(argv)

    try:
        boot = bootstrap(arguments.config, log_filename=AUDIT_LOG_FILENAME)
        specification = DatasetSpecification.from_config(boot.config)
        audit = audit_dataset(specification)
        report_path = write_json(
            Path(boot.paths.results) / specification.audit_filename, audit.as_dict()
        )
    except (DatasetValidationError, ConfigError, KeyError, OSError, ValueError) as error:
        _LOGGER.error("Dataset audit failed: %s", error)
        print(reporting.render([*reporting.banner(_TITLE), f"ERROR: {error}", ""]))
        print(reporting.render(reporting.closing("DATASET STATUS : FAIL")))
        return 1

    _LOGGER.info("Audit report written to %s.", report_path)
    print(render_audit(audit, report_path))
    return 0 if audit.passed else 1


def _issue_lines(audit: DatasetAudit) -> list[str]:
    """Render the issue section of the console report."""
    if not audit.issues:
        return reporting.entry("Issues", "None found")

    lines = reporting.entry(
        "Issues",
        f"{len(audit.errors)} error(s), {len(audit.warnings)} warning(s)",
    )
    for issue in audit.issues[:MAX_REPORTED_ISSUES]:
        marker = "ERROR" if issue.severity == SEVERITY_ERROR else "WARN "
        lines.extend(reporting.entry(f"{marker} {issue.category}", issue.location, issue.detail))

    remaining = len(audit.issues) - MAX_REPORTED_ISSUES
    if remaining > 0:
        lines.extend(reporting.entry("More", f"{remaining} further issue(s) in the JSON report"))
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
