"""Formatting primitives for the console reports printed by the command line tools.

Keeping the banner, indentation and rule widths in one place is what makes the
milestone reports look like they belong to the same program.
"""

from collections.abc import Iterable
from typing import Final

#: Width of the horizontal rules framing a report.
RULE_WIDTH: Final[int] = 50

MAJOR_RULE: Final[str] = "=" * RULE_WIDTH
MINOR_RULE: Final[str] = "-" * RULE_WIDTH
INDENT: Final[str] = " " * 4


def banner(title: str) -> list[str]:
    """Return the framed report title, followed by a blank line."""
    return [MAJOR_RULE, title, MAJOR_RULE, ""]


def entry(label: str, *values: str) -> list[str]:
    """Return a ``label:`` line followed by one indented line per value."""
    return [f"{label}:", *(f"{INDENT}{value}" for value in values), ""]


def entries(items: Iterable[tuple[str, str]]) -> list[str]:
    """Return the concatenation of :func:`entry` over ``(label, value)`` pairs."""
    lines: list[str] = []
    for label, value in items:
        lines.extend(entry(label, value))
    return lines


def rule() -> list[str]:
    """Return a minor rule separating two parts of a report, plus a blank line."""
    return [MINOR_RULE, ""]


def closing(status_line: str) -> list[str]:
    """Return the framed final status line of a report."""
    return [MAJOR_RULE, status_line, MAJOR_RULE]


def render(lines: Iterable[str]) -> str:
    """Join report lines into the final printable string."""
    return "\n".join(lines)
