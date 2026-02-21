#!/usr/bin/env python3
"""Tiny loader and brief generator for repo compositional memory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


INDEX_PATH = Path(__file__).resolve().parent / "index.toml"


class MemoryError(RuntimeError):
    """Raised when the compositional memory index cannot be loaded."""


def load_index(path: Path = INDEX_PATH) -> dict[str, Any]:
    """Load the compositional memory index from TOML."""
    if not path.exists():
        raise MemoryError(f"memory index not found at {path}")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise MemoryError("memory index TOML did not parse to a dictionary")
    return data


def _fundamentals_table(index: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the fundamentals table, falling back to legacy topics."""
    fundamentals = index.get("fundamentals")
    if isinstance(fundamentals, Mapping) and fundamentals:
        return fundamentals
    topics = index.get("topics")
    if isinstance(topics, Mapping) and topics:
        return topics
    return {}


def list_topics(index: Mapping[str, Any]) -> list[str]:
    """List known fundamentals (legacy name kept for compatibility)."""
    fundamentals = _fundamentals_table(index)
    return sorted(str(name) for name in fundamentals.keys())


def current_focus(index: Mapping[str, Any]) -> str | None:
    focus = index.get("current_focus", {})
    if not isinstance(focus, Mapping):
        return None
    fundamental = focus.get("fundamental") or focus.get("topic")
    return str(fundamental) if fundamental else None


def _as_lines(values: Sequence[Any]) -> list[str]:
    lines: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            lines.append(f"- {text}")
    return lines


def topic_brief(index: Mapping[str, Any], topic: str | None = None) -> str:
    fundamentals = _fundamentals_table(index)
    if not fundamentals:
        return "No fundamentals defined in memory index."

    focus = current_focus(index)
    topic_name = topic or focus
    if not topic_name:
        available = ", ".join(list_topics(index)) or "(none)"
        return f"No current focus set. Fundamentals: {available}"

    topic_data = fundamentals.get(topic_name)
    if not isinstance(topic_data, Mapping):
        available = ", ".join(list_topics(index)) or "(none)"
        return f"Unknown fundamental '{topic_name}'. Fundamentals: {available}"

    lines: list[str] = []
    focus_suffix = " (current focus)" if topic_name == focus else ""
    lines.append(f"Fundamental: {topic_name}{focus_suffix}")

    summary = topic_data.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")

    principles = topic_data.get("principles", [])
    if isinstance(principles, Sequence) and not isinstance(principles, (str, bytes)):
        principle_lines = _as_lines(principles)
        if principle_lines:
            lines.append("Principles:")
            lines.extend(principle_lines)

    heuristics = topic_data.get("heuristics", [])
    if isinstance(heuristics, Sequence) and not isinstance(heuristics, (str, bytes)):
        heuristic_lines = _as_lines(heuristics)
        if heuristic_lines:
            lines.append("Heuristics:")
            lines.extend(heuristic_lines)

    checklist = topic_data.get("checklist", [])
    if isinstance(checklist, Sequence) and not isinstance(checklist, (str, bytes)):
        checklist_lines = _as_lines(checklist)
        if checklist_lines:
            lines.append("Checklist:")
            lines.extend(checklist_lines)

    references = topic_data.get("references", [])
    if isinstance(references, Sequence) and not isinstance(references, (str, bytes)):
        refs = [str(ref) for ref in references if str(ref).strip()]
        if refs:
            lines.append("References:")
            lines.extend(f"- {ref}" for ref in refs)

    return "\n".join(lines)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX_PATH, help="Path to memory index TOML")
    parser.add_argument(
        "--fundamental",
        type=str,
        default=None,
        help="Fundamental to brief (defaults to current focus)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Deprecated alias for --fundamental",
    )
    parser.add_argument("--list", action="store_true", help="List available fundamentals")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        index = load_index(args.index)
    except MemoryError as exc:
        print(f"error: {exc}")
        return 2

    if args.list:
        topics = list_topics(index)
        if not topics:
            print("No fundamentals defined.")
            return 0
        print("Fundamentals:")
        for name in topics:
            print(f"- {name}")
        return 0

    requested = args.fundamental or args.topic
    print(topic_brief(index, topic=requested))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
