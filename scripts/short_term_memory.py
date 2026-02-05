#!/usr/bin/env python3
"""Repo-local short-term memory for conversation turns."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_STORE = Path(__file__).resolve().parents[1] / "memory" / "conversation.jsonl"
CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")


def sanitize_text(raw_text: str, keep_code: bool) -> str:
    text = raw_text if keep_code else CODE_FENCE_RE.sub("", raw_text)
    cleaned: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$ "):
            continue
        if stripped.startswith(("Exit code:", "Wall time:", "Output:")):
            continue
        cleaned.append(stripped)
    return " ".join(cleaned).strip()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_entry(store: Path, entry: Dict[str, str]) -> None:
    ensure_parent_dir(store)
    with store.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def iter_entries(store: Path) -> Iterable[Dict[str, str]]:
    if not store.exists():
        return
    with store.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_entries(store: Path, entries: List[Dict[str, str]]) -> None:
    ensure_parent_dir(store)
    with store.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def resolve_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def cmd_log(args: argparse.Namespace) -> int:
    raw_text = resolve_text(args)
    cleaned = sanitize_text(raw_text, keep_code=args.keep_code)
    if not cleaned:
        print("Nothing to log after filtering.", file=sys.stderr)
        return 1

    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "session_id": args.session,
        "role": args.role,
        "text": cleaned,
    }
    append_entry(args.store, entry)
    print(json.dumps(entry, ensure_ascii=True))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    entries = list(iter_entries(args.store))
    if args.session:
        entries = [e for e in entries if e.get("session_id") == args.session]
    if args.role:
        entries = [e for e in entries if e.get("role") == args.role]
    if args.limit:
        entries = entries[-args.limit :]

    if args.json:
        for entry in entries:
            print(json.dumps(entry, ensure_ascii=True))
        return 0

    for entry in entries:
        print(
            f"[{entry.get('timestamp', '?')}] "
            f"{entry.get('role', '?')}: {entry.get('text', '')}"
        )
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    if not args.yes:
        print("Refusing to clear without --yes.", file=sys.stderr)
        return 2
    if not args.store.exists():
        print("Nothing to clear.")
        return 0

    if args.session:
        entries = list(iter_entries(args.store))
        kept_entries = [e for e in entries if e.get("session_id") != args.session]
        removed = len(entries) - len(kept_entries)
        write_entries(args.store, kept_entries)
        print(f"Removed {removed} entries from session '{args.session}'.")
        return 0

    args.store.unlink()
    print("Cleared all memory entries.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Store short-term conversation memory for this repo."
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE,
        help=f"Path to JSONL store file (default: {DEFAULT_STORE})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    log_parser = subparsers.add_parser("log", help="Log one conversation turn.")
    log_parser.add_argument("--role", choices=["user", "assistant"], required=True)
    log_parser.add_argument("--text", help="Conversation text. If omitted, reads stdin.")
    log_parser.add_argument(
        "--session",
        default=dt.datetime.now().strftime("%Y-%m-%d"),
        help="Session identifier (default: today's date in local time).",
    )
    log_parser.add_argument(
        "--keep-code",
        action="store_true",
        help="Keep fenced code blocks instead of stripping them.",
    )
    log_parser.set_defaults(func=cmd_log)

    show_parser = subparsers.add_parser("show", help="Show stored conversation turns.")
    show_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum entries to show (default: 20).",
    )
    show_parser.add_argument("--session", help="Filter by session id.")
    show_parser.add_argument("--role", choices=["user", "assistant"], help="Filter by role.")
    show_parser.add_argument("--json", action="store_true", help="Print raw JSONL rows.")
    show_parser.set_defaults(func=cmd_show)

    clear_parser = subparsers.add_parser("clear", help="Clear stored memory.")
    clear_parser.add_argument("--session", help="Only clear entries for this session id.")
    clear_parser.add_argument("--yes", action="store_true", help="Confirm destructive action.")
    clear_parser.set_defaults(func=cmd_clear)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
