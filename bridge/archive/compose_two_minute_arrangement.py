#!/usr/bin/env python3
"""Archived legacy wrapper for compose_arrangement (not active runtime entrypoint)."""

from __future__ import annotations

import sys

import compose_arrangement as _impl
from compose_arrangement import main as _main


for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

__all__ = [name for name in dir(_impl) if not name.startswith("__")]


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
