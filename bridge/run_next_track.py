#!/usr/bin/env python3
"""Run a complete next-track workflow with direct arrangement composition."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import compose_arrangement as arrangement
import setup_marimba_environment as setup

DEFAULT_MARIMBA_PIANO_REGISTRY_PATH = (
    Path(__file__).resolve().parent / "config" / "instrument_registry.marimba_piano.v1.json"
)


def _resolve_duration_seed(requested_seed: int | None) -> int:
    if requested_seed is not None:
        return int(requested_seed)
    # Fresh composition seed by default so repeated runs are not locked.
    return max(1, int(time.time_ns() % 2_147_483_647))


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpm", type=float, required=True, help="Tempo in BPM")
    parser.add_argument(
        "--meter",
        required=True,
        help="Time signature as NUM/DEN (for example: 5/4)",
    )
    parser.add_argument("--key-name", default=None, help="Key label, e.g. 'G# minor'")
    parser.add_argument("--mood", default="Beautiful", help="Mood tag to log with the suggestion")
    parser.add_argument(
        "--minutes",
        type=float,
        default=5.0,
        help="Target minutes for the setup clip and composition",
    )
    parser.add_argument(
        "--track-name",
        default="Marimba",
        help="Legacy no-op argument (kept for CLI compatibility)",
    )
    parser.add_argument(
        "--clip-name",
        default="Marimba Blank",
        help="Legacy no-op argument (kept for CLI compatibility)",
    )
    parser.add_argument(
        "--save-policy",
        choices=("ephemeral", "current", "archive"),
        default="archive",
        help="Save policy for setup + composition",
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help="Archive directory when save policy is archive",
    )
    parser.add_argument(
        "--duration-seed",
        type=int,
        default=None,
        help="Arrangement duration seed (defaults to a fresh seed each run)",
    )
    parser.add_argument(
        "--marimba-strategy",
        choices=("auto", "ostinato_pulse", "broken_resonance", "chord_bloom", "lyrical_roll"),
        default="auto",
        help="Marimba identity strategy",
    )
    parser.add_argument(
        "--marimba-pair-mode",
        choices=("auto", "off", "attack_answer"),
        default="auto",
        help="Marimba pair mode",
    )
    parser.add_argument(
        "--instrument-registry-path",
        default=str(DEFAULT_MARIMBA_PIANO_REGISTRY_PATH),
        help="Instrument registry path for composition (default: marimba+piano registry)",
    )
    parser.add_argument(
        "--no-write-cache",
        action="store_true",
        help="Bypass arrangement write cache to force fresh clip writes",
    )
    parser.add_argument(
        "--human-feedback-mode",
        choices=("written", "verbal"),
        default="written",
        help="Human feedback mode stored in eval artifacts when feedback text is provided",
    )
    parser.add_argument(
        "--human-feedback-text",
        default=None,
        help="Optional written/verbal human feedback text to include in eval artifacts",
    )
    parser.add_argument(
        "--launch-ableton",
        dest="launch_ableton",
        action="store_true",
        default=True,
        help="Launch Ableton Live before setup (default: enabled)",
    )
    parser.add_argument(
        "--no-launch-ableton",
        dest="launch_ableton",
        action="store_false",
        help="Skip launching Ableton; assumes it is already open",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed plan and exit without sending OSC messages",
    )

    return parser.parse_args(list(argv))


def _compose_args(ns: argparse.Namespace) -> list[str]:
    sig_num, sig_den = setup.parse_meter(ns.meter)
    duration_seed = _resolve_duration_seed(ns.duration_seed)
    compose_argv = [
        "--bpm",
        str(ns.bpm),
        "--sig-num",
        str(sig_num),
        "--sig-den",
        str(sig_den),
        "--minutes",
        str(ns.minutes),
        "--mood",
        ns.mood,
        "--save-policy",
        ns.save_policy,
        "--duration-seed",
        str(duration_seed),
        "--marimba-strategy",
        ns.marimba_strategy,
        "--marimba-pair-mode",
        ns.marimba_pair_mode,
        "--instrument-registry-path",
        str(ns.instrument_registry_path),
        "--clip-write-mode",
        "single_clip",
        "--no-write-cache",
    ]
    if ns.key_name:
        compose_argv.extend(["--key-name", ns.key_name])
    if ns.no_write_cache:
        # Idempotent when already present; kept for explicit CLI compatibility.
        compose_argv.append("--no-write-cache")
    if ns.archive_dir:
        compose_argv.extend(["--archive-dir", ns.archive_dir])
    if ns.human_feedback_text:
        compose_argv.extend(["--human-feedback-mode", ns.human_feedback_mode])
        compose_argv.extend(["--human-feedback-text", ns.human_feedback_text])
    if ns.dry_run:
        compose_argv.append("--dry-run")
    return compose_argv


def run_next_track(ns: argparse.Namespace) -> int:
    if ns.launch_ableton:
        setup._launch_ableton_live(setup.DEFAULT_LAUNCH_WAIT_SECONDS)

    compose_cfg = arrangement.parse_args(_compose_args(ns))
    return arrangement.run(compose_cfg)


def main(argv: Iterable[str]) -> int:
    ns = parse_args(argv)
    try:
        return run_next_track(ns)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
