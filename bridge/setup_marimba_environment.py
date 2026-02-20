#!/usr/bin/env python3
"""Prepare a blank marimba arrangement clip from BPM/meter inputs."""

from __future__ import annotations

import argparse
import math
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import ableton_udp_bridge as bridge
import compose_kick_pattern as kick
from arrangement.artifacts import _archive_live_set


HOST = bridge.DEFAULT_HOST
PORT = bridge.DEFAULT_PORT
ACK_PORT = bridge.DEFAULT_ACK_PORT

DEFAULT_MINUTES = 5.0
DEFAULT_TRACK_NAME = "Marimba"
DEFAULT_CLIP_NAME = "Marimba Blank"
DEFAULT_ACK_TIMEOUT_S = 1.75
DEFAULT_LAUNCH_WAIT_SECONDS = 2.5
DEFAULT_ARCHIVE_DIR = Path("output/live_sets")


@dataclass(frozen=True)
class ClipPlan:
    minutes: float
    bpm: float
    sig_num: int
    sig_den: int
    beats_per_bar: float
    target_beats: float
    bars: int
    clip_start_beats: float
    clip_length_beats: float
    clip_end_beats: float


@dataclass(frozen=True)
class SetupConfig:
    bpm: float
    sig_num: int
    sig_den: int
    minutes: float
    start_beats: float
    track_name: str
    clip_name: str
    mood: str | None
    key_name: str | None
    ack_timeout_s: float
    launch_ableton: bool
    launch_wait_s: float
    save_policy: str
    archive_dir: str | None
    dry_run: bool


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


NOTE_TO_ROOT = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def parse_meter(text: str) -> tuple[int, int]:
    raw = str(text).strip()
    parts = raw.split("/", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"invalid meter '{text}'; expected NUM/DEN")
    num = int(parts[0].strip())
    den = int(parts[1].strip())
    if num <= 0 or den <= 0:
        raise ValueError(f"invalid meter '{text}'; numerator and denominator must be > 0")
    return num, den


def parse_key_name(text: str) -> tuple[int, str]:
    raw = str(text).strip()
    if not raw:
        raise ValueError("key name cannot be empty")
    parts = raw.split()
    if len(parts) < 2:
        raise ValueError(f"invalid key '{text}'; expected '<root> <major|minor>'")
    root_raw = parts[0].strip().upper().replace("♯", "#").replace("♭", "B")
    quality_raw = parts[1].strip().lower()
    if root_raw not in NOTE_TO_ROOT:
        raise ValueError(f"unsupported key root '{parts[0]}' in '{text}'")
    if quality_raw.startswith("maj"):
        scale_name = "Major"
    elif quality_raw.startswith("min"):
        scale_name = "Minor"
    else:
        raise ValueError(f"unsupported key quality '{parts[1]}' in '{text}'")
    return NOTE_TO_ROOT[root_raw], scale_name


def _safe_bpm_label(bpm: float) -> str:
    return f"{float(bpm):g}".replace(".", "_")


def build_meter_bpm_filename(sig_num: int, sig_den: int, bpm: float) -> str:
    _ = int(sig_den)  # Kept for call-site compatibility; filename uses numerator only.
    return f"{int(sig_num)}_{_safe_bpm_label(float(bpm))}.als"


def build_clip_plan(
    *,
    bpm: float,
    sig_num: int,
    sig_den: int,
    minutes: float,
    start_beats: float,
) -> ClipPlan:
    beats_per_bar = kick._beats_per_bar_from_signature(sig_num, sig_den)
    target_beats = float(bpm) * float(minutes)
    bars = max(1, int(math.ceil(float(target_beats) / float(beats_per_bar))))
    clip_length_beats = float(bars) * float(beats_per_bar)
    clip_start_beats = float(start_beats)
    clip_end_beats = clip_start_beats + clip_length_beats
    return ClipPlan(
        minutes=float(minutes),
        bpm=float(bpm),
        sig_num=int(sig_num),
        sig_den=int(sig_den),
        beats_per_bar=float(beats_per_bar),
        target_beats=float(target_beats),
        bars=int(bars),
        clip_start_beats=float(clip_start_beats),
        clip_length_beats=float(clip_length_beats),
        clip_end_beats=float(clip_end_beats),
    )


def parse_args(argv: Iterable[str]) -> SetupConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpm", type=_positive_float, required=True, help="Tempo in BPM")
    parser.add_argument(
        "--meter",
        default=None,
        help="Time signature as NUM/DEN (for example: 5/4). Overrides --sig-num/--sig-den when provided.",
    )
    parser.add_argument("--sig-num", type=_positive_int, default=4, help="Time signature numerator")
    parser.add_argument("--sig-den", type=_positive_int, default=4, help="Time signature denominator")
    parser.add_argument(
        "--minutes",
        type=_positive_float,
        default=DEFAULT_MINUTES,
        help=f"Target minutes (default: {DEFAULT_MINUTES})",
    )
    parser.add_argument(
        "--start-beats",
        type=_non_negative_float,
        default=0.0,
        help="Arrangement clip start time in beats (default: 0)",
    )
    parser.add_argument(
        "--track-name",
        default=DEFAULT_TRACK_NAME,
        help=f"Target track name (default: {DEFAULT_TRACK_NAME})",
    )
    parser.add_argument(
        "--clip-name",
        default=DEFAULT_CLIP_NAME,
        help=f"Arrangement clip name (default: {DEFAULT_CLIP_NAME})",
    )
    parser.add_argument("--mood", default=None, help="Optional mood label for logging")
    parser.add_argument(
        "--key-name",
        default=None,
        help="Optional key label in '<root> <major|minor>' form (for example: 'G# minor')",
    )
    parser.add_argument(
        "--ack-timeout",
        type=_positive_float,
        default=DEFAULT_ACK_TIMEOUT_S,
        help=f"Ack wait timeout in seconds (default: {DEFAULT_ACK_TIMEOUT_S})",
    )
    launch_group = parser.add_mutually_exclusive_group()
    launch_group.add_argument(
        "--launch-ableton",
        dest="launch_ableton",
        action="store_true",
        default=True,
        help="Launch Ableton Live before setup (default: enabled)",
    )
    launch_group.add_argument(
        "--no-launch-ableton",
        dest="launch_ableton",
        action="store_false",
        help="Skip launching Ableton and assume it is already open",
    )
    parser.add_argument(
        "--launch-wait-seconds",
        type=_non_negative_float,
        default=DEFAULT_LAUNCH_WAIT_SECONDS,
        help=f"Seconds to wait after launching Ableton (default: {DEFAULT_LAUNCH_WAIT_SECONDS})",
    )
    parser.add_argument(
        "--save-policy",
        choices=("ephemeral", "current", "archive"),
        default="ephemeral",
        help="Live set save policy after setup (default: ephemeral)",
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help="Archive directory used when --save-policy=archive",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the setup plan without sending OSC messages",
    )

    ns = parser.parse_args(list(argv))
    sig_num = int(ns.sig_num)
    sig_den = int(ns.sig_den)
    if ns.meter not in (None, ""):
        try:
            sig_num, sig_den = parse_meter(str(ns.meter))
        except ValueError as exc:
            parser.error(str(exc))
    if ns.key_name not in (None, ""):
        try:
            parse_key_name(str(ns.key_name))
        except ValueError as exc:
            parser.error(str(exc))

    return SetupConfig(
        bpm=float(ns.bpm),
        sig_num=int(sig_num),
        sig_den=int(sig_den),
        minutes=float(ns.minutes),
        start_beats=float(ns.start_beats),
        track_name=str(ns.track_name),
        clip_name=str(ns.clip_name),
        mood=None if ns.mood in (None, "") else str(ns.mood),
        key_name=None if ns.key_name in (None, "") else str(ns.key_name),
        ack_timeout_s=float(ns.ack_timeout),
        launch_ableton=bool(ns.launch_ableton),
        launch_wait_s=float(ns.launch_wait_seconds),
        save_policy=str(ns.save_policy),
        archive_dir=None if ns.archive_dir in (None, "") else str(ns.archive_dir),
        dry_run=bool(ns.dry_run),
    )


def _send_and_collect_acks(
    sock: socket.socket,
    ack_sock: socket.socket,
    command: bridge.OscCommand,
    timeout_s: float,
) -> list[tuple[str, list[bridge.OscArg]]]:
    print(f"sent: {bridge.describe_command(command)}")
    payload = bridge.encode_osc_message(command.address, command.args)
    sock.sendto(payload, (HOST, PORT))
    acks = bridge.wait_for_acks(ack_sock, timeout_s)
    kick._print_acks(acks)
    return acks


def _count_pong_acks(acks: Iterable[tuple[str, list[bridge.OscArg]]]) -> int:
    count = 0
    for address, args in acks:
        if address != "/ack" or not args:
            continue
        if str(args[0]) == "pong":
            count += 1
    return count


def _launch_ableton_live(wait_s: float) -> None:
    script = Path.home() / ".codex/skills/launch-ableton-live/scripts/launch-ableton-live.sh"
    if not script.exists():
        raise FileNotFoundError(f"launch script not found: {script}")
    subprocess.run([str(script)], check=True)
    if wait_s > 0:
        print(f"info: waiting {wait_s:g}s for Ableton startup")
        time.sleep(wait_s)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_archive_dir(raw_path: str | None) -> Path:
    if raw_path in (None, ""):
        return _repo_root() / DEFAULT_ARCHIVE_DIR
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _apply_live_key(
    sock: socket.socket,
    ack_sock: socket.socket,
    key_name: str,
    timeout_s: float,
) -> tuple[int, str]:
    root_note, scale_name = parse_key_name(key_name)
    kick._api_set(
        sock,
        ack_sock,
        "live_set",
        "root_note",
        int(root_note),
        "setup-root-note",
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        "live_set",
        "scale_name",
        str(scale_name),
        "setup-scale-name",
        timeout_s,
    )
    return int(root_note), str(scale_name)


def _ensure_live_transport(
    sock: socket.socket,
    ack_sock: socket.socket,
    *,
    bpm: float,
    sig_num: int,
    sig_den: int,
    timeout_s: float,
    max_attempts: int = 4,
) -> tuple[float | None, int | None, int | None]:
    observed_tempo: float | None = None
    observed_sig_num: int | None = None
    observed_sig_den: int | None = None
    target_bpm = float(bpm)
    target_sig_num = int(sig_num)
    target_sig_den = int(sig_den)

    for attempt in range(1, max(1, int(max_attempts)) + 1):
        kick._api_set(
            sock,
            ack_sock,
            "live_set",
            "tempo",
            target_bpm,
            f"setup-live-tempo-{attempt}",
            timeout_s,
        )
        kick._api_set(
            sock,
            ack_sock,
            "live_set",
            "signature_numerator",
            target_sig_num,
            f"setup-live-sig-num-{attempt}",
            timeout_s,
        )
        kick._api_set(
            sock,
            ack_sock,
            "live_set",
            "signature_denominator",
            target_sig_den,
            f"setup-live-sig-den-{attempt}",
            timeout_s,
        )

        observed_tempo = kick._as_float(
            kick._api_get(
                sock,
                ack_sock,
                "live_set",
                "tempo",
                f"setup-live-tempo-observed-{attempt}",
                timeout_s,
            )
        )
        observed_sig_num = kick._as_int(
            kick._api_get(
                sock,
                ack_sock,
                "live_set",
                "signature_numerator",
                f"setup-live-sig-num-observed-{attempt}",
                timeout_s,
            )
        )
        observed_sig_den = kick._as_int(
            kick._api_get(
                sock,
                ack_sock,
                "live_set",
                "signature_denominator",
                f"setup-live-sig-den-observed-{attempt}",
                timeout_s,
            )
        )

        tempo_ok = observed_tempo is not None and abs(float(observed_tempo) - target_bpm) <= 1e-6
        sig_num_ok = observed_sig_num == target_sig_num
        sig_den_ok = observed_sig_den == target_sig_den
        if tempo_ok and sig_num_ok and sig_den_ok:
            return observed_tempo, observed_sig_num, observed_sig_den

        if attempt < max_attempts:
            time.sleep(min(0.6, 0.15 * float(attempt)))

    return observed_tempo, observed_sig_num, observed_sig_den


def _resolve_track_path(
    sock: socket.socket,
    ack_sock: socket.socket,
    track_name: str,
    timeout_s: float,
) -> str | None:
    tracks_before = kick._get_children(
        sock,
        ack_sock,
        "live_set",
        "tracks",
        "setup-tracks-before",
        timeout_s,
    )
    if not tracks_before:
        return None

    track_index = kick._find_track_index_by_name(tracks_before, track_name)
    if track_index is None:
        kick._api_call(
            sock,
            ack_sock,
            "live_set",
            "create_midi_track",
            [-1],
            "setup-create-track",
            timeout_s,
        )
        tracks_after = kick._get_children(
            sock,
            ack_sock,
            "live_set",
            "tracks",
            "setup-tracks-after",
            timeout_s,
        )
        if len(tracks_after) <= len(tracks_before):
            return None
        track_index = len(tracks_after) - 1

    track_path = f"live_set tracks {track_index}"
    kick._api_set(
        sock,
        ack_sock,
        track_path,
        "name",
        track_name,
        "setup-track-name",
        timeout_s,
    )
    return track_path


def _delete_overlapping_clips(
    sock: socket.socket,
    ack_sock: socket.socket,
    track_path: str,
    clip_start: float,
    clip_end: float,
    timeout_s: float,
) -> int:
    clips = kick._get_children(
        sock,
        ack_sock,
        track_path,
        "arrangement_clips",
        "setup-arr-initial",
        timeout_s,
    )
    deleted = 0
    for idx, clip_info in enumerate(clips):
        clip_path_raw = clip_info.get("path")
        if not clip_path_raw:
            continue
        clip_path = kick._sanitize_live_path(str(clip_path_raw))
        start_existing = kick._as_float(
            kick._api_get(
                sock,
                ack_sock,
                clip_path,
                "start_time",
                f"setup-clip-start-{idx}",
                timeout_s,
            )
        )
        end_existing = kick._as_float(
            kick._api_get(
                sock,
                ack_sock,
                clip_path,
                "end_time",
                f"setup-clip-end-{idx}",
                timeout_s,
            )
        )
        if start_existing is None or end_existing is None:
            continue
        if not kick._overlaps(start_existing, end_existing, clip_start, clip_end):
            continue
        clip_desc = kick._api_describe(
            sock,
            ack_sock,
            clip_path,
            f"setup-clip-describe-{idx}",
            timeout_s,
        )
        clip_id = kick._as_int(clip_desc.get("id") if isinstance(clip_desc, dict) else None)
        if clip_id is None or clip_id <= 0:
            clip_id = kick._as_int(
                kick._api_get(
                    sock,
                    ack_sock,
                    clip_path,
                    "id",
                    f"setup-clip-id-{idx}",
                    timeout_s,
                )
            )
        if clip_id is None or clip_id <= 0:
            continue
        kick._api_call(
            sock,
            ack_sock,
            track_path,
            "delete_clip",
            [clip_id],
            f"setup-delete-clip-{idx}",
            timeout_s,
        )
        deleted += 1
    return deleted


def _create_blank_clip(
    sock: socket.socket,
    ack_sock: socket.socket,
    track_path: str,
    clip_start: float,
    clip_length: float,
    clip_name: str,
    sig_num: int,
    sig_den: int,
    timeout_s: float,
) -> str | None:
    arrangement_before = kick._get_children(
        sock,
        ack_sock,
        track_path,
        "arrangement_clips",
        "setup-arr-before",
        timeout_s,
    )
    create_result = kick._api_call(
        sock,
        ack_sock,
        track_path,
        "create_midi_clip",
        [clip_start, clip_length],
        "setup-create-clip",
        timeout_s,
    )
    clip_path: str | None = None
    clip_id = kick._extract_id_from_call_result(create_result)
    if clip_id is not None and clip_id > 0:
        id_path = f"id {clip_id}"
        resolved_id = kick._as_int(
            kick._api_get(
                sock,
                ack_sock,
                id_path,
                "id",
                "setup-created-clip-id",
                timeout_s,
            )
        )
        if resolved_id is not None and resolved_id > 0:
            clip_path = id_path

    arrangement_after = kick._get_children(
        sock,
        ack_sock,
        track_path,
        "arrangement_clips",
        "setup-arr-after",
        timeout_s,
    )
    if clip_path is None:
        clip_path = kick._new_child_path(arrangement_before, arrangement_after)
    if clip_path is None:
        # Some bridge/device states can return invalid id tokens; fall back to the
        # most recent arrangement clip that matches this run's start/length window.
        expected_end = float(clip_start) + float(clip_length)
        best_path: str | None = None
        best_index = -1
        for idx, clip_info in enumerate(arrangement_after):
            clip_path_raw = clip_info.get("path")
            if not clip_path_raw:
                continue
            candidate_path = kick._sanitize_live_path(str(clip_path_raw))
            start_value = kick._as_float(
                kick._api_get(
                    sock,
                    ack_sock,
                    candidate_path,
                    "start_time",
                    f"setup-resolve-clip-start-{idx}",
                    timeout_s,
                )
            )
            end_value = kick._as_float(
                kick._api_get(
                    sock,
                    ack_sock,
                    candidate_path,
                    "end_time",
                    f"setup-resolve-clip-end-{idx}",
                    timeout_s,
                )
            )
            item_index = int(clip_info.get("index", idx))
            if (
                start_value is not None
                and end_value is not None
                and abs(start_value - float(clip_start)) <= 1e-6
                and abs(end_value - expected_end) <= 1e-6
            ):
                if item_index >= best_index:
                    best_path = candidate_path
                    best_index = item_index
        if best_path is None and arrangement_after:
            tail_item = max(
                arrangement_after,
                key=lambda item: int(item.get("index", -1)),
            )
            tail_raw = tail_item.get("path")
            if tail_raw:
                best_path = kick._sanitize_live_path(str(tail_raw))
        clip_path = best_path
    if not clip_path:
        return None

    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "name",
        clip_name,
        "setup-clip-name",
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "signature_numerator",
        int(sig_num),
        "setup-clip-sig-num",
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "signature_denominator",
        int(sig_den),
        "setup-clip-sig-den",
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "loop_start",
        0.0,
        "setup-loop-start",
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "loop_end",
        clip_length,
        "setup-loop-end",
        timeout_s,
    )
    kick._api_call(
        sock,
        ack_sock,
        clip_path,
        "remove_notes_extended",
        {
            "from_pitch": 0,
            "pitch_span": 128,
            "from_time": 0.0,
            "time_span": float(clip_length),
        },
        "setup-clear-notes",
        max(timeout_s, 2.0),
    )
    return str(clip_path)


def run(cfg: SetupConfig) -> int:
    plan = build_clip_plan(
        bpm=cfg.bpm,
        sig_num=cfg.sig_num,
        sig_den=cfg.sig_den,
        minutes=cfg.minutes,
        start_beats=cfg.start_beats,
    )
    print("Marimba setup plan:")
    print(f"- tempo:       {plan.bpm:g}")
    print(f"- signature:   {plan.sig_num}/{plan.sig_den}")
    print(f"- minutes:     {plan.minutes:g}")
    print(f"- target beats:{plan.target_beats:g}")
    print(f"- bars:        {plan.bars}")
    print(f"- clip start:  {plan.clip_start_beats:g}")
    print(f"- clip length: {plan.clip_length_beats:g}")
    print(f"- clip end:    {plan.clip_end_beats:g}")
    print(f"- track:       {cfg.track_name}")
    print(f"- clip name:   {cfg.clip_name}")
    if cfg.mood:
        print(f"- mood:        {cfg.mood}")
    if cfg.key_name:
        print(f"- key:         {cfg.key_name}")
    print(f"- save policy: {cfg.save_policy}")

    if cfg.dry_run:
        print("\nDry run only. No OSC messages were sent.")
        return 0

    if cfg.launch_ableton:
        try:
            _launch_ableton_live(cfg.launch_wait_s)
        except Exception as exc:  # noqa: BLE001
            print(f"error: failed to launch Ableton Live: {exc}", file=sys.stderr)
            return 1

    bridge_cfg = bridge.BridgeConfig(
        host=HOST,
        port=PORT,
        ack_port=ACK_PORT,
        ack_timeout_s=cfg.ack_timeout_s,
        expect_ack=True,
        ping_first=False,
        status=False,
        tempo=None,
        sig_num=None,
        sig_den=None,
        create_midi_tracks=0,
        add_midi_tracks=0,
        midi_name="MIDI",
        create_audio_tracks=0,
        add_audio_tracks=0,
        audio_prefix="Audio",
        delete_audio_tracks=0,
        delete_midi_tracks=0,
        rename_track_index=None,
        rename_track_name=None,
        session_clip_track_index=None,
        session_clip_slot_index=None,
        session_clip_length=None,
        session_clip_notes_json=None,
        session_clip_name=None,
        append_session_clip_track_index=None,
        append_session_clip_slot_index=None,
        append_session_clip_notes_json=None,
        inspect_session_clip_track_index=None,
        inspect_session_clip_slot_index=None,
        ensure_midi_tracks=None,
        midi_ccs=(),
        cc64s=(),
        api_pings=(),
        api_gets=(),
        api_sets=(),
        api_calls=(),
        api_children=(),
        api_describes=(),
        ack_mode="per_command",
        ack_flush_interval=10,
        report_metrics=False,
        delay_ms=0,
        dry_run=False,
    )

    ack_sock = bridge.open_ack_socket(bridge_cfg)
    if ack_sock is None:
        print("error: failed to open ack socket", file=sys.stderr)
        return 1

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            print(f"\nTarget: udp://{HOST}:{PORT}")
            print(f"Ack:    udp://{HOST}:{ACK_PORT} (timeout {cfg.ack_timeout_s:.2f}s)")
            ping_acks = _send_and_collect_acks(sock, ack_sock, bridge.OscCommand("/ping"), cfg.ack_timeout_s)
            pong_count = _count_pong_acks(ping_acks)
            if pong_count == 0:
                print(
                    "error: bridge not responding on UDP 9000. Open your bridge-enabled set and keep exactly one "
                    "LiveUdpBridge device loaded (do not add a duplicate).",
                    file=sys.stderr,
                )
                return 5
            if pong_count > 1:
                print(
                    f"error: detected {pong_count} bridge responders on UDP 9000. Remove duplicate LiveUdpBridge "
                    "devices and retry.",
                    file=sys.stderr,
                )
                return 6

            for cmd in (
                bridge.OscCommand("/status"),
                bridge.OscCommand("/tempo", (float(plan.bpm),)),
                bridge.OscCommand("/sig_num", (int(plan.sig_num),)),
                bridge.OscCommand("/sig_den", (int(plan.sig_den),)),
            ):
                _send_and_collect_acks(sock, ack_sock, cmd, cfg.ack_timeout_s)

            tempo_verified, sig_num_verified, sig_den_verified = _ensure_live_transport(
                sock=sock,
                ack_sock=ack_sock,
                bpm=plan.bpm,
                sig_num=plan.sig_num,
                sig_den=plan.sig_den,
                timeout_s=cfg.ack_timeout_s,
            )
            if (
                tempo_verified is None
                or abs(float(tempo_verified) - float(plan.bpm)) > 1e-6
                or sig_num_verified != int(plan.sig_num)
                or sig_den_verified != int(plan.sig_den)
            ):
                print(
                    "warning: live_set transport did not fully verify; "
                    f"observed tempo/signature {tempo_verified} {sig_num_verified}/{sig_den_verified}",
                    file=sys.stderr,
                )

            if cfg.key_name:
                root_note, scale_name = _apply_live_key(
                    sock=sock,
                    ack_sock=ack_sock,
                    key_name=cfg.key_name,
                    timeout_s=cfg.ack_timeout_s,
                )
                print(f"info: applied live key root_note={root_note} scale_name={scale_name}")

            track_path = _resolve_track_path(sock, ack_sock, cfg.track_name, cfg.ack_timeout_s)
            if not track_path:
                print("error: could not resolve or create target track", file=sys.stderr)
                return 2
            print(f"info: target track path={track_path}")

            deleted = _delete_overlapping_clips(
                sock=sock,
                ack_sock=ack_sock,
                track_path=track_path,
                clip_start=plan.clip_start_beats,
                clip_end=plan.clip_end_beats,
                timeout_s=cfg.ack_timeout_s,
            )
            if deleted > 0:
                print(f"info: deleted {deleted} overlapping arrangement clip(s)")

            clip_path = _create_blank_clip(
                sock=sock,
                ack_sock=ack_sock,
                track_path=track_path,
                clip_start=plan.clip_start_beats,
                clip_length=plan.clip_length_beats,
                clip_name=cfg.clip_name,
                sig_num=plan.sig_num,
                sig_den=plan.sig_den,
                timeout_s=cfg.ack_timeout_s,
            )
            if not clip_path:
                print("error: could not create or resolve arrangement clip path", file=sys.stderr)
                return 3
            print(f"info: blank arrangement clip ready at {clip_path}")
            clip_sig_num_observed = kick._as_int(
                kick._api_get(
                    sock,
                    ack_sock,
                    clip_path,
                    "signature_numerator",
                    "setup-observe-clip-sig-num",
                    cfg.ack_timeout_s,
                )
            )
            clip_sig_den_observed = kick._as_int(
                kick._api_get(
                    sock,
                    ack_sock,
                    clip_path,
                    "signature_denominator",
                    "setup-observe-clip-sig-den",
                    cfg.ack_timeout_s,
                )
            )
            if clip_sig_num_observed and clip_sig_den_observed:
                print(
                    f"info: observed clip signature {clip_sig_num_observed}/{clip_sig_den_observed}"
                )

            tempo_observed = kick._as_float(
                kick._api_get(sock, ack_sock, "live_set", "tempo", "setup-observe-tempo", cfg.ack_timeout_s)
            )
            sig_num_observed = kick._as_int(
                kick._api_get(
                    sock,
                    ack_sock,
                    "live_set",
                    "signature_numerator",
                    "setup-observe-sig-num",
                    cfg.ack_timeout_s,
                )
            )
            sig_den_observed = kick._as_int(
                kick._api_get(
                    sock,
                    ack_sock,
                    "live_set",
                    "signature_denominator",
                    "setup-observe-sig-den",
                    cfg.ack_timeout_s,
                )
            )
            if tempo_observed is not None and sig_num_observed and sig_den_observed:
                print(
                    f"info: observed live_set tempo/signature {tempo_observed:g} "
                    f"{sig_num_observed}/{sig_den_observed}"
                )
            if cfg.key_name:
                root_observed = kick._as_int(
                    kick._api_get(
                        sock,
                        ack_sock,
                        "live_set",
                        "root_note",
                        "setup-observe-root-note",
                        cfg.ack_timeout_s,
                    )
                )
                scale_observed = kick._scalar(
                    kick._api_get(
                        sock,
                        ack_sock,
                        "live_set",
                        "scale_name",
                        "setup-observe-scale-name",
                        cfg.ack_timeout_s,
                    )
                )
                if root_observed is not None and scale_observed is not None:
                    print(
                        f"info: observed live_set key root_note={root_observed} "
                        f"scale_name={scale_observed}"
                    )

            if cfg.save_policy == "current":
                kick._api_call(
                    sock=sock,
                    ack_sock=ack_sock,
                    path="live_set",
                    method="save",
                    args=[],
                    request_id="setup-save-current",
                    timeout_s=max(cfg.ack_timeout_s, 2.0),
                )
                current_path = kick._api_get(
                    sock,
                    ack_sock,
                    "live_set",
                    "path",
                    "setup-save-path",
                    cfg.ack_timeout_s,
                )
                resolved_path = str(kick._scalar(current_path) or "").strip()
                if resolved_path:
                    print(f"info: saved current Live Set at {resolved_path}")
                else:
                    print("info: save current requested (Live returned no path)")
            elif cfg.save_policy == "archive":
                archive_dir = _resolve_archive_dir(cfg.archive_dir)
                date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                archive_name = build_meter_bpm_filename(
                    sig_num=plan.sig_num,
                    sig_den=plan.sig_den,
                    bpm=plan.bpm,
                )
                archive_path = archive_dir / date_slug / archive_name
                ok, save_message = _archive_live_set(
                    sock=sock,
                    ack_sock=ack_sock,
                    timeout_s=cfg.ack_timeout_s,
                    archive_path=archive_path,
                )
                if ok and archive_path.exists():
                    print(f"info: archived Live Set to {archive_path}")
                else:
                    print(
                        f"error: archive save failed for {archive_path} ({save_message})",
                        file=sys.stderr,
                    )
                    return 4
    finally:
        ack_sock.close()

    return 0


def main(argv: Iterable[str]) -> int:
    cfg = parse_args(argv)
    try:
        return run(cfg)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
