#!/usr/bin/env python3
"""Compose a foundational kick pattern into the Arrangement view via the UDP bridge."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import ableton_udp_bridge as bridge


HOST = bridge.DEFAULT_HOST
PORT = bridge.DEFAULT_PORT
ACK_PORT = bridge.DEFAULT_ACK_PORT

DEFAULT_GROOVE_NAME = "Hip Hop Loosely Flow 16ths 80 bpm"

OscAck = Tuple[str, List[bridge.OscArg]]


def _format_ack(ack: OscAck) -> str:
    address, args = ack
    if not args:
        return address
    return address + " " + " ".join(bridge.format_arg(a) for a in args)


def _print_acks(acks: Sequence[OscAck]) -> None:
    if not acks:
        print("ack:  (none received; bridge may not be loaded yet)")
        return
    for address, args in acks:
        for line in bridge.summarize_ack(address, args):
            print(line)


def _send_and_collect_acks(
    sock: socket.socket,
    ack_sock: socket.socket,
    command: bridge.OscCommand,
    timeout_s: float,
) -> List[OscAck]:
    payload = bridge.encode_osc_message(command.address, command.args)
    sock.sendto(payload, (HOST, PORT))
    return bridge.wait_for_acks(ack_sock, timeout_s)


def _extract_api_children(acks: Sequence[OscAck], request_id: str) -> List[dict]:
    for address, args in acks:
        if address != "/ack" or len(args) < 4:
            continue
        if args[0] != "api_children":
            continue
        if str(args[-1]) != request_id:
            continue
        payload = args[3]
        if not isinstance(payload, str):
            return []
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _extract_api_get(acks: Sequence[OscAck], request_id: str) -> object | None:
    for address, args in acks:
        if address != "/ack" or len(args) < 4:
            continue
        if args[0] != "api_get":
            continue
        if str(args[-1]) != request_id:
            continue
        value = args[3]
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return None


def _extract_api_describe(acks: Sequence[OscAck], request_id: str) -> dict | None:
    for address, args in acks:
        if address != "/ack" or len(args) < 3:
            continue
        if args[0] != "api_describe":
            continue
        if str(args[-1]) != request_id:
            continue
        payload = args[2]
        if not isinstance(payload, str):
            return payload if isinstance(payload, dict) else None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _scalar(value: object | None) -> object | None:
    """Extract a scalar value from LiveAPI-style responses."""
    if isinstance(value, list):
        if not value:
            return None
        return value[-1]
    return value


def _as_float(value: object | None) -> float | None:
    scalar = _scalar(value)
    if scalar is None:
        return None
    try:
        return float(scalar)
    except (TypeError, ValueError):
        return None


def _as_int(value: object | None) -> int | None:
    scalar = _scalar(value)
    if scalar is None:
        return None
    try:
        return int(float(scalar))
    except (TypeError, ValueError):
        return None


def _extract_api_call_result(acks: Sequence[OscAck], request_id: str) -> object | None:
    for address, args in acks:
        if address != "/ack" or len(args) < 4:
            continue
        if args[0] != "api_call":
            continue
        if str(args[-1]) != request_id:
            continue
        value = args[3]
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return None


def _children_request(path: str, child: str, request_id: str) -> bridge.OscCommand:
    return bridge.OscCommand("/api/children", (path, child, request_id))


def _get_children(
    sock: socket.socket,
    ack_sock: socket.socket,
    path: str,
    child: str,
    request_id: str,
    timeout_s: float,
) -> List[dict]:
    cmd = _children_request(path, child, request_id)
    print(f"sent: {bridge.describe_command(cmd)}")
    acks = _send_and_collect_acks(sock, ack_sock, cmd, timeout_s)
    _print_acks(acks)
    return _extract_api_children(acks, request_id)


def _api_set(
    sock: socket.socket,
    ack_sock: socket.socket,
    path: str,
    prop: str,
    value: object,
    request_id: str,
    timeout_s: float,
) -> None:
    cmd = bridge.OscCommand("/api/set", (path, prop, json.dumps(value), request_id))
    print(f"sent: {bridge.describe_command(cmd)}")
    _print_acks(_send_and_collect_acks(sock, ack_sock, cmd, timeout_s))


def _api_call(
    sock: socket.socket,
    ack_sock: socket.socket,
    path: str,
    method: str,
    args: object,
    request_id: str,
    timeout_s: float,
) -> object | None:
    cmd = bridge.OscCommand("/api/call", (path, method, json.dumps(args), request_id))
    print(f"sent: {bridge.describe_command(cmd)}")
    acks = _send_and_collect_acks(sock, ack_sock, cmd, timeout_s)
    _print_acks(acks)
    return _extract_api_call_result(acks, request_id)


def _api_get(
    sock: socket.socket,
    ack_sock: socket.socket,
    path: str,
    prop: str,
    request_id: str,
    timeout_s: float,
) -> object | None:
    cmd = bridge.OscCommand("/api/get", (path, prop, request_id))
    print(f"sent: {bridge.describe_command(cmd)}")
    acks = _send_and_collect_acks(sock, ack_sock, cmd, timeout_s)
    _print_acks(acks)
    return _extract_api_get(acks, request_id)


def _api_describe(
    sock: socket.socket,
    ack_sock: socket.socket,
    path: str,
    request_id: str,
    timeout_s: float,
) -> dict | None:
    cmd = bridge.OscCommand("/api/describe", (path, request_id))
    print(f"sent: {bridge.describe_command(cmd)}")
    acks = _send_and_collect_acks(sock, ack_sock, cmd, timeout_s)
    _print_acks(acks)
    return _extract_api_describe(acks, request_id)


def _beats_per_bar_from_signature(signature_numerator: int, signature_denominator: int) -> float:
    """Convert Live's time signature into quarter-note beat units."""
    if signature_numerator <= 0:
        raise ValueError("signature_numerator must be > 0")
    if signature_denominator <= 0:
        raise ValueError("signature_denominator must be > 0")
    return float(signature_numerator) * 4.0 / float(signature_denominator)


def _beat_step_from_denominator(signature_denominator: int) -> float:
    """Return the per-beat step size in quarter-note units."""
    if signature_denominator <= 0:
        raise ValueError("signature_denominator must be > 0")
    return 4.0 / float(signature_denominator)


def _beats_in_bar(beats_per_bar: float, beat_step: float) -> int:
    beats = int(round(float(beats_per_bar) / float(beat_step)))
    return max(1, beats)


def _five_four_grouping(bar_index: int) -> str:
    """Choose a 5/4 grouping per 4-bar phrase: 3-2 or 2-3."""
    phrase_index = bar_index // 4
    seed = _velocity_seed(phrase_index, 5, 0)
    return "3-2" if seed % 2 == 0 else "2-3"


def _five_four_hits(bar_index: int) -> List[int]:
    """Return beat indices to hit in 5/4 using 3-2 or 2-3 groupings."""
    grouping = _five_four_grouping(bar_index)
    if grouping == "3-2":
        patterns: Sequence[Sequence[int]] = (
            (0, 2, 3),  # 1, (skip 2), 3, start of "2"
            (0, 2),     # sparse 3-group anchor
            (0, 3),     # emphasize the 3-2 hinge
        )
    else:
        patterns = (
            (0, 1, 3),  # 1, 2, start of "3"
            (0, 1),     # sparse 2-group anchor
            (0, 3),     # emphasize the 2-3 hinge
        )

    seed = _velocity_seed(bar_index, 5, len(patterns))
    pick = seed % len(patterns)
    hits = list(patterns[pick])
    if 0 not in hits:
        hits.insert(0, 0)
    return sorted(set(hits))


def _patterns_for_meter(beats_in_bar: int) -> Sequence[Sequence[int]]:
    """Return sparse, meter-aware kick patterns that keep beat 1 obvious."""
    if beats_in_bar == 3:
        return (
            (0, 2),      # 1 + 3
            (0, 1),      # 1 + 2
            (0, 2),      # reinforce 3
            (0, 1, 2),   # occasional dense bar
        )
    if beats_in_bar == 4:
        return (
            (0, 2),      # classic anchor
            (0, 2, 3),   # late push
            (0, 1, 3),   # off-center support
            (0, 2),      # default anchor
        )
    if beats_in_bar == 6:
        return (
            (0, 3),         # 3+3 hinge
            (0, 2, 4),      # 2+2+2 feel
            (0, 3, 5),      # hinge plus lift
            (0, 2, 3, 5),   # slightly denser, still sparse
            (0, 4),         # long arc
        )
    if beats_in_bar == 7:
        return (
            (0, 3, 5),      # 3+2+2 hinge
            (0, 2, 4, 6),   # even spread
            (0, 3, 6),      # wide arc
            (0, 2, 5),      # asymmetric lift
        )

    # Fallback: beat 1 plus a hinge, with an occasional beat-1-only bar.
    hinge = max(1, beats_in_bar // 2)
    return (
        (0,),
        (0, hinge),
    )


def _pattern_hits_for_bar(bar_index: int, beats_in_bar: int) -> List[int]:
    """Choose a deterministic sparse pattern for the given bar/meter."""
    patterns = _patterns_for_meter(beats_in_bar)
    seed = _velocity_seed(bar_index, beats_in_bar, len(patterns))
    pick = seed % len(patterns)
    hits = [int(h) for h in patterns[pick] if 0 <= int(h) < beats_in_bar]
    if 0 not in hits:
        hits.insert(0, 0)
    return sorted(set(hits))


def _kick_times_for_bar(
    bar_index: int,
    bar_start: float,
    beats_per_bar: float,
    beat_step: float,
) -> List[float]:
    """Return kick times for a single bar, with meter-aware sparse patterns."""
    if beat_step <= 0:
        raise ValueError("beat_step must be > 0")

    beats_in_bar = _beats_in_bar(beats_per_bar, beat_step)
    beat_indices: Sequence[int]
    if beats_in_bar == 5:
        beat_indices = _five_four_hits(bar_index)
    else:
        beat_indices = _pattern_hits_for_bar(bar_index, beats_in_bar)

    bar_end = bar_start + beats_per_bar
    times: List[float] = []
    for beat_index in beat_indices:
        start_time = bar_start + float(beat_index) * float(beat_step)
        if start_time >= bar_end:
            continue
        times.append(round(start_time, 6))
    return times


def _chunk_notes(notes: Sequence[dict], chunk_size: int) -> List[List[dict]]:
    """Split note payloads into smaller UDP-safe chunks."""
    if chunk_size <= 0:
        return [list(notes)]
    return [list(notes[i : i + chunk_size]) for i in range(0, len(notes), chunk_size)]


def _fill_patterns(beats_per_bar: float, beat_step: float) -> List[List[float]]:
    """Return a small rotating set of end-of-phrase fill offsets."""
    return [
        # Late push.
        [beats_per_bar - (beat_step / 2.0)],
        # Earlier push.
        [beats_per_bar - (beat_step + (beat_step / 2.0))],
        # Very late push.
        [beats_per_bar - (beat_step / 4.0)],
    ]


def _clamp_velocity(value: float) -> int:
    return max(1, min(127, int(round(value))))


def _clamp_velocity_range(min_v: float, max_v: float) -> Tuple[int, int]:
    """Clamp a velocity range into Live's valid 1-127 window."""
    lo = _clamp_velocity(min_v)
    hi = _clamp_velocity(max_v)
    return (lo, hi) if lo <= hi else (hi, lo)


def _velocity_seed(bar_index: int, beat_index: int, variant_index: int = 0) -> int:
    """Create a deterministic pseudo-random seed per note."""
    seed = (
        (bar_index + 1) * 1103515245
        + (beat_index + 1) * 12345
        + (variant_index + 1) * 2654435761
    )
    return seed & 0x7FFFFFFF


def _velocity_in_range(min_v: int, max_v: int, seed: int) -> int:
    """Pick a deterministic velocity inside the provided inclusive range."""
    span = max_v - min_v + 1
    if span <= 1:
        return min_v
    return int(min_v + (seed % span))


def _classify_time(start_time: float, bar_start: float, beat_step: float) -> Tuple[bool, int]:
    """Return (on_beat, beat_index) for a note time within a bar."""
    beat_offset = max(0.0, float(start_time) - float(bar_start))
    raw_index = beat_offset / float(beat_step)
    beat_index = int(round(raw_index))
    on_beat = abs(raw_index - beat_index) <= 1e-6
    return on_beat, max(0, beat_index)


def build_kick_notes(
    bars: int,
    beats_per_bar: float = 4.0,
    beat_step: float = 1.0,
    pitch: int = 36,
    velocity: int = 110,
    duration: float = 0.25,
) -> Tuple[List[dict], float]:
    """Generate a kick pattern with beat-1 accents, variation, and phrase fills."""
    if bars <= 0:
        raise ValueError("bars must be > 0")
    if beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be > 0")
    if beat_step <= 0:
        raise ValueError("beat_step must be > 0")

    clip_length = float(bars) * float(beats_per_bar)
    notes: List[dict] = []
    fill_patterns = _fill_patterns(float(beats_per_bar), float(beat_step))
    fill_count = 0

    base_velocity = _clamp_velocity(velocity)
    velocity_offset = base_velocity - 110

    # Target ranges based on the user's guidance, gently offset by --velocity.
    accent_min, accent_max = _clamp_velocity_range(120 + velocity_offset, 125 + velocity_offset)
    beat_min, beat_max = _clamp_velocity_range(105 + velocity_offset, 115 + velocity_offset)
    grace_min, grace_max = _clamp_velocity_range(95 + velocity_offset, 100 + velocity_offset)

    for bar_index in range(bars):
        bar_start = float(bar_index) * float(beats_per_bar)
        fill_markers: dict[float, int] = {}

        # Default: kick on every beat. In 5/4, allow grouped spacing.
        times = _kick_times_for_bar(
            bar_index=bar_index,
            bar_start=bar_start,
            beats_per_bar=float(beats_per_bar),
            beat_step=float(beat_step),
        )

        # At phrase boundaries, add a one-off fill that rotates each time.
        if (bar_index + 1) % 4 == 0:
            pattern = fill_patterns[fill_count % len(fill_patterns)]
            fill_index = fill_count
            fill_count += 1
            for offset in pattern:
                if offset <= 0.0 or offset >= float(beats_per_bar):
                    continue
                fill_time = bar_start + float(offset)
                times.append(fill_time)
                fill_markers[round(fill_time, 6)] = fill_index

        # Dedupe and sort for deterministic output.
        unique_times = sorted({round(t, 6) for t in times})

        for start_time in unique_times:
            if start_time >= clip_length:
                continue
            on_beat, beat_index = _classify_time(start_time, bar_start, float(beat_step))
            fill_index = fill_markers.get(round(start_time, 6))
            variant_index = int(round(float(start_time) * 1000.0))
            if fill_index is not None:
                # Nudge fill notes into different slots within the grace range.
                variant_index += (fill_index + 1) * 7919
            seed = _velocity_seed(
                bar_index=bar_index,
                beat_index=beat_index,
                variant_index=variant_index,
            )

            if on_beat and beat_index == 0:
                note_velocity = _velocity_in_range(accent_min, accent_max, seed)
            elif on_beat:
                note_velocity = _velocity_in_range(beat_min, beat_max, seed)
            else:
                note_velocity = _velocity_in_range(grace_min, grace_max, seed)

            notes.append(
                {
                    "pitch": int(pitch),
                    "start_time": float(start_time),
                    "duration": float(duration),
                    "velocity": int(note_velocity),
                    "mute": 0,
                }
            )

    # Sort for readability and deterministic diffs in logs.
    notes.sort(key=lambda n: (n["start_time"], n["pitch"]))
    return notes, clip_length


def _sanitize_live_path(path: str) -> str:
    """Remove quoting artifacts that can appear in LiveAPI path strings."""
    return str(path).replace('"', "").strip()


def _new_child_path(before: Sequence[dict], after: Sequence[dict]) -> str | None:
    before_ids = {int(item.get("id", 0)) for item in before}
    candidates = [item for item in after if int(item.get("id", 0)) not in before_ids]
    if len(candidates) == 1:
        return _sanitize_live_path(str(candidates[0].get("path")))
    return None


def _extract_id_from_call_result(result: object | None) -> int | None:
    if isinstance(result, list):
        for idx, item in enumerate(result):
            if str(item) == "id" and idx + 1 < len(result):
                return _as_int(result[idx + 1])
        if len(result) == 1:
            return _as_int(result[0])
    if isinstance(result, dict):
        return _as_int(result.get("id"))
    return None


def _find_track_index_by_name(tracks: Sequence[dict], name: str) -> int | None:
    target = name.strip().lower()
    if not target:
        return None
    matches: List[int] = []
    for item in tracks:
        track_name = str(item.get("name", "")).strip().lower()
        if track_name == target:
            matches.append(int(item.get("index", -1)))
    if not matches:
        return None
    return max(matches)


def _find_groove_id_by_name(
    sock: socket.socket,
    ack_sock: socket.socket,
    groove_name: str,
    timeout_s: float,
) -> int | None:
    """Look up a groove id by its name in the groove pool."""
    target = groove_name.strip().lower()
    if not target:
        return None

    grooves = _get_children(sock, ack_sock, "live_set groove_pool", "grooves", "kick-grooves", timeout_s)
    if not grooves:
        return None

    for item in grooves:
        index = int(item.get("index", -1))
        if index < 0:
            continue
        groove_path = f"live_set groove_pool grooves {index}"
        describe = _api_describe(sock, ack_sock, groove_path, f"kick-groove-describe-{index}", timeout_s)
        if not describe:
            continue
        name_value = str(describe.get("name", "")).strip().lower()
        groove_id = _as_int(describe.get("id"))
        if groove_id and name_value == target:
            return groove_id
    return None


def _overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a < end_b and end_a > start_b


@dataclass(frozen=True)
class KickConfig:
    bars: int
    start_beats: float
    track_name: str
    clip_name: str
    device_name: str
    insert_device: bool
    beats_per_bar_override: float | None
    pitch: int
    velocity: int
    duration: float
    groove_name: str
    ack_timeout_s: float
    dry_run: bool


def parse_args(argv: Iterable[str]) -> KickConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=32, help="Arrangement length in bars")
    parser.add_argument(
        "--start-beats",
        type=float,
        default=0.0,
        help="Arrangement start time in beats (default: 0)",
    )
    parser.add_argument("--track-name", default="Kick", help="Name of the MIDI track")
    parser.add_argument("--clip-name", default="Kick 32", help="Name of the arrangement clip")
    parser.add_argument(
        "--device-name",
        default="Drum Rack",
        help="Native device to insert on the track (default: Drum Rack)",
    )
    parser.add_argument(
        "--no-device",
        action="store_true",
        help="Skip device insertion",
    )
    parser.add_argument(
        "--beats-per-bar",
        type=float,
        default=None,
        help="Override beats per bar (default: derive from Live time signature)",
    )
    parser.add_argument("--pitch", type=int, default=36, help="Kick MIDI note (default: 36)")
    parser.add_argument(
        "--velocity",
        type=int,
        default=110,
        help="Kick velocity (default: 110)",
    )
    parser.add_argument(
        "--groove-name",
        default=DEFAULT_GROOVE_NAME,
        help=f"Groove Pool name to apply (default: {DEFAULT_GROOVE_NAME})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.25,
        help="Note duration in beats (default: 0.25)",
    )
    parser.add_argument(
        "--ack-timeout",
        type=float,
        default=1.25,
        help="Ack wait timeout in seconds (default: 1.25)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without sending OSC messages",
    )

    ns = parser.parse_args(list(argv))

    if ns.bars <= 0:
        parser.error("--bars must be > 0")
    if ns.beats_per_bar is not None and ns.beats_per_bar <= 0:
        parser.error("--beats-per-bar must be > 0 when provided")
    if ns.ack_timeout <= 0:
        parser.error("--ack-timeout must be > 0")

    return KickConfig(
        bars=int(ns.bars),
        start_beats=float(ns.start_beats),
        track_name=str(ns.track_name),
        clip_name=str(ns.clip_name),
        device_name=str(ns.device_name),
        insert_device=not bool(ns.no_device),
        beats_per_bar_override=None if ns.beats_per_bar is None else float(ns.beats_per_bar),
        pitch=int(ns.pitch),
        velocity=int(ns.velocity),
        duration=float(ns.duration),
        groove_name=str(ns.groove_name),
        ack_timeout_s=float(ns.ack_timeout),
        dry_run=bool(ns.dry_run),
    )


def run(cfg: KickConfig) -> int:
    if cfg.dry_run:
        beats_per_bar = cfg.beats_per_bar_override if cfg.beats_per_bar_override else 4.0
        beat_step = 1.0
        notes, clip_length = build_kick_notes(
            bars=cfg.bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            pitch=cfg.pitch,
            velocity=cfg.velocity,
            duration=cfg.duration,
        )
        clip_start = float(cfg.start_beats)
        clip_end = clip_start + clip_length
        print("Kick pattern plan (dry run):")
        print(f"- bars:        {cfg.bars}")
        print(f"- beats/bar:   {beats_per_bar:g} (assumed)")
        print(f"- beat step:   {beat_step:g} (assumed quarter-note beats)")
        print(f"- clip start:  {clip_start:g}")
        print(f"- clip length: {clip_length:g} beats")
        print(f"- clip end:    {clip_end:g}")
        print(f"- note count:  {len(notes)}")
        print(f"- pitch:       {cfg.pitch}")
        print(f"- velocity:    {cfg.velocity}")
        print(f"- groove:      {cfg.groove_name}")
        print("\nDry run only. No OSC messages were sent.")
        return 0

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
        delay_ms=0,
        dry_run=False,
    )

    ack_sock = bridge.open_ack_socket(bridge_cfg)
    if ack_sock is None:
        print("error: failed to open ack socket", file=sys.stderr)
        return 1

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        print(f"\nTarget: udp://{HOST}:{PORT}")
        print(f"Ack:    udp://{HOST}:{ACK_PORT} (timeout {cfg.ack_timeout_s:.2f}s)")

        # Prime the bridge.
        for cmd in (bridge.OscCommand("/ping"), bridge.OscCommand("/status")):
            print(f"sent: {bridge.describe_command(cmd)}")
            _print_acks(_send_and_collect_acks(sock, ack_sock, cmd, cfg.ack_timeout_s))

        tracks_before = _get_children(
            sock,
            ack_sock,
            "live_set",
            "tracks",
            "kick-tracks-before",
            cfg.ack_timeout_s,
        )
        if not tracks_before:
            print("error: could not read tracks; reload the device in Live", file=sys.stderr)
            ack_sock.close()
            return 2

        tempo_value = _api_get(
            sock, ack_sock, "live_set", "tempo", "kick-tempo", cfg.ack_timeout_s
        )
        sig_num_value = _api_get(
            sock,
            ack_sock,
            "live_set",
            "signature_numerator",
            "kick-sig-num",
            cfg.ack_timeout_s,
        )
        sig_den_value = _api_get(
            sock,
            ack_sock,
            "live_set",
            "signature_denominator",
            "kick-sig-den",
            cfg.ack_timeout_s,
        )

        tempo = _as_float(tempo_value)
        signature_numerator = _as_int(sig_num_value) or 4
        signature_denominator = _as_int(sig_den_value) or 4
        beats_per_bar_from_sig = _beats_per_bar_from_signature(
            signature_numerator, signature_denominator
        )
        beat_step_from_sig = _beat_step_from_denominator(signature_denominator)
        beats_per_bar = (
            cfg.beats_per_bar_override
            if cfg.beats_per_bar_override is not None
            else beats_per_bar_from_sig
        )
        beat_step = beat_step_from_sig

        notes, clip_length = build_kick_notes(
            bars=cfg.bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            pitch=cfg.pitch,
            velocity=cfg.velocity,
            duration=cfg.duration,
        )

        clip_start = float(cfg.start_beats)
        clip_end = clip_start + clip_length

        print("\nKick pattern plan:")
        print(f"- tempo:       {tempo:g}" if tempo is not None else "- tempo:       (unknown)")
        print(f"- signature:   {signature_numerator}/{signature_denominator}")
        if cfg.beats_per_bar_override is None:
            print(f"- beats/bar:   {beats_per_bar:g} (from signature)")
        else:
            print(
                f"- beats/bar:   {beats_per_bar:g} "
                f"(override; signature implies {beats_per_bar_from_sig:g})"
            )
        print(f"- beat step:   {beat_step:g} (from signature denominator)")
        print(f"- bars:        {cfg.bars}")
        print(f"- clip start:  {clip_start:g}")
        print(f"- clip length: {clip_length:g} beats")
        print(f"- clip end:    {clip_end:g}")
        print(f"- note count:  {len(notes)}")
        print(f"- pitch:       {cfg.pitch}")
        print(f"- velocity:    {cfg.velocity}")

        existing_track_index = _find_track_index_by_name(tracks_before, cfg.track_name)
        if existing_track_index is None:
            create_track_req = "kick-create-track"
            _api_call(
                sock,
                ack_sock,
                "live_set",
                "create_midi_track",
                [-1],
                create_track_req,
                cfg.ack_timeout_s,
            )

            tracks_after = _get_children(
                sock,
                ack_sock,
                "live_set",
                "tracks",
                "kick-tracks-after",
                cfg.ack_timeout_s,
            )
            if len(tracks_after) <= len(tracks_before):
                print(
                    "error: MIDI track was not created; reload the device in Live",
                    file=sys.stderr,
                )
                ack_sock.close()
                return 3

            track_index = len(tracks_after) - 1
            track_path = f"live_set tracks {track_index}"
            print(f"info: created track_index={track_index} path={track_path}")
            track_was_created = True
        else:
            track_index = existing_track_index
            track_path = f"live_set tracks {track_index}"
            print(f"info: reusing track_index={track_index} path={track_path}")
            track_was_created = False

        _api_set(
            sock,
            ack_sock,
            track_path,
            "name",
            cfg.track_name,
            "kick-track-name",
            cfg.ack_timeout_s,
        )

        if cfg.insert_device and cfg.device_name.strip():
            devices = _get_children(
                sock,
                ack_sock,
                track_path,
                "devices",
                "kick-devices",
                cfg.ack_timeout_s,
            )
            if track_was_created or not devices:
                _api_call(
                    sock,
                    ack_sock,
                    track_path,
                    "insert_device",
                    [cfg.device_name],
                    "kick-insert-device",
                    cfg.ack_timeout_s,
                )
            else:
                print(
                    f"info: devices already present on track {track_index}; "
                    "skipping device insertion"
                )

        arrangement_initial = _get_children(
            sock,
            ack_sock,
            track_path,
            "arrangement_clips",
            "kick-arr-initial",
            cfg.ack_timeout_s,
        )

        deleted_overlaps = 0
        for idx, clip_info in enumerate(arrangement_initial):
            clip_path_raw = clip_info.get("path")
            if not clip_path_raw:
                continue
            clip_path = _sanitize_live_path(str(clip_path_raw))
            clip_start_existing = _as_float(
                _api_get(
                    sock,
                    ack_sock,
                    clip_path,
                    "start_time",
                    f"kick-clip-start-{idx}",
                    cfg.ack_timeout_s,
                )
            )
            clip_end_existing = _as_float(
                _api_get(
                    sock,
                    ack_sock,
                    clip_path,
                    "end_time",
                    f"kick-clip-end-{idx}",
                    cfg.ack_timeout_s,
                )
            )
            if clip_start_existing is None or clip_end_existing is None:
                continue
            if not _overlaps(clip_start_existing, clip_end_existing, clip_start, clip_end):
                continue

            clip_desc = _api_describe(
                sock,
                ack_sock,
                clip_path,
                f"kick-clip-describe-{idx}",
                cfg.ack_timeout_s,
            )
            clip_id = _as_int(clip_desc.get("id") if isinstance(clip_desc, dict) else None)
            if clip_id is None or clip_id <= 0:
                clip_id = _as_int(
                    _api_get(
                        sock,
                        ack_sock,
                        clip_path,
                        "id",
                        f"kick-clip-id-{idx}",
                        cfg.ack_timeout_s,
                    )
                )
            if clip_id is None or clip_id <= 0:
                print(f"warning: could not resolve clip id for {clip_path}; skipping delete")
                continue

            _api_call(
                sock,
                ack_sock,
                track_path,
                "delete_clip",
                [clip_id],
                f"kick-delete-clip-{idx}",
                cfg.ack_timeout_s,
            )
            deleted_overlaps += 1

        if deleted_overlaps > 0:
            print(f"info: deleted {deleted_overlaps} overlapping arrangement clip(s)")

        arrangement_before = _get_children(
            sock,
            ack_sock,
            track_path,
            "arrangement_clips",
            "kick-arr-before",
            cfg.ack_timeout_s,
        )

        create_result = _api_call(
            sock,
            ack_sock,
            track_path,
            "create_midi_clip",
            [clip_start, clip_length],
            "kick-create-clip",
            cfg.ack_timeout_s,
        )

        clip_id = _extract_id_from_call_result(create_result)
        if clip_id is not None and clip_id > 0:
            clip_path = f"id {clip_id}"
            print(f"info: arrangement clip id={clip_id} (using path '{clip_path}')")
        else:
            arrangement_after = _get_children(
                sock,
                ack_sock,
                track_path,
                "arrangement_clips",
                "kick-arr-after",
                cfg.ack_timeout_s,
            )
            clip_path = _new_child_path(arrangement_before, arrangement_after)
        if not clip_path:
            print(
                "error: could not identify the new arrangement clip; reload the device in Live",
                file=sys.stderr,
            )
            ack_sock.close()
            return 4

        print(f"info: arrangement clip path={clip_path}")

        _api_set(
            sock,
            ack_sock,
            clip_path,
            "name",
            cfg.clip_name,
            "kick-clip-name",
            cfg.ack_timeout_s,
        )
        _api_set(
            sock,
            ack_sock,
            clip_path,
            "loop_start",
            0.0,
            "kick-loop-start",
            cfg.ack_timeout_s,
        )
        _api_set(
            sock,
            ack_sock,
            clip_path,
            "loop_end",
            clip_length,
            "kick-loop-end",
            cfg.ack_timeout_s,
        )

        note_chunks = _chunk_notes(notes, chunk_size=48)
        for idx, chunk in enumerate(note_chunks, start=1):
            notes_json = {"notes": chunk}
            _api_call(
                sock,
                ack_sock,
                clip_path,
                "add_new_notes",
                notes_json,
                f"kick-add-notes-{idx}-of-{len(note_chunks)}",
                max(cfg.ack_timeout_s, 2.0),
            )

        groove_id = _find_groove_id_by_name(sock, ack_sock, cfg.groove_name, cfg.ack_timeout_s)
        if groove_id:
            # Groove strength is governed by the set-level groove_amount.
            _api_set(
                sock,
                ack_sock,
                "live_set",
                "groove_amount",
                1.0,
                "kick-groove-amount",
                cfg.ack_timeout_s,
            )
            _api_set(
                sock,
                ack_sock,
                clip_path,
                "groove",
                ["id", groove_id],
                "kick-clip-groove",
                cfg.ack_timeout_s,
            )
            print(f"info: applied groove '{cfg.groove_name}' (id={groove_id})")
        else:
            print(f"warning: groove '{cfg.groove_name}' not found in groove pool")

        note_dump = _api_call(
            sock,
            ack_sock,
            clip_path,
            "get_all_notes_extended",
            [],
            "kick-inspect",
            max(cfg.ack_timeout_s, 2.0),
        )
        if isinstance(note_dump, dict):
            note_count = len(note_dump.get("notes", [])) if isinstance(note_dump.get("notes"), list) else "?"
            print(f"info: Live reports note_count={note_count}")
        elif isinstance(note_dump, list):
            print(f"info: Live returned note payload list length={len(note_dump)}")

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
