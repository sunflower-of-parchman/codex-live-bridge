#!/usr/bin/env python3
"""
External controller for the Ableton Live UDP bridge.

This script sends OSC (Open Sound Control) messages to a Max for Live device
that listens on UDP port 9000 via `udpreceive 9000`. OSC encoding is
implemented using only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import select
import socket
import struct
from statistics import mean
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9000
DEFAULT_ACK_PORT = 9001

OscArg = Union[int, float, str]


@dataclass(frozen=True)
class OscCommand:
    address: str
    args: Tuple[OscArg, ...] = ()


@dataclass(frozen=True)
class BridgeConfig:
    host: str
    port: int
    ack_port: int
    ack_timeout_s: float
    expect_ack: bool
    ping_first: bool
    status: bool
    tempo: float | None
    sig_num: int | None
    sig_den: int | None
    create_midi_tracks: int
    add_midi_tracks: int
    midi_name: str
    create_audio_tracks: int
    add_audio_tracks: int
    audio_prefix: str
    delete_audio_tracks: int
    delete_midi_tracks: int
    rename_track_index: int | None
    rename_track_name: str | None
    session_clip_track_index: int | None
    session_clip_slot_index: int | None
    session_clip_length: float | None
    session_clip_notes_json: str | None
    session_clip_name: str | None
    append_session_clip_track_index: int | None
    append_session_clip_slot_index: int | None
    append_session_clip_notes_json: str | None
    inspect_session_clip_track_index: int | None
    inspect_session_clip_slot_index: int | None
    ensure_midi_tracks: int | None
    midi_ccs: Tuple[Tuple[int, int, int], ...]
    cc64s: Tuple[Tuple[int, int], ...]
    api_pings: Tuple[str | None, ...]
    api_gets: Tuple[Tuple[str, str, str | None], ...]
    api_sets: Tuple[Tuple[str, str, str, str | None], ...]
    api_calls: Tuple[Tuple[str, str, str, str | None], ...]
    api_children: Tuple[Tuple[str, str, str | None], ...]
    api_describes: Tuple[Tuple[str, str | None], ...]
    ack_mode: str
    ack_flush_interval: int
    report_metrics: bool
    delay_ms: int
    dry_run: bool


AckMode = Union[str]


@dataclass(frozen=True)
class SendMetrics:
    command_count: int
    send_durations_ms: Tuple[float, ...]
    ack_wait_durations_ms: Tuple[float, ...]
    acks_per_command: Tuple[int, ...]
    elapsed_ms: float


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (max(0.0, min(100.0, float(pct))) / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _format_ms(value: float) -> str:
    return f"{float(value):.2f}"


def _summarize_metrics(metrics: SendMetrics) -> List[str]:
    lines: List[str] = []
    lines.append(
        "metrics: commands={count} elapsed_ms={elapsed}".format(
            count=metrics.command_count,
            elapsed=_format_ms(metrics.elapsed_ms),
        )
    )

    if metrics.send_durations_ms:
        lines.append(
            "metrics: send_ms p50={p50} p95={p95} max={maxv}".format(
                p50=_format_ms(_percentile(metrics.send_durations_ms, 50.0)),
                p95=_format_ms(_percentile(metrics.send_durations_ms, 95.0)),
                maxv=_format_ms(max(metrics.send_durations_ms)),
            )
        )

    if metrics.ack_wait_durations_ms:
        lines.append(
            "metrics: ack_wait_ms p50={p50} p95={p95} max={maxv} mean={meanv}".format(
                p50=_format_ms(_percentile(metrics.ack_wait_durations_ms, 50.0)),
                p95=_format_ms(_percentile(metrics.ack_wait_durations_ms, 95.0)),
                maxv=_format_ms(max(metrics.ack_wait_durations_ms)),
                meanv=_format_ms(mean(metrics.ack_wait_durations_ms)),
            )
        )

    if metrics.acks_per_command:
        lines.append(
            "metrics: acks_per_command mean={meanv} max={maxv}".format(
                meanv=_format_ms(mean(float(v) for v in metrics.acks_per_command)),
                maxv=max(metrics.acks_per_command),
            )
        )
    return lines


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def midi_byte(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 127:
        raise argparse.ArgumentTypeError("value must be between 0 and 127")
    return parsed


def midi_channel(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 16:
        raise argparse.ArgumentTypeError("channel must be between 1 and 16")
    return parsed


def parse_args(argv: Iterable[str]) -> BridgeConfig:
    parser = argparse.ArgumentParser(
        description="Send OSC UDP commands to a Max for Live Ableton bridge."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="UDP host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UDP port")

    parser.add_argument(
        "--ack",
        action="store_true",
        help="Listen for OSC /ack responses on --ack-port",
    )
    parser.add_argument(
        "--ack-port",
        type=int,
        default=DEFAULT_ACK_PORT,
        help="UDP port to listen for acknowledgements",
    )
    parser.add_argument(
        "--ack-timeout",
        type=non_negative_float,
        default=0.6,
        help="How long to wait for acknowledgements after each send (seconds)",
    )
    parser.add_argument(
        "--ack-mode",
        choices=("per_command", "flush_end", "flush_interval"),
        default="per_command",
        help="Acknowledgement handling strategy (default: per_command)",
    )
    parser.add_argument(
        "--ack-flush-interval",
        type=positive_int,
        default=10,
        help="Flush interval in commands when --ack-mode=flush_interval (default: 10)",
    )
    parser.add_argument(
        "--no-ping-first",
        action="store_true",
        help="Skip sending /ping before the main commands when --ack is enabled",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Request bridge status via /status",
    )
    parser.add_argument(
        "--api-ping",
        nargs="?",
        action="append",
        default=[],
        metavar="REQUEST_ID",
        help="Send /api/ping with an optional request id",
    )
    parser.add_argument(
        "--api-get",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/get <path> <property> [request_id]",
    )
    parser.add_argument(
        "--api-set",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/set <path> <property> <value_json> [request_id]",
    )
    parser.add_argument(
        "--api-call",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/call <path> <method> <args_json> [request_id]",
    )
    parser.add_argument(
        "--api-children",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/children <path> <child_name> [request_id]",
    )
    parser.add_argument(
        "--api-describe",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/describe <path> [request_id]",
    )

    parser.add_argument(
        "--tempo",
        type=positive_float,
        default=120.0,
        help="Tempo in BPM (omit with --no-tempo)",
    )
    parser.add_argument(
        "--no-tempo",
        action="store_true",
        help="Do not send a tempo command",
    )
    parser.add_argument(
        "--sig-num",
        type=positive_int,
        default=4,
        help="Time signature numerator (omit with --no-signature)",
    )
    parser.add_argument(
        "--sig-den",
        type=positive_int,
        default=4,
        help="Time signature denominator (omit with --no-signature)",
    )
    parser.add_argument(
        "--no-signature",
        action="store_true",
        help="Do not send time signature commands",
    )
    parser.add_argument(
        "--create-midi-tracks",
        type=non_negative_int,
        default=0,
        help="How many /create_midi_track commands to send",
    )
    parser.add_argument(
        "--add-midi-tracks",
        type=non_negative_int,
        default=0,
        help="Create and name this many MIDI tracks via /add_midi_tracks",
    )
    parser.add_argument(
        "--midi-name",
        default="MIDI",
        help="Name used with --add-midi-tracks (default: MIDI)",
    )
    parser.add_argument(
        "--create-audio-tracks",
        type=non_negative_int,
        default=0,
        help="How many /create_audio_track commands to send",
    )
    parser.add_argument(
        "--add-audio-tracks",
        type=non_negative_int,
        default=0,
        help="Create and name this many audio tracks via /add_audio_tracks",
    )
    parser.add_argument(
        "--audio-prefix",
        default="Audio",
        help="Name prefix used with --add-audio-tracks (default: Audio)",
    )
    parser.add_argument(
        "--delete-audio-tracks",
        type=non_negative_int,
        default=0,
        help="Delete this many audio tracks via /delete_audio_tracks",
    )
    parser.add_argument(
        "--delete-midi-tracks",
        type=non_negative_int,
        default=0,
        help="Delete this many MIDI tracks via /delete_midi_tracks (track 0 is protected)",
    )
    parser.add_argument(
        "--rename-track-index",
        type=non_negative_int,
        default=None,
        help="Track index to rename via /rename_track",
    )
    parser.add_argument(
        "--rename-track-name",
        default=None,
        help="New name used with --rename-track-index",
    )
    parser.add_argument(
        "--session-clip-track-index",
        type=non_negative_int,
        default=None,
        help="Track index for /set_session_clip_notes",
    )
    parser.add_argument(
        "--session-clip-slot-index",
        type=non_negative_int,
        default=None,
        help="Clip slot index for /set_session_clip_notes",
    )
    parser.add_argument(
        "--session-clip-length",
        type=positive_float,
        default=None,
        help="Clip length in beats for /set_session_clip_notes",
    )
    parser.add_argument(
        "--session-clip-notes-json",
        default=None,
        help="JSON payload (string) for /set_session_clip_notes",
    )
    parser.add_argument(
        "--session-clip-name",
        default=None,
        help="Clip name for /set_session_clip_notes",
    )
    parser.add_argument(
        "--append-session-clip-track-index",
        type=non_negative_int,
        default=None,
        help="Track index for /append_session_clip_notes",
    )
    parser.add_argument(
        "--append-session-clip-slot-index",
        type=non_negative_int,
        default=None,
        help="Clip slot index for /append_session_clip_notes",
    )
    parser.add_argument(
        "--append-session-clip-notes-json",
        default=None,
        help="JSON payload (string) for /append_session_clip_notes",
    )
    parser.add_argument(
        "--inspect-session-clip-track-index",
        type=non_negative_int,
        default=None,
        help="Track index for /inspect_session_clip_notes",
    )
    parser.add_argument(
        "--inspect-session-clip-slot-index",
        type=non_negative_int,
        default=None,
        help="Clip slot index for /inspect_session_clip_notes",
    )
    parser.add_argument(
        "--ensure-midi-tracks",
        type=non_negative_int,
        default=None,
        help="Target MIDI track count",
    )
    parser.add_argument(
        "--midi-cc",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /midi_cc <controller> <value> [channel]",
    )
    parser.add_argument(
        "--cc64",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /cc64 <value> [channel]",
    )
    parser.add_argument(
        "--delay-ms",
        type=non_negative_int,
        default=40,
        help="Delay between messages in milliseconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print messages without sending them",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable command timing summaries",
    )

    ns = parser.parse_args(list(argv))

    tempo: float | None = None if ns.no_tempo else ns.tempo
    sig_num: int | None = None if ns.no_signature else ns.sig_num
    sig_den: int | None = None if ns.no_signature else ns.sig_den
    rename_track_index: int | None = ns.rename_track_index
    rename_track_name: str | None = (
        None if ns.rename_track_name is None else str(ns.rename_track_name)
    )

    if (rename_track_index is None) != (rename_track_name is None):
        parser.error("--rename-track-index and --rename-track-name must be provided together")

    session_clip_fields = [
        ns.session_clip_track_index,
        ns.session_clip_slot_index,
        ns.session_clip_length,
        ns.session_clip_notes_json,
    ]
    session_clip_any = any(field is not None for field in session_clip_fields)
    session_clip_all = all(field is not None for field in session_clip_fields)
    if session_clip_any and not session_clip_all:
        parser.error(
            "--session-clip-track-index, --session-clip-slot-index, "
            "--session-clip-length, and --session-clip-notes-json must be provided together"
        )

    session_clip_track_index: int | None = ns.session_clip_track_index
    session_clip_slot_index: int | None = ns.session_clip_slot_index
    session_clip_length: float | None = ns.session_clip_length
    session_clip_notes_json: str | None = (
        None if ns.session_clip_notes_json is None else str(ns.session_clip_notes_json)
    )
    session_clip_name: str | None = (
        None if ns.session_clip_name is None else str(ns.session_clip_name)
    )

    append_clip_track_index: int | None = ns.append_session_clip_track_index
    append_clip_slot_index: int | None = ns.append_session_clip_slot_index
    append_clip_notes_json: str | None = (
        None if ns.append_session_clip_notes_json is None else str(ns.append_session_clip_notes_json)
    )
    append_clip_fields = [append_clip_track_index, append_clip_slot_index, append_clip_notes_json]
    append_clip_any = any(field is not None for field in append_clip_fields)
    append_clip_all = all(field is not None for field in append_clip_fields)
    if append_clip_any and not append_clip_all:
        parser.error(
            "--append-session-clip-track-index, --append-session-clip-slot-index, "
            "and --append-session-clip-notes-json must be provided together"
        )

    inspect_track_index: int | None = ns.inspect_session_clip_track_index
    inspect_slot_index: int | None = ns.inspect_session_clip_slot_index
    if (inspect_track_index is None) != (inspect_slot_index is None):
        parser.error(
            "--inspect-session-clip-track-index and --inspect-session-clip-slot-index must be provided together"
        )

    def _optional_request_id(parts: Sequence[str], min_len: int) -> str | None:
        if len(parts) == min_len:
            return None
        return str(parts[-1])

    def _parse_api_get(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (2, 3):
                parser.error("--api-get expects: <path> <property> [request_id]")
            path, prop = str(parts[0]), str(parts[1])
            request_id = _optional_request_id(parts, 2)
            parsed.append((path, prop, request_id))
        return tuple(parsed)

    def _parse_api_set(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[str, str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (3, 4):
                parser.error("--api-set expects: <path> <property> <value_json> [request_id]")
            path, prop, value_json = str(parts[0]), str(parts[1]), str(parts[2])
            request_id = _optional_request_id(parts, 3)
            parsed.append((path, prop, value_json, request_id))
        return tuple(parsed)

    def _parse_api_call(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[str, str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (3, 4):
                parser.error("--api-call expects: <path> <method> <args_json> [request_id]")
            path, method, args_json = str(parts[0]), str(parts[1]), str(parts[2])
            request_id = _optional_request_id(parts, 3)
            parsed.append((path, method, args_json, request_id))
        return tuple(parsed)

    def _parse_api_children(
        entries: Sequence[Sequence[str]],
    ) -> Tuple[Tuple[str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (2, 3):
                parser.error("--api-children expects: <path> <child_name> [request_id]")
            path, child_name = str(parts[0]), str(parts[1])
            request_id = _optional_request_id(parts, 2)
            parsed.append((path, child_name, request_id))
        return tuple(parsed)

    def _parse_api_describe(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[str, str | None], ...]:
        parsed: List[Tuple[str, str | None]] = []
        for parts in entries:
            if len(parts) not in (1, 2):
                parser.error("--api-describe expects: <path> [request_id]")
            path = str(parts[0])
            request_id = _optional_request_id(parts, 1)
            parsed.append((path, request_id))
        return tuple(parsed)

    def _parse_midi_cc(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[int, int, int], ...]:
        parsed: List[Tuple[int, int, int]] = []
        for parts in entries:
            if len(parts) not in (2, 3):
                parser.error("--midi-cc expects: <controller> <value> [channel]")
            controller = midi_byte(str(parts[0]))
            value = midi_byte(str(parts[1]))
            channel = midi_channel(str(parts[2])) if len(parts) == 3 else 1
            parsed.append((controller, value, channel))
        return tuple(parsed)

    def _parse_cc64(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[int, int], ...]:
        parsed: List[Tuple[int, int]] = []
        for parts in entries:
            if len(parts) not in (1, 2):
                parser.error("--cc64 expects: <value> [channel]")
            value = midi_byte(str(parts[0]))
            channel = midi_channel(str(parts[1])) if len(parts) == 2 else 1
            parsed.append((value, channel))
        return tuple(parsed)

    api_pings: Tuple[str | None, ...] = tuple(
        None if value in (None, "") else str(value) for value in ns.api_ping
    )
    api_gets = _parse_api_get(ns.api_get)
    api_sets = _parse_api_set(ns.api_set)
    api_calls = _parse_api_call(ns.api_call)
    api_children = _parse_api_children(ns.api_children)
    api_describes = _parse_api_describe(ns.api_describe)
    midi_ccs = _parse_midi_cc(ns.midi_cc)
    cc64s = _parse_cc64(ns.cc64)

    return BridgeConfig(
        host=ns.host,
        port=ns.port,
        ack_port=ns.ack_port,
        ack_timeout_s=ns.ack_timeout,
        expect_ack=ns.ack,
        ping_first=ns.ack and not ns.no_ping_first,
        status=bool(ns.status),
        tempo=tempo,
        sig_num=sig_num,
        sig_den=sig_den,
        create_midi_tracks=ns.create_midi_tracks,
        add_midi_tracks=ns.add_midi_tracks,
        midi_name=str(ns.midi_name),
        create_audio_tracks=ns.create_audio_tracks,
        add_audio_tracks=ns.add_audio_tracks,
        audio_prefix=str(ns.audio_prefix),
        delete_audio_tracks=ns.delete_audio_tracks,
        delete_midi_tracks=ns.delete_midi_tracks,
        rename_track_index=rename_track_index,
        rename_track_name=rename_track_name,
        session_clip_track_index=session_clip_track_index,
        session_clip_slot_index=session_clip_slot_index,
        session_clip_length=session_clip_length,
        session_clip_notes_json=session_clip_notes_json,
        session_clip_name=session_clip_name,
        append_session_clip_track_index=append_clip_track_index,
        append_session_clip_slot_index=append_clip_slot_index,
        append_session_clip_notes_json=append_clip_notes_json,
        inspect_session_clip_track_index=inspect_track_index,
        inspect_session_clip_slot_index=inspect_slot_index,
        ensure_midi_tracks=ns.ensure_midi_tracks,
        midi_ccs=midi_ccs,
        cc64s=cc64s,
        api_pings=api_pings,
        api_gets=api_gets,
        api_sets=api_sets,
        api_calls=api_calls,
        api_children=api_children,
        api_describes=api_describes,
        ack_mode=str(ns.ack_mode),
        ack_flush_interval=int(ns.ack_flush_interval),
        report_metrics=not bool(ns.no_metrics),
        delay_ms=ns.delay_ms,
        dry_run=ns.dry_run,
    )


def _pad4(length: int) -> int:
    remainder = length % 4
    return 0 if remainder == 0 else 4 - remainder


def _encode_osc_string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    raw += b"\x00" * _pad4(len(raw))
    return raw


def _decode_osc_string(data: bytes, start: int) -> Tuple[str, int]:
    end = data.find(b"\x00", start)
    if end == -1:
        # Some OSC senders appear to omit the trailing NUL on the final string.
        # In that case, treat the remainder as the string and stop parsing.
        text = data[start:].decode("utf-8", errors="replace")
        return text, len(data)
    text = data[start:end].decode("utf-8", errors="replace")
    idx = end + 1
    idx += _pad4(idx)
    return text, idx


def encode_osc_message(address: str, args: Sequence[OscArg]) -> bytes:
    if not address.startswith("/"):
        raise ValueError(f"OSC address must start with '/': {address}")

    type_tags: List[str] = []
    payload = bytearray()

    for arg in args:
        if isinstance(arg, bool):
            type_tags.append("i")
            payload.extend(struct.pack(">i", int(arg)))
        elif isinstance(arg, int):
            type_tags.append("i")
            payload.extend(struct.pack(">i", arg))
        elif isinstance(arg, float):
            type_tags.append("f")
            payload.extend(struct.pack(">f", arg))
        elif isinstance(arg, str):
            type_tags.append("s")
            payload.extend(_encode_osc_string(arg))
        else:
            raise TypeError(f"Unsupported OSC argument type: {type(arg)}")

    type_tag_string = "," + "".join(type_tags)
    return _encode_osc_string(address) + _encode_osc_string(type_tag_string) + payload


def decode_osc_message(data: bytes) -> Tuple[str, List[OscArg]]:
    if data.startswith(b"#bundle"):
        raise ValueError("OSC bundles are not supported by this minimal decoder")

    address, idx = _decode_osc_string(data, 0)
    type_tags, idx = _decode_osc_string(data, idx)

    if not type_tags.startswith(","):
        raise ValueError(f"OSC type tags must start with ',': {type_tags}")

    args: List[OscArg] = []
    for tag in type_tags[1:]:
        if tag == "i":
            if idx + 4 > len(data):
                raise ValueError("OSC int argument truncated")
            value = struct.unpack(">i", data[idx : idx + 4])[0]
            idx += 4
            args.append(value)
        elif tag == "f":
            if idx + 4 > len(data):
                raise ValueError("OSC float argument truncated")
            value = struct.unpack(">f", data[idx : idx + 4])[0]
            idx += 4
            args.append(value)
        elif tag == "s":
            value, idx = _decode_osc_string(data, idx)
            args.append(value)
        else:
            raise ValueError(f"Unsupported OSC type tag: {tag}")

    return address, args


def format_arg(value: OscArg) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def describe_command(cmd: OscCommand) -> str:
    if not cmd.args:
        return cmd.address
    return cmd.address + " " + " ".join(format_arg(arg) for arg in cmd.args)


def _try_parse_json(value: OscArg) -> object | None:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _short_repr(value: object, max_len: int = 120) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, separators=(",", ":"))
    else:
        text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _rpc_ack_summary(args: Sequence[OscArg]) -> str | None:
    if not args:
        return None
    event = str(args[0])

    def _req_suffix(request_id: OscArg | None) -> str:
        return "" if request_id in (None, "") else f" req={request_id}"

    if event == "midi_cc" and len(args) >= 4:
        controller, value, channel = args[1], args[2], args[3]
        request_id = args[4] if len(args) >= 5 else None
        return (
            f"midi_cc ctrl={controller} value={value} ch={channel}"
            f"{_req_suffix(request_id)}"
        )

    if event == "cc64" and len(args) >= 3:
        value, channel = args[1], args[2]
        request_id = args[3] if len(args) >= 4 else None
        return f"cc64 value={value} ch={channel}{_req_suffix(request_id)}"

    if event == "api_get" and len(args) >= 4:
        path, prop, value = args[1], args[2], args[3]
        request_id = args[4] if len(args) >= 5 else None
        parsed = _try_parse_json(value)
        value_text = _short_repr(parsed if parsed is not None else value)
        return f"api_get {path} {prop} -> {value_text}{_req_suffix(request_id)}"

    if event == "api_set" and len(args) >= 4:
        path, prop, result = args[1], args[2], args[3]
        request_id = args[4] if len(args) >= 5 else None
        parsed = _try_parse_json(result)
        result_text = _short_repr(parsed if parsed is not None else result)
        return f"api_set {path} {prop} -> {result_text}{_req_suffix(request_id)}"

    if event == "api_call" and len(args) >= 4:
        path, method, result = args[1], args[2], args[3]
        request_id = args[4] if len(args) >= 5 else None
        parsed = _try_parse_json(result)
        result_text = _short_repr(parsed if parsed is not None else result)
        return f"api_call {path} {method} -> {result_text}{_req_suffix(request_id)}"

    if event == "api_children" and len(args) >= 4:
        path, child_name, children_json = args[1], args[2], args[3]
        request_id = args[4] if len(args) >= 5 else None
        parsed = _try_parse_json(children_json)
        count = len(parsed) if isinstance(parsed, list) else "?"
        preview = ""
        if isinstance(parsed, list) and parsed:
            names = [str(item.get("name", item.get("path"))) for item in parsed[:3]]
            preview = f" first={names}"
        return (
            f"api_children {path} {child_name} count={count}{preview}"
            f"{_req_suffix(request_id)}"
        )

    if event == "api_describe" and len(args) >= 3:
        path, describe_json = args[1], args[2]
        request_id = args[3] if len(args) >= 4 else None
        parsed = _try_parse_json(describe_json)
        if isinstance(parsed, dict):
            core = {
                "id": parsed.get("id"),
                "name": parsed.get("name"),
                "type": parsed.get("type"),
            }
            core = {k: v for k, v in core.items() if v not in (None, "")}
            core_text = _short_repr(core) if core else _short_repr(parsed)
        else:
            core_text = _short_repr(describe_json)
        return f"api_describe {path} -> {core_text}{_req_suffix(request_id)}"

    if event == "error" and len(args) >= 2 and str(args[1]).startswith("api_"):
        request_id = args[-1] if len(args) >= 3 else None
        detail = " ".join(str(a) for a in args[1:])
        return f"api_error {detail}{_req_suffix(request_id)}"

    return None


def summarize_ack(address: str, args: Sequence[OscArg]) -> List[str]:
    suffix = "" if not args else " " + " ".join(format_arg(a) for a in args)
    lines = [f"ack:  {address}{suffix}"]

    if address == "/ack":
        summary = _rpc_ack_summary(args)
        if summary:
            lines.append(f"ack:  {summary}")

    return lines


def build_commands(cfg: BridgeConfig) -> List[OscCommand]:
    commands: List[OscCommand] = []

    if cfg.ping_first:
        commands.append(OscCommand("/ping"))

    def _with_request_id(args: List[OscArg], request_id: str | None) -> Tuple[OscArg, ...]:
        if request_id is None:
            return tuple(args)
        return tuple(args + [request_id])

    # Additive LiveAPI RPC preflight surface.
    for request_id in cfg.api_pings:
        commands.append(OscCommand("/api/ping", _with_request_id([], request_id)))
    for path, prop, request_id in cfg.api_gets:
        commands.append(OscCommand("/api/get", _with_request_id([path, prop], request_id)))
    for path, prop, value_json, request_id in cfg.api_sets:
        commands.append(
            OscCommand("/api/set", _with_request_id([path, prop, value_json], request_id))
        )
    for path, method, args_json, request_id in cfg.api_calls:
        commands.append(
            OscCommand("/api/call", _with_request_id([path, method, args_json], request_id))
        )
    for path, child_name, request_id in cfg.api_children:
        commands.append(
            OscCommand("/api/children", _with_request_id([path, child_name], request_id))
        )
    for path, request_id in cfg.api_describes:
        commands.append(OscCommand("/api/describe", _with_request_id([path], request_id)))

    if cfg.status:
        commands.append(OscCommand("/status"))

    if cfg.delete_audio_tracks > 0:
        commands.append(OscCommand("/delete_audio_tracks", (cfg.delete_audio_tracks,)))

    if cfg.delete_midi_tracks > 0:
        commands.append(OscCommand("/delete_midi_tracks", (cfg.delete_midi_tracks,)))

    if cfg.tempo is not None:
        commands.append(OscCommand("/tempo", (cfg.tempo,)))

    if cfg.sig_num is not None:
        commands.append(OscCommand("/sig_num", (cfg.sig_num,)))

    if cfg.sig_den is not None:
        commands.append(OscCommand("/sig_den", (cfg.sig_den,)))

    for _ in range(cfg.create_midi_tracks):
        commands.append(OscCommand("/create_midi_track"))

    if cfg.add_midi_tracks > 0:
        commands.append(OscCommand("/add_midi_tracks", (cfg.add_midi_tracks, cfg.midi_name)))

    for _ in range(cfg.create_audio_tracks):
        commands.append(OscCommand("/create_audio_track"))

    if cfg.add_audio_tracks > 0:
        commands.append(
            OscCommand("/add_audio_tracks", (cfg.add_audio_tracks, cfg.audio_prefix))
        )

    if (
        cfg.session_clip_track_index is not None
        and cfg.session_clip_slot_index is not None
        and cfg.session_clip_length is not None
        and cfg.session_clip_notes_json is not None
    ):
        clip_name = "" if cfg.session_clip_name is None else cfg.session_clip_name
        commands.append(
            OscCommand(
                "/set_session_clip_notes",
                (
                    cfg.session_clip_track_index,
                    cfg.session_clip_slot_index,
                    cfg.session_clip_length,
                    cfg.session_clip_notes_json,
                    clip_name,
                ),
            )
        )

    if (
        cfg.append_session_clip_track_index is not None
        and cfg.append_session_clip_slot_index is not None
        and cfg.append_session_clip_notes_json is not None
    ):
        commands.append(
            OscCommand(
                "/append_session_clip_notes",
                (
                    cfg.append_session_clip_track_index,
                    cfg.append_session_clip_slot_index,
                    cfg.append_session_clip_notes_json,
                ),
            )
        )

    if (
        cfg.inspect_session_clip_track_index is not None
        and cfg.inspect_session_clip_slot_index is not None
    ):
        commands.append(
            OscCommand(
                "/inspect_session_clip_notes",
                (cfg.inspect_session_clip_track_index, cfg.inspect_session_clip_slot_index),
            )
        )

    if cfg.rename_track_index is not None and cfg.rename_track_name is not None:
        commands.append(
            OscCommand("/rename_track", (cfg.rename_track_index, cfg.rename_track_name))
        )

    if cfg.ensure_midi_tracks is not None:
        commands.append(OscCommand("/ensure_midi_tracks", (cfg.ensure_midi_tracks,)))

    for controller, value, channel in cfg.midi_ccs:
        commands.append(OscCommand("/midi_cc", (controller, value, channel)))

    for value, channel in cfg.cc64s:
        commands.append(OscCommand("/cc64", (value, channel)))

    return commands


def open_ack_socket(cfg: BridgeConfig) -> socket.socket | None:
    if not cfg.expect_ack or cfg.dry_run:
        return None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((cfg.host, cfg.ack_port))
    except OSError as exc:
        print(
            f"warning: could not bind ack socket on {cfg.host}:{cfg.ack_port}: {exc}",
            file=sys.stderr,
        )
        sock.close()
        return None

    sock.setblocking(False)
    return sock


def wait_for_acks(
    sock: socket.socket,
    timeout_s: float,
    quiet_window_s: float = 0.05,
) -> List[Tuple[str, List[OscArg]]]:
    if timeout_s <= 0:
        return []

    deadline = time.monotonic() + timeout_s
    received: List[Tuple[str, List[OscArg]]] = []
    quiet_window = max(0.0, float(quiet_window_s))

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        # Before first packet, wait up to the full timeout. Once we have at
        # least one ACK, only wait a short quiet window for follow-on packets.
        wait_timeout = remaining
        if received and quiet_window > 0.0:
            wait_timeout = min(wait_timeout, quiet_window)

        readable, _, _ = select.select([sock], [], [], wait_timeout)
        if not readable:
            if received:
                break
            break

        while True:
            try:
                packet, _addr = sock.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                return received

            try:
                address, args = decode_osc_message(packet)
                received.append((address, args))
            except Exception as exc:  # noqa: BLE001 - best-effort debug output
                received.append(("<unparsed>", [f"{exc}: {packet!r}"]))

    return received


def _drain_acks_nonblocking(sock: socket.socket) -> List[Tuple[str, List[OscArg]]]:
    drained: List[Tuple[str, List[OscArg]]] = []
    while True:
        try:
            packet, _addr = sock.recvfrom(65535)
        except BlockingIOError:
            break
        except OSError:
            break
        try:
            address, args = decode_osc_message(packet)
            drained.append((address, args))
        except Exception as exc:  # noqa: BLE001
            drained.append(("<unparsed>", [f"{exc}: {packet!r}"]))
    return drained


def _collect_and_print_acks(
    ack_sock: socket.socket,
    timeout_s: float,
    durations_ms: List[float],
    ack_counts: List[int],
) -> None:
    t0 = time.perf_counter()
    acks = wait_for_acks(ack_sock, timeout_s)
    durations_ms.append((time.perf_counter() - t0) * 1000.0)
    ack_counts.append(len(acks))

    if not acks:
        print(
            "ack:  (none received; bridge may not be loaded yet)",
            file=sys.stderr,
        )
        return

    for address, args in acks:
        for line in summarize_ack(address, args):
            print(line)


def send_commands(cfg: BridgeConfig, commands: Sequence[OscCommand]) -> SendMetrics:
    delay_s = cfg.delay_ms / 1000.0

    if cfg.dry_run:
        print(f"Target: udp://{cfg.host}:{cfg.port}")
        for cmd in commands:
            print(f"-> {describe_command(cmd)}")
        return SendMetrics(
            command_count=len(commands),
            send_durations_ms=(),
            ack_wait_durations_ms=(),
            acks_per_command=(),
            elapsed_ms=0.0,
        )

    ack_sock = open_ack_socket(cfg)
    send_durations_ms: List[float] = []
    ack_wait_durations_ms: List[float] = []
    ack_counts: List[int] = []
    flush_pending = 0

    t_all = time.perf_counter()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        print(f"Target: udp://{cfg.host}:{cfg.port}")
        if ack_sock is not None:
            print(f"Ack:    udp://{cfg.host}:{cfg.ack_port} (timeout {cfg.ack_timeout_s:.2f}s)")

        for idx, cmd in enumerate(commands):
            if ack_sock is not None:
                _drain_acks_nonblocking(ack_sock)

            payload = encode_osc_message(cmd.address, cmd.args)
            t_send = time.perf_counter()
            sock.sendto(payload, (cfg.host, cfg.port))
            send_durations_ms.append((time.perf_counter() - t_send) * 1000.0)
            print(f"sent: {describe_command(cmd)}")

            if ack_sock is not None:
                if cfg.ack_mode == "per_command":
                    _collect_and_print_acks(
                        ack_sock,
                        cfg.ack_timeout_s,
                        ack_wait_durations_ms,
                        ack_counts,
                    )
                else:
                    flush_pending += 1
                    should_flush = False
                    if cfg.ack_mode == "flush_end":
                        should_flush = idx == len(commands) - 1
                    elif cfg.ack_mode == "flush_interval":
                        should_flush = (
                            flush_pending >= max(1, int(cfg.ack_flush_interval))
                            or idx == len(commands) - 1
                        )

                    if should_flush:
                        _collect_and_print_acks(
                            ack_sock,
                            cfg.ack_timeout_s,
                            ack_wait_durations_ms,
                            ack_counts,
                        )
                        flush_pending = 0

            if delay_s > 0 and idx < len(commands) - 1:
                time.sleep(delay_s)

    if ack_sock is not None:
        ack_sock.close()

    metrics = SendMetrics(
        command_count=len(commands),
        send_durations_ms=tuple(send_durations_ms),
        ack_wait_durations_ms=tuple(ack_wait_durations_ms),
        acks_per_command=tuple(ack_counts),
        elapsed_ms=(time.perf_counter() - t_all) * 1000.0,
    )

    if cfg.report_metrics:
        for line in _summarize_metrics(metrics):
            print(line)

    return metrics


def main(argv: Iterable[str]) -> int:
    cfg = parse_args(argv)
    commands = build_commands(cfg)

    if not commands:
        print("No commands to send. Use --help for options.", file=sys.stderr)
        return 2

    try:
        send_commands(cfg, commands)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level CLI error handler
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
