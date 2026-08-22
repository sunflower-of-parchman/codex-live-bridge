#!/usr/bin/env python3
"""Full-surface smoke test for the Ableton Live UDP bridge."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import ableton_udp_bridge as bridge


HOST = bridge.DEFAULT_HOST
PORT = bridge.DEFAULT_PORT
ACK_PORT = bridge.DEFAULT_ACK_PORT
ACK_TIMEOUT_S = 1.0
MUTATING_FLAG = "--i-understand-this-mutates-live-set"

OscAck = Tuple[str, List[bridge.OscArg]]


def _send_and_collect_acks(
    sock: socket.socket,
    ack_sock: socket.socket,
    command: bridge.OscCommand,
    timeout_s: float = ACK_TIMEOUT_S,
    expected_request_id: str | None = None,
) -> List[OscAck]:
    payload = bridge.encode_osc_message(command.address, command.args)
    sock.sendto(payload, (HOST, PORT))
    return bridge.wait_for_acks(
        ack_sock,
        timeout_s,
        expected_sender_host=HOST,
        expected_request_id=expected_request_id,
    )


def _print_acks(acks: Sequence[OscAck]) -> None:
    if not acks:
        print("ack:  (none received; bridge may not be loaded yet)")
        return
    for address, args in acks:
        for line in bridge.summarize_ack(address, args):
            print(line)


@dataclass(frozen=True)
class Status:
    total_tracks: int
    midi_tracks: int
    audio_tracks: int
    return_tracks: int


def _extract_status(acks: Sequence[OscAck]) -> Status | None:
    for address, args in acks:
        if address != "/ack" or not args:
            continue
        if args[0] != "status":
            continue
        if len(args) < 5:
            continue
        try:
            return Status(
                total_tracks=int(args[1]),
                midi_tracks=int(args[2]),
                audio_tracks=int(args[3]),
                return_tracks=int(args[4]),
            )
        except (TypeError, ValueError):
            continue
    return None


def _extract_api_children(
    acks: Sequence[OscAck],
    request_id: str,
) -> List[dict]:
    for address, args in acks:
        event = bridge.parse_ack_event(address, args)
        if event.address != "/ack" or event.event != "api_children":
            continue
        if event.request_id != request_id:
            continue
        if (
            event.payload.get("path") != "live_set"
            or event.payload.get("child_name") != "tracks"
        ):
            continue
        children = event.payload.get("children")
        if not isinstance(children, list):
            continue
        if not all(isinstance(child, dict) for child in children):
            continue
        return children
    return []


def _smoke_request_id(label: str) -> str:
    return f"smoke-{label}-{secrets.token_hex(16)}"


def _extract_note_count(acks: Sequence[OscAck]) -> int | None:
    for address, args in acks:
        if address != "/ack" or len(args) < 4:
            continue
        if args[0] != "inspect_session_clip_notes":
            continue
        try:
            return int(args[3])
        except (TypeError, ValueError):
            return None
    return None


def _new_track_index(tracks_before: Sequence[dict], tracks_after: Sequence[dict]) -> int | None:
    if len(tracks_after) != len(tracks_before) + 1:
        return None
    expected_index = len(tracks_after) - 1
    try:
        actual_index = int(tracks_after[-1]["index"])
    except (KeyError, TypeError, ValueError):
        return None
    return expected_index if actual_index == expected_index else None


def _build_notes() -> Tuple[List[dict], float]:
    """Return a small, deterministic MIDI pattern."""
    length_beats = 8.0
    step = 0.5
    pitches = [60, 64, 67, 71]
    notes: List[dict] = []
    t = 0.0
    idx = 0
    while t < length_beats:
        notes.append(
            {
                "pitch": pitches[idx % len(pitches)],
                "start_time": t,
                "duration": step,
                "velocity": 100,
                "mute": 0,
            }
        )
        t += step
        idx += 1
    return notes, length_beats


def run(auth_token: str) -> int:
    cfg = bridge.BridgeConfig(
        host=HOST,
        port=PORT,
        ack_port=ACK_PORT,
        ack_timeout_s=ACK_TIMEOUT_S,
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
        auth_token=auth_token,
        delay_ms=0,
        dry_run=False,
    )

    ack_sock = bridge.open_ack_socket(cfg)
    if ack_sock is None:
        print("error: failed to open ack socket", file=sys.stderr)
        return 1

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        print(f"Target: udp://{HOST}:{PORT}")
        print(f"Ack:    udp://{HOST}:{ACK_PORT} (timeout {ACK_TIMEOUT_S:.2f}s)")

        # Prime the bridge using both legacy and API surfaces.
        for cmd in (
            bridge.OscCommand("/ping"),
            bridge.OscCommand("/api/describe", ("live_set",)),
            bridge.OscCommand("/api/children", ("live_set", "tracks")),
        ):
            print(f"sent: {bridge.describe_command(cmd)}")
            _print_acks(_send_and_collect_acks(sock, ack_sock, cmd))

        tracks_before_request_id = _smoke_request_id("tracks-before")
        tracks_before = _extract_api_children(
            _send_and_collect_acks(
                sock,
                ack_sock,
                bridge.OscCommand(
                    "/api/children",
                    ("live_set", "tracks", tracks_before_request_id),
                ),
                expected_request_id=tracks_before_request_id,
            ),
            tracks_before_request_id,
        )
        if not tracks_before:
            print("error: did not receive api_children tracks; reload the device in Live", file=sys.stderr)
            ack_sock.close()
            return 2

        track_count_before = len(tracks_before)
        print(f"info: tracks before={track_count_before}")

        # Create a new MIDI track using the generic RPC surface.
        create_track = bridge.OscCommand(
            "/api/call",
            bridge.authenticated_args(
                auth_token,
                ("live_set", "create_midi_track", json.dumps([-1])),
            ),
        )
        print(f"sent: {bridge.describe_command(create_track)}")
        _print_acks(_send_and_collect_acks(sock, ack_sock, create_track))

        tracks_after_request_id = _smoke_request_id("tracks-after")
        tracks_after = _extract_api_children(
            _send_and_collect_acks(
                sock,
                ack_sock,
                bridge.OscCommand(
                    "/api/children",
                    ("live_set", "tracks", tracks_after_request_id),
                ),
                expected_request_id=tracks_after_request_id,
            ),
            tracks_after_request_id,
        )
        track_count_after = len(tracks_after)
        new_track_index = _new_track_index(tracks_before, tracks_after)
        if new_track_index is None:
            print(
                "error: MIDI track creation did not add exactly one appended track; aborting before writes",
                file=sys.stderr,
            )
            ack_sock.close()
            return 3
        print(f"info: tracks after={track_count_after}; new_track_index={new_track_index}")

        track_path = f"live_set tracks {new_track_index}"
        set_name = bridge.OscCommand(
            "/api/set",
            bridge.authenticated_args(
                auth_token,
                (track_path, "name", json.dumps("Full Surface Smoke")),
            ),
        )
        print(f"sent: {bridge.describe_command(set_name)}")
        _print_acks(_send_and_collect_acks(sock, ack_sock, set_name))

        for prop, value in (
            ("tempo", 142.0),
            ("signature_numerator", 5),
            ("signature_denominator", 4),
        ):
            cmd = bridge.OscCommand(
                "/api/set",
                bridge.authenticated_args(
                    auth_token,
                    ("live_set", prop, json.dumps(value)),
                ),
            )
            print(f"sent: {bridge.describe_command(cmd)}")
            _print_acks(_send_and_collect_acks(sock, ack_sock, cmd))

        notes, clip_length = _build_notes()
        notes_json = json.dumps({"notes": notes}, separators=(",", ":"))
        set_clip = bridge.OscCommand(
            "/set_session_clip_notes",
            bridge.authenticated_args(
                auth_token,
                (
                    new_track_index,
                    0,
                    clip_length,
                    notes_json,
                    "Full Surface Smoke",
                ),
            ),
        )
        print(
            "sent: /set_session_clip_notes "
            f"{new_track_index} 0 {clip_length} <notes_json> Full Surface Smoke"
        )
        _print_acks(_send_and_collect_acks(sock, ack_sock, set_clip, timeout_s=1.5))

        inspect = bridge.OscCommand("/inspect_session_clip_notes", (new_track_index, 0))
        print(f"sent: {bridge.describe_command(inspect)}")
        inspect_acks = _send_and_collect_acks(sock, ack_sock, inspect, timeout_s=1.5)
        _print_acks(inspect_acks)
        note_count = _extract_note_count(inspect_acks)
        if note_count is None:
            print("error: did not receive the final clip note inspection", file=sys.stderr)
            ack_sock.close()
            return 4
        if note_count != len(notes):
            print(
                "error: final clip note count did not match the written pattern "
                f"(expected {len(notes)}, received {note_count})",
                file=sys.stderr,
            )
            ack_sock.close()
            return 4
        print(f"info: final note_count={note_count}")

        final_status_acks = _send_and_collect_acks(sock, ack_sock, bridge.OscCommand("/status"))
        final_status = _extract_status(final_status_acks)
        if final_status is None:
            _print_acks(final_status_acks)
            print("error: did not receive the final bridge status", file=sys.stderr)
            ack_sock.close()
            return 5
        print(
            "info: final status total_tracks="
            f"{final_status.total_tracks} midi={final_status.midi_tracks} "
            f"audio={final_status.audio_tracks} returns={final_status.return_tracks}"
        )

    ack_sock.close()
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full-surface Ableton bridge smoke test. This creates a MIDI "
            "track, writes a clip, and changes the set tempo and time signature."
        )
    )
    parser.add_argument(
        MUTATING_FLAG,
        action="store_true",
        required=True,
        help="Required confirmation that this smoke test mutates the active Live set.",
    )
    parser.add_argument(
        "--auth-token",
        default=os.environ.get(bridge.AUTH_TOKEN_ENV),
        required=os.environ.get(bridge.AUTH_TOKEN_ENV) is None,
        help=(
            "Capability token configured in the Max device "
            f"(default: {bridge.AUTH_TOKEN_ENV} environment variable)"
        ),
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        ns = parse_args(sys.argv[1:] if argv is None else argv)
        auth_token = bridge.normalize_auth_token(ns.auth_token)
    except SystemExit as exc:
        return int(exc.code or 0)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if auth_token is None:
        print("error: auth token is required", file=sys.stderr)
        return 2
    return run(auth_token)


if __name__ == "__main__":
    raise SystemExit(main())
