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
    expected_event, command_request_id = bridge._command_ack_expectation(command)
    if expected_request_id is not None and expected_request_id != command_request_id:
        raise bridge.BridgeAcknowledgementError("smoke command request ID does not match")
    bridge._drain_acks_nonblocking(ack_sock)
    payload = bridge.encode_osc_message(command.address, command.args)
    sock.sendto(payload, (HOST, PORT))
    return bridge.wait_for_acks(
        ack_sock,
        timeout_s,
        expected_sender_host=HOST,
        expected_request_id=command_request_id,
        expected_event=expected_event,
        expected_commands=(command,),
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
        if not all(bridge._ack_integer(value) for value in args[1:5]):
            continue
        if args[2] + args[3] > args[1]:
            continue
        return Status(*(int(value) for value in args[1:5]))
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
            bridge._normalized_live_path(event.payload.get("path")) != "live_set"
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


def _track_identities(tracks: Sequence[dict]) -> list[int] | None:
    ids: list[int] = []
    for index, track in enumerate(tracks):
        if (
            not isinstance(track, dict)
            or not bridge._ack_integer(track.get("index")) or track["index"] != index
            or not bridge._ack_integer(track.get("id"), minimum=1)
            or bridge._normalized_live_path(track.get("path")) != f"live_set tracks {index}"
        ):
            return None
        ids.append(int(track["id"]))
    return ids if len(ids) == len(set(ids)) else None


def _new_track_index(tracks_before: Sequence[dict], tracks_after: Sequence[dict]) -> int | None:
    if len(tracks_after) != len(tracks_before) + 1:
        return None
    before_ids = _track_identities(tracks_before)
    after_ids = _track_identities(tracks_after)
    if before_ids is None or after_ids is None or before_ids != after_ids[:-1]:
        return None
    return len(tracks_after) - 1


def _notes_match_pattern(actual: object, expected: Sequence[dict]) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    fields = ("pitch", "start_time", "duration", "velocity", "mute")
    normalized: list[tuple[object, ...]] = []
    for note in actual:
        if not isinstance(note, dict) or any(field not in note for field in fields):
            return False
        if not all(bridge._ack_number(note[field]) for field in fields[:-1]):
            return False
        if note["mute"] not in (0, 1, False, True):
            return False
        normalized.append(tuple(note[field] for field in fields))
    return sorted(normalized) == sorted(tuple(note[field] for field in fields) for note in expected)


def _scalar_property(value: object, property_name: str) -> object:
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        if len(value) == 2 and value[0] == property_name:
            return value[1]
    return value


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

    failure_code = 1
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            print(f"Target: udp://{HOST}:{PORT}")
            print(f"Ack:    udp://{HOST}:{ACK_PORT} (timeout {ACK_TIMEOUT_S:.2f}s)")

            def execute(command: bridge.OscCommand, timeout_s: float = ACK_TIMEOUT_S) -> List[OscAck]:
                acks = _send_and_collect_acks(sock, ack_sock, command, timeout_s=timeout_s)
                print(f"sent: {bridge.describe_command(command)}")
                bridge.validate_command_acks(command, acks)
                _print_acks(acks)
                return acks

            def require(condition: bool, message: str) -> None:
                if not condition:
                    raise bridge.BridgeAcknowledgementError(message)

            def tracks_snapshot(label: str) -> List[dict]:
                request_id = _smoke_request_id(label)
                command = bridge.OscCommand("/api/children", ("live_set", "tracks", request_id))
                return _extract_api_children(execute(command), request_id)

            execute(bridge.OscCommand("/ping"))
            execute(bridge.OscCommand("/api/describe", ("live_set", _smoke_request_id("describe"))))
            failure_code = 2
            initial_status = _extract_status(execute(bridge.OscCommand("/status")))
            tracks_before = tracks_snapshot("tracks-before")
            require(
                initial_status is not None and bool(tracks_before)
                and initial_status.total_tracks == len(tracks_before)
                and _track_identities(tracks_before) is not None,
                "initial track identities or status could not be verified; no mutations were sent",
            )
            assert initial_status is not None

            failure_code = 3
            execute(bridge.OscCommand(
                "/api/call",
                bridge.authenticated_args(auth_token, (
                    "live_set", "create_midi_track", json.dumps([-1]), _smoke_request_id("create-track")
                )),
            ))
            tracks_after = tracks_snapshot("tracks-after")
            new_track_index = _new_track_index(tracks_before, tracks_after)
            require(new_track_index is not None, "MIDI track creation did not preserve the existing tracks and append exactly one new track")
            assert new_track_index is not None
            print(f"info: tracks before={len(tracks_before)} after={len(tracks_after)}; new_track_index={new_track_index}")

            failure_code = 4
            track_path = f"live_set tracks {new_track_index}"
            name = "Full Surface Smoke"
            properties = (
                (track_path, "name", name),
                ("live_set", "tempo", 142.0),
                ("live_set", "signature_numerator", 5),
                ("live_set", "signature_denominator", 4),
            )
            for target, prop, value in properties:
                execute(bridge.OscCommand(
                    "/api/set",
                    bridge.authenticated_args(auth_token, (
                        target, prop, json.dumps(value), _smoke_request_id(f"set-{prop}")
                    )),
                ))

            notes, clip_length = _build_notes()
            execute(bridge.OscCommand(
                "/set_session_clip_notes",
                bridge.authenticated_args(auth_token, (
                    new_track_index, 0, clip_length,
                    json.dumps({"notes": notes}, separators=(",", ":")), name,
                )),
            ), timeout_s=1.5)
            inspect = bridge.OscCommand("/inspect_session_clip_notes", (new_track_index, 0))
            inspect_acks = execute(inspect, timeout_s=1.5)
            inspected = bridge.validate_command_acks(inspect, inspect_acks).payload["args"]
            inspected_notes = json.loads(inspected[7])["notes"]
            require(
                inspected[6] == clip_length and _notes_match_pattern(inspected_notes, notes),
                "final clip notes or length did not match the written pattern",
            )

            clip_path = f"{track_path} clip_slots 0 clip"
            for target, prop, expected_value in (*properties, (clip_path, "name", name), (clip_path, "length", clip_length)):
                command = bridge.OscCommand("/api/get", (target, prop, _smoke_request_id(f"read-{prop}")))
                event = bridge.validate_command_acks(command, execute(command))
                actual = _scalar_property(event.payload.get("value"), prop)
                require(actual == expected_value, f"readback did not match {target} {prop}")
            print(f"info: verified {len(notes)} exact notes, track/clip names, tempo, meter, and clip length")

            failure_code = 5
            final_status = _extract_status(execute(bridge.OscCommand("/status")))
            expected_status = Status(
                initial_status.total_tracks + 1, initial_status.midi_tracks + 1,
                initial_status.audio_tracks, initial_status.return_tracks,
            )
            require(final_status == expected_status, "final bridge status did not match the single appended MIDI track")
            print(f"info: final status total_tracks={final_status.total_tracks} midi={final_status.midi_tracks} audio={final_status.audio_tracks} returns={final_status.return_tracks}")
        return 0
    except (bridge.BridgeAcknowledgementError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return failure_code
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    finally:
        ack_sock.close()


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
