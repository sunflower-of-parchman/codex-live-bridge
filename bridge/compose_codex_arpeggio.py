#!/usr/bin/env python3
"""Compose a demo arpeggio clip in Ableton Live via the UDP bridge."""

from __future__ import annotations

import json
import socket
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import ableton_udp_bridge as bridge


HOST = bridge.DEFAULT_HOST
PORT = bridge.DEFAULT_PORT
ACK_PORT = bridge.DEFAULT_ACK_PORT
ACK_TIMEOUT_S = 1.0


OscAck = Tuple[str, List[bridge.OscArg]]


def _format_ack(ack: OscAck) -> str:
    address, args = ack
    if not args:
        return address
    return address + " " + " ".join(bridge.format_arg(a) for a in args)


@dataclass(frozen=True)
class Status:
    total_tracks: int
    midi_tracks: int
    audio_tracks: int
    return_tracks: int


def _send_and_collect_acks(
    sock: socket.socket,
    ack_sock: socket.socket,
    command: bridge.OscCommand,
    timeout_s: float = ACK_TIMEOUT_S,
) -> List[OscAck]:
    payload = bridge.encode_osc_message(command.address, command.args)
    sock.sendto(payload, (HOST, PORT))
    acks = bridge.wait_for_acks(ack_sock, timeout_s)
    return acks


def _extract_status(acks: Sequence[OscAck]) -> Status | None:
    for address, args in acks:
        if address != "/ack" or not args:
            continue
        if args[0] != "status":
            continue
        # /ack status <total> <midi> <audio> <returns> <path> <id>
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


def _build_chord_pitches(root: int, intervals: Iterable[int], octaves: int = 3) -> List[int]:
    pitches: List[int] = []
    for octave in range(octaves):
        base = root + 12 * octave
        for interval in intervals:
            pitches.append(base + interval)
    pitches.sort()
    return pitches


def _generate_arpeggio_notes() -> Tuple[List[dict], float]:
    """Generate an 8th-note arpeggio over ~1 minute in beat units."""
    bar_length = 5.0
    step = 0.5  # eighth notes in beat units
    steps_per_bar = int(bar_length / step)  # 10 eighth notes per bar

    # Keep the demo clip length tempo-agnostic by working only in beats.
    bars = 24
    clip_length = bar_length * bars

    chords = [
        _build_chord_pitches(52, [0, 3, 7]),  # E minor, root E3
        _build_chord_pitches(48, [0, 4, 7, 11]),  # Cmaj7, root C3
        _build_chord_pitches(45, [0, 3, 7]),  # A minor, root A2
    ]

    notes: List[dict] = []
    for bar_index in range(bars):
        chord = chords[bar_index % len(chords)]
        bar_start = bar_index * bar_length
        for step_index in range(steps_per_bar):
            start_time = bar_start + step_index * step
            pitch = chord[step_index % len(chord)]
            notes.append(
                {
                    "pitch": pitch,
                    "start_time": start_time,
                    "duration": step,
                    "velocity": 104,
                    "mute": 0,
                }
            )

    return notes, clip_length


def _chunk_notes(notes: Sequence[dict], chunk_size: int) -> List[List[dict]]:
    if chunk_size <= 0:
        return [list(notes)]
    return [list(notes[i : i + chunk_size]) for i in range(0, len(notes), chunk_size)]


def run() -> int:
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

        # Prime the bridge and read current status.
        for cmd in (bridge.OscCommand("/ping"), bridge.OscCommand("/status")):
            print(f"sent: {_format_ack((cmd.address, list(cmd.args)))}")
            acks = _send_and_collect_acks(sock, ack_sock, cmd)
            if not acks:
                print("ack:  (none received; bridge may not be loaded yet)")
            else:
                for ack in acks:
                    print(f"ack:  {_format_ack(ack)}")

        status_acks = _send_and_collect_acks(sock, ack_sock, bridge.OscCommand("/status"))
        status = _extract_status(status_acks)
        if status is None:
            print("error: did not receive /ack status; reload the device in Live", file=sys.stderr)
            ack_sock.close()
            return 2

        new_track_index = status.total_tracks
        print(f"info: status total_tracks={status.total_tracks}; new_track_index={new_track_index}")

        # Create the new MIDI track at the end.
        add_track = bridge.OscCommand("/add_midi_tracks", (1, "Codex Arp"))
        print(f"sent: {bridge.describe_command(add_track)}")
        acks = _send_and_collect_acks(sock, ack_sock, add_track)
        if not acks:
            print("ack:  (none received; bridge may not be loaded yet)")
        else:
            for ack in acks:
                print(f"ack:  {_format_ack(ack)}")

        notes, clip_length = _generate_arpeggio_notes()
        note_chunks = _chunk_notes(notes, chunk_size=60)

        first_notes_json = json.dumps({"notes": note_chunks[0]}, separators=(",", ":"))
        clip_cmd = bridge.OscCommand(
            "/set_session_clip_notes",
            (new_track_index, 0, clip_length, first_notes_json, "Codex Arp"),
        )
        print(
            "sent: /set_session_clip_notes "
            f"{new_track_index} 0 {clip_length} <notes_json chunk 1/{len(note_chunks)}> Codex Arp"
        )
        acks = _send_and_collect_acks(sock, ack_sock, clip_cmd, timeout_s=1.5)
        if not acks:
            print(
                "ack:  (none received; reload the LiveUdpBridge device and try again)",
                file=sys.stderr,
            )
        else:
            for ack in acks:
                print(f"ack:  {_format_ack(ack)}")

        for idx, chunk in enumerate(note_chunks[1:], start=2):
            chunk_json = json.dumps({"notes": chunk}, separators=(",", ":"))
            append_cmd = bridge.OscCommand(
                "/append_session_clip_notes",
                (new_track_index, 0, chunk_json),
            )
            print(
                "sent: /append_session_clip_notes "
                f"{new_track_index} 0 <notes_json chunk {idx}/{len(note_chunks)}>"
            )
            acks = _send_and_collect_acks(sock, ack_sock, append_cmd, timeout_s=1.5)
            if not acks:
                print(
                    "ack:  (none received for append; device may need reload)",
                    file=sys.stderr,
                )
            else:
                for ack in acks:
                    print(f"ack:  {_format_ack(ack)}")

        inspect_cmd = bridge.OscCommand("/inspect_session_clip_notes", (new_track_index, 0))
        print(f"sent: {bridge.describe_command(inspect_cmd)}")
        acks = _send_and_collect_acks(sock, ack_sock, inspect_cmd, timeout_s=1.5)
        if not acks:
            print("ack:  (none received for inspect; device may need reload)", file=sys.stderr)
        else:
            for ack in acks:
                print(f"ack:  {_format_ack(ack)}")

        # Final status check.
        final_status_acks = _send_and_collect_acks(sock, ack_sock, bridge.OscCommand("/status"))
        final_status = _extract_status(final_status_acks)
        if final_status is not None:
            print(
                "info: final status total_tracks="
                f"{final_status.total_tracks} midi={final_status.midi_tracks} "
                f"audio={final_status.audio_tracks} returns={final_status.return_tracks}"
            )
        else:
            for ack in final_status_acks:
                print(f"ack:  {_format_ack(ack)}")

    ack_sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
