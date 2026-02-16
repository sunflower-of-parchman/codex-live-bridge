#!/usr/bin/env python3
"""Compose a rim pattern into the Arrangement view via the UDP bridge."""

from __future__ import annotations

import argparse
import socket
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import ableton_udp_bridge as bridge
import compose_kick_pattern as kick


HOST = bridge.DEFAULT_HOST
PORT = bridge.DEFAULT_PORT
ACK_PORT = bridge.DEFAULT_ACK_PORT

DEFAULT_RIM_TRACK_NAME = "RIM"
DEFAULT_RIM_PITCH = 37
DEFAULT_RIM_CLIP_NAME = "RIM 1min"
DEFAULT_GROOVE_NAME = kick.DEFAULT_GROOVE_NAME


def _beats_in_bar(beats_per_bar: float, beat_step: float) -> int:
    beats = int(round(float(beats_per_bar) / float(beat_step)))
    return max(1, beats)


def _anchor_candidates(beats_in_bar: int) -> List[int]:
    """Prefer beat 2 and beat 4 where they exist."""
    candidates: List[int] = []
    if beats_in_bar >= 2:
        candidates.append(1)  # beat 2
    if beats_in_bar >= 4:
        candidates.append(3)  # beat 4
    if not candidates:
        candidates.append(0)  # fall back to beat 1
    return candidates


def build_rim_notes(
    bars: int,
    beats_per_bar: float = 4.0,
    beat_step: float = 1.0,
    pitch: int = DEFAULT_RIM_PITCH,
    velocity: int = 100,
    duration: float = 0.25,
) -> Tuple[List[dict], float]:
    """Generate a rim pattern that marks time without mirroring the kick."""
    if bars <= 0:
        raise ValueError("bars must be > 0")
    if beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be > 0")
    if beat_step <= 0:
        raise ValueError("beat_step must be > 0")

    clip_length = float(bars) * float(beats_per_bar)
    beats_in_bar = _beats_in_bar(beats_per_bar, beat_step)
    anchor_candidates = _anchor_candidates(beats_in_bar)

    eighth_step = float(beat_step) / 2.0
    sixteenth_step = float(beat_step) / 4.0

    base_velocity = kick._clamp_velocity(velocity)
    velocity_offset = base_velocity - 100

    anchor_min, anchor_max = kick._clamp_velocity_range(98 + velocity_offset, 112 + velocity_offset)
    beat_min, beat_max = kick._clamp_velocity_range(92 + velocity_offset, 106 + velocity_offset)
    sync_min, sync_max = kick._clamp_velocity_range(88 + velocity_offset, 100 + velocity_offset)

    sync_patterns: List[List[float]] = [
        [-eighth_step],
        [-sixteenth_step],
        [-eighth_step, -sixteenth_step],
    ]

    notes: List[dict] = []

    for bar_index in range(bars):
        bar_start = float(bar_index) * float(beats_per_bar)
        bar_end = bar_start + float(beats_per_bar)

        anchor_beat_index = anchor_candidates[bar_index % len(anchor_candidates)]
        anchor_time = bar_start + float(anchor_beat_index) * float(beat_step)

        times: List[float] = [anchor_time]

        # At phrase boundaries, add a simple syncopated pickup into the anchor.
        if (bar_index + 1) % 4 == 0:
            pattern = sync_patterns[(bar_index // 4) % len(sync_patterns)]
            for delta in pattern:
                sync_time = anchor_time + float(delta)
                if bar_start <= sync_time < bar_end:
                    times.append(sync_time)
        # Light additional movement every so often, but not every bar.
        elif (bar_index + 2) % 8 == 0:
            sync_time = anchor_time - eighth_step
            if bar_start <= sync_time < bar_end:
                times.append(sync_time)

        unique_times = sorted({round(t, 6) for t in times})

        for start_time in unique_times:
            if start_time >= clip_length:
                continue
            on_beat, beat_index = kick._classify_time(start_time, bar_start, float(beat_step))
            is_anchor = on_beat and beat_index == anchor_beat_index

            seed = kick._velocity_seed(
                bar_index=bar_index,
                beat_index=beat_index,
                variant_index=int(round(float(start_time) * 1000.0)),
            )

            if is_anchor:
                note_velocity = kick._velocity_in_range(anchor_min, anchor_max, seed)
            elif on_beat:
                note_velocity = kick._velocity_in_range(beat_min, beat_max, seed)
            else:
                note_velocity = kick._velocity_in_range(sync_min, sync_max, seed)

            notes.append(
                {
                    "pitch": int(pitch),
                    "start_time": float(start_time),
                    "duration": float(duration),
                    "velocity": int(note_velocity),
                    "mute": 0,
                }
            )

    notes.sort(key=lambda n: (n["start_time"], n["pitch"]))
    return notes, clip_length


@dataclass(frozen=True)
class RimConfig:
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


def parse_args(argv: Iterable[str]) -> RimConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=32, help="Arrangement length in bars")
    parser.add_argument(
        "--start-beats",
        type=float,
        default=0.0,
        help="Arrangement start time in beats (default: 0)",
    )
    parser.add_argument("--track-name", default=DEFAULT_RIM_TRACK_NAME, help="Name of the MIDI track")
    parser.add_argument("--clip-name", default=DEFAULT_RIM_CLIP_NAME, help="Name of the arrangement clip")
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
    parser.add_argument("--pitch", type=int, default=DEFAULT_RIM_PITCH, help="Rim MIDI note (default: 37)")
    parser.add_argument(
        "--velocity",
        type=int,
        default=100,
        help="Rim velocity center (default: 100)",
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

    return RimConfig(
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


def run(cfg: RimConfig) -> int:
    if cfg.dry_run:
        beats_per_bar = cfg.beats_per_bar_override if cfg.beats_per_bar_override else 4.0
        beat_step = 1.0
        notes, clip_length = build_rim_notes(
            bars=cfg.bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            pitch=cfg.pitch,
            velocity=cfg.velocity,
            duration=cfg.duration,
        )
        clip_start = float(cfg.start_beats)
        clip_end = clip_start + clip_length
        print("RIM pattern plan (dry run):")
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

        for cmd in (bridge.OscCommand("/ping"), bridge.OscCommand("/status")):
            print(f"sent: {bridge.describe_command(cmd)}")
            kick._print_acks(kick._send_and_collect_acks(sock, ack_sock, cmd, cfg.ack_timeout_s))

        tracks_before = kick._get_children(
            sock,
            ack_sock,
            "live_set",
            "tracks",
            "rim-tracks-before",
            cfg.ack_timeout_s,
        )
        if not tracks_before:
            print("error: could not read tracks; reload the device in Live", file=sys.stderr)
            ack_sock.close()
            return 2

        tempo_value = kick._api_get(sock, ack_sock, "live_set", "tempo", "rim-tempo", cfg.ack_timeout_s)
        sig_num_value = kick._api_get(
            sock,
            ack_sock,
            "live_set",
            "signature_numerator",
            "rim-sig-num",
            cfg.ack_timeout_s,
        )
        sig_den_value = kick._api_get(
            sock,
            ack_sock,
            "live_set",
            "signature_denominator",
            "rim-sig-den",
            cfg.ack_timeout_s,
        )

        tempo = kick._as_float(tempo_value)
        signature_numerator = kick._as_int(sig_num_value) or 4
        signature_denominator = kick._as_int(sig_den_value) or 4
        beats_per_bar_from_sig = kick._beats_per_bar_from_signature(signature_numerator, signature_denominator)
        beat_step_from_sig = kick._beat_step_from_denominator(signature_denominator)
        beats_per_bar = (
            cfg.beats_per_bar_override if cfg.beats_per_bar_override is not None else beats_per_bar_from_sig
        )
        beat_step = beat_step_from_sig

        notes, clip_length = build_rim_notes(
            bars=cfg.bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            pitch=cfg.pitch,
            velocity=cfg.velocity,
            duration=cfg.duration,
        )

        clip_start = float(cfg.start_beats)
        clip_end = clip_start + clip_length

        print("\nRIM pattern plan:")
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

        existing_track_index = kick._find_track_index_by_name(tracks_before, cfg.track_name)
        if existing_track_index is None:
            kick._api_call(
                sock,
                ack_sock,
                "live_set",
                "create_midi_track",
                [-1],
                "rim-create-track",
                cfg.ack_timeout_s,
            )

            tracks_after = kick._get_children(
                sock,
                ack_sock,
                "live_set",
                "tracks",
                "rim-tracks-after",
                cfg.ack_timeout_s,
            )
            if len(tracks_after) <= len(tracks_before):
                print("error: MIDI track was not created; reload the device in Live", file=sys.stderr)
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

        kick._api_set(sock, ack_sock, track_path, "name", cfg.track_name, "rim-track-name", cfg.ack_timeout_s)

        if cfg.insert_device and cfg.device_name.strip():
            devices = kick._get_children(sock, ack_sock, track_path, "devices", "rim-devices", cfg.ack_timeout_s)
            if track_was_created or not devices:
                kick._api_call(
                    sock,
                    ack_sock,
                    track_path,
                    "insert_device",
                    [cfg.device_name],
                    "rim-insert-device",
                    cfg.ack_timeout_s,
                )
            else:
                print(
                    f"info: devices already present on track {track_index}; "
                    "skipping device insertion"
                )

        arrangement_initial = kick._get_children(
            sock,
            ack_sock,
            track_path,
            "arrangement_clips",
            "rim-arr-initial",
            cfg.ack_timeout_s,
        )

        deleted_overlaps = 0
        for idx, clip_info in enumerate(arrangement_initial):
            clip_path_raw = clip_info.get("path")
            if not clip_path_raw:
                continue
            clip_path = kick._sanitize_live_path(str(clip_path_raw))
            clip_start_existing = kick._as_float(
                kick._api_get(sock, ack_sock, clip_path, "start_time", f"rim-clip-start-{idx}", cfg.ack_timeout_s)
            )
            clip_end_existing = kick._as_float(
                kick._api_get(sock, ack_sock, clip_path, "end_time", f"rim-clip-end-{idx}", cfg.ack_timeout_s)
            )
            if clip_start_existing is None or clip_end_existing is None:
                continue
            if not kick._overlaps(clip_start_existing, clip_end_existing, clip_start, clip_end):
                continue

            clip_desc = kick._api_describe(sock, ack_sock, clip_path, f"rim-clip-describe-{idx}", cfg.ack_timeout_s)
            clip_id = kick._as_int(clip_desc.get("id") if isinstance(clip_desc, dict) else None)
            if clip_id is None or clip_id <= 0:
                clip_id = kick._as_int(
                    kick._api_get(sock, ack_sock, clip_path, "id", f"rim-clip-id-{idx}", cfg.ack_timeout_s)
                )
            if clip_id is None or clip_id <= 0:
                print(f"warning: could not resolve clip id for {clip_path}; skipping delete")
                continue

            kick._api_call(
                sock,
                ack_sock,
                track_path,
                "delete_clip",
                [clip_id],
                f"rim-delete-clip-{idx}",
                cfg.ack_timeout_s,
            )
            deleted_overlaps += 1

        if deleted_overlaps > 0:
            print(f"info: deleted {deleted_overlaps} overlapping arrangement clip(s)")

        arrangement_before = kick._get_children(
            sock,
            ack_sock,
            track_path,
            "arrangement_clips",
            "rim-arr-before",
            cfg.ack_timeout_s,
        )

        create_result = kick._api_call(
            sock,
            ack_sock,
            track_path,
            "create_midi_clip",
            [clip_start, clip_length],
            "rim-create-clip",
            cfg.ack_timeout_s,
        )

        clip_id = kick._extract_id_from_call_result(create_result)
        if clip_id is not None and clip_id > 0:
            clip_path = f"id {clip_id}"
            print(f"info: arrangement clip id={clip_id} (using path '{clip_path}')")
        else:
            arrangement_after = kick._get_children(
                sock,
                ack_sock,
                track_path,
                "arrangement_clips",
                "rim-arr-after",
                cfg.ack_timeout_s,
            )
            clip_path = kick._new_child_path(arrangement_before, arrangement_after)
        if not clip_path:
            print(
                "error: could not identify the new arrangement clip; reload the device in Live",
                file=sys.stderr,
            )
            ack_sock.close()
            return 4

        print(f"info: arrangement clip path={clip_path}")

        kick._api_set(sock, ack_sock, clip_path, "name", cfg.clip_name, "rim-clip-name", cfg.ack_timeout_s)
        kick._api_set(sock, ack_sock, clip_path, "loop_start", 0.0, "rim-loop-start", cfg.ack_timeout_s)
        kick._api_set(sock, ack_sock, clip_path, "loop_end", clip_length, "rim-loop-end", cfg.ack_timeout_s)

        note_chunks = kick._chunk_notes(notes, chunk_size=48)
        for idx, chunk in enumerate(note_chunks, start=1):
            notes_json = {"notes": chunk}
            kick._api_call(
                sock,
                ack_sock,
                clip_path,
                "add_new_notes",
                notes_json,
                f"rim-add-notes-{idx}-of-{len(note_chunks)}",
                max(cfg.ack_timeout_s, 2.0),
            )

        groove_id = kick._find_groove_id_by_name(sock, ack_sock, cfg.groove_name, cfg.ack_timeout_s)
        if groove_id:
            kick._api_set(
                sock,
                ack_sock,
                "live_set",
                "groove_amount",
                1.0,
                "rim-groove-amount",
                cfg.ack_timeout_s,
            )
            kick._api_set(
                sock,
                ack_sock,
                clip_path,
                "groove",
                ["id", groove_id],
                "rim-clip-groove",
                cfg.ack_timeout_s,
            )
            print(f"info: applied groove '{cfg.groove_name}' (id={groove_id})")
        else:
            print(f"warning: groove '{cfg.groove_name}' not found in groove pool")

        note_dump = kick._api_call(
            sock,
            ack_sock,
            clip_path,
            "get_all_notes_extended",
            [],
            "rim-inspect",
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
