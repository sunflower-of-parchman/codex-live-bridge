#!/usr/bin/env python3
"""Compose a hi-hat pattern into the Arrangement view via the UDP bridge."""

from __future__ import annotations

import argparse
import socket
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import ableton_udp_bridge as bridge
import compose_kick_pattern as kick


HOST = bridge.DEFAULT_HOST
PORT = bridge.DEFAULT_PORT
ACK_PORT = bridge.DEFAULT_ACK_PORT

DEFAULT_HAT_TRACK_NAME = "HAT"
DEFAULT_HAT_PITCH = 42
DEFAULT_HAT_CLIP_NAME = "HAT 1min"
DEFAULT_GROOVE_NAME = kick.DEFAULT_GROOVE_NAME


def _beats_in_bar(beats_per_bar: float, beat_step: float) -> int:
    beats = int(round(float(beats_per_bar) / float(beat_step)))
    return max(1, beats)


def _anchor_candidates(beats_in_bar: int) -> List[int]:
    """Prefer beat 2 and beat 4 where they exist (rim-style anchors)."""
    candidates: List[int] = []
    if beats_in_bar >= 2:
        candidates.append(1)  # beat 2
    if beats_in_bar >= 4:
        candidates.append(3)  # beat 4
    if not candidates:
        candidates.append(0)
    return candidates


def _eighth_step(beat_step: float) -> float:
    return float(beat_step) / 2.0


def _steps_per_bar(beats_per_bar: float, eighth_step: float) -> int:
    steps = int(round(float(beats_per_bar) / float(eighth_step)))
    return max(1, steps)


def _step_index_for_offset(offset_beats: float, eighth_step: float) -> int:
    return max(0, int(round(float(offset_beats) / float(eighth_step))))


def _phrase_scalar(bar_index: int) -> float:
    """A gentle bar-to-bar velocity contour within 4-bar phrases."""
    phrase_position = bar_index % 4
    if phrase_position == 0:
        return 0.97
    if phrase_position == 1:
        return 1.00
    if phrase_position == 2:
        return 1.05
    return 0.99


def _accent_beat_indices(beats_in_bar: int) -> List[int]:
    """Prefer beats 2 and 4 as hat accents where available."""
    accents: List[int] = []
    if beats_in_bar >= 2:
        accents.append(1)
    if beats_in_bar >= 4:
        accents.append(3)
    return accents


def _accent_steps(accent_beats: Sequence[int], steps_per_bar: int) -> set[int]:
    return {2 * beat_index for beat_index in accent_beats if 2 * beat_index < steps_per_bar}


def _removal_count(bar_index: int, steps_per_bar: int, accent_steps: set[int]) -> int:
    """Keep the hat dense: remove at most one eighth in most bars."""
    if steps_per_bar <= len(accent_steps) + 2:
        return 0
    if (bar_index + 1) % 4 == 0:
        # Phrase boundaries already get extra space near fills.
        return 0
    seed = kick._velocity_seed(bar_index, steps_per_bar, len(accent_steps))
    # About 25% of bars keep all eighths; the rest remove one.
    return 0 if seed % 4 == 0 else 1


def _choose_removal_steps(
    bar_index: int,
    steps_per_bar: int,
    accent_steps: set[int],
    removal_count: int,
) -> set[int]:
    candidates = [step for step in range(steps_per_bar) if step not in accent_steps]
    removals: set[int] = set()
    for idx in range(removal_count):
        if not candidates:
            break
        seed = kick._velocity_seed(bar_index, idx, steps_per_bar)
        pick_index = seed % len(candidates)
        removals.add(candidates.pop(pick_index))
    return removals


def _ensure_min_density(
    active_steps: set[int],
    steps_per_bar: int,
    accent_steps: set[int],
    bar_index: int,
) -> None:
    """Ensure we keep a nearly full eighth-note backbone."""
    min_steps = max(len(accent_steps) + 2, steps_per_bar - 2)
    if len(active_steps) >= min_steps:
        return
    missing_steps = [step for step in range(steps_per_bar) if step not in active_steps]
    while len(active_steps) < min_steps and missing_steps:
        seed = kick._velocity_seed(bar_index, len(active_steps), steps_per_bar)
        pick_index = seed % len(missing_steps)
        active_steps.add(missing_steps.pop(pick_index))


def build_hat_notes(
    bars: int,
    beats_per_bar: float = 4.0,
    beat_step: float = 1.0,
    pitch: int = DEFAULT_HAT_PITCH,
    velocity: int = 88,
    duration: float = 0.25,
) -> Tuple[List[dict], float]:
    """Generate an eighth-note hat part with gaps and light syncopation."""
    if bars <= 0:
        raise ValueError("bars must be > 0")
    if beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be > 0")
    if beat_step <= 0:
        raise ValueError("beat_step must be > 0")

    clip_length = float(bars) * float(beats_per_bar)
    beats_in_bar = _beats_in_bar(beats_per_bar, beat_step)
    anchor_candidates = _anchor_candidates(beats_in_bar)
    accent_beats = _accent_beat_indices(beats_in_bar)

    eighth_step = _eighth_step(beat_step)
    steps_per_bar = _steps_per_bar(beats_per_bar, eighth_step)
    accent_steps = _accent_steps(accent_beats, steps_per_bar)

    base_velocity = kick._clamp_velocity(velocity)
    velocity_offset = base_velocity - 88

    on_min, on_max = kick._clamp_velocity_range(80 + velocity_offset, 100 + velocity_offset)
    off_min, off_max = kick._clamp_velocity_range(68 + velocity_offset, 92 + velocity_offset)
    accent_min, accent_max = kick._clamp_velocity_range(94 + velocity_offset, 110 + velocity_offset)

    notes: List[dict] = []

    for bar_index in range(bars):
        bar_start = float(bar_index) * float(beats_per_bar)
        bar_end = bar_start + float(beats_per_bar)

        rim_anchor_index = anchor_candidates[bar_index % len(anchor_candidates)]
        rim_anchor_step = rim_anchor_index * 2

        active_steps = set(range(steps_per_bar))

        removal_count = _removal_count(bar_index, steps_per_bar, accent_steps)
        active_steps.difference_update(
            _choose_removal_steps(bar_index, steps_per_bar, accent_steps, removal_count)
        )

        # Make space for rim anchors occasionally, but keep density high.
        rim_space_seed = kick._velocity_seed(bar_index, rim_anchor_step, steps_per_bar)
        if rim_anchor_step < steps_per_bar and rim_space_seed % 3 == 0:
            active_steps.discard(rim_anchor_step)

        # At phrase boundaries, make space for kick and rim fills near the bar end.
        if (bar_index + 1) % 4 == 0:
            fill_offsets = kick._fill_patterns(float(beats_per_bar), float(beat_step))
            pattern = fill_offsets[(bar_index // 4) % len(fill_offsets)]
            for offset in pattern:
                step_index = _step_index_for_offset(offset, eighth_step)
                active_steps.discard(step_index)
                previous_seed = kick._velocity_seed(bar_index, step_index, steps_per_bar)
                if step_index - 1 >= 0 and previous_seed % 2 == 0:
                    active_steps.discard(step_index - 1)

        _ensure_min_density(active_steps, steps_per_bar, accent_steps, bar_index)

        step_times = sorted(
            {
                round(bar_start + float(step) * eighth_step, 6)
                for step in active_steps
                if bar_start <= bar_start + float(step) * eighth_step < bar_end
            }
        )

        scalar = _phrase_scalar(bar_index)
        scaled_on_min, scaled_on_max = kick._clamp_velocity_range(on_min * scalar, on_max * scalar)
        scaled_off_min, scaled_off_max = kick._clamp_velocity_range(off_min * scalar, off_max * scalar)
        scaled_accent_min, scaled_accent_max = kick._clamp_velocity_range(
            accent_min * scalar, accent_max * scalar
        )

        for start_time in step_times:
            if start_time >= clip_length:
                continue
            on_beat, beat_index = kick._classify_time(start_time, bar_start, float(beat_step))
            is_accent_beat = on_beat and beat_index in accent_beats

            step_index = int(round((start_time - bar_start) / eighth_step))
            seed = kick._velocity_seed(
                bar_index=bar_index,
                beat_index=beat_index,
                variant_index=step_index,
            )

            if is_accent_beat:
                note_velocity = kick._velocity_in_range(scaled_accent_min, scaled_accent_max, seed)
            elif on_beat:
                note_velocity = kick._velocity_in_range(scaled_on_min, scaled_on_max, seed)
            else:
                note_velocity = kick._velocity_in_range(scaled_off_min, scaled_off_max, seed)

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
class HatConfig:
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


def parse_args(argv: Iterable[str]) -> HatConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=32, help="Arrangement length in bars")
    parser.add_argument(
        "--start-beats",
        type=float,
        default=0.0,
        help="Arrangement start time in beats (default: 0)",
    )
    parser.add_argument("--track-name", default=DEFAULT_HAT_TRACK_NAME, help="Name of the MIDI track")
    parser.add_argument("--clip-name", default=DEFAULT_HAT_CLIP_NAME, help="Name of the arrangement clip")
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
    parser.add_argument("--pitch", type=int, default=DEFAULT_HAT_PITCH, help="Hat MIDI note (default: 42)")
    parser.add_argument(
        "--velocity",
        type=int,
        default=88,
        help="Hat velocity center (default: 88)",
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

    return HatConfig(
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


def run(cfg: HatConfig) -> int:
    if cfg.dry_run:
        beats_per_bar = cfg.beats_per_bar_override if cfg.beats_per_bar_override else 4.0
        beat_step = 1.0
        notes, clip_length = build_hat_notes(
            bars=cfg.bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            pitch=cfg.pitch,
            velocity=cfg.velocity,
            duration=cfg.duration,
        )
        clip_start = float(cfg.start_beats)
        clip_end = clip_start + clip_length
        print("HAT pattern plan (dry run):")
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
            "hat-tracks-before",
            cfg.ack_timeout_s,
        )
        if not tracks_before:
            print("error: could not read tracks; reload the device in Live", file=sys.stderr)
            ack_sock.close()
            return 2

        tempo_value = kick._api_get(sock, ack_sock, "live_set", "tempo", "hat-tempo", cfg.ack_timeout_s)
        sig_num_value = kick._api_get(
            sock,
            ack_sock,
            "live_set",
            "signature_numerator",
            "hat-sig-num",
            cfg.ack_timeout_s,
        )
        sig_den_value = kick._api_get(
            sock,
            ack_sock,
            "live_set",
            "signature_denominator",
            "hat-sig-den",
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

        notes, clip_length = build_hat_notes(
            bars=cfg.bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            pitch=cfg.pitch,
            velocity=cfg.velocity,
            duration=cfg.duration,
        )

        clip_start = float(cfg.start_beats)
        clip_end = clip_start + clip_length

        print("\nHAT pattern plan:")
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
                "hat-create-track",
                cfg.ack_timeout_s,
            )

            tracks_after = kick._get_children(
                sock,
                ack_sock,
                "live_set",
                "tracks",
                "hat-tracks-after",
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

        kick._api_set(sock, ack_sock, track_path, "name", cfg.track_name, "hat-track-name", cfg.ack_timeout_s)

        if cfg.insert_device and cfg.device_name.strip():
            devices = kick._get_children(sock, ack_sock, track_path, "devices", "hat-devices", cfg.ack_timeout_s)
            if track_was_created or not devices:
                kick._api_call(
                    sock,
                    ack_sock,
                    track_path,
                    "insert_device",
                    [cfg.device_name],
                    "hat-insert-device",
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
            "hat-arr-initial",
            cfg.ack_timeout_s,
        )

        deleted_overlaps = 0
        for idx, clip_info in enumerate(arrangement_initial):
            clip_path_raw = clip_info.get("path")
            if not clip_path_raw:
                continue
            clip_path = kick._sanitize_live_path(str(clip_path_raw))
            clip_start_existing = kick._as_float(
                kick._api_get(sock, ack_sock, clip_path, "start_time", f"hat-clip-start-{idx}", cfg.ack_timeout_s)
            )
            clip_end_existing = kick._as_float(
                kick._api_get(sock, ack_sock, clip_path, "end_time", f"hat-clip-end-{idx}", cfg.ack_timeout_s)
            )
            if clip_start_existing is None or clip_end_existing is None:
                continue
            if not kick._overlaps(clip_start_existing, clip_end_existing, clip_start, clip_end):
                continue

            clip_desc = kick._api_describe(sock, ack_sock, clip_path, f"hat-clip-describe-{idx}", cfg.ack_timeout_s)
            clip_id = kick._as_int(clip_desc.get("id") if isinstance(clip_desc, dict) else None)
            if clip_id is None or clip_id <= 0:
                clip_id = kick._as_int(
                    kick._api_get(sock, ack_sock, clip_path, "id", f"hat-clip-id-{idx}", cfg.ack_timeout_s)
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
                f"hat-delete-clip-{idx}",
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
            "hat-arr-before",
            cfg.ack_timeout_s,
        )

        create_result = kick._api_call(
            sock,
            ack_sock,
            track_path,
            "create_midi_clip",
            [clip_start, clip_length],
            "hat-create-clip",
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
                "hat-arr-after",
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

        kick._api_set(sock, ack_sock, clip_path, "name", cfg.clip_name, "hat-clip-name", cfg.ack_timeout_s)
        kick._api_set(sock, ack_sock, clip_path, "loop_start", 0.0, "hat-loop-start", cfg.ack_timeout_s)
        kick._api_set(sock, ack_sock, clip_path, "loop_end", clip_length, "hat-loop-end", cfg.ack_timeout_s)

        note_chunks = kick._chunk_notes(notes, chunk_size=48)
        for idx, chunk in enumerate(note_chunks, start=1):
            notes_json = {"notes": chunk}
            kick._api_call(
                sock,
                ack_sock,
                clip_path,
                "add_new_notes",
                notes_json,
                f"hat-add-notes-{idx}-of-{len(note_chunks)}",
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
                "hat-groove-amount",
                cfg.ack_timeout_s,
            )
            kick._api_set(
                sock,
                ack_sock,
                clip_path,
                "groove",
                ["id", groove_id],
                "hat-clip-groove",
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
            "hat-inspect",
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
