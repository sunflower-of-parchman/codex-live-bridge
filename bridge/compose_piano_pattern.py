#!/usr/bin/env python3
"""Compose a phrase-level piano part into the Arrangement view via the UDP bridge."""

from __future__ import annotations

import argparse
import math
import random
import socket
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import ableton_udp_bridge as bridge
import compose_kick_pattern as kick


HOST = bridge.DEFAULT_HOST
PORT = bridge.DEFAULT_PORT
ACK_PORT = bridge.DEFAULT_ACK_PORT

DEFAULT_PIANO_TRACK_NAME = "Piano"
DEFAULT_PIANO_CLIP_NAME = "Piano Cm7-AbMaj7-Fm7 1min"
DEFAULT_GROOVE_NAME = kick.DEFAULT_GROOVE_NAME

# Harmonic rhythm lives at the phrase level: 2 bars per chord, 8-bar phrases.
DEFAULT_SEGMENT_BARS = 2
PHRASE_BARS = 8


ChordName = str


CHORD_VOICINGS: Dict[ChordName, Sequence[Sequence[int]]] = {
    # C minor 7: C Eb G Bb
    "Cm7": (
        (36, 48, 55, 58, 63),
        (36, 48, 58, 63, 67),
        (36, 51, 55, 60, 63),
    ),
    # Ab major 7: Ab C Eb G
    "AbMaj7": (
        (44, 56, 60, 63, 67),
        (44, 56, 60, 67, 72),
        (44, 55, 60, 63, 68),
    ),
    # F minor 7: F Ab C Eb
    "Fm7": (
        (41, 53, 56, 60, 63),
        (41, 53, 60, 63, 68),
        (41, 56, 60, 63, 65),
    ),
}

# Four distinct 8-bar phrase shapes to avoid looping a single progression verbatim.
PHRASE_PATTERNS: Sequence[Sequence[ChordName]] = (
    ("Cm7", "Cm7", "AbMaj7", "Fm7"),
    ("Cm7", "Fm7", "AbMaj7", "Cm7"),
    ("Cm7", "Cm7", "Fm7", "AbMaj7"),
    ("Cm7", "AbMaj7", "Fm7", "Cm7"),
)


def _beats_in_bar(beats_per_bar: float, beat_step: float) -> int:
    beats = int(round(float(beats_per_bar) / float(beat_step)))
    return max(1, beats)


def _segment_bars(segment_bars: int) -> int:
    if segment_bars <= 0:
        raise ValueError("segment_bars must be > 0")
    return max(1, int(segment_bars))


def _segment_count(bars: int, segment_bars: int) -> int:
    return int(math.ceil(float(bars) / float(segment_bars)))


def _pattern_for_phrase(phrase_index: int) -> Sequence[ChordName]:
    return PHRASE_PATTERNS[phrase_index % len(PHRASE_PATTERNS)]


def _chord_for_segment(segment_index: int, segments_per_phrase: int) -> ChordName:
    phrase_index = segment_index // segments_per_phrase
    segment_in_phrase = segment_index % segments_per_phrase
    pattern = _pattern_for_phrase(phrase_index)
    return pattern[segment_in_phrase % len(pattern)]


def _voicing_for_chord(chord: ChordName, segment_index: int) -> Sequence[int]:
    variants = CHORD_VOICINGS[chord]
    seed = kick._velocity_seed(segment_index, len(variants), len(chord))
    pick = seed % len(variants)
    return variants[pick]


def _phrase_scalar(bar_index: int) -> float:
    """A light phrase contour so later bars breathe slightly."""
    phrase_pos = bar_index % 4
    if phrase_pos == 0:
        return 0.98
    if phrase_pos == 1:
        return 1.00
    if phrase_pos == 2:
        return 1.04
    return 1.01


def _spread_offsets(beat_step: float, note_count: int) -> List[float]:
    """Subtle strum to avoid a flat, machine block."""
    base = float(beat_step) * 0.06
    return [round(base * i, 6) for i in range(max(1, note_count))]


def _eighth_step(beat_step: float) -> float:
    return float(beat_step) / 2.0


def _grid_candidates(step: float, start: float, end: float) -> List[float]:
    """Enumerate grid-aligned offsets between start and end."""
    if step <= 0:
        return []
    values: List[float] = []
    cursor = float(start)
    epsilon = float(step) * 0.001
    while cursor < float(end) - epsilon:
        values.append(round(cursor, 6))
        cursor += float(step)
    return values


def _motion_offsets(segment_beats: float, beat_step: float, segment_index: int) -> List[float]:
    """Generate grid-aware, deterministic motion inside a chord segment."""
    if segment_beats <= float(beat_step) * 0.5:
        return []

    start_offset = float(beat_step) * 0.5
    end_offset = float(segment_beats) - float(beat_step) * 0.2
    if end_offset <= start_offset:
        return []

    # Use a mix of musically meaningful subdivisions:
    # half notes, quarter notes, eighths, triplets, and rare sixteenths.
    half_step = float(beat_step) * 2.0
    quarter_step = float(beat_step)
    eighth_step = _eighth_step(beat_step)
    triplet_step = float(beat_step) / 3.0
    sixteenth_step = float(beat_step) / 4.0

    candidate_set: set[float] = set()
    for step, include_every in (
        (half_step, 1),
        (quarter_step, 1),
        (eighth_step, 1),
        (triplet_step, 2),
        (sixteenth_step, 6),
    ):
        grid = _grid_candidates(step, start_offset, end_offset)
        if not grid:
            continue
        # Thin certain grids so we do not end up overly dense.
        for idx, value in enumerate(grid):
            if include_every <= 1 or idx % include_every == 0:
                candidate_set.add(value)

    candidates = sorted(candidate_set)

    if not candidates:
        return []

    seed = kick._velocity_seed(segment_index, len(candidates), int(segment_beats * 1000))
    rng = random.Random(seed)

    base_count = max(3, int(segment_beats / float(beat_step)) + 2)
    jitter = rng.randint(-1, 3)
    target_count = max(3, min(16, base_count + jitter))
    target_count = min(target_count, len(candidates))

    offsets = sorted(rng.sample(candidates, target_count))

    # Ensure the motion is not all clustered right at the start.
    late_threshold = start_offset + float(beat_step) * 1.5
    if offsets and all(o < late_threshold for o in offsets):
        late_candidates = [o for o in candidates if o >= late_threshold]
        if late_candidates:
            offsets[-1] = late_candidates[min(len(late_candidates) - 1, seed % len(late_candidates))]
            offsets = sorted(set(offsets))

    return offsets


def _motion_pool(pitches: Sequence[int]) -> List[int]:
    """Prefer upper-register chord tones for motion to avoid reattacks."""
    if not pitches:
        return []
    ordered = sorted(int(p) for p in pitches)
    held_set = set(ordered)
    upper = ordered[1:] if len(ordered) > 1 else ordered

    motion_set: set[int] = set()
    for pitch in upper:
        # Push motion into a higher register so sustained chord tones
        # are not re-attacked at the same pitch.
        for octave in (12, 24):
            candidate = pitch + octave
            if 0 <= candidate <= 127 and candidate not in held_set:
                motion_set.add(candidate)

    if not motion_set:
        # Fallback: at least avoid the very lowest tone.
        highest = max(ordered)
        for octave in (12, 24, 36):
            candidate = highest + octave
            if 0 <= candidate <= 127 and candidate not in held_set:
                return [candidate]
        return [min(127, highest + 12)]

    return sorted(motion_set)


def _legato_chord_duration(segment_beats: float, beat_step: float) -> float:
    """Hold chord tones for most of the segment when no pedal is available."""
    gap = float(beat_step) * 0.02
    min_duration = float(beat_step) * 0.75
    duration = max(min_duration, float(segment_beats) - gap)
    return round(duration, 6)


def _motion_duration(
    segment_start: float,
    segment_beats: float,
    start_time: float,
    next_start_time: float | None,
    beat_step: float,
) -> float:
    segment_end = segment_start + float(segment_beats)
    remaining = segment_end - float(start_time)
    if remaining <= 0:
        return 0.0
    target_end = segment_end if next_start_time is None else min(segment_end, float(next_start_time))
    desired = max(0.0, target_end - float(start_time))
    overlap = float(beat_step) * 0.03
    min_duration = float(beat_step) * 0.25
    duration = min(remaining, max(min_duration, desired + overlap))
    return round(duration, 6)


def _velocity_ranges(velocity_center: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    base_velocity = kick._clamp_velocity(velocity_center)
    offset = base_velocity - 92

    # Keep lows gentle; do not slam the bottom of the piano.
    low_min, low_max = kick._clamp_velocity_range(72 + offset, 90 + offset)
    mid_min, mid_max = kick._clamp_velocity_range(78 + offset, 98 + offset)
    top_min, top_max = kick._clamp_velocity_range(74 + offset, 94 + offset)
    return (low_min, low_max), (mid_min, mid_max), (top_min, top_max)


def _transpose_pitch(pitch: int, semitones: int) -> int:
    shifted = int(pitch) + int(semitones)
    return max(0, min(127, shifted))


def build_piano_layers(
    bars: int,
    beats_per_bar: float = 4.0,
    beat_step: float = 1.0,
    velocity_center: int = 92,
    segment_bars: int = DEFAULT_SEGMENT_BARS,
    transpose_semitones: int = 0,
) -> Tuple[List[dict], List[dict], float]:
    """Generate chord and motion layers for the piano part.

    The part emphasizes phrase-scale harmony: each chord holds for 1, 2, 4
    and 8-bar phrases vary their chord order to avoid verbatim loops.
    """
    if bars <= 0:
        raise ValueError("bars must be > 0")
    if beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be > 0")
    if beat_step <= 0:
        raise ValueError("beat_step must be > 0")

    segment_bars = _segment_bars(segment_bars)
    clip_length = float(bars) * float(beats_per_bar)

    segment_count = _segment_count(bars, segment_bars)
    segments_per_phrase = max(1, PHRASE_BARS // segment_bars)

    beats_in_bar = _beats_in_bar(beats_per_bar, beat_step)

    (low_min, low_max), (mid_min, mid_max), (top_min, top_max) = _velocity_ranges(velocity_center)

    chord_notes: List[dict] = []
    motion_notes: List[dict] = []

    for segment_index in range(segment_count):
        start_bar = segment_index * segment_bars
        remaining_bars = max(0, bars - start_bar)
        if remaining_bars <= 0:
            break

        effective_bars = min(segment_bars, remaining_bars)
        segment_start = float(start_bar) * float(beats_per_bar)
        segment_beats_effective = float(effective_bars) * float(beats_per_bar)

        chord_name = _chord_for_segment(segment_index, segments_per_phrase)
        raw_pitches = list(_voicing_for_chord(chord_name, segment_index))
        pitches = [
            _transpose_pitch(pitch, transpose_semitones) for pitch in raw_pitches
        ]
        offsets = _spread_offsets(beat_step, len(pitches))

        bar_index_for_scalar = min(bars - 1, start_bar)
        scalar = _phrase_scalar(bar_index_for_scalar)

        low_scalar = 0.94
        scaled_low = kick._clamp_velocity_range(low_min * scalar * low_scalar, low_max * scalar * low_scalar)
        scaled_mid = kick._clamp_velocity_range(mid_min * scalar, mid_max * scalar)
        scaled_top = kick._clamp_velocity_range(top_min * scalar, top_max * scalar)

        # Treat the lowest note as the anchor, the middle notes as body, and the
        # highest note as the air/edge.
        lowest_pitch = min(pitches)
        highest_pitch = max(pitches)
        chord_duration = _legato_chord_duration(segment_beats_effective, beat_step)

        for idx, pitch in enumerate(pitches):
            start_time = segment_start + offsets[idx]
            if start_time >= clip_length:
                continue

            on_beat, beat_index = kick._classify_time(start_time, segment_start, float(beat_step))
            variant_index = idx + (beats_in_bar * segment_index)
            seed = kick._velocity_seed(segment_index, beat_index, variant_index)

            if pitch == lowest_pitch:
                vel_min, vel_max = scaled_low
            elif pitch == highest_pitch:
                vel_min, vel_max = scaled_top
            else:
                vel_min, vel_max = scaled_mid

            note_velocity = kick._velocity_in_range(vel_min, vel_max, seed)

            chord_notes.append(
                {
                    "pitch": int(pitch),
                    "start_time": float(round(start_time, 6)),
                    "duration": float(chord_duration),
                    "velocity": int(note_velocity),
                    "mute": 0,
                }
            )

        motion_pool = _motion_pool(pitches)
        motion_offsets = _motion_offsets(segment_beats_effective, beat_step, segment_index)
        for motion_idx, offset in enumerate(motion_offsets):
            start_time = segment_start + float(offset)
            if start_time >= clip_length:
                continue
            if start_time >= segment_start + segment_beats_effective:
                continue

            next_start_time: float | None = None
            if motion_idx + 1 < len(motion_offsets):
                next_start_time = segment_start + float(motion_offsets[motion_idx + 1])

            pitch_seed = kick._velocity_seed(
                segment_index,
                motion_idx,
                int(round(offset * 1000)),
            )
            pitch = motion_pool[pitch_seed % len(motion_pool)]

            duration = _motion_duration(
                segment_start,
                segment_beats_effective,
                start_time,
                next_start_time,
                beat_step,
            )
            if duration <= 0:
                continue

            on_beat, beat_index = kick._classify_time(start_time, segment_start, float(beat_step))
            variant_index = motion_idx + beats_in_bar * (segment_index + 1)
            seed = kick._velocity_seed(segment_index, beat_index, variant_index)

            # Motion should be present but not overpower the chord anchor.
            if pitch == highest_pitch:
                vel_min, vel_max = scaled_top
            else:
                vel_min, vel_max = scaled_mid

            motion_scalar = 0.97 if not on_beat else 1.0
            motion_min, motion_max = kick._clamp_velocity_range(vel_min * motion_scalar, vel_max * motion_scalar)
            note_velocity = kick._velocity_in_range(motion_min, motion_max, seed)

            motion_notes.append(
                {
                    "pitch": int(pitch),
                    "start_time": float(round(start_time, 6)),
                    "duration": float(duration),
                    "velocity": int(note_velocity),
                    "mute": 0,
                }
            )

    chord_notes.sort(key=lambda n: (n["start_time"], n["pitch"]))
    motion_notes.sort(key=lambda n: (n["start_time"], n["pitch"]))
    return chord_notes, motion_notes, clip_length


def build_piano_notes(
    bars: int,
    beats_per_bar: float = 4.0,
    beat_step: float = 1.0,
    velocity_center: int = 92,
    segment_bars: int = DEFAULT_SEGMENT_BARS,
    transpose_semitones: int = 0,
    layer_mode: str = "full",
) -> Tuple[List[dict], float]:
    """Generate a full piano part using Cm7, AbMaj7, and Fm7.

    This preserves the existing interface while allowing optional transposition.
    """
    chord_notes, motion_notes, clip_length = build_piano_layers(
        bars=bars,
        beats_per_bar=beats_per_bar,
        beat_step=beat_step,
        velocity_center=velocity_center,
        segment_bars=segment_bars,
        transpose_semitones=transpose_semitones,
    )
    resolved_mode = str(layer_mode).strip().lower()
    if resolved_mode == "chords":
        notes = chord_notes
    elif resolved_mode == "motion":
        notes = motion_notes
    else:
        notes = chord_notes + motion_notes
    notes.sort(key=lambda n: (n["start_time"], n["pitch"]))
    return notes, clip_length


@dataclass(frozen=True)
class PianoConfig:
    bars: int
    start_beats: float
    track_name: str
    clip_name: str
    device_name: str
    insert_device: bool
    beats_per_bar_override: float | None
    velocity_center: int
    groove_name: str
    segment_bars: int
    layer_mode: str
    ack_timeout_s: float
    dry_run: bool


def parse_args(argv: Iterable[str]) -> PianoConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=32, help="Arrangement length in bars")
    parser.add_argument(
        "--start-beats",
        type=float,
        default=0.0,
        help="Arrangement start time in beats (default: 0)",
    )
    parser.add_argument("--track-name", default=DEFAULT_PIANO_TRACK_NAME, help="Name of the MIDI track")
    parser.add_argument("--clip-name", default=DEFAULT_PIANO_CLIP_NAME, help="Name of the arrangement clip")
    parser.add_argument(
        "--device-name",
        default="Grand Piano",
        help="Native device to insert on the track (default: Grand Piano)",
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
    parser.add_argument(
        "--velocity-center",
        type=int,
        default=92,
        help="Piano velocity center (default: 92)",
    )
    parser.add_argument(
        "--segment-bars",
        type=int,
        default=DEFAULT_SEGMENT_BARS,
        help="Bars per chord segment (default: 2)",
    )
    parser.add_argument(
        "--layer-mode",
        choices=("full", "chords", "motion"),
        default="full",
        help="Piano layer mode (default: full)",
    )
    parser.add_argument(
        "--groove-name",
        default=DEFAULT_GROOVE_NAME,
        help=f"Groove Pool name to apply (default: {DEFAULT_GROOVE_NAME})",
    )
    parser.add_argument(
        "--ack-timeout",
        type=float,
        default=1.5,
        help="Ack wait timeout in seconds (default: 1.5)",
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
    if ns.segment_bars <= 0:
        parser.error("--segment-bars must be > 0")
    if ns.ack_timeout <= 0:
        parser.error("--ack-timeout must be > 0")

    return PianoConfig(
        bars=int(ns.bars),
        start_beats=float(ns.start_beats),
        track_name=str(ns.track_name),
        clip_name=str(ns.clip_name),
        device_name=str(ns.device_name),
        insert_device=not bool(ns.no_device),
        beats_per_bar_override=None if ns.beats_per_bar is None else float(ns.beats_per_bar),
        velocity_center=int(ns.velocity_center),
        groove_name=str(ns.groove_name),
        segment_bars=int(ns.segment_bars),
        layer_mode=str(ns.layer_mode),
        ack_timeout_s=float(ns.ack_timeout),
        dry_run=bool(ns.dry_run),
    )


def run(cfg: PianoConfig) -> int:
    if cfg.dry_run:
        beats_per_bar = cfg.beats_per_bar_override if cfg.beats_per_bar_override else 4.0
        beat_step = 1.0
        notes, clip_length = build_piano_notes(
            bars=cfg.bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity_center=cfg.velocity_center,
            segment_bars=cfg.segment_bars,
            layer_mode=cfg.layer_mode,
        )
        clip_start = float(cfg.start_beats)
        clip_end = clip_start + clip_length
        print("Piano pattern plan (dry run):")
        print(f"- bars:        {cfg.bars}")
        print(f"- beats/bar:   {beats_per_bar:g} (assumed)")
        print(f"- beat step:   {beat_step:g} (assumed quarter-note beats)")
        print(f"- segment:     {cfg.segment_bars} bars per chord")
        print(f"- clip start:  {clip_start:g}")
        print(f"- clip length: {clip_length:g} beats")
        print(f"- clip end:    {clip_end:g}")
        print(f"- note count:  {len(notes)}")
        print(f"- velocity:    {cfg.velocity_center}")
        print(f"- layer mode:  {cfg.layer_mode}")
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
        ack_mode="per_command",
        ack_flush_interval=10,
        report_metrics=True,
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
            "piano-tracks-before",
            cfg.ack_timeout_s,
        )
        if not tracks_before:
            print("error: could not read tracks; reload the device in Live", file=sys.stderr)
            ack_sock.close()
            return 2

        tempo_value = kick._api_get(sock, ack_sock, "live_set", "tempo", "piano-tempo", cfg.ack_timeout_s)
        sig_num_value = kick._api_get(
            sock,
            ack_sock,
            "live_set",
            "signature_numerator",
            "piano-sig-num",
            cfg.ack_timeout_s,
        )
        sig_den_value = kick._api_get(
            sock,
            ack_sock,
            "live_set",
            "signature_denominator",
            "piano-sig-den",
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

        notes, clip_length = build_piano_notes(
            bars=cfg.bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity_center=cfg.velocity_center,
            segment_bars=cfg.segment_bars,
            layer_mode=cfg.layer_mode,
        )

        clip_start = float(cfg.start_beats)
        clip_end = clip_start + clip_length

        print("\nPiano pattern plan:")
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
        print(f"- segment:     {cfg.segment_bars} bars per chord")
        print(f"- bars:        {cfg.bars}")
        print(f"- clip start:  {clip_start:g}")
        print(f"- clip length: {clip_length:g} beats")
        print(f"- clip end:    {clip_end:g}")
        print(f"- note count:  {len(notes)}")
        print(f"- layer mode:  {cfg.layer_mode}")
        print("- palette:     Cm7, AbMaj7, Fm7")

        existing_track_index = kick._find_track_index_by_name(tracks_before, cfg.track_name)
        if existing_track_index is None:
            kick._api_call(
                sock,
                ack_sock,
                "live_set",
                "create_midi_track",
                [-1],
                "piano-create-track",
                cfg.ack_timeout_s,
            )

            tracks_after = kick._get_children(
                sock,
                ack_sock,
                "live_set",
                "tracks",
                "piano-tracks-after",
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

        kick._api_set(sock, ack_sock, track_path, "name", cfg.track_name, "piano-track-name", cfg.ack_timeout_s)

        if cfg.insert_device and cfg.device_name.strip():
            devices = kick._get_children(sock, ack_sock, track_path, "devices", "piano-devices", cfg.ack_timeout_s)
            if track_was_created or not devices:
                kick._api_call(
                    sock,
                    ack_sock,
                    track_path,
                    "insert_device",
                    [cfg.device_name],
                    "piano-insert-device",
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
            "piano-arr-initial",
            cfg.ack_timeout_s,
        )

        deleted_overlaps = 0
        for idx, clip_info in enumerate(arrangement_initial):
            clip_path_raw = clip_info.get("path")
            if not clip_path_raw:
                continue
            clip_path = kick._sanitize_live_path(str(clip_path_raw))
            clip_start_existing = kick._as_float(
                kick._api_get(sock, ack_sock, clip_path, "start_time", f"piano-clip-start-{idx}", cfg.ack_timeout_s)
            )
            clip_end_existing = kick._as_float(
                kick._api_get(sock, ack_sock, clip_path, "end_time", f"piano-clip-end-{idx}", cfg.ack_timeout_s)
            )
            if clip_start_existing is None or clip_end_existing is None:
                continue
            if not kick._overlaps(clip_start_existing, clip_end_existing, clip_start, clip_end):
                continue

            clip_desc = kick._api_describe(sock, ack_sock, clip_path, f"piano-clip-describe-{idx}", cfg.ack_timeout_s)
            clip_id = kick._as_int(clip_desc.get("id") if isinstance(clip_desc, dict) else None)
            if clip_id is None or clip_id <= 0:
                clip_id = kick._as_int(
                    kick._api_get(sock, ack_sock, clip_path, "id", f"piano-clip-id-{idx}", cfg.ack_timeout_s)
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
                f"piano-delete-clip-{idx}",
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
            "piano-arr-before",
            cfg.ack_timeout_s,
        )

        create_result = kick._api_call(
            sock,
            ack_sock,
            track_path,
            "create_midi_clip",
            [clip_start, clip_length],
            "piano-create-clip",
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
                "piano-arr-after",
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

        kick._api_set(sock, ack_sock, clip_path, "name", cfg.clip_name, "piano-clip-name", cfg.ack_timeout_s)
        kick._api_set(sock, ack_sock, clip_path, "loop_start", 0.0, "piano-loop-start", cfg.ack_timeout_s)
        kick._api_set(sock, ack_sock, clip_path, "loop_end", clip_length, "piano-loop-end", cfg.ack_timeout_s)

        note_chunks = kick._chunk_notes(notes, chunk_size=40)
        for idx, chunk in enumerate(note_chunks, start=1):
            notes_json = {"notes": chunk}
            kick._api_call(
                sock,
                ack_sock,
                clip_path,
                "add_new_notes",
                notes_json,
                f"piano-add-notes-{idx}-of-{len(note_chunks)}",
                max(cfg.ack_timeout_s, 2.5),
            )

        groove_id = kick._find_groove_id_by_name(sock, ack_sock, cfg.groove_name, cfg.ack_timeout_s)
        if groove_id:
            kick._api_set(
                sock,
                ack_sock,
                "live_set",
                "groove_amount",
                1.0,
                "piano-groove-amount",
                cfg.ack_timeout_s,
            )
            kick._api_set(
                sock,
                ack_sock,
                clip_path,
                "groove",
                ["id", groove_id],
                "piano-clip-groove",
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
            "piano-inspect",
            max(cfg.ack_timeout_s, 2.5),
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
