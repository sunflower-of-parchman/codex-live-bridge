from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from arrangement.base import (
    InstrumentSpec,
    MarimbaIdentityConfig,
    Section,
    _fit_pitch_to_register,
)


NOTE_TO_PITCH_CLASS = {
    "c": 0,
    "b#": 0,
    "c#": 1,
    "db": 1,
    "d": 2,
    "d#": 3,
    "eb": 3,
    "e": 4,
    "fb": 4,
    "e#": 5,
    "f": 5,
    "f#": 6,
    "gb": 6,
    "g": 7,
    "g#": 8,
    "ab": 8,
    "a": 9,
    "a#": 10,
    "bb": 10,
    "b": 11,
    "cb": 11,
}


def _find_track_key(mapping: Mapping[str, Any], wanted: str) -> str | None:
    target = str(wanted).strip().lower()
    if not target:
        return None
    for key in mapping.keys():
        if str(key).strip().lower() == target:
            return str(key)
    return None

def _limit_note_count(notes: Sequence[dict], max_count: int) -> List[dict]:
    if max_count <= 0:
        return []
    if len(notes) <= max_count:
        return [dict(n) for n in notes]
    if not notes:
        return []
    stride = float(len(notes)) / float(max_count)
    picked: List[dict] = []
    for idx in range(max_count):
        source_idx = min(len(notes) - 1, int(round(idx * stride)))
        picked.append(dict(notes[source_idx]))
    picked.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
    return picked

def _seed_notes_for_section(
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict]]]],
    section_index: int,
    *,
    preferred_tracks: Sequence[str],
) -> List[dict]:
    for preferred in preferred_tracks:
        key = _find_track_key(arranged_by_track, preferred)
        if key is None:
            continue
        payload = arranged_by_track.get(key, ())
        if section_index < 0 or section_index >= len(payload):
            continue
        notes = payload[section_index][1]
        if notes:
            return [dict(n) for n in notes]
    return []

def _clamp_velocity(value: float) -> int:
    return max(1, min(127, int(round(float(value)))))

def _normalize_note_token(token: str) -> str:
    text = str(token).strip().lower()
    text = text.replace("♯", "#").replace("♭", "b")
    text = text.replace("sharp", "#").replace("flat", "b")
    return text

def _key_pitch_classes(key_name: str) -> set[int] | None:
    if not key_name:
        return None
    parts = str(key_name).strip().split()
    if len(parts) < 2:
        return None
    root = _normalize_note_token(parts[0])
    quality = str(parts[1]).strip().lower()
    root_pc = NOTE_TO_PITCH_CLASS.get(root)
    if root_pc is None:
        return None
    if quality.startswith("maj"):
        intervals = (0, 2, 4, 5, 7, 9, 11)
    elif quality.startswith("min"):
        intervals = (0, 2, 3, 5, 7, 8, 10)
    else:
        return None
    return {(root_pc + interval) % 12 for interval in intervals}


def _key_scale_pitch_classes(key_name: str) -> list[int] | None:
    if not key_name:
        return None
    parts = str(key_name).strip().split()
    if len(parts) < 2:
        return None
    root = _normalize_note_token(parts[0])
    quality = str(parts[1]).strip().lower()
    root_pc = NOTE_TO_PITCH_CLASS.get(root)
    if root_pc is None:
        return None
    if quality.startswith("maj"):
        intervals = (0, 2, 4, 5, 7, 9, 11)
    elif quality.startswith("min"):
        intervals = (0, 2, 3, 5, 7, 8, 10)
    else:
        return None
    return [int((root_pc + interval) % 12) for interval in intervals]

def _nearest_pitch_in_key(
    pitch: int,
    *,
    allowed_pitch_classes: set[int],
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> int:
    low = max(0, min(127, int(midi_min_pitch)))
    high = max(0, min(127, int(midi_max_pitch)))
    if low > high:
        low, high = high, low
    in_range = max(low, min(high, int(pitch)))
    if in_range % 12 in allowed_pitch_classes:
        return in_range
    candidates = [value for value in range(low, high + 1) if value % 12 in allowed_pitch_classes]
    if not candidates:
        return in_range
    return min(candidates, key=lambda value: (abs(value - in_range), value))

def _constrain_notes_to_key(
    notes: Sequence[dict],
    *,
    key_name: str,
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> List[dict]:
    allowed = _key_pitch_classes(key_name)
    if not allowed:
        return [dict(n) for n in notes]
    constrained: List[dict] = []
    for note in sorted(notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0)))):
        copied = dict(note)
        copied["pitch"] = int(
            _nearest_pitch_in_key(
                int(copied.get("pitch", 60)),
                allowed_pitch_classes=allowed,
                midi_min_pitch=midi_min_pitch,
                midi_max_pitch=midi_max_pitch,
            )
        )
        constrained.append(copied)
    return constrained


def _as_positive_int_sequence(raw: Any, fallback: Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [int(v) for v in raw if isinstance(v, (int, float)) and int(v) > 0]
        if values:
            return tuple(values)
    safe_fallback = [int(v) for v in fallback if int(v) > 0]
    return tuple(safe_fallback or [2, 3, 4])


def _tempo_rhythm_profile(
    *,
    bpm: float,
    grid_step_beats: float,
    payload: Mapping[str, Any],
) -> tuple[float, str]:
    base_step = max(1e-6, float(grid_step_beats))
    threshold_raw = payload.get("fast_tempo_bpm_threshold", 124.0)
    try:
        fast_threshold = float(threshold_raw)
    except (TypeError, ValueError):
        fast_threshold = 124.0
    fast_step_raw = payload.get("fast_tempo_rhythm_step_beats", max(base_step * 2.0, 0.5))
    try:
        fast_step = float(fast_step_raw)
    except (TypeError, ValueError):
        fast_step = max(base_step * 2.0, 0.5)
    fast_step = max(base_step, fast_step)
    if float(bpm) >= float(fast_threshold):
        return float(fast_step), "fast"
    return float(base_step), "normal"


def _mutation_windows(total_bars: int, windows: Sequence[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if total_bars <= 0:
        return out
    safe_windows = [max(1, int(v)) for v in windows if int(v) > 0]
    if not safe_windows:
        safe_windows = [2, 3, 4]
    cursor = 0
    idx = 0
    while cursor < int(total_bars):
        size = int(safe_windows[idx % len(safe_windows)])
        remaining = int(total_bars) - cursor
        if size > remaining:
            size = remaining
        out.append((int(cursor), int(size)))
        cursor += int(size)
        idx += 1
    return out


def _build_harmony_plan(
    *,
    total_bars: int,
    block_sizes: Sequence[int],
    progression: Sequence[int],
) -> tuple[list[int], list[tuple[int, int, int]]]:
    if total_bars <= 0:
        return [], []
    safe_blocks = [max(1, int(v)) for v in block_sizes if int(v) > 0]
    if not safe_blocks:
        safe_blocks = [16, 24, 32]
    safe_progression = [int(v) for v in progression] if progression else [0]
    plan: list[int] = []
    blocks: list[tuple[int, int, int]] = []
    cursor = 0
    idx = 0
    while cursor < int(total_bars):
        block_size = int(safe_blocks[idx % len(safe_blocks)])
        remaining = int(total_bars) - cursor
        if block_size > remaining:
            block_size = remaining
        degree = int(safe_progression[idx % len(safe_progression)])
        blocks.append((int(cursor), int(block_size), int(degree)))
        for _ in range(int(block_size)):
            plan.append(int(degree))
        cursor += int(block_size)
        idx += 1
    return plan, blocks


def _arc_scale(
    progress: float,
    *,
    start: float,
    peak: float,
    end: float,
    peak_ratio: float,
) -> float:
    p = max(0.0, min(1.0, float(progress)))
    pivot = max(1e-6, min(0.999999, float(peak_ratio)))
    if p <= pivot:
        t = p / pivot
        return float(start + ((peak - start) * t))
    t = (p - pivot) / (1.0 - pivot)
    return float(peak + ((end - peak) * t))


def _apply_form_arc(
    notes: Sequence[dict],
    *,
    beats_per_bar: float,
    total_bars: int,
    peak_ratio: float,
    density_start_scale: float,
    density_peak_scale: float,
    density_end_scale: float,
    velocity_start_scale: float,
    velocity_peak_scale: float,
    velocity_end_scale: float,
) -> List[dict]:
    if not notes or total_bars <= 0:
        return [dict(note) for note in notes]
    grouped: dict[int, list[dict]] = {}
    bar_span = max(float(beats_per_bar), 1e-6)
    for note in notes:
        bar_index = int(float(note.get("start_time", 0.0)) // bar_span)
        grouped.setdefault(bar_index, []).append(dict(note))
    shaped: List[dict] = []
    for bar_index in sorted(grouped.keys()):
        bar_notes = sorted(grouped[bar_index], key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
        progress = (float(bar_index) + 0.5) / max(1.0, float(total_bars))
        density_scale = _arc_scale(
            progress,
            start=float(density_start_scale),
            peak=float(density_peak_scale),
            end=float(density_end_scale),
            peak_ratio=float(peak_ratio),
        )
        velocity_scale = _arc_scale(
            progress,
            start=float(velocity_start_scale),
            peak=float(velocity_peak_scale),
            end=float(velocity_end_scale),
            peak_ratio=float(peak_ratio),
        )
        target_count = max(1, int(round(len(bar_notes) * max(0.05, float(density_scale)))))
        if target_count < len(bar_notes):
            def _priority(note: dict) -> tuple[int, float, int, float, int]:
                local_beat = float(note.get("start_time", 0.0)) - (float(bar_index) * bar_span)
                on_downbeat = 1 if abs(local_beat - round(local_beat)) <= 1e-4 else 0
                return (
                    int(on_downbeat),
                    float(note.get("duration", 0.0)),
                    int(note.get("velocity", 0)),
                    -float(note.get("start_time", 0.0)),
                    -int(note.get("pitch", 0)),
                )

            ranked = sorted(bar_notes, key=_priority, reverse=True)
            bar_notes = ranked[:target_count]
            bar_notes.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
        for note in bar_notes:
            copied = dict(note)
            copied["velocity"] = int(_clamp_velocity(float(copied.get("velocity", 96)) * float(velocity_scale)))
            shaped.append(copied)
    shaped.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
    return shaped


def _harmony_degree_cycle(profile: str) -> tuple[int, ...]:
    name = str(profile).strip().lower()
    if name in {"toggle", "tonic_dominant_toggle"}:
        return (0, 4, 0, 4, 0, 5, 4, 0)
    if name in {"modal_wave", "wave"}:
        return (0, 2, 4, 3, 5, 4, 2, 0)
    return (0, 3, 4, 0, 5, 3, 4, 0)


def _triad_pitch_classes(scale_pitch_classes: Sequence[int], degree_index: int) -> tuple[int, int, int]:
    if not scale_pitch_classes:
        return (0, 4, 7)
    scale = [int(v) % 12 for v in scale_pitch_classes]
    idx = int(degree_index) % len(scale)
    return (
        int(scale[idx]),
        int(scale[(idx + 2) % len(scale)]),
        int(scale[(idx + 4) % len(scale)]),
    )


def _nearest_pitch_for_pitch_class(
    pitch_class: int,
    *,
    target_pitch: int,
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> int:
    low = max(0, min(127, int(midi_min_pitch)))
    high = max(0, min(127, int(midi_max_pitch)))
    if low > high:
        low, high = high, low
    pc = int(pitch_class) % 12
    target = max(low, min(high, int(target_pitch)))
    candidates = [value for value in range(low, high + 1) if value % 12 == pc]
    if not candidates:
        return _fit_pitch_to_register(target, low, high)
    return min(candidates, key=lambda value: (abs(value - target), value))


def _quantize_to_grid(value: float, grid_step_beats: float) -> float:
    step = max(1e-6, float(grid_step_beats))
    return float(round(round(float(value) / step) * step, 6))


def _quantize_notes_to_grid(
    notes: Sequence[dict],
    *,
    grid_step_beats: float,
    clip_length_beats: float,
) -> List[dict]:
    if not notes:
        return []
    step = max(1e-6, float(grid_step_beats))
    min_duration = max(0.05, step)
    clip_length = max(0.0, float(clip_length_beats))
    out: List[dict] = []
    for note in sorted(notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0)))):
        copied = dict(note)
        start = _quantize_to_grid(float(copied.get("start_time", 0.0)), step)
        duration = _quantize_to_grid(float(copied.get("duration", min_duration)), step)
        duration = max(min_duration, duration)
        if start < 0.0:
            start = 0.0
        if start >= clip_length:
            continue
        max_duration = max(min_duration, clip_length - start)
        duration = min(duration, max_duration)
        copied["start_time"] = float(round(start, 6))
        copied["duration"] = float(round(duration, 6))
        copied["velocity"] = _clamp_velocity(float(copied.get("velocity", 96)))
        out.append(copied)
    out.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
    return out


def _enforce_max_simultaneous_notes(notes: Sequence[dict], max_simultaneous: int) -> List[dict]:
    if max_simultaneous <= 0:
        return []
    ordered = [dict(note) for note in sorted(notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))]
    out: List[dict] = []
    active_ends: list[float] = []
    epsilon = 1e-6
    for note in ordered:
        start = float(note.get("start_time", 0.0))
        duration = max(0.05, float(note.get("duration", 0.25)))
        active_ends = [end for end in active_ends if end > start + epsilon]
        if len(active_ends) >= int(max_simultaneous):
            continue
        out.append(note)
        active_ends.append(start + duration)
    return out


def _limit_notes_per_bar(
    notes: Sequence[dict],
    *,
    beats_per_bar: float,
    max_notes_per_bar: int,
) -> List[dict]:
    if max_notes_per_bar <= 0:
        return []
    if not notes:
        return []
    grouped: dict[int, list[dict]] = {}
    for note in notes:
        bar_index = int(float(note.get("start_time", 0.0)) // max(float(beats_per_bar), 1e-6))
        grouped.setdefault(bar_index, []).append(dict(note))
    out: List[dict] = []
    for bar_index in sorted(grouped.keys()):
        bar_notes = sorted(grouped[bar_index], key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
        if len(bar_notes) > int(max_notes_per_bar):
            bar_notes = bar_notes[: int(max_notes_per_bar)]
        out.extend(bar_notes)
    out.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
    return out


def _merge_section_notes_to_timeline(
    payloads: Sequence[tuple[Section, Sequence[dict]]],
    *,
    beats_per_bar: float,
) -> List[dict]:
    merged: List[dict] = []
    for section, notes in payloads:
        section_start = float(section.start_bar) * float(beats_per_bar)
        for note in notes:
            copied = dict(note)
            copied["start_time"] = float(round(section_start + float(note.get("start_time", 0.0)), 6))
            merged.append(copied)
    merged.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
    return merged


def _split_timeline_notes_by_sections(
    notes: Sequence[dict],
    *,
    sections: Sequence[Section],
    beats_per_bar: float,
    grid_step_beats: float,
) -> List[tuple[Section, List[dict]]]:
    out: List[tuple[Section, List[dict]]] = []
    epsilon = max(1e-6, float(grid_step_beats) * 0.02)
    min_duration = max(0.05, float(grid_step_beats))
    for section in sections:
        start = float(section.start_bar) * float(beats_per_bar)
        end = start + (float(section.bar_count) * float(beats_per_bar))
        section_notes: List[dict] = []
        for note in notes:
            absolute_start = float(note.get("start_time", 0.0))
            if absolute_start < start or absolute_start >= end:
                continue
            max_duration = max(min_duration, end - absolute_start - epsilon)
            copied = dict(note)
            copied["start_time"] = float(round(absolute_start - start, 6))
            copied["duration"] = float(round(min(max_duration, float(note.get("duration", min_duration))), 6))
            section_notes.append(copied)
        section_notes.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
        out.append((section, section_notes))
    return out


def _compose_evolving_ostinato_timeline(
    *,
    total_bars: int,
    beats_per_bar: float,
    grid_step_beats: float,
    windows: Sequence[tuple[int, int]],
    scale_pitch_classes: Sequence[int],
    harmony_profile: str,
    harmony_plan_by_bar: Sequence[int] | None,
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> List[dict]:
    if total_bars <= 0:
        return []
    step = max(1e-6, float(grid_step_beats))
    steps_per_bar = max(1, int(round(float(beats_per_bar) / step)))
    patterns = (
        (0, 2, 4, 6, 8, 10, 12, 14),
        (0, 1, 4, 6, 8, 9, 12, 14),
        (0, 2, 5, 6, 8, 10, 13, 14),
        (0, 3, 4, 7, 8, 11, 12, 15),
    )
    progression = _harmony_degree_cycle(harmony_profile)
    lane_low = max(int(midi_min_pitch), 60)
    lane_high = min(int(midi_max_pitch), 90)
    if lane_low > lane_high:
        lane_low, lane_high = int(midi_min_pitch), int(midi_max_pitch)

    notes: List[dict] = []
    for window_index, (window_bar_start, window_bars) in enumerate(windows):
        base = patterns[window_index % len(patterns)]
        shift = (window_index * 2) % max(1, steps_per_bar)
        pattern = sorted({(int(step_idx) + shift) % max(1, steps_per_bar) for step_idx in base if int(step_idx) < max(1, steps_per_bar)})
        if window_index % 3 == 1 and len(pattern) > 6:
            pattern = pattern[:-1]
        if window_index % 4 == 2:
            pattern = sorted({*pattern, (window_index + window_bar_start) % max(1, steps_per_bar)})
        degree = progression[window_index % len(progression)]
        chord_pcs = _triad_pitch_classes(scale_pitch_classes, degree)

        for offset in range(int(window_bars)):
            absolute_bar = int(window_bar_start) + int(offset)
            if harmony_plan_by_bar and absolute_bar < len(harmony_plan_by_bar):
                degree = int(harmony_plan_by_bar[absolute_bar])
                chord_pcs = _triad_pitch_classes(scale_pitch_classes, degree)
            bar_start_time = float(absolute_bar) * float(beats_per_bar)
            for pattern_idx, step_idx in enumerate(pattern):
                start_time = bar_start_time + (float(step_idx) * step)
                pitch_class = chord_pcs[(pattern_idx + absolute_bar) % len(chord_pcs)]
                target_pitch = lane_low + ((absolute_bar * 5 + pattern_idx * 3) % max(1, (lane_high - lane_low + 1)))
                duration = float(beats_per_bar) if int(step_idx) == 0 and absolute_bar % 4 == 0 else (step * 2.0 if pattern_idx % 4 == 0 else step)
                velocity = _clamp_velocity(70 + ((absolute_bar * 11 + pattern_idx * 7 + window_index * 5) % 35))
                notes.append(
                    {
                        "pitch": int(
                            _nearest_pitch_for_pitch_class(
                                pitch_class,
                                target_pitch=int(target_pitch),
                                midi_min_pitch=lane_low,
                                midi_max_pitch=lane_high,
                            )
                        ),
                        "start_time": float(round(start_time, 6)),
                        "duration": float(round(duration, 6)),
                        "velocity": int(velocity),
                        "mute": 0,
                    }
                )
    notes.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
    return notes


def _compose_left_hand_right_hand_timeline(
    *,
    total_bars: int,
    beats_per_bar: float,
    grid_step_beats: float,
    windows: Sequence[tuple[int, int]],
    scale_pitch_classes: Sequence[int],
    harmony_profile: str,
    harmony_plan_by_bar: Sequence[int] | None,
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> List[dict]:
    if total_bars <= 0:
        return []
    step = max(1e-6, float(grid_step_beats))
    steps_per_bar = max(1, int(round(float(beats_per_bar) / step)))
    left_patterns = (
        (0, 4, 8, 12),
        (0, 2, 4, 8, 10, 12),
        (0, 4, 7, 8, 12, 15),
    )
    right_patterns = (
        (2, 6, 10, 14),
        (1, 5, 9, 13),
        (3, 7, 11, 15),
    )
    progression = _harmony_degree_cycle(harmony_profile)
    left_low = max(int(midi_min_pitch), 55)
    left_high = min(int(midi_max_pitch), 74)
    right_low = max(int(midi_min_pitch), 72)
    right_high = min(int(midi_max_pitch), 96)
    if left_low > left_high:
        left_low, left_high = int(midi_min_pitch), int(midi_max_pitch)
    if right_low > right_high:
        right_low, right_high = int(midi_min_pitch), int(midi_max_pitch)

    melody_cursor = 0
    notes: List[dict] = []
    for window_index, (window_bar_start, window_bars) in enumerate(windows):
        degree = progression[window_index % len(progression)]
        chord_pcs = _triad_pitch_classes(scale_pitch_classes, degree)
        left_pattern = [int(v) % max(1, steps_per_bar) for v in left_patterns[window_index % len(left_patterns)] if int(v) < max(1, steps_per_bar)]
        right_pattern = [int(v) % max(1, steps_per_bar) for v in right_patterns[window_index % len(right_patterns)] if int(v) < max(1, steps_per_bar)]
        if not left_pattern:
            left_pattern = [0]
        if not right_pattern:
            right_pattern = [max(1, steps_per_bar // 2)]

        for offset in range(int(window_bars)):
            absolute_bar = int(window_bar_start) + int(offset)
            bar_start_time = float(absolute_bar) * float(beats_per_bar)
            if harmony_plan_by_bar and absolute_bar < len(harmony_plan_by_bar):
                local_degree = int(harmony_plan_by_bar[absolute_bar])
            else:
                local_degree = progression[(window_index + offset) % len(progression)]
            local_chord = _triad_pitch_classes(scale_pitch_classes, local_degree)

            for pattern_idx, step_idx in enumerate(left_pattern):
                start_time = bar_start_time + (float(step_idx) * step)
                left_pc = local_chord[pattern_idx % len(local_chord)] if pattern_idx % 2 == 0 else local_chord[2]
                target_pitch = left_low + ((absolute_bar * 3 + pattern_idx * 2) % max(1, (left_high - left_low + 1)))
                notes.append(
                    {
                        "pitch": int(
                            _nearest_pitch_for_pitch_class(
                                left_pc,
                                target_pitch=int(target_pitch),
                                midi_min_pitch=left_low,
                                midi_max_pitch=left_high,
                            )
                        ),
                        "start_time": float(round(start_time, 6)),
                        "duration": float(round(step * 2.0, 6)),
                        "velocity": int(_clamp_velocity(68 + ((absolute_bar * 9 + pattern_idx * 4) % 28))),
                        "mute": 0,
                    }
                )

            melody_count = max(2, min(4, len(right_pattern)))
            for pattern_idx, step_idx in enumerate(right_pattern[:melody_count]):
                start_time = bar_start_time + (float(step_idx) * step)
                melody_cursor = (melody_cursor + (2 if (absolute_bar + pattern_idx) % 3 == 0 else 1)) % len(scale_pitch_classes)
                melody_pc = scale_pitch_classes[melody_cursor]
                if pattern_idx % 2 == 0:
                    melody_pc = chord_pcs[pattern_idx % len(chord_pcs)]
                target_pitch = right_low + ((absolute_bar * 5 + pattern_idx * 7 + window_index * 3) % max(1, (right_high - right_low + 1)))
                if pattern_idx == 0 and absolute_bar % 4 == 0:
                    duration = max(step * 4.0, float(beats_per_bar) * 0.5)
                elif pattern_idx == melody_count - 1 and absolute_bar % 8 == 7:
                    duration = float(beats_per_bar)
                else:
                    duration = step * (2.0 + float(pattern_idx % 2))
                notes.append(
                    {
                        "pitch": int(
                            _nearest_pitch_for_pitch_class(
                                melody_pc,
                                target_pitch=int(target_pitch),
                                midi_min_pitch=right_low,
                                midi_max_pitch=right_high,
                            )
                        ),
                        "start_time": float(round(start_time, 6)),
                        "duration": float(round(duration, 6)),
                        "velocity": int(_clamp_velocity(82 + ((absolute_bar * 7 + pattern_idx * 9) % 30))),
                        "mute": 0,
                    }
                )
    notes.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
    return notes

def _apply_rhythmic_variation(
    notes: Sequence[dict],
    *,
    section: Section,
    beats_per_bar: float,
    beat_step: float,
    preferred_durations_beats: Sequence[float],
) -> List[dict]:
    if not notes:
        return []
    duration_pool = [float(value) for value in preferred_durations_beats if float(value) > 0.0]
    if not duration_pool:
        duration_pool = [max(float(beat_step) * 0.25, 0.25), max(float(beat_step) * 0.5, 0.5), float(beat_step)]
    duration_pool = sorted(set(round(value, 6) for value in duration_pool))
    section_length = float(section.bar_count) * float(beats_per_bar)
    epsilon = max(float(beat_step) * 0.02, 1e-4)
    varied: List[dict] = []
    for idx, note in enumerate(sorted(notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))):
        copied = dict(note)
        start_time = max(0.0, float(copied.get("start_time", 0.0)))
        bar_index = int(start_time // max(float(beats_per_bar), 1e-6))
        local_beat = start_time - (float(bar_index) * float(beats_per_bar))
        on_downbeat = abs(local_beat) <= max(float(beat_step) * 0.04, 1e-4)
        if on_downbeat:
            downbeat_pool = [value for value in duration_pool if value >= max(float(beat_step) * 0.5, 0.25)]
            pool = downbeat_pool or duration_pool
        else:
            pool = duration_pool
        pick = (int(section.index) * 5 + int(bar_index) * 3 + idx) % len(pool)
        preferred = float(pool[pick])
        max_duration = max(0.0, section_length - start_time - epsilon)
        duration = max(0.05, min(preferred, max_duration))
        copied["duration"] = float(round(duration, 6))
        varied.append(copied)
    return varied

def _enforce_max_leap(
    notes: Sequence[dict],
    *,
    max_leap_semitones: int,
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> List[dict]:
    if max_leap_semitones <= 0:
        return [dict(n) for n in notes]
    out: List[dict] = []
    prev_pitch: int | None = None
    for note in sorted(notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0)))):
        new_note = dict(note)
        pitch = int(new_note.get("pitch", 60))
        if prev_pitch is not None:
            leap = int(pitch) - int(prev_pitch)
            if abs(leap) > max_leap_semitones:
                direction = 1 if leap > 0 else -1
                pitch = int(prev_pitch) + (direction * int(max_leap_semitones))
        pitch = _fit_pitch_to_register(pitch, midi_min_pitch, midi_max_pitch)
        new_note["pitch"] = int(pitch)
        out.append(new_note)
        prev_pitch = int(pitch)
    return out

def _extract_strategy(
    identity: MarimbaIdentityConfig,
    strategy_name: str,
) -> tuple[str, Mapping[str, Any]]:
    strategy_pool = identity.payload.get("strategies", {})
    if not isinstance(strategy_pool, Mapping):
        strategy_pool = {}
    fallback = str(identity.strategy_default or "ostinato_pulse")
    wanted = str(strategy_name or "").strip() or fallback
    if wanted == "auto":
        wanted = fallback
    raw = strategy_pool.get(wanted)
    if not isinstance(raw, Mapping):
        raw = strategy_pool.get(fallback)
        if not isinstance(raw, Mapping):
            raw = {}
        wanted = fallback
    return wanted, raw

def _strategy_for_section(
    section: Section,
    requested: str,
    identity: MarimbaIdentityConfig,
) -> str:
    raw = str(requested or "").strip().lower()
    if raw and raw != "auto":
        return raw
    label = str(section.label).strip().lower()
    if label in {"intro", "release", "afterglow"}:
        return "lyrical_roll"
    if label in {"build"}:
        rotation = ("ostinato_pulse", "broken_resonance", "chord_bloom", "lyrical_roll")
        return rotation[int(section.index) % len(rotation)]
    if label in {"pre_climax"}:
        return "broken_resonance"
    if label in {"climax"}:
        return "broken_resonance"
    return str(identity.strategy_default or "ostinato_pulse")

def _marimba_strategy_ostinato(
    notes: Sequence[dict],
    *,
    section: Section,
    beats_per_bar: float,
    beat_step: float,
    strategy: Mapping[str, Any],
    constraints: Mapping[str, Any],
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> List[dict]:
    if not notes:
        return []
    steps_per_bar = max(1, int(round(float(beats_per_bar) / max(float(beat_step), 1e-6))))
    anchor_steps = strategy.get("anchor_steps_6_8", [0, max(1, steps_per_bar // 2)])
    if not isinstance(anchor_steps, Sequence):
        anchor_steps = [0, max(1, steps_per_bar // 2)]
    base_steps = sorted({int(step) % max(1, steps_per_bar) for step in anchor_steps})
    if not base_steps:
        base_steps = [0, max(1, steps_per_bar // 2)]
    pulse_duration = max(float(beat_step) * 0.5, float(strategy.get("pulse_duration_beats", beat_step)))
    accent_cycle = constraints.get("accent_cycle_6_8", [1.0] * steps_per_bar)
    if not isinstance(accent_cycle, Sequence) or not accent_cycle:
        accent_cycle = [1.0] * steps_per_bar

    source = sorted(notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 60))))
    unique_pitches = [int(n.get("pitch", 60)) for n in source]
    if not unique_pitches:
        unique_pitches = [72]

    output: List[dict] = []
    pitch_idx = 0
    for bar in range(max(1, int(section.bar_count))):
        bar_offset = float(bar) * float(beats_per_bar)
        pattern_variants = (
            base_steps,
            sorted({*base_steps, max(1, steps_per_bar // 2)}),
            sorted({(step + 1) % steps_per_bar for step in base_steps}),
            sorted({*base_steps, max(0, steps_per_bar - 1)}),
        )
        anchor_pattern = pattern_variants[(int(section.index) + bar) % len(pattern_variants)]
        for step in anchor_pattern:
            step_idx = int(step) % max(1, steps_per_bar)
            start_time = bar_offset + (float(step_idx) * float(beat_step))
            if start_time >= float(section.bar_count) * float(beats_per_bar):
                continue
            source_note = source[pitch_idx % len(source)]
            pitch = _fit_pitch_to_register(
                int(unique_pitches[pitch_idx % len(unique_pitches)]),
                midi_min_pitch,
                midi_max_pitch,
            )
            accent = float(accent_cycle[step_idx % len(accent_cycle)])
            velocity = _clamp_velocity(float(source_note.get("velocity", 100)) * accent)
            output.append(
                {
                    "pitch": int(pitch),
                    "start_time": float(round(start_time, 6)),
                    "duration": float(round(pulse_duration, 6)),
                    "velocity": int(velocity),
                    "mute": int(source_note.get("mute", 0)),
                }
            )
            pitch_idx += 1
    return output

def _marimba_strategy_broken_resonance(
    notes: Sequence[dict],
    *,
    section: Section,
    beats_per_bar: float,
    beat_step: float,
    strategy: Mapping[str, Any],
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> List[dict]:
    if not notes:
        return []
    anchors = strategy.get("anchor_steps_6_8", [0, 2, 4])
    if not isinstance(anchors, Sequence):
        anchors = [0, 2, 4]
    pulse_duration = max(float(beat_step), float(strategy.get("pulse_duration_beats", beat_step * 1.25)))
    source = sorted(notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 60))))
    base_pitches = [int(n.get("pitch", 60)) for n in source] or [72]
    octave_pattern = (0, 12, -12, 0)
    out: List[dict] = []
    cursor = 0
    for bar in range(max(1, int(section.bar_count))):
        bar_offset = float(bar) * float(beats_per_bar)
        shifted = sorted({(int(step) + ((int(section.index) + bar) % 2)) % max(1, int(round(beats_per_bar / max(beat_step, 1e-6)))) for step in anchors})
        anchor_pattern = shifted if shifted else [0]
        if (bar + int(section.index)) % 3 == 2:
            anchor_pattern = sorted({*anchor_pattern, 0})
        for step in anchor_pattern:
            idx = int(step) % max(1, int(round(beats_per_bar / max(beat_step, 1e-6))))
            st = bar_offset + (float(idx) * float(beat_step))
            if st >= float(section.bar_count) * float(beats_per_bar):
                continue
            template = source[cursor % len(source)]
            base_pitch = int(base_pitches[cursor % len(base_pitches)])
            shaped_pitch = base_pitch + int(octave_pattern[cursor % len(octave_pattern)])
            shaped_pitch = _fit_pitch_to_register(shaped_pitch, midi_min_pitch, midi_max_pitch)
            vel = _clamp_velocity(float(template.get("velocity", 96)) * (0.92 + (0.06 * (cursor % 2))))
            out.append(
                {
                    "pitch": int(shaped_pitch),
                    "start_time": float(round(st, 6)),
                    "duration": float(round(pulse_duration, 6)),
                    "velocity": int(vel),
                    "mute": int(template.get("mute", 0)),
                }
            )
            cursor += 1
    return out


def _marimba_strategy_chord_bloom(
    notes: Sequence[dict],
    *,
    section: Section,
    beats_per_bar: float,
    beat_step: float,
    strategy: Mapping[str, Any],
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> List[dict]:
    if not notes:
        return []

    steps_per_bar = max(1, int(round(float(beats_per_bar) / max(float(beat_step), 1e-6))))
    attack_steps_raw = strategy.get("attack_steps_6_8", strategy.get("anchor_steps_6_8", [0, 2, 4]))
    if not isinstance(attack_steps_raw, Sequence):
        attack_steps_raw = [0, 2, 4]
    attack_steps = sorted(
        {int(step) % max(1, steps_per_bar) for step in attack_steps_raw}
    ) or [0, max(1, steps_per_bar // 2)]

    intervals_raw = strategy.get("chord_intervals", [0, 3, 7, 10])
    if isinstance(intervals_raw, Sequence):
        intervals = [int(value) for value in intervals_raw if isinstance(value, (int, float))]
    else:
        intervals = [0, 3, 7, 10]
    if not intervals:
        intervals = [0, 3, 7, 10]

    chord_size = max(2, min(4, int(strategy.get("chord_size", 3) or 3)))
    pulse_duration = max(float(beat_step), float(strategy.get("pulse_duration_beats", beat_step * 1.5)))
    bloom_offset = max(0.0, float(strategy.get("bloom_offset_beats", beat_step * 0.25)))
    stack_every = max(1, int(strategy.get("stack_every", 2) or 2))
    section_length = float(section.bar_count) * float(beats_per_bar)

    source = sorted(notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 60))))
    pitch_pool = [int(note.get("pitch", 60)) for note in source] or [72]

    out: List[dict] = []
    cursor = 0
    for bar in range(max(1, int(section.bar_count))):
        bar_offset = float(bar) * float(beats_per_bar)
        shifted = sorted(
            {
                (int(step) + ((bar + int(section.index)) % 2)) % max(1, steps_per_bar)
                for step in attack_steps
            }
        )
        anchor_pattern = shifted if shifted else [0]
        for event_idx, step in enumerate(anchor_pattern):
            step_idx = int(step) % max(1, steps_per_bar)
            start_time = bar_offset + (float(step_idx) * float(beat_step))
            if start_time >= section_length:
                continue
            template = source[cursor % len(source)]
            root_pitch = _fit_pitch_to_register(
                int(pitch_pool[cursor % len(pitch_pool)]),
                midi_min_pitch,
                midi_max_pitch,
            )
            stacked_attack = ((bar + event_idx + int(section.index)) % stack_every) == 0
            for voice_idx in range(chord_size):
                interval = int(intervals[voice_idx % len(intervals)])
                voice_pitch = _fit_pitch_to_register(
                    int(root_pitch) + interval,
                    midi_min_pitch,
                    midi_max_pitch,
                )
                voice_start = start_time if stacked_attack else start_time + (float(voice_idx) * bloom_offset)
                if voice_start >= section_length:
                    continue
                remaining = max(0.05, section_length - voice_start)
                voice_duration = max(
                    float(beat_step) * 0.5,
                    float(pulse_duration) * max(0.62, 1.0 - (0.12 * float(voice_idx))),
                )
                velocity_scale = max(0.62, 1.0 - (0.1 * float(voice_idx)))
                out.append(
                    {
                        "pitch": int(voice_pitch),
                        "start_time": float(round(voice_start, 6)),
                        "duration": float(round(min(voice_duration, remaining), 6)),
                        "velocity": _clamp_velocity(float(template.get("velocity", 96)) * velocity_scale),
                        "mute": int(template.get("mute", 0)),
                    }
                )
            cursor += 1
    out.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
    return out


def _marimba_strategy_lyrical_roll(
    notes: Sequence[dict],
    *,
    strategy: Mapping[str, Any],
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> List[dict]:
    if not notes:
        return []
    subdivision = max(0.125, float(strategy.get("roll_subdivision_beats", 0.25)))
    roll_min_duration = max(subdivision, float(strategy.get("roll_min_duration_beats", 0.75)))
    out: List[dict] = []
    for note in sorted(notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 60)))):
        start = float(note.get("start_time", 0.0))
        dur = max(0.05, float(note.get("duration", 0.5)))
        pitch = _fit_pitch_to_register(int(note.get("pitch", 60)), midi_min_pitch, midi_max_pitch)
        velocity = int(note.get("velocity", 96))
        mute = int(note.get("mute", 0))
        if dur < roll_min_duration:
            out.append(
                {
                    "pitch": int(pitch),
                    "start_time": float(round(start, 6)),
                    "duration": float(round(dur, 6)),
                    "velocity": int(velocity),
                    "mute": int(mute),
                }
            )
            continue
        roll_count = max(1, int(round(dur / subdivision)))
        for idx in range(roll_count):
            roll_start = start + (float(idx) * subdivision)
            if roll_start >= start + dur:
                break
            accent = 1.0 if idx % 2 == 0 else 0.88
            out.append(
                {
                    "pitch": int(pitch),
                    "start_time": float(round(roll_start, 6)),
                    "duration": float(round(min(subdivision * 0.92, (start + dur) - roll_start), 6)),
                    "velocity": _clamp_velocity(float(velocity) * accent),
                    "mute": int(mute),
                }
            )
    return out

def _apply_marimba_pair_attack_answer(
    marimba_notes: Sequence[dict],
    vibraphone_notes: Sequence[dict],
    *,
    section_length_beats: float,
    rules: Mapping[str, Any],
) -> tuple[List[dict], List[dict]]:
    min_sep = max(0.0, float(rules.get("min_start_separation_beats", 0.25)))
    marimba_max_dur = max(0.05, float(rules.get("marimba_max_duration_beats", 0.55)))
    vib_min_dur = max(0.05, float(rules.get("vibraphone_min_duration_beats", 0.7)))
    vib_vel_scale = float(rules.get("vibraphone_velocity_scale", 0.9))

    mar_out: List[dict] = []
    mar_starts: List[float] = []
    for note in sorted(marimba_notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0)))):
        new_note = dict(note)
        new_note["duration"] = float(round(min(float(new_note.get("duration", 0.5)), marimba_max_dur), 6))
        mar_out.append(new_note)
        mar_starts.append(float(new_note.get("start_time", 0.0)))

    vib_out: List[dict] = []
    for note in sorted(vibraphone_notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0)))):
        new_note = dict(note)
        start_time = float(new_note.get("start_time", 0.0))
        duration = max(vib_min_dur, float(new_note.get("duration", vib_min_dur)))
        if any(abs(start_time - mar_start) <= min_sep for mar_start in mar_starts):
            start_time += min_sep
        if start_time + duration > float(section_length_beats):
            duration = max(0.05, float(section_length_beats) - start_time)
        if duration <= 0:
            continue
        new_note["start_time"] = float(round(start_time, 6))
        new_note["duration"] = float(round(duration, 6))
        new_note["velocity"] = _clamp_velocity(float(new_note.get("velocity", 90)) * vib_vel_scale)
        vib_out.append(new_note)

    vib_out.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
    return mar_out, vib_out

def _apply_marimba_identity(
    *,
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict]]]],
    specs: Sequence[InstrumentSpec],
    sections: Sequence[Section],
    beats_per_bar: float,
    beat_step: float,
    identity: MarimbaIdentityConfig | None,
    requested_strategy: str,
    key_name: str,
    pair_mode: MarimbaPairMode,
    focus_track: str | None,
    pair_track: str | None,
    bpm: float = 120.0,
) -> tuple[dict[str, List[tuple[Section, List[dict]]]], dict[str, Any]]:
    copied: dict[str, List[tuple[Section, List[dict]]]] = {
        track: [(section, [dict(n) for n in notes]) for section, notes in payloads]
        for track, payloads in arranged_by_track.items()
    }
    metadata: dict[str, Any] = {"enabled": False}
    if identity is None:
        metadata["status"] = "identity_disabled_or_missing"
        return copied, metadata

    spec_by_name = {spec.name.lower(): spec for spec in specs}
    marimba_target = (focus_track or identity.track_name).strip() if focus_track else identity.track_name
    pair_target = (pair_track or identity.pair_track_name).strip() if pair_track else identity.pair_track_name
    marimba_key = _find_track_key(copied, marimba_target)
    if marimba_key is None:
        metadata["status"] = "marimba_track_not_found"
        metadata["requested_track"] = marimba_target
        return copied, metadata
    marimba_spec = spec_by_name.get(str(marimba_key).strip().lower())
    if marimba_spec is None:
        metadata["status"] = "marimba_spec_not_found"
        metadata["requested_track"] = marimba_key
        return copied, metadata

    constraints = identity.payload.get("constraints", {})
    if not isinstance(constraints, Mapping):
        constraints = {}

    max_leap = int(constraints.get("max_leap_semitones", 12) or 12)
    density_max = int(constraints.get("max_density_notes_per_bar", 8) or 8)
    preferred_durations_raw = constraints.get("preferred_durations_beats", (0.25, 0.5, 0.75, 1.0))
    if isinstance(preferred_durations_raw, Sequence):
        preferred_durations = [
            float(value)
            for value in preferred_durations_raw
            if isinstance(value, (int, float)) and float(value) > 0.0
        ]
    else:
        preferred_durations = [0.25, 0.5, 0.75, 1.0]
    if not preferred_durations:
        preferred_durations = [0.25, 0.5, 0.75, 1.0]
    pair_rules_pool = identity.payload.get("pair_rules", {})
    if not isinstance(pair_rules_pool, Mapping):
        pair_rules_pool = {}

    requested_strategy_token = str(requested_strategy or "").strip().lower()
    explicit_strategy_requested = requested_strategy_token not in {"", "auto"}
    composition_family = str(identity.payload.get("composition_family_default", "legacy_sectional")).strip().lower()
    strategy_override_names = {"ostinato_pulse", "broken_resonance", "lyrical_roll", "chord_bloom"}
    if explicit_strategy_requested and requested_strategy_token in strategy_override_names:
        # Explicit strategy requests should take precedence over family defaults.
        composition_family = "legacy_sectional"
    hand_model = str(identity.payload.get("hand_model_default", "four_mallet")).strip().lower()
    if hand_model not in {"two_mallet", "four_mallet"}:
        hand_model = "four_mallet"
    grid_step_beats = float(
        identity.payload.get("grid_step_beats", max(float(beat_step) / 4.0, 0.25))
        or max(float(beat_step) / 4.0, 0.25)
    )
    grid_step_beats = max(0.0625, float(grid_step_beats))
    rhythm_step_beats, tempo_profile = _tempo_rhythm_profile(
        bpm=float(bpm),
        grid_step_beats=grid_step_beats,
        payload=identity.payload,
    )
    mutation_window_bars = _as_positive_int_sequence(identity.payload.get("mutation_window_bars"), (2, 3, 4))
    harmony_block_bars = _as_positive_int_sequence(identity.payload.get("harmony_block_bars"), (16, 24, 32))
    harmony_motion_profile = str(
        identity.payload.get("harmony_motion_profile", "tonic_subdominant_dominant_cycle")
    ).strip().lower()
    max_simultaneous = 2 if hand_model == "two_mallet" else 4

    strategy_usage: dict[str, int] = {}
    marimba_payload = copied.get(marimba_key, [])
    if composition_family in {"evolving_ostinato", "left_hand_ostinato_right_hand_melody"} and sections:
        total_bars = sum(max(1, int(section.bar_count)) for section in sections)
        windows = _mutation_windows(total_bars, mutation_window_bars)
        progression = _harmony_degree_cycle(harmony_motion_profile)
        harmony_plan_by_bar, harmony_blocks = _build_harmony_plan(
            total_bars=total_bars,
            block_sizes=harmony_block_bars,
            progression=progression,
        )
        scale_pitch_classes = _key_scale_pitch_classes(key_name)
        if not scale_pitch_classes:
            scale_pitch_classes = [0, 2, 3, 5, 7, 8, 10]
        if composition_family == "evolving_ostinato":
            timeline_notes = _compose_evolving_ostinato_timeline(
                total_bars=total_bars,
                beats_per_bar=beats_per_bar,
                grid_step_beats=rhythm_step_beats,
                windows=windows,
                scale_pitch_classes=scale_pitch_classes,
                harmony_profile=harmony_motion_profile,
                harmony_plan_by_bar=harmony_plan_by_bar,
                midi_min_pitch=marimba_spec.midi_min_pitch,
                midi_max_pitch=marimba_spec.midi_max_pitch,
            )
        else:
            timeline_notes = _compose_left_hand_right_hand_timeline(
                total_bars=total_bars,
                beats_per_bar=beats_per_bar,
                grid_step_beats=rhythm_step_beats,
                windows=windows,
                scale_pitch_classes=scale_pitch_classes,
                harmony_profile=harmony_motion_profile,
                harmony_plan_by_bar=harmony_plan_by_bar,
                midi_min_pitch=marimba_spec.midi_min_pitch,
                midi_max_pitch=marimba_spec.midi_max_pitch,
            )
        timeline_notes = _quantize_notes_to_grid(
            timeline_notes,
            grid_step_beats=rhythm_step_beats,
            clip_length_beats=float(total_bars) * float(beats_per_bar),
        )
        timeline_notes = _enforce_max_leap(
            timeline_notes,
            max_leap_semitones=max_leap,
            midi_min_pitch=marimba_spec.midi_min_pitch,
            midi_max_pitch=marimba_spec.midi_max_pitch,
        )
        timeline_notes = _constrain_notes_to_key(
            timeline_notes,
            key_name=key_name,
            midi_min_pitch=marimba_spec.midi_min_pitch,
            midi_max_pitch=marimba_spec.midi_max_pitch,
        )
        form_arc_raw = identity.payload.get("form_arc", {})
        if not isinstance(form_arc_raw, Mapping):
            form_arc_raw = {}
        form_arc_enabled = bool(form_arc_raw.get("enabled", True))
        form_arc_peak_ratio = float(form_arc_raw.get("peak_ratio", 0.68))
        density_start_scale = float(form_arc_raw.get("density_start_scale", 0.8))
        density_peak_scale = float(form_arc_raw.get("density_peak_scale", 1.0))
        density_end_scale = float(form_arc_raw.get("density_end_scale", 0.78))
        velocity_start_scale = float(form_arc_raw.get("velocity_start_scale", 0.9))
        velocity_peak_scale = float(form_arc_raw.get("velocity_peak_scale", 1.15))
        velocity_end_scale = float(form_arc_raw.get("velocity_end_scale", 0.9))
        if form_arc_enabled:
            timeline_notes = _apply_form_arc(
                timeline_notes,
                beats_per_bar=beats_per_bar,
                total_bars=total_bars,
                peak_ratio=form_arc_peak_ratio,
                density_start_scale=density_start_scale,
                density_peak_scale=density_peak_scale,
                density_end_scale=density_end_scale,
                velocity_start_scale=velocity_start_scale,
                velocity_peak_scale=velocity_peak_scale,
                velocity_end_scale=velocity_end_scale,
            )
        timeline_notes = _limit_notes_per_bar(
            timeline_notes,
            beats_per_bar=beats_per_bar,
            max_notes_per_bar=max(1, density_max),
        )
        timeline_notes = _enforce_max_simultaneous_notes(
            timeline_notes,
            max_simultaneous=max_simultaneous,
        )
        copied[marimba_key] = _split_timeline_notes_by_sections(
            timeline_notes,
            sections=sections,
            beats_per_bar=beats_per_bar,
            grid_step_beats=grid_step_beats,
        )
        strategy_usage[composition_family] = len(sections)
    else:
        harmony_blocks = []
        form_arc_enabled = False
        form_arc_peak_ratio = 0.68
        composition_family = "legacy_sectional"
        for idx, (section, notes) in enumerate(marimba_payload):
            seed_notes = [dict(n) for n in notes]
            if not seed_notes:
                seed_notes = _seed_notes_for_section(
                    copied,
                    idx,
                    preferred_tracks=("Piano", "Rhodes", "Vibraphone", "Acoustic Guitar"),
                )
            if not seed_notes:
                seed_notes = [
                    {
                        "pitch": 72,
                        "start_time": 0.0,
                        "duration": max(beat_step, 0.5),
                        "velocity": 96,
                        "mute": 0,
                    }
                ]
            resolved_name, strategy = _extract_strategy(
                identity,
                _strategy_for_section(section, requested_strategy, identity),
            )
            strategy_usage[resolved_name] = strategy_usage.get(resolved_name, 0) + 1
            if resolved_name == "broken_resonance":
                shaped = _marimba_strategy_broken_resonance(
                    seed_notes,
                    section=section,
                    beats_per_bar=beats_per_bar,
                    beat_step=beat_step,
                    strategy=strategy,
                    midi_min_pitch=marimba_spec.midi_min_pitch,
                    midi_max_pitch=marimba_spec.midi_max_pitch,
                )
            elif resolved_name == "lyrical_roll":
                shaped = _marimba_strategy_lyrical_roll(
                    seed_notes,
                    strategy=strategy,
                    midi_min_pitch=marimba_spec.midi_min_pitch,
                    midi_max_pitch=marimba_spec.midi_max_pitch,
                )
            elif resolved_name == "chord_bloom":
                shaped = _marimba_strategy_chord_bloom(
                    seed_notes,
                    section=section,
                    beats_per_bar=beats_per_bar,
                    beat_step=beat_step,
                    strategy=strategy,
                    midi_min_pitch=marimba_spec.midi_min_pitch,
                    midi_max_pitch=marimba_spec.midi_max_pitch,
                )
            else:
                shaped = _marimba_strategy_ostinato(
                    seed_notes,
                    section=section,
                    beats_per_bar=beats_per_bar,
                    beat_step=beat_step,
                    strategy=strategy,
                    constraints=constraints,
                    midi_min_pitch=marimba_spec.midi_min_pitch,
                    midi_max_pitch=marimba_spec.midi_max_pitch,
                )

            shaped = _enforce_max_leap(
                shaped,
                max_leap_semitones=max_leap,
                midi_min_pitch=marimba_spec.midi_min_pitch,
                midi_max_pitch=marimba_spec.midi_max_pitch,
            )
            shaped = _apply_rhythmic_variation(
                shaped,
                section=section,
                beats_per_bar=beats_per_bar,
                beat_step=beat_step,
                preferred_durations_beats=preferred_durations,
            )
            max_section_notes = max(1, int(section.bar_count) * max(1, density_max))
            shaped = _limit_note_count(shaped, max_section_notes)
            shaped = _constrain_notes_to_key(
                shaped,
                key_name=key_name,
                midi_min_pitch=marimba_spec.midi_min_pitch,
                midi_max_pitch=marimba_spec.midi_max_pitch,
            )
            copied[marimba_key][idx] = (section, shaped)

    resolved_pair_mode: str = str(pair_mode)
    if resolved_pair_mode == "auto":
        resolved_pair_mode = str(identity.pair_mode_default or "attack_answer")
    pair_key = _find_track_key(copied, pair_target) if resolved_pair_mode != "off" else None
    if resolved_pair_mode == "attack_answer" and pair_key is not None:
        rule = pair_rules_pool.get("attack_answer", {})
        if not isinstance(rule, Mapping):
            rule = {}
        for idx, section in enumerate(sections):
            mar_notes = copied[marimba_key][idx][1]
            vib_notes = copied[pair_key][idx][1]
            section_length = float(section.bar_count) * float(beats_per_bar)
            mar_out, vib_out = _apply_marimba_pair_attack_answer(
                mar_notes,
                vib_notes,
                section_length_beats=section_length,
                rules=rule,
            )
            copied[marimba_key][idx] = (section, mar_out)
            copied[pair_key][idx] = (section, vib_out)

    metadata = {
        "enabled": True,
        "status": "applied",
        "marimba_track": marimba_key,
        "pair_track": pair_key,
        "requested_strategy": requested_strategy,
        "resolved_pair_mode": resolved_pair_mode,
        "composition_family": composition_family,
        "hand_model": hand_model,
        "grid_step_beats": float(grid_step_beats),
        "rhythm_step_beats": float(rhythm_step_beats),
        "tempo_profile": str(tempo_profile),
        "tempo_bpm": float(bpm),
        "mutation_window_bars": list(mutation_window_bars),
        "harmony_block_bars": list(harmony_block_bars),
        "harmony_blocks": [list(item) for item in harmony_blocks],
        "form_arc_enabled": bool(form_arc_enabled),
        "form_arc_peak_ratio": float(form_arc_peak_ratio),
        "harmony_motion_profile": harmony_motion_profile,
        "strategy_usage": strategy_usage,
        "identity_path": str(identity.path),
    }
    return copied, metadata
