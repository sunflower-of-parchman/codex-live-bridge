from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from arrangement.base import InstrumentSpec, Section, _fit_pitch_to_register


@dataclass(frozen=True)
class PassReport:
    pass_index: int
    pass_name: str
    note_count_before: int
    note_count_after: int
    changed_tracks: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_index": int(self.pass_index),
            "pass_name": str(self.pass_name),
            "note_count_before": int(self.note_count_before),
            "note_count_after": int(self.note_count_after),
            "changed_tracks": list(self.changed_tracks),
            "changed_track_count": len(self.changed_tracks),
        }


def _clone_arranged_by_track(
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
) -> dict[str, list[tuple[Section, list[dict[str, Any]]]]]:
    return {
        str(track): [(section, [dict(note) for note in notes]) for section, notes in payloads]
        for track, payloads in arranged_by_track.items()
    }


def _count_notes(arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]]) -> int:
    return sum(len(notes) for payloads in arranged_by_track.values() for _section, notes in payloads)


def _track_payload_hash(payloads: Sequence[tuple[Section, Sequence[dict[str, Any]]]]) -> str:
    rows: list[str] = []
    for section, notes in payloads:
        rows.append(f"s:{int(section.index)}|bars:{int(section.bar_count)}")
        for note in sorted(
            notes,
            key=lambda n: (
                float(n.get("start_time", 0.0)),
                int(n.get("pitch", 0)),
                float(n.get("duration", 0.0)),
                int(n.get("velocity", 0)),
                int(n.get("mute", 0)),
            ),
        ):
            rows.append(
                "n:"
                f"{int(note.get('pitch', 0))}|"
                f"{float(note.get('start_time', 0.0)):.6f}|"
                f"{float(note.get('duration', 0.0)):.6f}|"
                f"{int(note.get('velocity', 0))}|"
                f"{int(note.get('mute', 0))}"
            )
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return digest


def _changed_tracks(
    before: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
    after: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
) -> tuple[str, ...]:
    names = sorted(set(before.keys()) | set(after.keys()))
    changed: list[str] = []
    for name in names:
        left = before.get(name, ())
        right = after.get(name, ())
        if _track_payload_hash(left) != _track_payload_hash(right):
            changed.append(str(name))
    return tuple(changed)


def _seed_rank(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12], 16)


def _clamp_velocity(value: float) -> int:
    return max(1, min(127, int(round(float(value)))))


def _density_ratio_for_label(label: str) -> float:
    table = {
        "intro": 0.75,
        "build": 0.92,
        "pre_climax": 1.03,
        "climax": 1.18,
        "release": 0.72,
        "afterglow": 0.82,
    }
    return float(table.get(str(label).strip().lower(), 1.0))


def _velocity_scale_for_label(label: str) -> float:
    table = {
        "intro": 0.82,
        "build": 0.95,
        "pre_climax": 1.04,
        "climax": 1.16,
        "release": 0.86,
        "afterglow": 0.9,
    }
    return float(table.get(str(label).strip().lower(), 1.0))


def _is_percussion(spec: InstrumentSpec | None) -> bool:
    if spec is None:
        return False
    source = str(spec.source).strip().lower()
    if source in {"kick", "rim", "hat"}:
        return True
    role = str(spec.role).strip().lower()
    return role in {"pulse", "clock", "drum", "percussion"}


def _is_melodic(spec: InstrumentSpec | None, track_name: str) -> bool:
    if spec is not None:
        source = str(spec.source).strip().lower()
        if source in {"piano_motion", "piano_chords"}:
            return True
        if _is_percussion(spec):
            return False
        role = str(spec.role).strip().lower()
        if role in {"motif", "lead", "harmony", "counter"}:
            return True
    name = str(track_name).strip().lower()
    return any(token in name for token in ("marimba", "vibraphone", "piano", "rhodes", "keys"))


def _thin_notes(notes: Sequence[dict[str, Any]], target_count: int, seed_parts: Sequence[object]) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    if target_count >= len(notes):
        return [dict(note) for note in notes]
    ranked = sorted(
        (dict(note) for note in notes),
        key=lambda note: (
            _seed_rank(
                *seed_parts,
                int(note.get("pitch", 0)),
                round(float(note.get("start_time", 0.0)), 6),
                round(float(note.get("duration", 0.0)), 6),
            ),
            float(note.get("start_time", 0.0)),
            int(note.get("pitch", 0)),
        ),
    )
    kept = ranked[:target_count]
    kept.sort(key=lambda note: (float(note.get("start_time", 0.0)), int(note.get("pitch", 0))))
    return kept


def _densify_notes(
    notes: Sequence[dict[str, Any]],
    target_count: int,
    section_length_beats: float,
    beat_step: float,
    seed_parts: Sequence[object],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [dict(note) for note in notes]
    if not notes or target_count <= len(out):
        out.sort(key=lambda note: (float(note.get("start_time", 0.0)), int(note.get("pitch", 0))))
        return out

    epsilon = max(float(beat_step) * 0.02, 1e-4)
    min_duration = max(float(beat_step) * 0.1, 0.05)
    existing_keys = {
        (
            int(note.get("pitch", 0)),
            round(float(note.get("start_time", 0.0)), 6),
            round(float(note.get("duration", 0.0)), 6),
        )
        for note in out
    }
    shifts = [max(float(beat_step) * 0.5, 0.125), float(beat_step), max(float(beat_step) * 1.5, 0.25)]

    idx = 0
    max_attempts = max(32, target_count * 10)
    while len(out) < target_count and idx < max_attempts:
        template = dict(notes[idx % len(notes)])
        shift = shifts[idx % len(shifts)]
        if idx % 2 == 1:
            shift *= -1.0
        start = float(template.get("start_time", 0.0)) + shift
        start = max(0.0, min(start, max(0.0, float(section_length_beats) - min_duration - epsilon)))

        duration = min(float(template.get("duration", beat_step)), max(min_duration, float(section_length_beats) - start - epsilon))
        if duration <= min_duration:
            idx += 1
            continue

        candidate = dict(template)
        candidate["start_time"] = float(round(start, 6))
        candidate["duration"] = float(round(duration, 6))
        candidate["velocity"] = _clamp_velocity(float(candidate.get("velocity", 96)) * 0.9)

        key = (
            int(candidate.get("pitch", 0)),
            round(float(candidate.get("start_time", 0.0)), 6),
            round(float(candidate.get("duration", 0.0)), 6),
        )
        if key in existing_keys:
            idx += 1
            continue

        rank = _seed_rank(*seed_parts, idx, key[0], key[1])
        if rank % 100 < 85:
            out.append(candidate)
            existing_keys.add(key)
        idx += 1

    out.sort(key=lambda note: (float(note.get("start_time", 0.0)), int(note.get("pitch", 0))))
    return out


def _reshape_density(
    notes: Sequence[dict[str, Any]],
    density_ratio: float,
    section_length_beats: float,
    beat_step: float,
    seed_parts: Sequence[object],
) -> list[dict[str, Any]]:
    if not notes:
        return []
    ratio = max(0.0, float(density_ratio))
    target = int(round(len(notes) * ratio))
    if ratio > 0.0:
        target = max(1, target)

    if target == len(notes):
        return [dict(note) for note in notes]
    if target < len(notes):
        return _thin_notes(notes, target, seed_parts)
    return _densify_notes(notes, target, section_length_beats, beat_step, seed_parts)


def _extract_reference_phrase(notes: Sequence[dict[str, Any]], max_notes: int = 4) -> list[dict[str, Any]]:
    if not notes:
        return []
    ordered = sorted(
        (dict(note) for note in notes),
        key=lambda note: (float(note.get("start_time", 0.0)), int(note.get("pitch", 0))),
    )
    picked = ordered[: max(1, int(max_notes))]
    anchor = float(picked[0].get("start_time", 0.0))
    phrase: list[dict[str, Any]] = []
    for note in picked:
        copied = dict(note)
        copied["start_time"] = float(round(float(note.get("start_time", 0.0)) - anchor, 6))
        phrase.append(copied)
    return phrase


def _inject_reference_phrase(
    notes: Sequence[dict[str, Any]],
    phrase: Sequence[dict[str, Any]],
    section_length_beats: float,
    beat_step: float,
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> list[dict[str, Any]]:
    if not phrase:
        return [dict(note) for note in notes]

    out = [dict(note) for note in notes]
    if not out:
        return out

    epsilon = max(float(beat_step) * 0.02, 1e-4)
    existing_positions = {
        (
            int(note.get("pitch", 0)),
            round(float(note.get("start_time", 0.0)), 6),
        )
        for note in out
    }

    base_pitch = int(out[0].get("pitch", 60))
    phrase_root = int(phrase[0].get("pitch", base_pitch))

    for phrase_note in phrase[:2]:
        rel = float(phrase_note.get("start_time", 0.0))
        start = rel
        if start >= float(section_length_beats) - epsilon:
            continue
        duration = min(float(phrase_note.get("duration", beat_step)), float(section_length_beats) - start - epsilon)
        if duration <= max(float(beat_step) * 0.1, 0.05):
            continue

        interval = int(phrase_note.get("pitch", phrase_root)) - phrase_root
        pitch = _fit_pitch_to_register(base_pitch + interval, midi_min_pitch, midi_max_pitch)
        key = (int(pitch), round(float(start), 6))
        if key in existing_positions:
            continue

        out.append(
            {
                "pitch": int(pitch),
                "start_time": float(round(start, 6)),
                "duration": float(round(duration, 6)),
                "velocity": _clamp_velocity(float(phrase_note.get("velocity", 90)) * 0.94),
                "mute": int(phrase_note.get("mute", 0)),
            }
        )
        existing_positions.add(key)

    out.sort(key=lambda note: (float(note.get("start_time", 0.0)), int(note.get("pitch", 0))))
    return out


def _transpose_answer_shape(
    notes: Sequence[dict[str, Any]],
    midi_min_pitch: int,
    midi_max_pitch: int,
    semitones: int,
) -> list[dict[str, Any]]:
    if not notes:
        return []
    out: list[dict[str, Any]] = []
    for idx, note in enumerate(sorted(notes, key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))):
        copied = dict(note)
        if idx % 3 == 1:
            pitch = int(copied.get("pitch", 60)) + int(semitones)
            copied["pitch"] = _fit_pitch_to_register(pitch, midi_min_pitch, midi_max_pitch)
            copied["velocity"] = _clamp_velocity(float(copied.get("velocity", 96)) * 0.96)
        out.append(copied)
    return out


def _apply_form_density_pass(
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
    spec_by_track: Mapping[str, InstrumentSpec],
    beats_per_bar: float,
    beat_step: float,
    seed: int,
) -> dict[str, list[tuple[Section, list[dict[str, Any]]]]]:
    shaped: dict[str, list[tuple[Section, list[dict[str, Any]]]]] = {}
    for track_name, payloads in arranged_by_track.items():
        spec = spec_by_track.get(str(track_name).strip().lower())
        next_payloads: list[tuple[Section, list[dict[str, Any]]]] = []
        for section, notes in payloads:
            section_length = float(section.bar_count) * float(beats_per_bar)
            copied = [dict(note) for note in notes]
            if copied and not _is_percussion(spec):
                ratio = _density_ratio_for_label(section.label)
                copied = _reshape_density(
                    copied,
                    density_ratio=ratio,
                    section_length_beats=section_length,
                    beat_step=beat_step,
                    seed_parts=("form", seed, track_name, section.index, section.label),
                )
            copied.sort(key=lambda note: (float(note.get("start_time", 0.0)), int(note.get("pitch", 0))))
            next_payloads.append((section, copied))
        shaped[str(track_name)] = next_payloads
    return shaped


def _apply_development_pass(
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
    spec_by_track: Mapping[str, InstrumentSpec],
    beats_per_bar: float,
    beat_step: float,
) -> dict[str, list[tuple[Section, list[dict[str, Any]]]]]:
    developed: dict[str, list[tuple[Section, list[dict[str, Any]]]]] = {}
    for track_name, payloads in arranged_by_track.items():
        spec = spec_by_track.get(str(track_name).strip().lower())
        melodic = _is_melodic(spec, str(track_name))
        phrase = []
        if melodic:
            for _section, notes in payloads:
                if notes:
                    phrase = _extract_reference_phrase(notes)
                    if phrase:
                        break

        midi_min = 0 if spec is None else int(spec.midi_min_pitch)
        midi_max = 127 if spec is None else int(spec.midi_max_pitch)

        next_payloads: list[tuple[Section, list[dict[str, Any]]]] = []
        for section, notes in payloads:
            section_length = float(section.bar_count) * float(beats_per_bar)
            copied = [dict(note) for note in notes]
            label = str(section.label).strip().lower()

            if melodic and copied:
                if section.index > 0 and label in {"build", "pre_climax", "climax"}:
                    copied = _inject_reference_phrase(
                        copied,
                        phrase,
                        section_length_beats=section_length,
                        beat_step=beat_step,
                        midi_min_pitch=midi_min,
                        midi_max_pitch=midi_max,
                    )
                if section.index % 2 == 1 and label in {"build", "pre_climax", "climax", "afterglow"}:
                    copied = _transpose_answer_shape(
                        copied,
                        midi_min_pitch=midi_min,
                        midi_max_pitch=midi_max,
                        semitones=2,
                    )

            copied.sort(key=lambda note: (float(note.get("start_time", 0.0)), int(note.get("pitch", 0))))
            next_payloads.append((section, copied))
        developed[str(track_name)] = next_payloads
    return developed


def _apply_dynamic_pass(
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
    spec_by_track: Mapping[str, InstrumentSpec],
    beats_per_bar: float,
) -> dict[str, list[tuple[Section, list[dict[str, Any]]]]]:
    dynamic: dict[str, list[tuple[Section, list[dict[str, Any]]]]] = {}
    for track_name, payloads in arranged_by_track.items():
        spec = spec_by_track.get(str(track_name).strip().lower())
        percussion = _is_percussion(spec)
        next_payloads: list[tuple[Section, list[dict[str, Any]]]] = []
        for section, notes in payloads:
            section_length = max(1e-6, float(section.bar_count) * float(beats_per_bar))
            label_scale = _velocity_scale_for_label(section.label)
            if percussion:
                label_scale *= 0.92
            shaped_notes: list[dict[str, Any]] = []
            for note in notes:
                copied = dict(note)
                progress = max(0.0, min(1.0, float(copied.get("start_time", 0.0)) / section_length))
                contour = 0.9 + (0.2 * progress)
                copied["velocity"] = _clamp_velocity(float(copied.get("velocity", 96)) * label_scale * contour)
                shaped_notes.append(copied)
            shaped_notes.sort(key=lambda item: (float(item.get("start_time", 0.0)), int(item.get("pitch", 0))))
            next_payloads.append((section, shaped_notes))
        dynamic[str(track_name)] = next_payloads
    return dynamic


def _normalize_note_token(token: str) -> str:
    text = str(token).strip().lower()
    text = text.replace("♯", "#").replace("♭", "b")
    text = text.replace("sharp", "#").replace("flat", "b")
    text = text.replace("-", "")
    return text


def _key_root_pitch_class(key_name: str) -> int | None:
    if not key_name:
        return None
    root = _normalize_note_token(str(key_name).split()[0])
    mapping = {
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
    return mapping.get(root)


def _nearest_pitch_for_class(
    existing_pitches: Sequence[int],
    pitch_class: int,
    midi_min_pitch: int,
    midi_max_pitch: int,
) -> int:
    low = max(0, min(127, int(midi_min_pitch)))
    high = max(0, min(127, int(midi_max_pitch)))
    if low > high:
        low, high = high, low

    center = int(round(sum(existing_pitches) / len(existing_pitches))) if existing_pitches else int((low + high) / 2)
    candidates = [pitch for pitch in range(low, high + 1) if pitch % 12 == int(pitch_class) % 12]
    if not candidates:
        return _fit_pitch_to_register(center, low, high)
    return min(candidates, key=lambda value: (abs(value - center), value))


def _apply_cadence_pass(
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
    spec_by_track: Mapping[str, InstrumentSpec],
    sections: Sequence[Section],
    beats_per_bar: float,
    beat_step: float,
    key_name: str,
) -> dict[str, list[tuple[Section, list[dict[str, Any]]]]]:
    cadenced = _clone_arranged_by_track(arranged_by_track)
    if not sections:
        return cadenced

    root_class = _key_root_pitch_class(key_name)
    if root_class is None:
        return cadenced

    # Prefer marimba for cadence; otherwise first melodic track.
    target_track: str | None = None
    for name in cadenced.keys():
        if "marimba" in str(name).strip().lower():
            target_track = str(name)
            break
    if target_track is None:
        for name in cadenced.keys():
            spec = spec_by_track.get(str(name).strip().lower())
            if _is_melodic(spec, str(name)):
                target_track = str(name)
                break
    if target_track is None:
        return cadenced

    payloads = cadenced.get(target_track)
    if payloads is None or not payloads:
        return cadenced

    last_index = len(payloads) - 1
    last_section, last_notes = payloads[last_index]
    section_length = float(last_section.bar_count) * float(beats_per_bar)
    if section_length <= 0:
        return cadenced

    spec = spec_by_track.get(str(target_track).strip().lower())
    midi_min = 0 if spec is None else int(spec.midi_min_pitch)
    midi_max = 127 if spec is None else int(spec.midi_max_pitch)
    existing_pitches = [int(note.get("pitch", 60)) for note in last_notes] or [60]
    cadence_pitch = _nearest_pitch_for_class(existing_pitches, root_class, midi_min, midi_max)

    cadence_start = max(0.0, float(section_length) - float(beats_per_bar))
    cadence_duration = max(float(beat_step), min(float(beats_per_bar), float(section_length) - cadence_start))

    epsilon = max(float(beat_step) * 0.02, 1e-4)
    has_cadence = any(
        int(note.get("pitch", 0)) % 12 == cadence_pitch % 12
        and abs(float(note.get("start_time", 0.0)) - cadence_start) <= epsilon
        for note in last_notes
    )

    if not has_cadence:
        cadence_velocity = _clamp_velocity(max(96, max((int(note.get("velocity", 0)) for note in last_notes), default=96)))
        last_notes.append(
            {
                "pitch": int(cadence_pitch),
                "start_time": float(round(cadence_start, 6)),
                "duration": float(round(cadence_duration, 6)),
                "velocity": int(cadence_velocity),
                "mute": 0,
            }
        )
        last_notes.sort(key=lambda note: (float(note.get("start_time", 0.0)), int(note.get("pitch", 0))))
        payloads[last_index] = (last_section, last_notes)
        cadenced[target_track] = payloads

    return cadenced


def _apply_polish_pass(
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
    spec_by_track: Mapping[str, InstrumentSpec],
    pass_number: int,
) -> dict[str, list[tuple[Section, list[dict[str, Any]]]]]:
    polished = _clone_arranged_by_track(arranged_by_track)
    amount = 2 if int(pass_number) % 2 == 0 else 1
    for track_name, payloads in polished.items():
        spec = spec_by_track.get(str(track_name).strip().lower())
        if _is_percussion(spec):
            continue
        for idx, (section, notes) in enumerate(payloads):
            next_notes: list[dict[str, Any]] = []
            for note in notes:
                copied = dict(note)
                rank = _seed_rank("polish", pass_number, track_name, section.index, copied.get("start_time", 0.0), copied.get("pitch", 0))
                delta = amount if rank % 2 == 0 else -amount
                copied["velocity"] = _clamp_velocity(int(copied.get("velocity", 96)) + delta)
                next_notes.append(copied)
            next_notes.sort(key=lambda item: (float(item.get("start_time", 0.0)), int(item.get("pitch", 0))))
            payloads[idx] = (section, next_notes)
        polished[str(track_name)] = payloads
    return polished


def run_multi_pass_pipeline(
    *,
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
    specs: Sequence[InstrumentSpec],
    sections: Sequence[Section],
    beats_per_bar: float,
    beat_step: float,
    key_name: str,
    pass_count: int,
    seed: int,
) -> tuple[dict[str, list[tuple[Section, list[dict[str, Any]]]]], list[dict[str, Any]]]:
    total_passes = max(1, int(pass_count))
    spec_by_track = {str(spec.name).strip().lower(): spec for spec in specs}

    current = _clone_arranged_by_track(arranged_by_track)
    reports: list[dict[str, Any]] = []

    pass_names = [
        "seed_layout",
        "form_density",
        "repetition_development",
        "dynamic_arc",
        "cadence",
    ]

    for pass_index in range(1, total_passes + 1):
        pass_name = pass_names[pass_index - 1] if pass_index <= len(pass_names) else f"polish_{pass_index - len(pass_names)}"
        before = _clone_arranged_by_track(current)
        note_count_before = _count_notes(before)

        if pass_name == "seed_layout":
            after = _clone_arranged_by_track(before)
        elif pass_name == "form_density":
            after = _apply_form_density_pass(
                before,
                spec_by_track=spec_by_track,
                beats_per_bar=beats_per_bar,
                beat_step=beat_step,
                seed=seed,
            )
        elif pass_name == "repetition_development":
            after = _apply_development_pass(
                before,
                spec_by_track=spec_by_track,
                beats_per_bar=beats_per_bar,
                beat_step=beat_step,
            )
        elif pass_name == "dynamic_arc":
            after = _apply_dynamic_pass(
                before,
                spec_by_track=spec_by_track,
                beats_per_bar=beats_per_bar,
            )
        elif pass_name == "cadence":
            after = _apply_cadence_pass(
                before,
                spec_by_track=spec_by_track,
                sections=sections,
                beats_per_bar=beats_per_bar,
                beat_step=beat_step,
                key_name=key_name,
            )
        else:
            after = _apply_polish_pass(
                before,
                spec_by_track=spec_by_track,
                pass_number=pass_index,
            )

        note_count_after = _count_notes(after)
        changed = _changed_tracks(before, after)
        report = PassReport(
            pass_index=pass_index,
            pass_name=pass_name,
            note_count_before=note_count_before,
            note_count_after=note_count_after,
            changed_tracks=changed,
        )
        reports.append(report.as_dict())
        current = after

    return current, reports
