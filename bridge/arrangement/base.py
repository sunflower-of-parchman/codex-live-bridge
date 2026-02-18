from __future__ import annotations

import hashlib
import json
import math
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Literal, Mapping, Sequence, Tuple

import ableton_udp_bridge as bridge
import compose_hat_pattern as hat
import compose_kick_pattern as kick
import compose_piano_pattern as piano
import compose_rim_pattern as rim


HOST = bridge.DEFAULT_HOST

PORT = bridge.DEFAULT_PORT

ACK_PORT = bridge.DEFAULT_ACK_PORT

DEFAULT_MINUTES_MIN = 3.0

DEFAULT_MINUTES_MAX = 6.0

DEFAULT_BPM = 137.0

DEFAULT_SIG_NUM = 6

DEFAULT_SIG_DEN = 4

DEFAULT_SECTION_BARS = 8

DEFAULT_TRANSPOSE = 2

DEFAULT_MOOD = "Energetic"

DEFAULT_KEY = "D minor"

DEFAULT_KICK_TRACK = "Kick Drum"

DEFAULT_RIM_TRACK = rim.DEFAULT_RIM_TRACK_NAME

DEFAULT_HAT_TRACK = hat.DEFAULT_HAT_TRACK_NAME

DEFAULT_PIANO_TRACK = piano.DEFAULT_PIANO_TRACK_NAME

DEFAULT_GROOVE_NAME = kick.DEFAULT_GROOVE_NAME

DEFAULT_NOTE_CHUNK_SIZE = 40

DEFAULT_WRITE_STRATEGY = "full_replace"

DEFAULT_WRITE_CACHE_PATH = Path("bridge/.cache/arrangement_write_cache.json")

DEFAULT_INSTRUMENT_REGISTRY_PATH = Path("bridge/config/instrument_registry.marimba.v1.json")

DEFAULT_MARIMBA_IDENTITY_PATH = Path("bridge/config/marimba_identity.v1.json")

DEFAULT_SAVE_POLICY = "ephemeral"

DEFAULT_ARCHIVE_DIR = Path("output/live_sets")

DEFAULT_COMPOSITION_PRINT_DIR = Path("memory/composition_prints")

OscAck = Tuple[str, List[bridge.OscArg]]

PianoMode = Literal["full", "chords", "motion", "out"]

HatDensity = Literal["quarter", "eighth", "sixteenth"]
SectionProfileFamily = Literal["legacy_arc", "lift_release", "wave_train"]

WriteStrategy = Literal["full_replace", "delta_update"]

SavePolicy = Literal["ephemeral", "archive"]

TrackNamingMode = Literal["slot", "registry"]

MarimbaPairMode = Literal["auto", "off", "attack_answer"]

ClipWriteMode = Literal["section_clips", "single_clip"]

DEFAULT_TRACK_NAMING_MODE: TrackNamingMode = "registry"

DEFAULT_CLIP_WRITE_MODE: ClipWriteMode = "section_clips"

def _req_id(*parts: object) -> str:
    raw = "-".join(str(p) for p in parts if p is not None)
    return raw.replace(" ", "_")

def _slug_token(text: str) -> str:
    raw = str(text).strip().lower()
    if not raw:
        return "untitled"
    slug = re.sub(r"[^a-z0-9]+", "_", raw)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "untitled"

def _safe_bpm_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "_")

def _run_label(sig_num: int, sig_den: int, bpm: float, mood: str) -> str:
    _ = int(sig_den)  # Label uses meter numerator only.
    _ = str(mood)  # Label no longer encodes mood.
    return f"{int(sig_num)}_{_safe_bpm_label(bpm)}"

def _slot_track_name(index: int) -> str:
    return f"Instrument {index + 1:02d}"

def _build_live_track_names(
    specs: Sequence["InstrumentSpec"],
    mode: TrackNamingMode,
) -> dict[str, str]:
    names: dict[str, str] = {}
    for idx, spec in enumerate(specs):
        if mode == "slot":
            names[spec.name] = _slot_track_name(idx)
        else:
            names[spec.name] = spec.name
    return names

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

def _beats_per_bar(sig_num: int, sig_den: int) -> float:
    return kick._beats_per_bar_from_signature(sig_num, sig_den)

def _beat_step(sig_den: int) -> float:
    return kick._beat_step_from_denominator(sig_den)

def _bars_for_minutes(bpm: float, beats_per_bar: float, minutes: float) -> int:
    if bpm <= 0:
        raise ValueError("bpm must be > 0")
    if beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be > 0")
    if minutes <= 0:
        raise ValueError("minutes must be > 0")
    total_beats_target = float(bpm) * float(minutes)
    bars = int(math.ceil(total_beats_target / float(beats_per_bar)))
    return max(1, bars)

@dataclass(frozen=True)
class Section:
    index: int
    start_bar: int
    bar_count: int
    label: str
    kick_on: bool
    rim_on: bool
    hat_on: bool
    piano_mode: PianoMode
    kick_keep_ratio: float
    rim_keep_ratio: float
    hat_keep_ratio: float
    hat_density: HatDensity

def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

def _section_profile_landmarks(
    index: int,
    total_sections: int,
) -> tuple[int, int]:
    if total_sections <= 0:
        raise ValueError("total_sections must be > 0")
    last_index = total_sections - 1
    if total_sections <= 3:
        climax_index = last_index
    else:
        climax_index = max(1, int(math.floor(float(total_sections) * (2.0 / 3.0))))
    pre_climax_index = max(1, climax_index - 1)
    return pre_climax_index, climax_index


def _section_profile_legacy(
    index: int,
    total_sections: int,
) -> tuple[str, bool, bool, bool, PianoMode, float, float, float, HatDensity]:
    """Return the original arc-shaped arrangement profile."""
    if total_sections <= 0:
        raise ValueError("total_sections must be > 0")

    last_index = total_sections - 1
    pre_climax_index, climax_index = _section_profile_landmarks(index, total_sections)

    # Intro: light elements only, with hats marking time gently.
    if index == 0:
        return ("intro", False, False, True, "chords", 0.0, 0.0, 0.70, "quarter")

    # Early build: keep percussion light and save kick/rim for later power.
    if index < pre_climax_index:
        return ("build", False, False, True, "chords", 0.0, 0.0, 0.85, "eighth")

    # Climax: all parts in near the final third.
    if index == climax_index:
        return ("climax", True, True, True, "chords", 1.0, 1.0, 1.0, "sixteenth")

    # Pre-climax: introduce kick and rim more sparingly.
    if index == pre_climax_index:
        return ("pre_climax", True, True, True, "chords", 0.70, 0.45, 1.0, "eighth")

    # Release: thin back out after the climax.
    if index >= last_index:
        return ("release", True, False, True, "chords", 0.65, 0.0, 0.70, "quarter")

    return ("afterglow", True, False, True, "chords", 0.75, 0.0, 0.85, "eighth")


def _section_profile_lift_release(
    index: int,
    total_sections: int,
) -> tuple[str, bool, bool, bool, PianoMode, float, float, float, HatDensity]:
    """Return a profile with clearer harmonic lift and explicit release thinning."""
    if total_sections <= 0:
        raise ValueError("total_sections must be > 0")
    last_index = total_sections - 1
    pre_climax_index, climax_index = _section_profile_landmarks(index, total_sections)

    if index == 0:
        return ("intro", False, False, True, "out", 0.0, 0.0, 0.62, "quarter")
    if index == climax_index:
        return ("climax", True, True, True, "full", 1.0, 0.85, 1.0, "sixteenth")
    if index == pre_climax_index:
        return ("pre_climax", True, False, True, "chords", 0.72, 0.0, 0.95, "eighth")
    if index >= last_index:
        return ("release", True, False, True, "out", 0.52, 0.0, 0.58, "quarter")
    if index < pre_climax_index:
        return ("build", False, False, True, "motion", 0.0, 0.0, 0.82, "eighth")
    return ("afterglow", True, False, True, "motion", 0.62, 0.0, 0.72, "eighth")


def _section_profile_wave_train(
    index: int,
    total_sections: int,
) -> tuple[str, bool, bool, bool, PianoMode, float, float, float, HatDensity]:
    """Return a profile with repeating intensity waves before final release."""
    if total_sections <= 0:
        raise ValueError("total_sections must be > 0")
    last_index = total_sections - 1
    if total_sections <= 3:
        return _section_profile_legacy(index, total_sections)

    if index == 0:
        return ("intro", False, False, True, "chords", 0.0, 0.0, 0.70, "quarter")
    if index >= last_index:
        return ("release", True, False, True, "out", 0.55, 0.0, 0.62, "quarter")
    if index == last_index - 1:
        return ("afterglow", True, False, True, "chords", 0.66, 0.0, 0.74, "eighth")

    phase = (index - 1) % 3
    if phase == 0:
        return ("build", False, False, True, "motion", 0.0, 0.0, 0.86, "eighth")
    if phase == 1:
        return ("pre_climax", True, False, True, "chords", 0.72, 0.0, 0.96, "eighth")
    return ("climax", True, True, True, "full", 0.92, 0.72, 1.0, "sixteenth")


def _section_profile(
    index: int,
    total_sections: int,
    profile_family: SectionProfileFamily = "legacy_arc",
) -> tuple[str, bool, bool, bool, PianoMode, float, float, float, HatDensity]:
    family = str(profile_family).strip().lower()
    if family == "lift_release":
        return _section_profile_lift_release(index, total_sections)
    if family == "wave_train":
        return _section_profile_wave_train(index, total_sections)
    return _section_profile_legacy(index, total_sections)


def _select_section_profile_family(
    *,
    sig_num: int,
    sig_den: int,
    bpm: float,
    mood: str,
    seed: int,
) -> SectionProfileFamily:
    mood_token = str(mood).strip().lower()
    candidates: tuple[SectionProfileFamily, ...] = ("legacy_arc", "lift_release", "wave_train")
    if any(token in mood_token for token in ("ambient", "calm", "dream", "soft", "beautiful")):
        candidates = ("legacy_arc", "lift_release")
    elif any(token in mood_token for token in ("energetic", "driving", "aggressive", "tense")):
        candidates = ("wave_train", "legacy_arc")
    elif float(bpm) >= 132.0:
        candidates = ("wave_train", "lift_release", "legacy_arc")

    value = _stable_hash_to_unit(seed, sig_num, sig_den, bpm, mood_token, "section_profile_family")
    pick = int(math.floor(value * len(candidates)))
    if pick >= len(candidates):
        pick = len(candidates) - 1
    return candidates[pick]


def _build_sections(
    total_bars: int,
    section_bars: int,
    profile_family: SectionProfileFamily = "legacy_arc",
) -> List[Section]:
    if total_bars <= 0:
        raise ValueError("total_bars must be > 0")
    if section_bars <= 0:
        raise ValueError("section_bars must be > 0")

    total_sections = int(math.ceil(float(total_bars) / float(section_bars)))
    sections: List[Section] = []
    start_bar = 0
    index = 0
    while start_bar < total_bars:
        remaining = total_bars - start_bar
        bar_count = min(section_bars, remaining)
        (
            label,
            kick_on,
            rim_on,
            hat_on,
            piano_mode,
            kick_keep_ratio,
            rim_keep_ratio,
            hat_keep_ratio,
            hat_density,
        ) = _section_profile(index, total_sections, profile_family=profile_family)
        sections.append(
            Section(
                index=index,
                start_bar=start_bar,
                bar_count=bar_count,
                label=label,
                kick_on=kick_on,
                rim_on=rim_on,
                hat_on=hat_on,
                piano_mode=piano_mode,
                kick_keep_ratio=_clamp_ratio(kick_keep_ratio),
                rim_keep_ratio=_clamp_ratio(rim_keep_ratio),
                hat_keep_ratio=_clamp_ratio(hat_keep_ratio),
                hat_density=hat_density,
            )
        )
        start_bar += bar_count
        index += 1
    return sections

def _section_bounds(section: Section, beats_per_bar: float) -> tuple[float, float, float]:
    start = float(section.start_bar) * float(beats_per_bar)
    length = float(section.bar_count) * float(beats_per_bar)
    end = start + length
    return start, end, length

@dataclass(frozen=True)
class ArrangementClipRef:
    path: str
    clip_id: int | None
    start_time: float
    end_time: float

@dataclass(frozen=True)
class NoteDelta:
    remove_note_ids: Tuple[int, ...]
    modification_notes: Tuple[dict, ...]
    add_notes: Tuple[dict, ...]
    requires_full_replace: bool

@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    source: str
    role: str
    priority: int
    required: bool
    active_min: int
    active_max: int
    pitch_shift: int
    velocity_scale: float
    keep_ratio_scale: float
    allow_labels: Tuple[str, ...]
    apply_groove: bool
    mute_by_default: bool = False
    midi_min_pitch: int = 0
    midi_max_pitch: int = 127

@dataclass(frozen=True)
class MarimbaIdentityConfig:
    path: Path
    payload: dict[str, Any]
    track_name: str
    pair_track_name: str
    strategy_default: str
    pair_mode_default: str

def _repo_root() -> Path:
    # base.py lives at bridge/arrangement/base.py; repo root is two levels up.
    return Path(__file__).resolve().parents[2]

def _resolve_registry_path(raw_path: str | None) -> Path:
    if raw_path in (None, ""):
        return _repo_root() / DEFAULT_INSTRUMENT_REGISTRY_PATH
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return _repo_root() / path

def _resolve_marimba_identity_path(raw_path: str | None) -> Path:
    if raw_path in (None, ""):
        return _repo_root() / DEFAULT_MARIMBA_IDENTITY_PATH
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return _repo_root() / path

def _load_marimba_identity(path: Path) -> MarimbaIdentityConfig | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid marimba identity config: {path}")
    if not bool(payload.get("enabled", True)):
        return None
    track_name = str(payload.get("track_name", "Marimba")).strip() or "Marimba"
    pair_track_name = str(payload.get("pair_track_name", "Vibraphone")).strip() or "Vibraphone"
    strategy_default = str(payload.get("strategy_default", "ostinato_pulse")).strip() or "ostinato_pulse"
    pair_mode_default = str(payload.get("pair_mode_default", "attack_answer")).strip() or "attack_answer"
    return MarimbaIdentityConfig(
        path=path,
        payload=payload,
        track_name=track_name,
        pair_track_name=pair_track_name,
        strategy_default=strategy_default,
        pair_mode_default=pair_mode_default,
    )

def _as_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, parsed)

def _as_positive_int(value: Any, default: int = 1) -> int:
    parsed = _as_non_negative_int(value, default=default)
    return parsed if parsed > 0 else int(default)

def _as_float_clamped(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return max(float(minimum), min(float(maximum), parsed))

def _load_instrument_registry(path: Path) -> List[InstrumentSpec]:
    if not path.exists():
        raise FileNotFoundError(f"instrument registry not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    instruments = data.get("instruments") if isinstance(data, dict) else None
    if not isinstance(instruments, list) or not instruments:
        raise ValueError(f"invalid instrument registry (missing instruments list): {path}")

    seen: set[str] = set()
    specs: List[InstrumentSpec] = []
    for idx, entry in enumerate(instruments):
        if not isinstance(entry, dict):
            raise ValueError(f"instrument entry at index {idx} is not an object")

        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"instrument entry at index {idx} is missing name")
        if name in seen:
            raise ValueError(f"duplicate instrument name in registry: {name}")
        seen.add(name)

        source = str(entry.get("source", "")).strip().lower()
        if source not in {"kick", "rim", "hat", "piano_chords", "piano_motion"}:
            raise ValueError(f"unsupported source '{source}' for instrument '{name}'")

        role = str(entry.get("role", "support")).strip().lower() or "support"
        priority = _as_positive_int(entry.get("priority", 50), default=50)
        required = bool(entry.get("required", False))
        active_min = _as_non_negative_int(entry.get("active_min", 0), default=0)
        active_max = _as_non_negative_int(entry.get("active_max", active_min), default=active_min)
        if active_max < active_min:
            active_max = active_min

        labels_raw = entry.get("allow_labels", ())
        labels: List[str] = []
        if isinstance(labels_raw, list):
            for item in labels_raw:
                text = str(item).strip().lower()
                if text:
                    labels.append(text)

        midi_min_pitch = _as_non_negative_int(entry.get("midi_min_pitch", 0), default=0)
        midi_max_pitch = _as_non_negative_int(entry.get("midi_max_pitch", 127), default=127)
        midi_min_pitch = max(0, min(127, int(midi_min_pitch)))
        midi_max_pitch = max(0, min(127, int(midi_max_pitch)))
        if midi_min_pitch > midi_max_pitch:
            midi_min_pitch, midi_max_pitch = midi_max_pitch, midi_min_pitch

        spec = InstrumentSpec(
            name=name,
            source=source,
            role=role,
            priority=priority,
            required=required,
            active_min=active_min,
            active_max=active_max,
            pitch_shift=int(entry.get("pitch_shift", 0) or 0),
            velocity_scale=_as_float_clamped(entry.get("velocity_scale", 1.0), 1.0, 0.1, 2.0),
            keep_ratio_scale=_as_float_clamped(entry.get("keep_ratio_scale", 1.0), 1.0, 0.0, 2.0),
            allow_labels=tuple(labels),
            apply_groove=bool(entry.get("apply_groove", False)),
            mute_by_default=bool(entry.get("mute_by_default", False)),
            midi_min_pitch=midi_min_pitch,
            midi_max_pitch=midi_max_pitch,
        )
        specs.append(spec)

    return specs

def _stable_hash_to_unit(*parts: object) -> float:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16)
    return float(value % 10_000_000) / 10_000_000.0

def _select_minutes(
    *,
    explicit_minutes: float | None,
    minutes_min: float,
    minutes_max: float,
    seed: int,
    bpm: float,
    sig_num: int,
    sig_den: int,
    mood: str,
    key_name: str,
) -> float:
    if explicit_minutes is not None:
        return float(explicit_minutes)
    if minutes_max <= minutes_min:
        return float(minutes_min)
    fraction = _stable_hash_to_unit(seed, bpm, sig_num, sig_den, mood, key_name)
    selected = float(minutes_min) + (float(minutes_max) - float(minutes_min)) * fraction
    # Quarter-minute granularity keeps section math more legible.
    quantized = round(selected * 4.0) / 4.0
    return max(float(minutes_min), min(float(minutes_max), float(quantized)))

def _pick_section_indices(
    candidates: Sequence[int],
    count: int,
    seed_parts: Sequence[object],
) -> set[int]:
    ranked = sorted(
        candidates,
        key=lambda idx: _stable_hash_to_unit(*seed_parts, idx),
    )
    return set(ranked[: max(0, min(len(ranked), int(count)))])

def _build_activation_mask(
    specs: Sequence[InstrumentSpec],
    sections: Sequence[Section],
    seed: int,
    force_active_names: Sequence[str] | None = None,
) -> dict[str, set[int]]:
    total_sections = len(sections)
    section_indices = [s.index for s in sections]
    labels_by_idx = {s.index: str(s.label).lower() for s in sections}

    mask: dict[str, set[int]] = {}
    for spec in specs:
        if spec.required:
            mask[spec.name] = set(section_indices)
            continue

        candidates = list(section_indices)
        if spec.allow_labels:
            allowed = set(spec.allow_labels)
            candidates = [idx for idx in section_indices if labels_by_idx.get(idx, "") in allowed]
        if not candidates:
            candidates = list(section_indices)

        min_count = max(0, min(total_sections, int(spec.active_min)))
        max_count = max(min_count, min(total_sections, int(spec.active_max)))
        count_float = float(min_count) + (float(max_count - min_count) * _stable_hash_to_unit(seed, spec.name, "count"))
        desired_count = int(round(count_float))
        desired_count = max(min_count, min(max_count, desired_count))
        mask[spec.name] = _pick_section_indices(candidates, desired_count, (seed, spec.name, "pick"))

    # Cap section density to avoid every instrument stacking all the time.
    max_active_per_section = max(4, int(math.ceil(len(specs) * 0.5)))
    specs_by_name = {spec.name: spec for spec in specs}
    for section_idx in section_indices:
        active_names = [name for name, idxs in mask.items() if section_idx in idxs]
        if len(active_names) <= max_active_per_section:
            continue
        active_names.sort(
            key=lambda name: (
                0 if specs_by_name[name].required else 1,
                specs_by_name[name].priority,
                _stable_hash_to_unit(seed, "cap", section_idx, name),
            )
        )
        for name in active_names[max_active_per_section:]:
            if specs_by_name[name].required:
                continue
            mask[name].discard(section_idx)

    forced: set[str] = set()
    if force_active_names:
        forced = {str(name).strip().lower() for name in force_active_names if str(name).strip()}
    if forced:
        all_sections = set(section_indices)
        for spec in specs:
            if str(spec.name).strip().lower() in forced:
                mask[spec.name] = set(all_sections)
    return mask

def _fit_pitch_to_register(pitch: int, midi_min_pitch: int, midi_max_pitch: int) -> int:
    low = max(0, min(127, int(midi_min_pitch)))
    high = max(0, min(127, int(midi_max_pitch)))
    if low > high:
        low, high = high, low

    # Keep pitch class when possible by moving in octaves.
    candidates: list[int] = []
    base_pitch = int(pitch)
    for octave in range(-10, 11):
        candidate = base_pitch + (12 * octave)
        if low <= candidate <= high:
            candidates.append(candidate)
    if candidates:
        return int(min(candidates, key=lambda value: (abs(value - base_pitch), value)))

    # If no same pitch-class note exists in the range, clamp to the nearest bound.
    return max(low, min(high, base_pitch))

def _transform_instrument_notes(
    notes: Sequence[dict],
    spec: InstrumentSpec,
    section: Section,
    beats_per_bar: float,
    beat_step: float,
) -> List[dict]:
    transformed: List[dict] = []
    for note in notes:
        out = dict(note)
        pitch = int(out.get("pitch", 60)) + int(spec.pitch_shift)
        out["pitch"] = _fit_pitch_to_register(pitch, spec.midi_min_pitch, spec.midi_max_pitch)
        velocity = int(round(int(out.get("velocity", 100)) * float(spec.velocity_scale)))
        out["velocity"] = max(1, min(127, velocity))
        out["mute"] = 1 if spec.mute_by_default else int(out.get("mute", 0))
        transformed.append(out)

    ratio = _clamp_ratio(1.0)
    if spec.keep_ratio_scale < 0.999:
        ratio = _clamp_ratio(float(spec.keep_ratio_scale))
    if ratio < 0.999 and transformed:
        transformed = _thin_notes(transformed, section, beats_per_bar, beat_step, ratio)
    transformed.sort(key=lambda n: (n["start_time"], n["pitch"]))
    return transformed

def _slice_and_clamp_notes(
    notes: Sequence[dict],
    start: float,
    end: float,
    beat_step: float,
) -> List[dict]:
    """Keep notes that start within [start, end) and shift them to section time."""
    if end <= start:
        return []
    gap = float(beat_step) * 0.02
    min_duration = float(beat_step) * 0.1
    section_notes: List[dict] = []

    for note in notes:
        start_time = float(note.get("start_time", 0.0))
        if start_time < start or start_time >= end:
            continue
        duration = float(note.get("duration", 0.0))
        max_duration = max(0.0, float(end) - start_time - gap)
        clamped_duration = min(duration, max_duration)
        if clamped_duration <= min_duration:
            continue

        shifted = dict(note)
        shifted["start_time"] = float(round(start_time - start, 6))
        shifted["duration"] = float(round(clamped_duration, 6))
        section_notes.append(shifted)

    section_notes.sort(key=lambda n: (n["start_time"], n["pitch"]))
    return section_notes

def _thin_notes(
    notes: Sequence[dict],
    section: Section,
    beats_per_bar: float,
    beat_step: float,
    keep_ratio: float,
) -> List[dict]:
    """Thin notes deterministically while preserving beat-1 anchors."""
    ratio = _clamp_ratio(keep_ratio)
    if ratio >= 0.999:
        return list(notes)
    if ratio <= 0.0:
        return []

    keep_threshold = int(round(ratio * 100.0))
    thinned: List[dict] = []
    for note in notes:
        start_time = float(note.get("start_time", 0.0))
        local_bar_index = int(start_time // float(beats_per_bar))
        absolute_bar_index = int(section.start_bar) + local_bar_index
        bar_start = float(local_bar_index) * float(beats_per_bar)
        on_beat, beat_index = kick._classify_time(start_time, bar_start, float(beat_step))

        # Always keep the downbeat when it is present.
        if on_beat and beat_index == 0:
            thinned.append(dict(note))
            continue

        seed = kick._velocity_seed(
            absolute_bar_index,
            int(beat_index),
            int(round(start_time * 1000.0)),
        )
        if seed % 100 < keep_threshold:
            thinned.append(dict(note))

    thinned.sort(key=lambda n: (n["start_time"], n["pitch"]))
    return thinned

def _hat_grid_step(beat_step: float, density: HatDensity) -> float:
    if density == "quarter":
        return float(beat_step)
    if density == "sixteenth":
        return float(beat_step) / 4.0
    return float(beat_step) / 2.0

def _aligns_to_grid(time_value: float, step: float) -> bool:
    if step <= 0:
        return False
    units = float(time_value) / float(step)
    epsilon = float(step) * 0.02
    return abs(units - round(units)) <= epsilon

def _ensure_hat_downbeats(
    notes: Sequence[dict],
    section: Section,
    beats_per_bar: float,
    beat_step: float,
) -> List[dict]:
    """Ensure each bar has a hat on beat 1 to keep time clear."""
    if not notes:
        return []

    epsilon = float(beat_step) * 0.02
    pitch = int(notes[0].get("pitch", hat.DEFAULT_HAT_PITCH))
    duration = float(notes[0].get("duration", 0.25))

    existing_times = {round(float(n.get("start_time", 0.0)), 6) for n in notes}
    downbeat_notes: List[dict] = [dict(n) for n in notes]

    for local_bar_index in range(int(section.bar_count)):
        bar_start = float(local_bar_index) * float(beats_per_bar)
        has_downbeat = any(abs(float(t) - bar_start) <= epsilon for t in existing_times)
        if has_downbeat:
            continue

        absolute_bar_index = int(section.start_bar) + local_bar_index
        seed = kick._velocity_seed(absolute_bar_index, 0, int(section.index + 1))
        velocity = kick._velocity_in_range(82, 96, seed)
        downbeat = {
            "pitch": pitch,
            "start_time": float(round(bar_start, 6)),
            "duration": duration,
            "velocity": int(velocity),
            "mute": 0,
        }
        downbeat_notes.append(downbeat)
        existing_times.add(round(bar_start, 6))

    downbeat_notes.sort(key=lambda n: (n["start_time"], n["pitch"]))
    return downbeat_notes

def _densify_hats_to_sixteenth(
    notes: Sequence[dict],
    section: Section,
    beats_per_bar: float,
    beat_step: float,
) -> List[dict]:
    """Add sixteenth-grid hats between existing hits for extra drive."""
    if not notes:
        return []

    step = float(beat_step) / 4.0
    epsilon = step * 0.02
    existing_times = {round(float(n.get("start_time", 0.0)), 6) for n in notes}
    dense_notes: List[dict] = [dict(n) for n in notes]

    for note in notes:
        start_time = float(note.get("start_time", 0.0))
        local_bar_index = int(start_time // float(beats_per_bar))
        bar_start = float(local_bar_index) * float(beats_per_bar)
        bar_end = bar_start + float(beats_per_bar)
        candidate_time = start_time + step

        if candidate_time >= bar_end - epsilon:
            continue
        if not _aligns_to_grid(candidate_time, step):
            continue

        key = round(candidate_time, 6)
        if key in existing_times:
            continue

        absolute_bar_index = int(section.start_bar) + local_bar_index
        seed = kick._velocity_seed(
            absolute_bar_index,
            local_bar_index,
            int(round(candidate_time * 1000.0)),
        )
        base_velocity = int(note.get("velocity", 90))
        ghost_velocity = max(1, min(127, int(round(base_velocity * 0.92)) - (seed % 4)))

        extra_note = dict(note)
        extra_note["start_time"] = float(key)
        extra_note["velocity"] = int(ghost_velocity)
        dense_notes.append(extra_note)
        existing_times.add(key)

    dense_notes.sort(key=lambda n: (n["start_time"], n["pitch"]))
    return dense_notes

def _shape_hats(
    section: Section,
    notes: Sequence[dict],
    beats_per_bar: float,
    beat_step: float,
) -> List[dict]:
    """Shape hat density per section: quarter, eighth, or sixteenth."""
    if not notes:
        return []

    density_step = _hat_grid_step(beat_step, section.hat_density)
    filtered = [
        dict(n)
        for n in notes
        if _aligns_to_grid(float(n.get("start_time", 0.0)), density_step)
    ]

    if section.hat_density == "sixteenth":
        filtered = _densify_hats_to_sixteenth(filtered, section, beats_per_bar, beat_step)

    return _ensure_hat_downbeats(filtered, section, beats_per_bar, beat_step)

def _arrange_drums(
    base_notes: Sequence[dict],
    sections: Sequence[Section],
    beats_per_bar: float,
    beat_step: float,
    enabled: Callable[[Section], bool],
    keep_ratio: Callable[[Section], float] | None = None,
    post_process: Callable[[Section, Sequence[dict], float, float], List[dict]] | None = None,
) -> List[tuple[Section, List[dict]]]:
    arranged: List[tuple[Section, List[dict]]] = []
    for section in sections:
        start, end, _ = _section_bounds(section, beats_per_bar)
        if enabled(section):
            notes = _slice_and_clamp_notes(base_notes, start, end, beat_step)
            if keep_ratio is not None and notes:
                notes = _thin_notes(notes, section, beats_per_bar, beat_step, keep_ratio(section))
            if post_process is not None and notes:
                notes = post_process(section, notes, beats_per_bar, beat_step)
        else:
            notes = []
        arranged.append((section, notes))
    return arranged

def _arrange_piano(
    chord_notes: Sequence[dict],
    motion_notes: Sequence[dict],
    sections: Sequence[Section],
    beats_per_bar: float,
    beat_step: float,
) -> List[tuple[Section, List[dict]]]:
    arranged: List[tuple[Section, List[dict]]] = []
    for section in sections:
        start, end, _ = _section_bounds(section, beats_per_bar)
        chords = _slice_and_clamp_notes(chord_notes, start, end, beat_step)
        motion = _slice_and_clamp_notes(motion_notes, start, end, beat_step)

        if section.piano_mode == "out":
            notes = []
        elif section.piano_mode == "chords":
            notes = chords
        elif section.piano_mode == "motion":
            notes = motion
        else:
            notes = chords + motion

        notes = list(notes)
        if notes:
            notes.sort(key=lambda n: (n["start_time"], n["pitch"]))
        arranged.append((section, notes))
    return arranged

def _build_source_sections(
    *,
    sections: Sequence[Section],
    bars: int,
    beats_per_bar: float,
    beat_step: float,
    transpose_semitones: int,
) -> dict[str, List[tuple[Section, List[dict]]]]:
    kick_notes, _ = kick.build_kick_notes(
        bars=bars,
        beats_per_bar=beats_per_bar,
        beat_step=beat_step,
        pitch=36,
        velocity=110,
        duration=0.25,
    )
    rim_notes, _ = rim.build_rim_notes(
        bars=bars,
        beats_per_bar=beats_per_bar,
        beat_step=beat_step,
        pitch=rim.DEFAULT_RIM_PITCH,
        velocity=100,
        duration=0.25,
    )
    hat_notes, _ = hat.build_hat_notes(
        bars=bars,
        beats_per_bar=beats_per_bar,
        beat_step=beat_step,
        pitch=hat.DEFAULT_HAT_PITCH,
        velocity=88,
        duration=0.25,
    )
    chord_notes, motion_notes, _ = piano.build_piano_layers(
        bars=bars,
        beats_per_bar=beats_per_bar,
        beat_step=beat_step,
        velocity_center=92,
        segment_bars=piano.DEFAULT_SEGMENT_BARS,
        transpose_semitones=transpose_semitones,
    )

    kick_sections = _arrange_drums(
        kick_notes,
        sections,
        beats_per_bar,
        beat_step,
        enabled=lambda s: s.kick_on,
        keep_ratio=lambda s: s.kick_keep_ratio,
    )
    rim_sections = _arrange_drums(
        rim_notes,
        sections,
        beats_per_bar,
        beat_step,
        enabled=lambda s: s.rim_on,
        keep_ratio=lambda s: s.rim_keep_ratio,
    )
    hat_sections = _arrange_drums(
        hat_notes,
        sections,
        beats_per_bar,
        beat_step,
        enabled=lambda s: s.hat_on,
        keep_ratio=lambda s: s.hat_keep_ratio,
        post_process=_shape_hats,
    )
    piano_chords = _arrange_piano(
        chord_notes,
        [],
        sections,
        beats_per_bar,
        beat_step,
    )
    piano_motion = _arrange_piano(
        [],
        motion_notes,
        sections,
        beats_per_bar,
        beat_step,
    )
    return {
        "kick": kick_sections,
        "rim": rim_sections,
        "hat": hat_sections,
        "piano_chords": piano_chords,
        "piano_motion": piano_motion,
    }

def _arrange_from_registry(
    *,
    specs: Sequence[InstrumentSpec],
    sources: Mapping[str, Sequence[tuple[Section, Sequence[dict]]]],
    sections: Sequence[Section],
    activation_mask: Mapping[str, set[int]],
    beats_per_bar: float,
    beat_step: float,
) -> dict[str, List[tuple[Section, List[dict]]]]:
    arranged_by_track: dict[str, List[tuple[Section, List[dict]]]] = {}
    for spec in specs:
        source_payload = sources.get(spec.source)
        if source_payload is None:
            raise ValueError(f"source '{spec.source}' not found for instrument '{spec.name}'")
        active_indices = activation_mask.get(spec.name, set())
        sections_payload: List[tuple[Section, List[dict]]] = []
        for section, notes in source_payload:
            if section.index not in active_indices:
                sections_payload.append((section, []))
                continue
            transformed = _transform_instrument_notes(
                notes,
                spec,
                section,
                beats_per_bar,
                beat_step,
            )
            sections_payload.append((section, transformed))
        arranged_by_track[spec.name] = sections_payload
    return arranged_by_track
