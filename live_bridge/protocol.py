"""Command envelope parsing and payload validation for the Live bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import uuid


class ProtocolError(ValueError):
    """Raised when a command envelope or payload is invalid."""


@dataclass(frozen=True)
class CommandEnvelope:
    command_id: str
    command: str
    payload: Dict[str, Any]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_int(payload: Dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError(f"'{field}' must be an integer.")
    return value


def _require_number(payload: Dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if not _is_number(value):
        raise ProtocolError(f"'{field}' must be a number.")
    return float(value)


def _optional_number(payload: Dict[str, Any], field: str) -> Optional[float]:
    if field not in payload or payload[field] is None:
        return None
    return _require_number(payload, field)


def _require_bool(payload: Dict[str, Any], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise ProtocolError(f"'{field}' must be a boolean.")
    return value


def _require_str(payload: Dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"'{field}' must be a non-empty string.")
    return value


def _require_note_dict(note: Dict[str, Any], idx: int) -> None:
    for key in ("pitch", "start_time", "duration", "velocity"):
        if key not in note:
            raise ProtocolError(f"notes[{idx}] is missing '{key}'.")

    pitch = note["pitch"]
    if not isinstance(pitch, int) or pitch < 0 or pitch > 127:
        raise ProtocolError(f"notes[{idx}].pitch must be an integer from 0 to 127.")

    start_time = note["start_time"]
    duration = note["duration"]
    velocity = note["velocity"]
    mute = note.get("mute", False)

    if not _is_number(start_time) or float(start_time) < 0:
        raise ProtocolError(f"notes[{idx}].start_time must be a number >= 0.")
    if not _is_number(duration) or float(duration) <= 0:
        raise ProtocolError(f"notes[{idx}].duration must be a number > 0.")
    if not isinstance(velocity, int) or velocity < 1 or velocity > 127:
        raise ProtocolError(f"notes[{idx}].velocity must be an integer from 1 to 127.")
    if not isinstance(mute, bool):
        raise ProtocolError(f"notes[{idx}].mute must be a boolean.")


def _validate_note_insert(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    _require_int(payload, "clip_slot_index")
    notes = payload.get("notes")
    if not isinstance(notes, list) or not notes:
        raise ProtocolError("'notes' must be a non-empty list.")
    for idx, note in enumerate(notes):
        if not isinstance(note, dict):
            raise ProtocolError(f"notes[{idx}] must be an object.")
        _require_note_dict(note, idx)


def _validate_create_midi_clip(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    _require_int(payload, "clip_slot_index")
    length_beats = _require_number(payload, "length_beats")
    if length_beats <= 0:
        raise ProtocolError("'length_beats' must be > 0.")


def _validate_fire_clip(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    _require_int(payload, "clip_slot_index")


def _validate_stop_track(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")


def _validate_track_toggle(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    _require_bool(payload, "value")


def _validate_note_velocity(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    _require_int(payload, "clip_slot_index")
    _require_int(payload, "pitch")
    _require_number(payload, "start_time")
    _require_number(payload, "duration")
    velocity = _require_int(payload, "velocity")
    if velocity < 1 or velocity > 127:
        raise ProtocolError("'velocity' must be from 1 to 127.")


def _validate_automation(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    _require_int(payload, "clip_slot_index")
    _require_int(payload, "device_index")
    _require_int(payload, "parameter_index")
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ProtocolError("'points' must be a non-empty list.")
    for idx, point in enumerate(points):
        if not isinstance(point, dict):
            raise ProtocolError(f"points[{idx}] must be an object.")
        if "time" not in point or "value" not in point:
            raise ProtocolError(f"points[{idx}] must include 'time' and 'value'.")
        if not _is_number(point["time"]) or float(point["time"]) < 0:
            raise ProtocolError(f"points[{idx}].time must be >= 0.")
        if not _is_number(point["value"]):
            raise ProtocolError(f"points[{idx}].value must be numeric.")


def _validate_mix_volume(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    _require_number(payload, "value")


def _validate_mix_pan(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    value = _require_number(payload, "value")
    if value < -1.0 or value > 1.0:
        raise ProtocolError("'value' must be between -1.0 and 1.0 for panning.")


def _validate_send_level(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    _require_int(payload, "send_index")
    _require_number(payload, "value")


def _validate_device_param(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    _require_int(payload, "device_index")
    _require_int(payload, "parameter_index")
    _require_number(payload, "value")


def _validate_eq3(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    if "device_index" in payload:
        _require_int(payload, "device_index")
    if all(field not in payload for field in ("low_gain", "mid_gain", "high_gain", "low_on", "mid_on", "high_on")):
        raise ProtocolError("eq3 payload must include at least one gain or on/off field.")
    for field in ("low_gain", "mid_gain", "high_gain"):
        _optional_number(payload, field)
    for field in ("low_on", "mid_on", "high_on"):
        if field in payload:
            _require_bool(payload, field)


def _validate_eq8_gain(payload: Dict[str, Any]) -> None:
    _require_int(payload, "track_index")
    _require_int(payload, "device_index")
    band = _require_int(payload, "band")
    if band < 1 or band > 8:
        raise ProtocolError("'band' must be between 1 and 8 for EQ Eight.")
    _require_number(payload, "gain")


def _validate_tempo(payload: Dict[str, Any]) -> None:
    bpm = _require_number(payload, "bpm")
    if bpm <= 0:
        raise ProtocolError("'bpm' must be > 0.")


def _validate_global_key(payload: Dict[str, Any]) -> None:
    root_note = _require_int(payload, "root_note")
    if root_note < 0 or root_note > 11:
        raise ProtocolError("'root_note' must be between 0 and 11.")
    if "scale_name" in payload:
        _require_str(payload, "scale_name")
    if "scale_intervals" in payload:
        intervals = payload["scale_intervals"]
        if not isinstance(intervals, list) or not intervals:
            raise ProtocolError("'scale_intervals' must be a non-empty list when provided.")
        for idx, value in enumerate(intervals):
            if not isinstance(value, int):
                raise ProtocolError(f"scale_intervals[{idx}] must be an integer.")


VALIDATORS: Dict[str, Callable[[Dict[str, Any]], None]] = {
    "note_insert": _validate_note_insert,
    "create_midi_clip": _validate_create_midi_clip,
    "fire_clip": _validate_fire_clip,
    "stop_track": _validate_stop_track,
    "set_note_velocity": _validate_note_velocity,
    "create_automation": _validate_automation,
    "set_track_mute": _validate_track_toggle,
    "set_track_solo": _validate_track_toggle,
    "set_track_volume": _validate_mix_volume,
    "set_track_pan": _validate_mix_pan,
    "set_send_level": _validate_send_level,
    "set_device_parameter": _validate_device_param,
    "set_eq3": _validate_eq3,
    "set_eq8_band_gain": _validate_eq8_gain,
    "set_tempo": _validate_tempo,
    "set_global_key": _validate_global_key,
}


def supported_commands() -> List[str]:
    return sorted(VALIDATORS.keys())


def parse_envelope(raw: Dict[str, Any]) -> CommandEnvelope:
    if not isinstance(raw, dict):
        raise ProtocolError("Command body must be a JSON object.")
    command = raw.get("command")
    payload = raw.get("payload", {})
    command_id = raw.get("id") or str(uuid.uuid4())

    if not isinstance(command, str) or not command.strip():
        raise ProtocolError("'command' must be a non-empty string.")
    if not isinstance(payload, dict):
        raise ProtocolError("'payload' must be an object.")
    if not isinstance(command_id, str):
        raise ProtocolError("'id' must be a string when provided.")

    envelope = CommandEnvelope(command_id=command_id, command=command.strip(), payload=payload)
    validate_payload(envelope.command, envelope.payload)
    return envelope


def validate_payload(command: str, payload: Dict[str, Any]) -> None:
    validator = VALIDATORS.get(command)
    if validator is None:
        supported = ", ".join(supported_commands())
        raise ProtocolError(f"Unsupported command '{command}'. Supported commands: {supported}")
    validator(payload)
