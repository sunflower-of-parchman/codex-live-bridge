#!/usr/bin/env python3
"""
External controller for the Ableton Live UDP bridge.

This script sends OSC (Open Sound Control) messages to a Max for Live device.
Its Node-for-Max receiver listens on loopback UDP port 9000. OSC encoding is
implemented using only the Python standard library.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import select
import socket
import struct
from statistics import mean
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple, Union


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9000
DEFAULT_ACK_PORT = 9001
AUTH_TOKEN_ENV = "CODEX_LIVE_BRIDGE_TOKEN"
AUTH_TOKEN_PLACEHOLDER = "CHANGE_ME_BEFORE_USE"
MIN_AUTH_TOKEN_BYTES = 16
MAX_AUTH_TOKEN_BYTES = 256
MAX_TRACKS_PER_COMMAND = 32
MAX_TRACK_TARGET = 256
PROTECTED_OSC_ADDRESSES = frozenset(
    {
        "/tempo",
        "/sig_num",
        "/sig_den",
        "/create_midi_track",
        "/add_midi_tracks",
        "/create_audio_track",
        "/add_audio_tracks",
        "/delete_audio_tracks",
        "/delete_midi_tracks",
        "/rename_track",
        "/set_session_clip_notes",
        "/append_session_clip_notes",
        "/ensure_midi_tracks",
        "/api/set",
        "/api/call",
        "/api_observe",
        "/api_unobserve",
        "/api_clear_observers",
        "/api/parameter_set",
        "/api/insert_device",
        "/api/insert_chain",
        "/api/drum_chain_in_note",
        "/midi_cc",
        "/cc64",
    }
)
SESSION_CLIP_INSPECTION_MAX_NOTES = 4096
SESSION_CLIP_INSPECTION_MAX_DEVICES = 256
SESSION_CLIP_INSPECTION_MAX_FRAGMENTS = 1024
SESSION_CLIP_INSPECTION_MAX_ACTIVE_ASSEMBLIES = 16
SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES = 4096
SESSION_CLIP_INSPECTION_MAX_RETAINED_BYTES = (
    SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES
    * SESSION_CLIP_INSPECTION_MAX_FRAGMENTS
)
SESSION_CLIP_INSPECTION_MAX_MISSING_DIAGNOSTIC_INDEXES = 16
SESSION_CLIP_INSPECTION_MAX_UNRELATED_ACKS = 32
SESSION_CLIP_INSPECTION_MAX_CORRELATED_ACKS = (
    SESSION_CLIP_INSPECTION_MAX_FRAGMENTS + 1
)
ARRANGEMENT_INSPECTION_PACKET_BUDGET_BYTES = 4096
ARRANGEMENT_INSPECTION_MAX_ITEMS = 256
ARRANGEMENT_INSPECTION_MAX_NOTES = 4096
ARRANGEMENT_INSPECTION_MAX_FRAGMENTS = 1024
ARRANGEMENT_INSPECTION_MAX_ACTIVE_ASSEMBLIES = 16
ARRANGEMENT_INSPECTION_MAX_RETAINED_BYTES = 4 * 1024 * 1024
ARRANGEMENT_INSPECTION_EVENTS = frozenset(
    {
        "api_arrangement_project_inspect",
        "api_arrangement_track_inspect",
        "api_arrangement_clip_inspect",
    }
)
ACK_DRAIN_MAX_PACKETS = 32
SESSION_CLIP_INSPECTION_MAX_PACKETS_PER_SELECT = 16
ACK_PACKET_BUDGET_BYTES = 65_507
ACK_WAIT_MAX_PACKETS = 256
ACK_WAIT_MAX_RETAINED_ACKS = 64
ACK_WAIT_MAX_RETAINED_BYTES = 64 * 1024
ACK_MAX_BATCH_COMMANDS = 32
ACK_MAX_PACKETS_PER_SELECT = 16
ACK_MAX_DIAGNOSTIC_CHARS = 256

OscArg = Union[int, float, str]


@dataclass(frozen=True)
class OscCommand:
    address: str
    args: Tuple[OscArg, ...] = ()


@dataclass(frozen=True)
class BridgeConfig:
    host: str
    port: int
    ack_port: int
    ack_timeout_s: float
    expect_ack: bool
    ping_first: bool
    status: bool
    tempo: float | None
    sig_num: int | None
    sig_den: int | None
    create_midi_tracks: int
    add_midi_tracks: int
    midi_name: str
    create_audio_tracks: int
    add_audio_tracks: int
    audio_prefix: str
    delete_audio_tracks: int
    delete_midi_tracks: int
    rename_track_index: int | None
    rename_track_name: str | None
    session_clip_track_index: int | None
    session_clip_slot_index: int | None
    session_clip_length: float | None
    session_clip_notes_json: str | None
    session_clip_name: str | None
    append_session_clip_track_index: int | None
    append_session_clip_slot_index: int | None
    append_session_clip_notes_json: str | None
    inspect_session_clip_track_index: int | None
    inspect_session_clip_slot_index: int | None
    ensure_midi_tracks: int | None
    midi_ccs: Tuple[Tuple[int, int, int], ...]
    cc64s: Tuple[Tuple[int, int], ...]
    api_pings: Tuple[str | None, ...]
    api_gets: Tuple[Tuple[str, str, str | None], ...]
    api_sets: Tuple[Tuple[str, str, str, str | None], ...]
    api_calls: Tuple[Tuple[str, str, str, str | None], ...]
    api_children: Tuple[Tuple[str, str, str | None], ...]
    api_describes: Tuple[Tuple[str, str | None], ...]
    auth_token: str | None = field(default=None, repr=False)
    api_observes: Tuple[Tuple[str, str, str, str | None], ...] = ()
    api_unobserves: Tuple[Tuple[str, str | None], ...] = ()
    api_observers: Tuple[str | None, ...] = ()
    api_clear_observers: Tuple[str | None, ...] = ()
    api_session_contexts: Tuple[str | None, ...] = ()
    api_theory_statuses: Tuple[str | None, ...] = ()
    api_tuning_statuses: Tuple[str | None, ...] = ()
    api_device_lists: Tuple[Tuple[str, str | None], ...] = ()
    api_device_parameters: Tuple[Tuple[str, str | None], ...] = ()
    api_parameter_sets: Tuple[Tuple[str, str, str | None], ...] = ()
    api_mixer_statuses: Tuple[Tuple[str, str | None], ...] = ()
    api_insert_devices: Tuple[Tuple[str, str, str, str | None], ...] = ()
    api_insert_chains: Tuple[Tuple[str, str, str | None], ...] = ()
    api_drum_chain_in_notes: Tuple[Tuple[str, int, str | None], ...] = ()
    api_session_clip_inspects: Tuple[Tuple[int, int, str], ...] = ()
    api_arrangement_project_inspects: Tuple[str, ...] = ()
    api_arrangement_track_inspects: Tuple[Tuple[int, str], ...] = ()
    api_arrangement_clip_inspects: Tuple[Tuple[int, int, bool, str], ...] = ()
    ack_mode: str = "per_command"
    ack_flush_interval: int = 10
    listen: bool = False
    listen_timeout_s: float = 0.0
    listen_max_events: int = 0
    report_metrics: bool = False
    delay_ms: int = 0
    dry_run: bool = False


AckMode = Union[str]


@dataclass(frozen=True)
class SendMetrics:
    command_count: int
    send_durations_ms: Tuple[float, ...]
    ack_wait_durations_ms: Tuple[float, ...]
    acks_per_command: Tuple[int, ...]
    elapsed_ms: float


@dataclass(frozen=True)
class AckEvent:
    address: str
    event: str | None
    request_id: str | None
    payload: dict[str, object]
    is_error: bool = False


class BridgeAcknowledgementError(RuntimeError):
    """Raised when a requested bridge acknowledgement cannot be verified."""


class SessionClipInspectionAssemblyError(ValueError):
    """Raised when session clip inspection fragments cannot be assembled safely."""


class SessionClipInspectionAssembler:
    """Assemble packet-bounded session clip inspection fragments."""

    SCHEMA = "codex-live-bridge.session-midi-clip-inspection"
    SCHEMA_VERSION = 1
    PRODUCER_VERSION = "3.1.0"
    PACKET_BUDGET_BYTES = SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES
    MAX_NOTES = SESSION_CLIP_INSPECTION_MAX_NOTES
    MAX_DEVICES = SESSION_CLIP_INSPECTION_MAX_DEVICES
    MAX_FRAGMENTS = SESSION_CLIP_INSPECTION_MAX_FRAGMENTS
    MAX_ACTIVE_ASSEMBLIES = SESSION_CLIP_INSPECTION_MAX_ACTIVE_ASSEMBLIES
    MAX_RETAINED_BYTES = SESSION_CLIP_INSPECTION_MAX_RETAINED_BYTES
    _FRAGMENT_KINDS = {"complete", "context", "device_page", "note_page"}
    _ROOT_KEYS = {
        "schema",
        "schema_version",
        "producer_version",
        "inspection_id",
        "correlation",
        "snapshot",
        "transfer",
        "completeness",
        "data",
    }
    _CORRELATION_KEYS = {"request_id", "track_index", "slot_index"}
    _SNAPSHOT_KEYS = {
        "started_ms",
        "completed_ms",
        "atomic",
        "consistent",
    }
    _TRANSFER_KEYS = {
        "fragment_index",
        "fragment_count",
        "fragment_kind",
        "is_last",
        "packet_budget_bytes",
    }
    _COMPLETENESS_KEYS = {
        "track",
        "clip",
        "devices",
        "notes",
        "missing_fields",
    }
    _CONTEXT_DATA_KEYS = {"context", "track", "clip", "summary"}
    _TRACK_KEYS = {"index", "path", "id", "name"}
    _CLIP_KEYS = {
        "slot_index",
        "path",
        "id",
        "name",
        "start_marker",
        "end_marker",
        "live_length",
        "looping",
        "loop_start",
        "loop_end",
    }
    _SUMMARY_KEYS = {"note_count", "pitch_min", "pitch_max"}
    _DEVICE_PAGE_KEYS = {
        "device_offset",
        "device_count",
        "device_total",
        "devices",
    }
    _NOTE_PAGE_KEYS = {"note_offset", "note_count", "note_total", "notes"}
    _DEVICE_KEYS = {"index", "path", "id", "name", "class_name", "type"}
    _DEVICE_TYPES = {0, 1, 2, 4}
    _NOTE_KEYS = {
        "note_id",
        "pitch",
        "start_time",
        "duration",
        "velocity",
        "mute",
        "probability",
        "velocity_deviation",
        "release_velocity",
    }

    def __init__(self) -> None:
        self._states: dict[tuple[str, str], dict[str, object]] = {}
        self._retained_bytes = 0

    @staticmethod
    def _require_dict(value: object, label: str) -> dict[str, object]:
        if not isinstance(value, dict):
            raise SessionClipInspectionAssemblyError(f"malformed fragment: {label}")
        return value

    @staticmethod
    def _require_list(value: object, label: str) -> list[object]:
        if not isinstance(value, list):
            raise SessionClipInspectionAssemblyError(f"malformed fragment: {label}")
        return value

    @staticmethod
    def _require_non_negative_int(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SessionClipInspectionAssemblyError(f"malformed fragment: {label}")
        return value

    @staticmethod
    def _require_int_range(
        value: object,
        label: str,
        minimum: int,
        maximum: int,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or value > maximum
        ):
            raise SessionClipInspectionAssemblyError(
                f"malformed fragment: {label}"
            )
        return value

    @staticmethod
    def _require_finite_number(
        value: object,
        label: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SessionClipInspectionAssemblyError(
                f"malformed fragment: {label}"
            )
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise SessionClipInspectionAssemblyError(
                f"malformed fragment: {label}"
            ) from exc
        if not math.isfinite(numeric):
            raise SessionClipInspectionAssemblyError(
                f"malformed fragment: {label}"
            )
        if minimum is not None and numeric < minimum:
            raise SessionClipInspectionAssemblyError(
                f"malformed fragment: {label}"
            )
        if maximum is not None and numeric > maximum:
            raise SessionClipInspectionAssemblyError(
                f"malformed fragment: {label}"
            )
        return numeric

    @staticmethod
    def _require_non_empty_string(value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SessionClipInspectionAssemblyError(
                f"malformed fragment: {label}"
            )
        return value

    @staticmethod
    def _require_nullable_string(value: object, label: str) -> str | None:
        if value is not None and not isinstance(value, str):
            raise SessionClipInspectionAssemblyError(
                f"malformed fragment: {label}"
            )
        return value

    @staticmethod
    def _require_exact_keys(
        value: dict[str, object],
        expected: set[str],
        label: str,
    ) -> None:
        if set(value) != expected:
            raise SessionClipInspectionAssemblyError(
                f"malformed fragment: {label} keys"
            )

    @staticmethod
    def _copy_fields(
        value: dict[str, object],
        fields: Sequence[str],
    ) -> dict[str, object]:
        return {field: value[field] for field in fields if field in value}

    @staticmethod
    def _canonical(value: object) -> str:
        try:
            return json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise SessionClipInspectionAssemblyError(
                "malformed fragment: not JSON serializable"
            ) from exc

    def _validate_track(
        self,
        track: object,
        correlation: dict[str, object],
    ) -> dict[str, object]:
        value = self._require_dict(track, "track")
        self._require_exact_keys(value, self._TRACK_KEYS, "track")
        track_index = self._require_non_negative_int(value["index"], "track.index")
        self._require_non_empty_string(value["path"], "track.path")
        self._require_non_negative_int(value["id"], "track.id")
        self._require_nullable_string(value["name"], "track.name")
        if track_index != correlation["track_index"]:
            raise SessionClipInspectionAssemblyError("mixed metadata")
        return value

    def _validate_clip(
        self,
        clip: object,
        correlation: dict[str, object],
    ) -> dict[str, object]:
        value = self._require_dict(clip, "clip")
        self._require_exact_keys(value, self._CLIP_KEYS, "clip")
        slot_index = self._require_non_negative_int(
            value["slot_index"], "clip.slot_index"
        )
        self._require_non_empty_string(value["path"], "clip.path")
        self._require_non_negative_int(value["id"], "clip.id")
        self._require_nullable_string(value["name"], "clip.name")
        start_marker = self._require_finite_number(
            value["start_marker"], "clip.start_marker"
        )
        end_marker = self._require_finite_number(
            value["end_marker"], "clip.end_marker"
        )
        self._require_finite_number(
            value["live_length"], "clip.live_length", minimum=0
        )
        if not isinstance(value["looping"], bool):
            raise SessionClipInspectionAssemblyError(
                "malformed fragment: clip.looping"
            )
        loop_start = self._require_finite_number(
            value["loop_start"], "clip.loop_start"
        )
        loop_end = self._require_finite_number(
            value["loop_end"], "clip.loop_end"
        )
        if (
            end_marker < start_marker
            or not math.isfinite(end_marker - start_marker)
            or loop_end < loop_start
            or not math.isfinite(loop_end - loop_start)
        ):
            raise SessionClipInspectionAssemblyError(
                "malformed fragment: clip ranges"
            )
        if slot_index != correlation["slot_index"]:
            raise SessionClipInspectionAssemblyError("mixed metadata")
        return value

    def _validate_summary(self, summary: object) -> dict[str, object]:
        value = self._require_dict(summary, "summary")
        self._require_exact_keys(value, self._SUMMARY_KEYS, "summary")
        note_count = self._require_non_negative_int(
            value["note_count"], "summary.note_count"
        )
        if note_count > self.MAX_NOTES:
            raise SessionClipInspectionAssemblyError(
                "resource limit exceeded: summary.note_count"
            )
        pitch_min = value["pitch_min"]
        pitch_max = value["pitch_max"]
        if note_count == 0:
            if pitch_min is not None or pitch_max is not None:
                raise SessionClipInspectionAssemblyError(
                    "malformed fragment: summary pitches"
                )
        else:
            minimum = self._require_int_range(
                pitch_min, "summary.pitch_min", 0, 127
            )
            maximum = self._require_int_range(
                pitch_max, "summary.pitch_max", 0, 127
            )
            if minimum > maximum:
                raise SessionClipInspectionAssemblyError(
                    "malformed fragment: summary pitches"
                )
        return value

    def _validate_context_data(
        self,
        data: dict[str, object],
        correlation: dict[str, object],
    ) -> None:
        if data["context"] != "session":
            raise SessionClipInspectionAssemblyError(
                "malformed fragment: context"
            )
        self._validate_track(data["track"], correlation)
        self._validate_clip(data["clip"], correlation)
        self._validate_summary(data["summary"])

    def _validate_device(self, device: object, expected_index: int) -> None:
        value = self._require_dict(device, "device")
        self._require_exact_keys(value, self._DEVICE_KEYS, "device")
        index = self._require_non_negative_int(value["index"], "device.index")
        if index != expected_index:
            raise SessionClipInspectionAssemblyError(
                "noncontiguous device indexes"
            )
        self._require_non_empty_string(value["path"], "device.path")
        self._require_non_negative_int(value["id"], "device.id")
        self._require_nullable_string(value["name"], "device.name")
        self._require_nullable_string(value["class_name"], "device.class_name")
        device_type = value["type"]
        if device_type is not None and (
            isinstance(device_type, bool)
            or not isinstance(device_type, int)
            or device_type not in self._DEVICE_TYPES
        ):
            raise SessionClipInspectionAssemblyError(
                "malformed fragment: device.type"
            )

    def _validate_note(self, note: object) -> None:
        value = self._require_dict(note, "note")
        self._require_exact_keys(value, self._NOTE_KEYS, "note")
        self._require_non_negative_int(value["note_id"], "note.note_id")
        self._require_int_range(value["pitch"], "note.pitch", 0, 127)
        start_time = self._require_finite_number(
            value["start_time"], "note.start_time"
        )
        duration = self._require_finite_number(
            value["duration"], "note.duration", minimum=0
        )
        if not math.isfinite(start_time + duration):
            raise SessionClipInspectionAssemblyError(
                "malformed fragment: note.end_time"
            )
        self._require_finite_number(
            value["velocity"], "note.velocity", minimum=0, maximum=127
        )
        mute = value["mute"]
        if not isinstance(mute, bool):
            self._require_finite_number(
                mute, "note.mute", minimum=0, maximum=1
            )
            if float(mute) not in (0.0, 1.0):
                raise SessionClipInspectionAssemblyError(
                    "malformed fragment: note.mute"
                )
        self._require_finite_number(
            value["probability"],
            "note.probability",
            minimum=0,
            maximum=1,
        )
        self._require_finite_number(
            value["velocity_deviation"],
            "note.velocity_deviation",
            minimum=-127,
            maximum=127,
        )
        self._require_finite_number(
            value["release_velocity"],
            "note.release_velocity",
            minimum=0,
            maximum=127,
        )

    def _validate_page(
        self,
        data: dict[str, object],
        *,
        offset_field: str,
        count_field: str,
        total_field: str,
        items_field: str,
        item_kind: str,
    ) -> None:
        offset = self._require_non_negative_int(data.get(offset_field), offset_field)
        count = self._require_non_negative_int(data.get(count_field), count_field)
        total = self._require_non_negative_int(data.get(total_field), total_field)
        maximum = self.MAX_DEVICES if item_kind == "device" else self.MAX_NOTES
        if count > maximum or total > maximum:
            raise SessionClipInspectionAssemblyError(
                f"resource limit exceeded: {item_kind} count"
            )
        items = self._require_list(data.get(items_field), items_field)
        if len(items) > maximum:
            raise SessionClipInspectionAssemblyError(
                f"resource limit exceeded: {items_field}"
            )
        if count != len(items) or offset + count > total:
            raise SessionClipInspectionAssemblyError("inconsistent counts")
        for item_index, item in enumerate(items):
            if item_kind == "device":
                self._validate_device(item, offset + item_index)
            else:
                self._validate_note(item)

    def _validate_fragment(
        self,
        fragment: object,
        request_id: str | None,
    ) -> tuple[
        dict[str, object],
        tuple[str, str],
        int,
        int,
        str,
        str,
    ]:
        value = self._require_dict(fragment, "root")
        self._require_exact_keys(value, self._ROOT_KEYS, "root")
        if value.get("schema") != self.SCHEMA:
            raise SessionClipInspectionAssemblyError("malformed fragment: schema")
        schema_version = value.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != self.SCHEMA_VERSION
        ):
            raise SessionClipInspectionAssemblyError("malformed fragment: schema_version")
        if value.get("producer_version") != self.PRODUCER_VERSION:
            raise SessionClipInspectionAssemblyError("malformed fragment: producer_version")

        inspection_id = value.get("inspection_id")
        self._require_non_empty_string(inspection_id, "inspection_id")
        assert isinstance(inspection_id, str)

        correlation = self._require_dict(value.get("correlation"), "correlation")
        self._require_exact_keys(
            correlation, self._CORRELATION_KEYS, "correlation"
        )
        fragment_request_id = correlation.get("request_id")
        if not isinstance(fragment_request_id, str) or not fragment_request_id:
            raise SessionClipInspectionAssemblyError("malformed fragment: request_id")
        try:
            request_id_bytes = fragment_request_id.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise SessionClipInspectionAssemblyError(
                "malformed fragment: request_id"
            ) from exc
        if len(request_id_bytes) > 128:
            raise SessionClipInspectionAssemblyError("malformed fragment: request_id")
        if request_id is not None and fragment_request_id != request_id:
            raise SessionClipInspectionAssemblyError("mixed metadata")
        self._require_non_negative_int(
            correlation.get("track_index"), "track_index"
        )
        self._require_non_negative_int(
            correlation.get("slot_index"), "slot_index"
        )

        snapshot = self._require_dict(value.get("snapshot"), "snapshot")
        self._require_exact_keys(snapshot, self._SNAPSHOT_KEYS, "snapshot")
        started_ms = self._require_non_negative_int(
            snapshot.get("started_ms"), "started_ms"
        )
        completed_ms = self._require_non_negative_int(
            snapshot.get("completed_ms"), "completed_ms"
        )
        if completed_ms < started_ms:
            raise SessionClipInspectionAssemblyError(
                "malformed fragment: snapshot range"
            )
        if snapshot.get("atomic") is not False or snapshot.get("consistent") is not True:
            raise SessionClipInspectionAssemblyError("malformed fragment: snapshot")

        transfer = self._require_dict(value.get("transfer"), "transfer")
        self._require_exact_keys(transfer, self._TRANSFER_KEYS, "transfer")
        fragment_index = self._require_non_negative_int(
            transfer.get("fragment_index"), "fragment_index"
        )
        fragment_count = self._require_non_negative_int(
            transfer.get("fragment_count"), "fragment_count"
        )
        if fragment_count > self.MAX_FRAGMENTS:
            raise SessionClipInspectionAssemblyError(
                "resource limit exceeded: fragment_count"
            )
        if fragment_count <= 0 or fragment_index >= fragment_count:
            raise SessionClipInspectionAssemblyError("inconsistent counts")
        fragment_kind = transfer.get("fragment_kind")
        if fragment_kind not in self._FRAGMENT_KINDS:
            raise SessionClipInspectionAssemblyError("malformed fragment: fragment_kind")
        if not isinstance(transfer.get("is_last"), bool):
            raise SessionClipInspectionAssemblyError(
                "malformed fragment: is_last"
            )
        if transfer.get("is_last") is not (fragment_index == fragment_count - 1):
            raise SessionClipInspectionAssemblyError("mixed transfer metadata")
        packet_budget_bytes = transfer.get("packet_budget_bytes")
        if (
            isinstance(packet_budget_bytes, bool)
            or not isinstance(packet_budget_bytes, int)
            or packet_budget_bytes != self.PACKET_BUDGET_BYTES
        ):
            raise SessionClipInspectionAssemblyError("mixed transfer metadata")

        completeness = self._require_dict(value.get("completeness"), "completeness")
        self._require_exact_keys(
            completeness, self._COMPLETENESS_KEYS, "completeness"
        )
        for field in ("track", "clip", "devices", "notes"):
            if completeness.get(field) != "complete":
                raise SessionClipInspectionAssemblyError("malformed fragment: completeness")
        if completeness.get("missing_fields") != []:
            raise SessionClipInspectionAssemblyError("malformed fragment: missing_fields")

        data = self._require_dict(value.get("data"), "data")
        if fragment_kind == "complete":
            if fragment_count != 1 or fragment_index != 0:
                raise SessionClipInspectionAssemblyError("mixed transfer metadata")
            self._require_exact_keys(
                data,
                self._CONTEXT_DATA_KEYS
                | self._DEVICE_PAGE_KEYS
                | self._NOTE_PAGE_KEYS,
                "complete data",
            )
            self._validate_context_data(data, correlation)
            self._validate_page(
                data,
                offset_field="device_offset",
                count_field="device_count",
                total_field="device_total",
                items_field="devices",
                item_kind="device",
            )
            self._validate_page(
                data,
                offset_field="note_offset",
                count_field="note_count",
                total_field="note_total",
                items_field="notes",
                item_kind="note",
            )
            if data.get("device_offset") != 0 or data.get("note_offset") != 0:
                raise SessionClipInspectionAssemblyError("noncontiguous page offsets")
            if (
                data.get("device_count") != data.get("device_total")
                or data.get("note_count") != data.get("note_total")
            ):
                raise SessionClipInspectionAssemblyError("inconsistent counts")
        elif fragment_kind == "context":
            if fragment_index != 0:
                raise SessionClipInspectionAssemblyError(
                    "mixed transfer metadata"
                )
            self._require_exact_keys(
                data, self._CONTEXT_DATA_KEYS, "context data"
            )
            self._validate_context_data(data, correlation)
        elif fragment_kind == "device_page":
            if fragment_index == 0:
                raise SessionClipInspectionAssemblyError(
                    "mixed transfer metadata"
                )
            self._require_exact_keys(
                data, self._DEVICE_PAGE_KEYS, "device page"
            )
            self._validate_page(
                data,
                offset_field="device_offset",
                count_field="device_count",
                total_field="device_total",
                items_field="devices",
                item_kind="device",
            )
        elif fragment_kind == "note_page":
            if fragment_index == 0:
                raise SessionClipInspectionAssemblyError(
                    "mixed transfer metadata"
                )
            self._require_exact_keys(data, self._NOTE_PAGE_KEYS, "note page")
            self._validate_page(
                data,
                offset_field="note_offset",
                count_field="note_count",
                total_field="note_total",
                items_field="notes",
                item_kind="note",
            )

        metadata = {
            "schema": value["schema"],
            "schema_version": value["schema_version"],
            "producer_version": value["producer_version"],
            "inspection_id": inspection_id,
            "correlation": correlation,
            "snapshot": snapshot,
            "completeness": completeness,
            "fragment_count": fragment_count,
            "packet_budget_bytes": transfer["packet_budget_bytes"],
        }
        key = (fragment_request_id, inspection_id)
        return (
            value,
            key,
            fragment_index,
            fragment_count,
            str(fragment_kind),
            self._canonical(metadata),
        )

    @staticmethod
    def _candidate_state_key(
        fragment: object,
        request_id: str | None,
    ) -> tuple[str, str] | None:
        if not isinstance(fragment, dict):
            return None
        inspection_id = fragment.get("inspection_id")
        correlation = fragment.get("correlation")
        if not isinstance(inspection_id, str) or not isinstance(correlation, dict):
            return None
        fragment_request_id = correlation.get("request_id")
        if not isinstance(fragment_request_id, str):
            return None
        if request_id is not None and request_id != fragment_request_id:
            return None
        return (fragment_request_id, inspection_id)

    @staticmethod
    def _missing_fragment_message(
        fragment_count: int,
        fragments: dict[object, object],
    ) -> str | None:
        shown: list[int] = []
        missing_count = 0
        for index in range(fragment_count):
            if index in fragments:
                continue
            missing_count += 1
            if len(shown) < SESSION_CLIP_INSPECTION_MAX_MISSING_DIAGNOSTIC_INDEXES:
                shown.append(index)
        if missing_count == 0:
            return None
        message = "missing fragment indexes: " + ",".join(
            str(index) for index in shown
        )
        omitted = missing_count - len(shown)
        if omitted > 0:
            message += f" ... (+{omitted} more)"
        return message

    def _assemble_and_evict(
        self,
        key: tuple[str, str],
        state: dict[str, object],
    ) -> dict[str, object]:
        try:
            return self._assemble_state(key, state)
        finally:
            self._evict_state(key)

    def _evict_state(self, key: tuple[str, str]) -> None:
        state = self._states.pop(key, None)
        if state is None:
            return
        retained_bytes = state.get("retained_bytes", 0)
        if isinstance(retained_bytes, int):
            self._retained_bytes = max(0, self._retained_bytes - retained_bytes)

    def add_event(self, event: AckEvent) -> dict[str, object] | None:
        if event.event != "api_session_clip_inspect":
            raise SessionClipInspectionAssemblyError("malformed fragment event")
        return self.add_fragment(event.payload.get("fragment"), event.request_id)

    def add_fragment(
        self,
        fragment: object,
        request_id: str | None = None,
    ) -> dict[str, object] | None:
        candidate_key = self._candidate_state_key(fragment, request_id)
        try:
            (
                value,
                key,
                fragment_index,
                fragment_count,
                _fragment_kind,
                metadata_key,
            ) = self._validate_fragment(fragment, request_id)
            canonical_fragment = self._canonical(value)
            fragment_bytes = len(canonical_fragment.encode("utf-8"))
            if fragment_bytes > self.PACKET_BUDGET_BYTES:
                raise SessionClipInspectionAssemblyError(
                    "resource limit exceeded: fragment bytes"
                )
            state = self._states.get(key)
            if state is None:
                if len(self._states) >= self.MAX_ACTIVE_ASSEMBLIES:
                    raise SessionClipInspectionAssemblyError(
                        "active assembly limit exceeded"
                    )
                state = {
                    "metadata_key": metadata_key,
                    "fragment_count": fragment_count,
                    "fragments": {},
                    "retained_bytes": 0,
                }
                self._states[key] = state
            elif (
                state["metadata_key"] != metadata_key
                or state["fragment_count"] != fragment_count
            ):
                raise SessionClipInspectionAssemblyError("mixed metadata")

            fragments = state["fragments"]
            assert isinstance(fragments, dict)
            existing = fragments.get(fragment_index)
            if existing is not None:
                _existing_fragment, existing_canonical = existing
                if existing_canonical != canonical_fragment:
                    raise SessionClipInspectionAssemblyError(
                        f"conflicting duplicate fragment index {fragment_index}"
                    )
                return None

            if self._retained_bytes + fragment_bytes > self.MAX_RETAINED_BYTES:
                raise SessionClipInspectionAssemblyError(
                    "resource limit exceeded: retained bytes"
                )
            fragments[fragment_index] = (value, canonical_fragment)
            state_retained_bytes = state.get("retained_bytes", 0)
            assert isinstance(state_retained_bytes, int)
            state["retained_bytes"] = state_retained_bytes + fragment_bytes
            self._retained_bytes += fragment_bytes
            if len(fragments) == fragment_count:
                return self._assemble_and_evict(key, state)
            return None
        except SessionClipInspectionAssemblyError:
            if candidate_key is not None:
                self._evict_state(candidate_key)
            raise

    def assemble(self, request_id: str, inspection_id: str) -> dict[str, object]:
        key = (request_id, inspection_id)
        state = self._states.get(key)
        if state is None:
            raise SessionClipInspectionAssemblyError("missing fragment indexes: all")
        fragment_count = int(state["fragment_count"])
        fragments = state["fragments"]
        assert isinstance(fragments, dict)
        missing_message = self._missing_fragment_message(
            fragment_count,
            fragments,
        )
        if missing_message is not None:
            self._evict_state(key)
            raise SessionClipInspectionAssemblyError(missing_message)
        return self._assemble_and_evict(key, state)

    def _assemble_pages(
        self,
        pages: list[dict[str, object]],
        *,
        offset_field: str,
        count_field: str,
        total_field: str,
        items_field: str,
        label: str,
    ) -> list[object]:
        if not pages:
            return []
        expected_offset = 0
        expected_total = int(pages[0][total_field])
        if expected_total == 0:
            raise SessionClipInspectionAssemblyError("inconsistent counts")
        items: list[object] = []
        for page in pages:
            if int(page[total_field]) != expected_total:
                raise SessionClipInspectionAssemblyError("mixed metadata")
            if int(page[offset_field]) != expected_offset:
                raise SessionClipInspectionAssemblyError(
                    f"noncontiguous {label} offsets"
                )
            if int(page[count_field]) <= 0:
                raise SessionClipInspectionAssemblyError("inconsistent counts")
            page_items = self._require_list(page[items_field], items_field)
            items.extend(page_items)
            expected_offset += int(page[count_field])
        if expected_offset != expected_total:
            raise SessionClipInspectionAssemblyError("inconsistent counts")
        return items

    def _assemble_state(
        self,
        key: tuple[str, str],
        state: dict[str, object],
    ) -> dict[str, object]:
        fragments = state["fragments"]
        assert isinstance(fragments, dict)
        ordered = [fragments[index][0] for index in sorted(fragments)]
        first = ordered[0]
        transfer = self._require_dict(first["transfer"], "transfer")

        if len(ordered) == 1 and transfer["fragment_kind"] == "complete":
            data = self._require_dict(first["data"], "data")
            devices = list(self._require_list(data["devices"], "devices"))
            notes = list(self._require_list(data["notes"], "notes"))
            context_data = data
        else:
            kinds = [
                self._require_dict(fragment["transfer"], "transfer")[
                    "fragment_kind"
                ]
                for fragment in ordered
            ]
            if not kinds or kinds[0] != "context":
                raise SessionClipInspectionAssemblyError(
                    "malformed fragments: context"
                )
            page_phase = "device_page"
            device_pages: list[dict[str, object]] = []
            note_pages: list[dict[str, object]] = []
            for fragment, kind in zip(ordered[1:], kinds[1:]):
                data = self._require_dict(fragment["data"], "data")
                if kind == "device_page":
                    if page_phase == "note_page":
                        raise SessionClipInspectionAssemblyError(
                            "invalid fragment ordering"
                        )
                    device_pages.append(data)
                elif kind == "note_page":
                    page_phase = "note_page"
                    note_pages.append(data)
                else:
                    raise SessionClipInspectionAssemblyError(
                        "invalid fragment ordering"
                    )
            context_data = self._require_dict(ordered[0]["data"], "data")
            devices = self._assemble_pages(
                device_pages,
                offset_field="device_offset",
                count_field="device_count",
                total_field="device_total",
                items_field="devices",
                label="device",
            )
            notes = self._assemble_pages(
                note_pages,
                offset_field="note_offset",
                count_field="note_count",
                total_field="note_total",
                items_field="notes",
                label="note",
            )

        summary = self._require_dict(context_data["summary"], "summary")
        if summary.get("note_count") != len(notes):
            raise SessionClipInspectionAssemblyError("inconsistent counts")
        note_pitches = [
            self._require_dict(note, "note")["pitch"] for note in notes
        ]
        expected_pitch_min = min(note_pitches) if note_pitches else None
        expected_pitch_max = max(note_pitches) if note_pitches else None
        if (
            summary.get("pitch_min") != expected_pitch_min
            or summary.get("pitch_max") != expected_pitch_max
        ):
            raise SessionClipInspectionAssemblyError("inconsistent summary")

        correlation = self._require_dict(first["correlation"], "correlation")
        snapshot = self._require_dict(first["snapshot"], "snapshot")
        completeness = self._require_dict(first["completeness"], "completeness")
        fragment_count = int(state["fragment_count"])
        track = self._require_dict(context_data["track"], "track")
        clip = self._require_dict(context_data["clip"], "clip")
        device_values = [
            self._require_dict(device, "device") for device in devices
        ]
        note_values = [self._require_dict(note, "note") for note in notes]
        return {
            "schema": first["schema"],
            "schema_version": first["schema_version"],
            "producer_version": first["producer_version"],
            "inspection_id": key[1],
            "correlation": self._copy_fields(
                correlation,
                ("request_id", "track_index", "slot_index"),
            ),
            "snapshot": self._copy_fields(
                snapshot,
                ("started_ms", "completed_ms", "atomic", "consistent"),
            ),
            "completeness": self._copy_fields(
                completeness,
                ("track", "clip", "devices", "notes", "missing_fields"),
            ),
            "context": context_data["context"],
            "track": self._copy_fields(track, ("index", "path", "id", "name")),
            "clip": self._copy_fields(
                clip,
                (
                    "slot_index",
                    "path",
                    "id",
                    "name",
                    "start_marker",
                    "end_marker",
                    "live_length",
                    "looping",
                    "loop_start",
                    "loop_end",
                ),
            ),
            "summary": self._copy_fields(
                summary,
                ("note_count", "pitch_min", "pitch_max"),
            ),
            "devices": [
                self._copy_fields(
                    device,
                    ("index", "path", "id", "name", "class_name", "type"),
                )
                for device in device_values
            ],
            "notes": [
                self._copy_fields(
                    note,
                    (
                        "note_id",
                        "pitch",
                        "start_time",
                        "duration",
                        "velocity",
                        "mute",
                        "probability",
                        "velocity_deviation",
                        "release_velocity",
                    ),
                )
                for note in note_values
            ],
            "transport": {
                "complete": True,
                "fragment_count": fragment_count,
                "received_fragment_count": len(fragments),
                "fragment_indexes": sorted(int(index) for index in fragments),
                "packet_budget_bytes": self.PACKET_BUDGET_BYTES,
            },
        }


class ArrangementInspectionAssemblyError(ValueError):
    """Raised when Arrangement inspection fragments fail strict validation."""


class ArrangementInspectionAssembler:
    """Assemble bounded, correlated Arrangement facts without implicit notes."""

    SCHEMA = "codex-live-bridge.arrangement-inspection"
    SCHEMA_VERSION = 1
    PRODUCER_VERSION = "3.2.0"
    PACKET_BUDGET_BYTES = ARRANGEMENT_INSPECTION_PACKET_BUDGET_BYTES
    MAX_ITEMS = ARRANGEMENT_INSPECTION_MAX_ITEMS
    MAX_NOTES = ARRANGEMENT_INSPECTION_MAX_NOTES
    MAX_FRAGMENTS = ARRANGEMENT_INSPECTION_MAX_FRAGMENTS
    MAX_ACTIVE_ASSEMBLIES = ARRANGEMENT_INSPECTION_MAX_ACTIVE_ASSEMBLIES
    MAX_RETAINED_BYTES = ARRANGEMENT_INSPECTION_MAX_RETAINED_BYTES
    _ROOT_KEYS = {
        "schema", "schema_version", "producer_version", "inspection_id",
        "correlation", "snapshot", "transfer", "data",
    }
    _CORRELATION_KEYS = {"request_id", "scope", "track_index", "clip_index"}
    _SNAPSHOT_KEYS = {"started_ms", "completed_ms", "atomic", "consistent"}
    _TRANSFER_KEYS = {
        "fragment_index", "fragment_count", "fragment_kind", "is_last",
        "packet_budget_bytes",
    }
    _TRACK_KEYS = {
        "index", "path", "id", "name", "has_midi_input", "has_audio_input",
        "mute", "solo", "arrangement_clip_count", "device_count",
    }
    _CLIP_KEYS = {
        "index", "path", "id", "name", "start_time", "end_time",
        "start_marker", "end_marker", "live_length", "looping", "loop_start",
        "loop_end", "is_midi_clip", "is_audio_clip",
    }
    _DEVICE_KEYS = {"index", "path", "id", "name", "class_name", "type"}
    _AUX_TRACK_KEYS = {"index", "path", "id", "name", "device_count", "mute", "solo"}

    def __init__(self) -> None:
        self._states: dict[tuple[str, str], dict[str, object]] = {}
        self._retained_bytes = 0

    @staticmethod
    def _fail(label: str) -> None:
        raise ArrangementInspectionAssemblyError(f"malformed fragment: {label}")

    @classmethod
    def _require_dict(
        cls, value: object, keys: set[str], label: str
    ) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != keys:
            cls._fail(label)
        assert isinstance(value, dict)
        return value

    @classmethod
    def _require_index(
        cls, value: object, label: str, maximum: int | None = None
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            cls._fail(label)
        assert isinstance(value, int)
        if maximum is not None and value > maximum:
            raise ArrangementInspectionAssemblyError(
                f"resource limit exceeded: {label}"
            )
        return value

    @classmethod
    def _require_text(cls, value: object, label: str) -> str:
        if not isinstance(value, str) or not value:
            cls._fail(label)
        assert isinstance(value, str)
        return value

    @classmethod
    def _require_nullable_text(cls, value: object, label: str) -> None:
        if value is not None and not isinstance(value, str):
            cls._fail(label)

    @classmethod
    def _require_nullable_bool(cls, value: object, label: str) -> None:
        if value is not None and not isinstance(value, bool):
            cls._fail(label)

    @classmethod
    def _require_nullable_number(cls, value: object, label: str) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            cls._fail(label)
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            cls._fail(label)
            return
        if not finite:
            cls._fail(label)

    @classmethod
    def _canonical(cls, value: object) -> str:
        try:
            return json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ArrangementInspectionAssemblyError(
                "malformed fragment: not JSON serializable"
            ) from exc

    def _evict_state(self, key: tuple[str, str]) -> None:
        state = self._states.pop(key, None)
        if state is not None:
            self._retained_bytes = max(
                0, self._retained_bytes - int(state.get("retained_bytes", 0))
            )

    def _validate_track(
        self, value: object, expected_index: int | None = None
    ) -> dict[str, object]:
        track = self._require_dict(value, self._TRACK_KEYS, "track keys")
        index = self._require_index(track["index"], "track.index")
        if expected_index is not None and index != expected_index:
            raise ArrangementInspectionAssemblyError("mixed metadata")
        self._require_text(track["path"], "track.path")
        self._require_index(track["id"], "track.id")
        self._require_nullable_text(track["name"], "track.name")
        for field in ("has_midi_input", "has_audio_input", "mute", "solo"):
            self._require_nullable_bool(track[field], f"track.{field}")
        self._require_index(
            track["arrangement_clip_count"], "track.arrangement_clip_count", self.MAX_ITEMS
        )
        self._require_index(track["device_count"], "track.device_count", self.MAX_ITEMS)
        return track

    def _validate_clip(
        self, value: object, expected_index: int | None = None
    ) -> dict[str, object]:
        clip = self._require_dict(value, self._CLIP_KEYS, "clip keys")
        index = self._require_index(clip["index"], "clip.index")
        if expected_index is not None and index != expected_index:
            raise ArrangementInspectionAssemblyError("mixed metadata")
        self._require_text(clip["path"], "clip.path")
        self._require_index(clip["id"], "clip.id")
        self._require_nullable_text(clip["name"], "clip.name")
        for field in (
            "start_time", "end_time", "start_marker", "end_marker", "live_length",
            "loop_start", "loop_end",
        ):
            self._require_nullable_number(clip[field], f"clip.{field}")
        if (
            clip["start_time"] is None
            or clip["end_time"] is None
            or float(clip["end_time"]) < float(clip["start_time"])
            or not math.isfinite(float(clip["end_time"]) - float(clip["start_time"]))
        ):
            self._fail("clip time range")
        if clip["live_length"] is not None and float(clip["live_length"]) < 0:
            self._fail("clip.live_length")
        for start_field, end_field in (
            ("start_marker", "end_marker"), ("loop_start", "loop_end")
        ):
            start = clip[start_field]
            end = clip[end_field]
            if start is not None and end is not None and (
                float(end) < float(start)
                or not math.isfinite(float(end) - float(start))
            ):
                self._fail("clip ranges")
        for field in ("looping", "is_midi_clip", "is_audio_clip"):
            self._require_nullable_bool(clip[field], f"clip.{field}")
        return clip

    def _validate_devices(self, value: object, expected_count: int) -> None:
        if not isinstance(value, list) or len(value) != expected_count:
            self._fail("device count")
        for expected_index, item in enumerate(value):
            device = self._require_dict(item, self._DEVICE_KEYS, "device keys")
            if self._require_index(device["index"], "device.index") != expected_index:
                self._fail("device indexes")
            self._require_text(device["path"], "device.path")
            self._require_index(device["id"], "device.id")
            self._require_nullable_text(device["name"], "device.name")
            self._require_nullable_text(device["class_name"], "device.class_name")
            device_type = device["type"]
            if device_type is not None and (
                isinstance(device_type, bool)
                or not isinstance(device_type, int)
                or device_type not in {0, 1, 2, 4}
            ):
                self._fail("device.type")

    def _validate_aux_track(
        self, value: object, expected_index: int | None
    ) -> None:
        track = self._require_dict(value, self._AUX_TRACK_KEYS, "aux track keys")
        if track["index"] != expected_index or isinstance(track["index"], bool):
            self._fail("aux track index")
        self._require_text(track["path"], "aux track.path")
        self._require_index(track["id"], "aux track.id")
        self._require_nullable_text(track["name"], "aux track.name")
        self._require_index(track["device_count"], "aux track.device_count", self.MAX_ITEMS)
        self._require_nullable_bool(track["mute"], "aux track.mute")
        self._require_nullable_bool(track["solo"], "aux track.solo")

    def _validate_payload(
        self, payload: object, correlation: dict[str, object]
    ) -> dict[str, object]:
        if not isinstance(payload, dict):
            self._fail("payload")
        assert isinstance(payload, dict)
        scope = correlation["scope"]
        if payload.get("context") != "arrangement" or payload.get("scope") != scope:
            raise ArrangementInspectionAssemblyError("mixed metadata")
        privacy = self._require_dict(
            payload.get("privacy"), {"notes_requested", "notes_included"}, "privacy"
        )
        requested = privacy["notes_requested"]
        included = privacy["notes_included"]
        if not isinstance(requested, bool) or not isinstance(included, bool):
            self._fail("privacy flags")
        if requested != included or (scope != "clip" and (requested or included)):
            raise ArrangementInspectionAssemblyError("privacy contract violation")

        if scope == "project":
            if set(payload) != {
                "context", "scope", "project", "tracks", "return_tracks", "main_track", "privacy"
            }:
                self._fail("project payload keys")
            project = self._require_dict(
                payload["project"],
                {
                    "tempo", "signature_numerator", "signature_denominator",
                    "track_count", "return_track_count",
                },
                "project keys",
            )
            for field in ("tempo", "signature_numerator", "signature_denominator"):
                self._require_nullable_number(project[field], f"project.{field}")
            track_count = self._require_index(
                project["track_count"], "project.track_count", self.MAX_ITEMS
            )
            return_count = self._require_index(
                project["return_track_count"], "project.return_track_count", self.MAX_ITEMS
            )
            tracks = payload["tracks"]
            if not isinstance(tracks, list) or len(tracks) != track_count:
                self._fail("track count")
            for index, track in enumerate(tracks):
                self._validate_track(track, index)
            returns = payload["return_tracks"]
            if not isinstance(returns, list) or len(returns) != return_count:
                self._fail("return track count")
            for index, track in enumerate(returns):
                self._validate_aux_track(track, index)
            self._validate_aux_track(payload["main_track"], None)
            return payload

        track = self._validate_track(payload.get("track"), int(correlation["track_index"]))
        if scope == "track":
            if set(payload) != {"context", "scope", "track", "clips", "devices", "privacy"}:
                self._fail("track payload keys")
            clips = payload["clips"]
            if not isinstance(clips, list) or len(clips) != track["arrangement_clip_count"]:
                self._fail("clip count")
            for index, clip in enumerate(clips):
                self._validate_clip(clip, index)
            self._validate_devices(payload["devices"], int(track["device_count"]))
            return payload

        expected_keys = {"context", "scope", "track", "clip", "devices", "privacy"}
        if requested:
            expected_keys.update({"notes", "summary"})
        if set(payload) != expected_keys:
            raise ArrangementInspectionAssemblyError("privacy contract violation")
        clip = self._validate_clip(payload["clip"], int(correlation["clip_index"]))
        self._validate_devices(payload["devices"], int(track["device_count"]))
        if requested:
            if clip["is_midi_clip"] is not True:
                raise ArrangementInspectionAssemblyError("privacy contract violation")
            notes = payload["notes"]
            if not isinstance(notes, list) or len(notes) > self.MAX_NOTES:
                raise ArrangementInspectionAssemblyError("resource limit exceeded: notes")
            validator = SessionClipInspectionAssembler()
            try:
                for note in notes:
                    validator._validate_note(note)
                summary = validator._validate_summary(payload["summary"])
            except SessionClipInspectionAssemblyError as exc:
                raise ArrangementInspectionAssemblyError(str(exc)) from exc
            pitches = [int(note["pitch"]) for note in notes]
            if (
                summary["note_count"] != len(notes)
                or summary["pitch_min"] != (min(pitches) if pitches else None)
                or summary["pitch_max"] != (max(pitches) if pitches else None)
            ):
                self._fail("note summary")
        return payload

    def add_event(self, event: AckEvent) -> dict[str, object] | None:
        if event.event not in ARRANGEMENT_INSPECTION_EVENTS:
            raise ArrangementInspectionAssemblyError("malformed fragment event")
        return self.add_fragment(event.payload.get("fragment"), event.request_id, event.event)

    def add_fragment(
        self,
        fragment: object,
        request_id: str | None = None,
        event_name: str | None = None,
    ) -> dict[str, object] | None:
        candidate_key: tuple[str, str] | None = None
        if isinstance(fragment, dict) and isinstance(fragment.get("correlation"), dict):
            candidate_request = fragment["correlation"].get("request_id")
            candidate_id = fragment.get("inspection_id")
            if (
                isinstance(candidate_request, str)
                and isinstance(candidate_id, str)
                and (request_id is None or request_id == candidate_request)
            ):
                candidate_key = (candidate_request, candidate_id)
        try:
            value = self._require_dict(fragment, self._ROOT_KEYS, "root keys")
            if (
                value["schema"] != self.SCHEMA
                or isinstance(value["schema_version"], bool)
                or not isinstance(value["schema_version"], int)
                or value["schema_version"] != self.SCHEMA_VERSION
                or value["producer_version"] != self.PRODUCER_VERSION
            ):
                self._fail("schema")
            inspection_id = self._require_text(value["inspection_id"], "inspection_id")
            if len(inspection_id.encode("utf-8")) > 128:
                self._fail("inspection_id")
            correlation = self._require_dict(
                value["correlation"], self._CORRELATION_KEYS, "correlation keys"
            )
            correlated_request = self._require_text(
                correlation["request_id"], "correlation.request_id"
            )
            if len(correlated_request.encode("utf-8")) > 128:
                self._fail("correlation.request_id")
            if request_id is not None and correlated_request != request_id:
                raise ArrangementInspectionAssemblyError("request ID mismatch")
            scope = correlation["scope"]
            if not isinstance(scope, str) or scope not in {"project", "track", "clip"}:
                self._fail("correlation.scope")
            expected_event = f"api_arrangement_{scope}_inspect"
            if event_name is not None and event_name != expected_event:
                raise ArrangementInspectionAssemblyError("mixed metadata")
            track_index = correlation["track_index"]
            clip_index = correlation["clip_index"]
            if scope == "project":
                if track_index is not None or clip_index is not None:
                    self._fail("correlation indexes")
            else:
                self._require_index(track_index, "correlation.track_index")
                if scope == "track" and clip_index is not None:
                    self._fail("correlation.clip_index")
                if scope == "clip":
                    self._require_index(clip_index, "correlation.clip_index")
            snapshot = self._require_dict(value["snapshot"], self._SNAPSHOT_KEYS, "snapshot keys")
            started = self._require_index(snapshot["started_ms"], "snapshot.started_ms")
            completed = self._require_index(snapshot["completed_ms"], "snapshot.completed_ms")
            if completed < started or snapshot["atomic"] is not False or snapshot["consistent"] is not True:
                self._fail("snapshot")
            transfer = self._require_dict(value["transfer"], self._TRANSFER_KEYS, "transfer keys")
            count = self._require_index(transfer["fragment_count"], "transfer.fragment_count", self.MAX_FRAGMENTS)
            index = self._require_index(transfer["fragment_index"], "transfer.fragment_index")
            if count == 0 or index >= count:
                self._fail("transfer indexes")
            kind = transfer["fragment_kind"]
            if kind != ("complete" if count == 1 else "chunk"):
                self._fail("transfer.fragment_kind")
            if transfer["is_last"] is not (index == count - 1):
                self._fail("transfer.is_last")
            if (
                isinstance(transfer["packet_budget_bytes"], bool)
                or not isinstance(transfer["packet_budget_bytes"], int)
                or transfer["packet_budget_bytes"] != self.PACKET_BUDGET_BYTES
            ):
                self._fail("transfer.packet_budget_bytes")
            data = self._require_dict(value["data"], {"payload_chunk"}, "data keys")
            self._require_text(data["payload_chunk"], "data.payload_chunk")

            canonical = self._canonical(value)
            try:
                fragment_bytes = len(canonical.encode("utf-8"))
            except UnicodeError as exc:
                raise ArrangementInspectionAssemblyError("malformed fragment: UTF-8") from exc
            if fragment_bytes > self.PACKET_BUDGET_BYTES:
                raise ArrangementInspectionAssemblyError("resource limit exceeded: fragment bytes")
            key = (correlated_request, inspection_id)
            metadata = self._canonical(
                {
                    "correlation": correlation,
                    "snapshot": snapshot,
                    "count": count,
                    "event": expected_event,
                }
            )
            state = self._states.get(key)
            if state is None:
                if len(self._states) >= self.MAX_ACTIVE_ASSEMBLIES:
                    raise ArrangementInspectionAssemblyError("active assembly limit exceeded")
                state = {"metadata": metadata, "count": count, "fragments": {}, "retained_bytes": 0}
                self._states[key] = state
            elif state["metadata"] != metadata:
                raise ArrangementInspectionAssemblyError("mixed metadata")
            fragments = state["fragments"]
            assert isinstance(fragments, dict)
            existing = fragments.get(index)
            if existing is not None:
                if existing[1] != canonical:
                    raise ArrangementInspectionAssemblyError("conflicting duplicate fragment")
                return None
            if self._retained_bytes + fragment_bytes > self.MAX_RETAINED_BYTES:
                raise ArrangementInspectionAssemblyError("resource limit exceeded: retained bytes")
            fragments[index] = (data["payload_chunk"], canonical)
            state["retained_bytes"] = int(state["retained_bytes"]) + fragment_bytes
            self._retained_bytes += fragment_bytes
            if len(fragments) != count:
                return None
            try:
                payload_text = "".join(str(fragments[i][0]) for i in range(count))
                if len(payload_text.encode("utf-8")) > self.MAX_RETAINED_BYTES:
                    raise ArrangementInspectionAssemblyError("resource limit exceeded: payload bytes")
                payload = json.loads(
                    payload_text,
                    parse_constant=lambda _value: (_ for _ in ()).throw(
                        ValueError("non-finite number")
                    ),
                )
                validated = self._validate_payload(payload, correlation)
                return {
                    "schema": self.SCHEMA,
                    "schema_version": self.SCHEMA_VERSION,
                    "producer_version": self.PRODUCER_VERSION,
                    "inspection_id": inspection_id,
                    "correlation": dict(correlation),
                    "snapshot": dict(snapshot),
                    "data": validated,
                    "transport": {
                        "complete": True,
                        "fragment_count": count,
                        "received_fragment_count": count,
                        "fragment_indexes": list(range(count)),
                        "packet_budget_bytes": self.PACKET_BUDGET_BYTES,
                    },
                }
            except (json.JSONDecodeError, ValueError, UnicodeError) as exc:
                if isinstance(exc, ArrangementInspectionAssemblyError):
                    raise
                raise ArrangementInspectionAssemblyError("malformed payload JSON") from exc
            finally:
                self._evict_state(key)
        except ArrangementInspectionAssemblyError:
            if candidate_key is not None:
                self._evict_state(candidate_key)
            raise


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (max(0.0, min(100.0, float(pct))) / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _format_ms(value: float) -> str:
    return f"{float(value):.2f}"


def _summarize_metrics(metrics: SendMetrics) -> List[str]:
    lines: List[str] = []
    lines.append(
        "metrics: commands={count} elapsed_ms={elapsed}".format(
            count=metrics.command_count,
            elapsed=_format_ms(metrics.elapsed_ms),
        )
    )

    if metrics.send_durations_ms:
        lines.append(
            "metrics: send_ms p50={p50} p95={p95} max={maxv}".format(
                p50=_format_ms(_percentile(metrics.send_durations_ms, 50.0)),
                p95=_format_ms(_percentile(metrics.send_durations_ms, 95.0)),
                maxv=_format_ms(max(metrics.send_durations_ms)),
            )
        )

    if metrics.ack_wait_durations_ms:
        lines.append(
            "metrics: ack_wait_ms p50={p50} p95={p95} max={maxv} mean={meanv}".format(
                p50=_format_ms(_percentile(metrics.ack_wait_durations_ms, 50.0)),
                p95=_format_ms(_percentile(metrics.ack_wait_durations_ms, 95.0)),
                maxv=_format_ms(max(metrics.ack_wait_durations_ms)),
                meanv=_format_ms(mean(metrics.ack_wait_durations_ms)),
            )
        )

    if metrics.acks_per_command:
        lines.append(
            "metrics: acks_per_command mean={meanv} max={maxv}".format(
                meanv=_format_ms(mean(float(v) for v in metrics.acks_per_command)),
                maxv=max(metrics.acks_per_command),
            )
        )
    return lines


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and > 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and >= 0")
    return parsed


def midi_byte(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 127:
        raise argparse.ArgumentTypeError("value must be between 0 and 127")
    return parsed


def midi_channel(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 16:
        raise argparse.ArgumentTypeError("channel must be between 1 and 16")
    return parsed


def normalize_auth_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    byte_length = len(token.encode("utf-8"))
    if (
        token == AUTH_TOKEN_PLACEHOLDER
        or byte_length < MIN_AUTH_TOKEN_BYTES
        or byte_length > MAX_AUTH_TOKEN_BYTES
    ):
        raise ValueError(
            f"auth token must be {MIN_AUTH_TOKEN_BYTES} to "
            f"{MAX_AUTH_TOKEN_BYTES} UTF-8 bytes and must not be the placeholder"
        )
    return token


def loopback_host(value: str) -> str:
    host = str(value).strip()
    if host.lower() == "localhost":
        return DEFAULT_HOST
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "host must be an IPv4 loopback address or localhost"
        ) from exc
    if address.version != 4 or not address.is_loopback:
        raise argparse.ArgumentTypeError(
            "host must be an IPv4 loopback address or localhost"
        )
    return host


def authenticated_args(
    auth_token: str | None,
    args: Sequence[OscArg] = (),
) -> Tuple[OscArg, ...]:
    token = normalize_auth_token(auth_token)
    if token is None:
        raise ValueError(
            "Mutating commands require --auth-token or "
            f"the {AUTH_TOKEN_ENV} environment variable"
        )
    return (token, *tuple(args))


def parse_args(argv: Iterable[str]) -> BridgeConfig:
    parser = argparse.ArgumentParser(
        description="Send OSC UDP commands to a Max for Live Ableton bridge."
    )
    parser.add_argument(
        "--host",
        type=loopback_host,
        default=DEFAULT_HOST,
        help="Loopback UDP host (default: 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UDP port")
    parser.add_argument(
        "--auth-token",
        default=os.environ.get(AUTH_TOKEN_ENV),
        help=(
            "Capability token for mutating commands "
            f"(default: {AUTH_TOKEN_ENV} environment variable)"
        ),
    )

    parser.add_argument(
        "--ack",
        action="store_true",
        help="Listen for OSC /ack responses on --ack-port",
    )
    parser.add_argument(
        "--ack-port",
        type=int,
        default=DEFAULT_ACK_PORT,
        help="UDP port to listen for acknowledgements",
    )
    parser.add_argument(
        "--ack-timeout",
        type=positive_float,
        default=0.6,
        help="How long to wait for acknowledgements after each send (seconds)",
    )
    parser.add_argument(
        "--ack-mode",
        choices=("per_command", "flush_end", "flush_interval"),
        default="per_command",
        help=(
            "Acknowledgement handling strategy (default: per_command); flush modes "
            "batch only distinct request IDs and serialize uncorrelated or fragmented commands"
        ),
    )
    parser.add_argument(
        "--ack-flush-interval",
        type=positive_int,
        default=10,
        help="Flush interval in commands when --ack-mode=flush_interval (default: 10)",
    )
    parser.add_argument(
        "--no-ping-first",
        action="store_true",
        help="Skip sending /ping before the main commands when --ack is enabled",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Request bridge status via /status",
    )
    parser.add_argument(
        "--api-ping",
        nargs="?",
        action="append",
        default=[],
        metavar="REQUEST_ID",
        help="Send /api/ping with an optional request id",
    )
    parser.add_argument(
        "--api-get",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/get <path> <property> [request_id]",
    )
    parser.add_argument(
        "--api-set",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/set <path> <property> <value_json> [request_id]",
    )
    parser.add_argument(
        "--api-call",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/call <path> <method> <args_json> [request_id]",
    )
    parser.add_argument(
        "--api-children",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/children <path> <child_name> [request_id]",
    )
    parser.add_argument(
        "--api-describe",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/describe <path> [request_id]",
    )
    parser.add_argument(
        "--api-observe",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api_observe <path> <property_or_child> <options_json> [request_id]",
    )
    parser.add_argument(
        "--api-unobserve",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api_unobserve <observer_id> [request_id]",
    )
    parser.add_argument(
        "--api-observers",
        nargs="?",
        action="append",
        default=[],
        metavar="REQUEST_ID",
        help="Send /api_observers with an optional request id",
    )
    parser.add_argument(
        "--api-clear-observers",
        nargs="?",
        action="append",
        default=[],
        metavar="REQUEST_ID",
        help="Send /api_clear_observers with an optional request id",
    )
    parser.add_argument(
        "--api-session-context",
        nargs="?",
        action="append",
        default=[],
        metavar="REQUEST_ID",
        help="Send /api/session_context with an optional request id",
    )
    parser.add_argument(
        "--api-theory-status",
        nargs="?",
        action="append",
        default=[],
        metavar="REQUEST_ID",
        help="Send /api/theory_status with an optional request id",
    )
    parser.add_argument(
        "--api-tuning-status",
        nargs="?",
        action="append",
        default=[],
        metavar="REQUEST_ID",
        help="Send /api/tuning_status with an optional request id",
    )
    parser.add_argument(
        "--api-device-list",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/device_list <track_ref|all> [request_id]",
    )
    parser.add_argument(
        "--api-device-parameters",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/device_parameters <device_path> [request_id]",
    )
    parser.add_argument(
        "--api-parameter-set",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/parameter_set <parameter_path> <value_json> [request_id]",
    )
    parser.add_argument(
        "--api-mixer-status",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/mixer_status <track_ref|master|return:N> [request_id]",
    )
    parser.add_argument(
        "--api-insert-device",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/insert_device <track_or_chain_path> <native_device_name> <target_index_or_empty> [request_id]",
    )
    parser.add_argument(
        "--api-insert-chain",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/insert_chain <rack_device_path> <target_index_or_empty> [request_id]",
    )
    parser.add_argument(
        "--api-drum-chain-in-note",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /api/drum_chain_in_note <drum_chain_path> <note|-1> [request_id]",
    )
    parser.add_argument(
        "--api-session-clip-inspect",
        nargs=3,
        action="append",
        default=[],
        metavar=("TRACK_INDEX", "SLOT_INDEX", "REQUEST_ID"),
        help="Send /api/session_clip_inspect <track_index> <slot_index> 1 <request_id>",
    )
    parser.add_argument(
        "--api-arrangement-project-inspect",
        action="append",
        default=[],
        metavar="REQUEST_ID",
        help="Read bounded Arrangement project/track metadata without MIDI notes",
    )
    parser.add_argument(
        "--api-arrangement-track-inspect",
        nargs=2,
        action="append",
        default=[],
        metavar=("TRACK_INDEX", "REQUEST_ID"),
        help="Read bounded Arrangement track, clip, and device metadata without MIDI notes",
    )
    parser.add_argument(
        "--api-arrangement-clip-inspect",
        nargs=3,
        action="append",
        default=[],
        metavar=("TRACK_INDEX", "CLIP_INDEX", "REQUEST_ID"),
        help="Read one Arrangement clip without retrieving or returning MIDI notes",
    )
    parser.add_argument(
        "--api-arrangement-clip-inspect-notes",
        nargs=3,
        action="append",
        default=[],
        metavar=("TRACK_INDEX", "CLIP_INDEX", "REQUEST_ID"),
        help="Explicitly opt in to bounded raw MIDI notes for one Arrangement clip",
    )

    parser.add_argument(
        "--tempo",
        type=positive_float,
        default=None,
        help="Set tempo in BPM",
    )
    parser.add_argument(
        "--no-tempo",
        action="store_true",
        help="Do not send a tempo command",
    )
    parser.add_argument(
        "--sig-num",
        type=positive_int,
        default=None,
        help="Set time signature numerator (default denominator: 4)",
    )
    parser.add_argument(
        "--sig-den",
        type=positive_int,
        default=None,
        help="Set time signature denominator (default numerator: 4)",
    )
    parser.add_argument(
        "--no-signature",
        action="store_true",
        help="Do not send time signature commands",
    )
    parser.add_argument(
        "--create-midi-tracks",
        type=non_negative_int,
        default=0,
        help="How many /create_midi_track commands to send",
    )
    parser.add_argument(
        "--add-midi-tracks",
        type=non_negative_int,
        default=0,
        help="Create and name this many MIDI tracks via /add_midi_tracks",
    )
    parser.add_argument(
        "--midi-name",
        default="MIDI",
        help="Name used with --add-midi-tracks (default: MIDI)",
    )
    parser.add_argument(
        "--create-audio-tracks",
        type=non_negative_int,
        default=0,
        help="How many /create_audio_track commands to send",
    )
    parser.add_argument(
        "--add-audio-tracks",
        type=non_negative_int,
        default=0,
        help="Create and name this many audio tracks via /add_audio_tracks",
    )
    parser.add_argument(
        "--audio-prefix",
        default="Audio",
        help="Name prefix used with --add-audio-tracks (default: Audio)",
    )
    parser.add_argument(
        "--delete-audio-tracks",
        type=non_negative_int,
        default=0,
        help="Delete this many audio tracks via /delete_audio_tracks",
    )
    parser.add_argument(
        "--delete-midi-tracks",
        type=non_negative_int,
        default=0,
        help="Delete this many MIDI tracks via /delete_midi_tracks (track 0 is protected)",
    )
    parser.add_argument(
        "--rename-track-index",
        type=non_negative_int,
        default=None,
        help="Track index to rename via /rename_track",
    )
    parser.add_argument(
        "--rename-track-name",
        default=None,
        help="New name used with --rename-track-index",
    )
    parser.add_argument(
        "--session-clip-track-index",
        type=non_negative_int,
        default=None,
        help="Track index for /set_session_clip_notes",
    )
    parser.add_argument(
        "--session-clip-slot-index",
        type=non_negative_int,
        default=None,
        help="Clip slot index for /set_session_clip_notes",
    )
    parser.add_argument(
        "--session-clip-length",
        type=positive_float,
        default=None,
        help="Clip length in beats for /set_session_clip_notes",
    )
    parser.add_argument(
        "--session-clip-notes-json",
        default=None,
        help="JSON payload (string) for /set_session_clip_notes",
    )
    parser.add_argument(
        "--session-clip-name",
        default=None,
        help="Clip name for /set_session_clip_notes",
    )
    parser.add_argument(
        "--append-session-clip-track-index",
        type=non_negative_int,
        default=None,
        help="Track index for /append_session_clip_notes",
    )
    parser.add_argument(
        "--append-session-clip-slot-index",
        type=non_negative_int,
        default=None,
        help="Clip slot index for /append_session_clip_notes",
    )
    parser.add_argument(
        "--append-session-clip-notes-json",
        default=None,
        help="JSON payload (string) for /append_session_clip_notes",
    )
    parser.add_argument(
        "--inspect-session-clip-track-index",
        type=non_negative_int,
        default=None,
        help="Track index for /inspect_session_clip_notes",
    )
    parser.add_argument(
        "--inspect-session-clip-slot-index",
        type=non_negative_int,
        default=None,
        help="Clip slot index for /inspect_session_clip_notes",
    )
    parser.add_argument(
        "--ensure-midi-tracks",
        type=non_negative_int,
        default=None,
        help="Target MIDI track count",
    )
    parser.add_argument(
        "--midi-cc",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /midi_cc <controller> <value> [channel]",
    )
    parser.add_argument(
        "--cc64",
        nargs="+",
        action="append",
        default=[],
        metavar="ARGS",
        help="Send /cc64 <value> [channel]",
    )
    parser.add_argument(
        "--delay-ms",
        type=non_negative_int,
        default=40,
        help="Delay between messages in milliseconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print messages without sending them",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Listen for ACK and observer events after sending commands; allows no send commands",
    )
    parser.add_argument(
        "--listen-timeout",
        type=non_negative_float,
        default=0.0,
        help="Listen timeout in seconds; 0 listens until interrupted or --listen-max-events",
    )
    parser.add_argument(
        "--listen-max-events",
        type=non_negative_int,
        default=0,
        help="Maximum events to print in --listen mode; 0 means unlimited",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable command timing summaries",
    )

    ns = parser.parse_args(list(argv))

    tempo: float | None = None if ns.no_tempo else ns.tempo
    if ns.no_signature or (ns.sig_num is None and ns.sig_den is None):
        sig_num: int | None = None
        sig_den: int | None = None
    else:
        sig_num = 4 if ns.sig_num is None else ns.sig_num
        sig_den = 4 if ns.sig_den is None else ns.sig_den
    try:
        auth_token = normalize_auth_token(ns.auth_token)
    except ValueError as exc:
        parser.error(str(exc))
    rename_track_index: int | None = ns.rename_track_index
    rename_track_name: str | None = (
        None if ns.rename_track_name is None else str(ns.rename_track_name)
    )

    if (rename_track_index is None) != (rename_track_name is None):
        parser.error("--rename-track-index and --rename-track-name must be provided together")

    session_clip_fields = [
        ns.session_clip_track_index,
        ns.session_clip_slot_index,
        ns.session_clip_length,
        ns.session_clip_notes_json,
    ]
    session_clip_any = any(field is not None for field in session_clip_fields)
    session_clip_all = all(field is not None for field in session_clip_fields)
    if session_clip_any and not session_clip_all:
        parser.error(
            "--session-clip-track-index, --session-clip-slot-index, "
            "--session-clip-length, and --session-clip-notes-json must be provided together"
        )

    session_clip_track_index: int | None = ns.session_clip_track_index
    session_clip_slot_index: int | None = ns.session_clip_slot_index
    session_clip_length: float | None = ns.session_clip_length
    session_clip_notes_json: str | None = (
        None if ns.session_clip_notes_json is None else str(ns.session_clip_notes_json)
    )
    session_clip_name: str | None = (
        None if ns.session_clip_name is None else str(ns.session_clip_name)
    )

    append_clip_track_index: int | None = ns.append_session_clip_track_index
    append_clip_slot_index: int | None = ns.append_session_clip_slot_index
    append_clip_notes_json: str | None = (
        None if ns.append_session_clip_notes_json is None else str(ns.append_session_clip_notes_json)
    )
    append_clip_fields = [append_clip_track_index, append_clip_slot_index, append_clip_notes_json]
    append_clip_any = any(field is not None for field in append_clip_fields)
    append_clip_all = all(field is not None for field in append_clip_fields)
    if append_clip_any and not append_clip_all:
        parser.error(
            "--append-session-clip-track-index, --append-session-clip-slot-index, "
            "and --append-session-clip-notes-json must be provided together"
        )

    inspect_track_index: int | None = ns.inspect_session_clip_track_index
    inspect_slot_index: int | None = ns.inspect_session_clip_slot_index
    if (inspect_track_index is None) != (inspect_slot_index is None):
        parser.error(
            "--inspect-session-clip-track-index and --inspect-session-clip-slot-index must be provided together"
        )

    def _optional_request_id(parts: Sequence[str], min_len: int) -> str | None:
        if len(parts) == min_len:
            return None
        return str(parts[-1])

    def _parse_api_get(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (2, 3):
                parser.error("--api-get expects: <path> <property> [request_id]")
            path, prop = str(parts[0]), str(parts[1])
            request_id = _optional_request_id(parts, 2)
            parsed.append((path, prop, request_id))
        return tuple(parsed)

    def _parse_api_set(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[str, str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (3, 4):
                parser.error("--api-set expects: <path> <property> <value_json> [request_id]")
            path, prop, value_json = str(parts[0]), str(parts[1]), str(parts[2])
            request_id = _optional_request_id(parts, 3)
            parsed.append((path, prop, value_json, request_id))
        return tuple(parsed)

    def _parse_api_call(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[str, str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (3, 4):
                parser.error("--api-call expects: <path> <method> <args_json> [request_id]")
            path, method, args_json = str(parts[0]), str(parts[1]), str(parts[2])
            request_id = _optional_request_id(parts, 3)
            parsed.append((path, method, args_json, request_id))
        return tuple(parsed)

    def _parse_api_children(
        entries: Sequence[Sequence[str]],
    ) -> Tuple[Tuple[str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (2, 3):
                parser.error("--api-children expects: <path> <child_name> [request_id]")
            path, child_name = str(parts[0]), str(parts[1])
            request_id = _optional_request_id(parts, 2)
            parsed.append((path, child_name, request_id))
        return tuple(parsed)

    def _parse_api_describe(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[str, str | None], ...]:
        parsed: List[Tuple[str, str | None]] = []
        for parts in entries:
            if len(parts) not in (1, 2):
                parser.error("--api-describe expects: <path> [request_id]")
            path = str(parts[0])
            request_id = _optional_request_id(parts, 1)
            parsed.append((path, request_id))
        return tuple(parsed)

    def _parse_api_observe(
        entries: Sequence[Sequence[str]],
    ) -> Tuple[Tuple[str, str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (3, 4):
                parser.error(
                    "--api-observe expects: <path> <property_or_child> <options_json> [request_id]"
                )
            path, property_name, options_json = str(parts[0]), str(parts[1]), str(parts[2])
            request_id = _optional_request_id(parts, 3)
            parsed.append((path, property_name, options_json, request_id))
        return tuple(parsed)

    def _parse_api_unobserve(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[str, str | None], ...]:
        parsed: List[Tuple[str, str | None]] = []
        for parts in entries:
            if len(parts) not in (1, 2):
                parser.error("--api-unobserve expects: <observer_id> [request_id]")
            observer_id = str(parts[0])
            request_id = _optional_request_id(parts, 1)
            parsed.append((observer_id, request_id))
        return tuple(parsed)

    def _parse_single_arg_optional_req(
        entries: Sequence[Sequence[str]],
        flag_name: str,
        arg_name: str,
    ) -> Tuple[Tuple[str, str | None], ...]:
        parsed: List[Tuple[str, str | None]] = []
        for parts in entries:
            if len(parts) not in (1, 2):
                parser.error(f"{flag_name} expects: <{arg_name}> [request_id]")
            parsed.append((str(parts[0]), _optional_request_id(parts, 1)))
        return tuple(parsed)

    def _parse_api_parameter_set(
        entries: Sequence[Sequence[str]],
    ) -> Tuple[Tuple[str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (2, 3):
                parser.error("--api-parameter-set expects: <parameter_path> <value_json> [request_id]")
            parsed.append((str(parts[0]), str(parts[1]), _optional_request_id(parts, 2)))
        return tuple(parsed)

    def _parse_api_insert_device(
        entries: Sequence[Sequence[str]],
    ) -> Tuple[Tuple[str, str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (3, 4):
                parser.error(
                    "--api-insert-device expects: <track_or_chain_path> <native_device_name> "
                    "<target_index_or_empty> [request_id]"
                )
            parsed.append((str(parts[0]), str(parts[1]), str(parts[2]), _optional_request_id(parts, 3)))
        return tuple(parsed)

    def _parse_api_insert_chain(
        entries: Sequence[Sequence[str]],
    ) -> Tuple[Tuple[str, str, str | None], ...]:
        parsed: List[Tuple[str, str, str | None]] = []
        for parts in entries:
            if len(parts) not in (2, 3):
                parser.error("--api-insert-chain expects: <rack_device_path> <target_index_or_empty> [request_id]")
            parsed.append((str(parts[0]), str(parts[1]), _optional_request_id(parts, 2)))
        return tuple(parsed)

    def _parse_api_drum_chain_in_note(
        entries: Sequence[Sequence[str]],
    ) -> Tuple[Tuple[str, int, str | None], ...]:
        parsed: List[Tuple[str, int, str | None]] = []
        for parts in entries:
            if len(parts) not in (2, 3):
                parser.error("--api-drum-chain-in-note expects: <drum_chain_path> <note|-1> [request_id]")
            try:
                note = int(parts[1])
            except (TypeError, ValueError):
                parser.error("--api-drum-chain-in-note note must be an integer between -1 and 127")
            if not -1 <= note <= 127:
                parser.error("--api-drum-chain-in-note note must be between -1 and 127")
            parsed.append((str(parts[0]), note, _optional_request_id(parts, 2)))
        return tuple(parsed)

    def _parse_api_session_clip_inspect(
        entries: Sequence[Sequence[str]],
    ) -> Tuple[Tuple[int, int, str], ...]:
        parsed: List[Tuple[int, int, str]] = []
        for parts in entries:
            try:
                track_index = non_negative_int(str(parts[0]))
                slot_index = non_negative_int(str(parts[1]))
            except (TypeError, ValueError, argparse.ArgumentTypeError):
                parser.error(
                    "--api-session-clip-inspect indexes must be non-negative integers"
                )
            request_id = str(parts[2])
            if not request_id:
                parser.error(
                    "--api-session-clip-inspect request_id must be non-empty"
                )
            if len(request_id.encode("utf-8")) > 128:
                parser.error(
                    "--api-session-clip-inspect request_id must be at most 128 UTF-8 bytes"
                )
            parsed.append((track_index, slot_index, request_id))
        return tuple(parsed)

    def _arrangement_request_id(value: object, option: str) -> str:
        request_id = str(value)
        if not request_id:
            parser.error(f"{option} request_id must be non-empty")
        if len(request_id.encode("utf-8")) > 128:
            parser.error(f"{option} request_id must be at most 128 UTF-8 bytes")
        return request_id

    def _arrangement_index(value: object, option: str) -> int:
        try:
            return non_negative_int(str(value))
        except (TypeError, ValueError, argparse.ArgumentTypeError):
            parser.error(f"{option} indexes must be non-negative integers")

    def _parse_api_arrangement_track_inspect(
        entries: Sequence[Sequence[str]],
    ) -> Tuple[Tuple[int, str], ...]:
        option = "--api-arrangement-track-inspect"
        return tuple(
            (
                _arrangement_index(parts[0], option),
                _arrangement_request_id(parts[1], option),
            )
            for parts in entries
        )

    def _parse_api_arrangement_clip_inspect(
        entries: Sequence[Sequence[str]],
        *,
        include_notes: bool,
    ) -> Tuple[Tuple[int, int, bool, str], ...]:
        option = (
            "--api-arrangement-clip-inspect-notes"
            if include_notes
            else "--api-arrangement-clip-inspect"
        )
        return tuple(
            (
                _arrangement_index(parts[0], option),
                _arrangement_index(parts[1], option),
                include_notes,
                _arrangement_request_id(parts[2], option),
            )
            for parts in entries
        )

    def _parse_midi_cc(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[int, int, int], ...]:
        parsed: List[Tuple[int, int, int]] = []
        for parts in entries:
            if len(parts) not in (2, 3):
                parser.error("--midi-cc expects: <controller> <value> [channel]")
            try:
                controller = midi_byte(str(parts[0]))
                value = midi_byte(str(parts[1]))
                channel = midi_channel(str(parts[2])) if len(parts) == 3 else 1
            except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
                parser.error(f"--midi-cc: {exc}")
            parsed.append((controller, value, channel))
        return tuple(parsed)

    def _parse_cc64(entries: Sequence[Sequence[str]]) -> Tuple[Tuple[int, int], ...]:
        parsed: List[Tuple[int, int]] = []
        for parts in entries:
            if len(parts) not in (1, 2):
                parser.error("--cc64 expects: <value> [channel]")
            try:
                value = midi_byte(str(parts[0]))
                channel = midi_channel(str(parts[1])) if len(parts) == 2 else 1
            except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
                parser.error(f"--cc64: {exc}")
            parsed.append((value, channel))
        return tuple(parsed)

    api_pings: Tuple[str | None, ...] = tuple(
        None if value in (None, "") else str(value) for value in ns.api_ping
    )
    api_gets = _parse_api_get(ns.api_get)
    api_sets = _parse_api_set(ns.api_set)
    api_calls = _parse_api_call(ns.api_call)
    api_children = _parse_api_children(ns.api_children)
    api_describes = _parse_api_describe(ns.api_describe)
    api_observes = _parse_api_observe(ns.api_observe)
    api_unobserves = _parse_api_unobserve(ns.api_unobserve)
    api_observers: Tuple[str | None, ...] = tuple(
        None if value in (None, "") else str(value) for value in ns.api_observers
    )
    api_clear_observers: Tuple[str | None, ...] = tuple(
        None if value in (None, "") else str(value) for value in ns.api_clear_observers
    )
    api_session_contexts: Tuple[str | None, ...] = tuple(
        None if value in (None, "") else str(value) for value in ns.api_session_context
    )
    api_theory_statuses: Tuple[str | None, ...] = tuple(
        None if value in (None, "") else str(value) for value in ns.api_theory_status
    )
    api_tuning_statuses: Tuple[str | None, ...] = tuple(
        None if value in (None, "") else str(value) for value in ns.api_tuning_status
    )
    api_device_lists = _parse_single_arg_optional_req(ns.api_device_list, "--api-device-list", "track_ref")
    api_device_parameters = _parse_single_arg_optional_req(
        ns.api_device_parameters, "--api-device-parameters", "device_path"
    )
    api_parameter_sets = _parse_api_parameter_set(ns.api_parameter_set)
    api_mixer_statuses = _parse_single_arg_optional_req(ns.api_mixer_status, "--api-mixer-status", "track_ref")
    api_insert_devices = _parse_api_insert_device(ns.api_insert_device)
    api_insert_chains = _parse_api_insert_chain(ns.api_insert_chain)
    api_drum_chain_in_notes = _parse_api_drum_chain_in_note(ns.api_drum_chain_in_note)
    api_session_clip_inspects = _parse_api_session_clip_inspect(
        ns.api_session_clip_inspect
    )
    api_arrangement_project_inspects = tuple(
        _arrangement_request_id(value, "--api-arrangement-project-inspect")
        for value in ns.api_arrangement_project_inspect
    )
    api_arrangement_track_inspects = _parse_api_arrangement_track_inspect(
        ns.api_arrangement_track_inspect
    )
    api_arrangement_clip_inspects = (
        _parse_api_arrangement_clip_inspect(
            ns.api_arrangement_clip_inspect, include_notes=False
        )
        + _parse_api_arrangement_clip_inspect(
            ns.api_arrangement_clip_inspect_notes, include_notes=True
        )
    )
    midi_ccs = _parse_midi_cc(ns.midi_cc)
    cc64s = _parse_cc64(ns.cc64)
    expect_ack = bool(ns.ack or ns.listen)

    return BridgeConfig(
        host=ns.host,
        port=ns.port,
        ack_port=ns.ack_port,
        ack_timeout_s=ns.ack_timeout,
        expect_ack=expect_ack,
        ping_first=expect_ack and not ns.no_ping_first and not ns.listen,
        status=bool(ns.status),
        tempo=tempo,
        sig_num=sig_num,
        sig_den=sig_den,
        create_midi_tracks=ns.create_midi_tracks,
        add_midi_tracks=ns.add_midi_tracks,
        midi_name=str(ns.midi_name),
        create_audio_tracks=ns.create_audio_tracks,
        add_audio_tracks=ns.add_audio_tracks,
        audio_prefix=str(ns.audio_prefix),
        delete_audio_tracks=ns.delete_audio_tracks,
        delete_midi_tracks=ns.delete_midi_tracks,
        rename_track_index=rename_track_index,
        rename_track_name=rename_track_name,
        session_clip_track_index=session_clip_track_index,
        session_clip_slot_index=session_clip_slot_index,
        session_clip_length=session_clip_length,
        session_clip_notes_json=session_clip_notes_json,
        session_clip_name=session_clip_name,
        append_session_clip_track_index=append_clip_track_index,
        append_session_clip_slot_index=append_clip_slot_index,
        append_session_clip_notes_json=append_clip_notes_json,
        inspect_session_clip_track_index=inspect_track_index,
        inspect_session_clip_slot_index=inspect_slot_index,
        ensure_midi_tracks=ns.ensure_midi_tracks,
        midi_ccs=midi_ccs,
        cc64s=cc64s,
        api_pings=api_pings,
        api_gets=api_gets,
        api_sets=api_sets,
        api_calls=api_calls,
        api_children=api_children,
        api_describes=api_describes,
        auth_token=auth_token,
        api_observes=api_observes,
        api_unobserves=api_unobserves,
        api_observers=api_observers,
        api_clear_observers=api_clear_observers,
        api_session_contexts=api_session_contexts,
        api_theory_statuses=api_theory_statuses,
        api_tuning_statuses=api_tuning_statuses,
        api_device_lists=api_device_lists,
        api_device_parameters=api_device_parameters,
        api_parameter_sets=api_parameter_sets,
        api_mixer_statuses=api_mixer_statuses,
        api_insert_devices=api_insert_devices,
        api_insert_chains=api_insert_chains,
        api_drum_chain_in_notes=api_drum_chain_in_notes,
        api_session_clip_inspects=api_session_clip_inspects,
        api_arrangement_project_inspects=api_arrangement_project_inspects,
        api_arrangement_track_inspects=api_arrangement_track_inspects,
        api_arrangement_clip_inspects=api_arrangement_clip_inspects,
        ack_mode=str(ns.ack_mode),
        ack_flush_interval=int(ns.ack_flush_interval),
        listen=bool(ns.listen),
        listen_timeout_s=float(ns.listen_timeout),
        listen_max_events=int(ns.listen_max_events),
        report_metrics=not bool(ns.no_metrics),
        delay_ms=ns.delay_ms,
        dry_run=ns.dry_run,
    )


def _pad4(length: int) -> int:
    remainder = length % 4
    return 0 if remainder == 0 else 4 - remainder


def _encode_osc_string(value: str) -> bytes:
    if "\x00" in value:
        raise ValueError("OSC strings cannot contain embedded NUL bytes")
    raw = value.encode("utf-8") + b"\x00"
    raw += b"\x00" * _pad4(len(raw))
    return raw


def _decode_osc_string(data: bytes, start: int) -> Tuple[str, int]:
    if start < 0 or start >= len(data):
        raise ValueError("OSC string is truncated")
    end = data.find(b"\x00", start)
    if end == -1:
        raise ValueError("OSC string is missing its NUL terminator")
    try:
        text = data[start:end].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("OSC string is not valid UTF-8") from exc
    idx = end + 1
    padded_idx = idx + _pad4(idx)
    if padded_idx > len(data):
        raise ValueError("OSC string padding is truncated")
    if any(byte != 0 for byte in data[idx:padded_idx]):
        raise ValueError("OSC string padding must be zero")
    idx = padded_idx
    return text, idx


def encode_osc_message(address: str, args: Sequence[OscArg]) -> bytes:
    if not address.startswith("/"):
        raise ValueError(f"OSC address must start with '/': {address}")

    type_tags: List[str] = []
    payload = bytearray()

    for arg in args:
        if isinstance(arg, bool):
            type_tags.append("i")
            payload.extend(struct.pack(">i", int(arg)))
        elif isinstance(arg, int):
            type_tags.append("i")
            payload.extend(struct.pack(">i", arg))
        elif isinstance(arg, float):
            type_tags.append("f")
            payload.extend(struct.pack(">f", arg))
        elif isinstance(arg, str):
            type_tags.append("s")
            payload.extend(_encode_osc_string(arg))
        else:
            raise TypeError(f"Unsupported OSC argument type: {type(arg)}")

    type_tag_string = "," + "".join(type_tags)
    return _encode_osc_string(address) + _encode_osc_string(type_tag_string) + payload


def decode_osc_message(data: bytes) -> Tuple[str, List[OscArg]]:
    if data.startswith(b"#bundle"):
        raise ValueError("OSC bundles are not supported by this minimal decoder")

    address, idx = _decode_osc_string(data, 0)
    if not address.startswith("/"):
        raise ValueError(f"OSC address must start with '/': {address}")
    type_tags, idx = _decode_osc_string(data, idx)

    if not type_tags.startswith(","):
        raise ValueError(f"OSC type tags must start with ',': {type_tags}")

    args: List[OscArg] = []
    for tag in type_tags[1:]:
        if tag == "i":
            if idx + 4 > len(data):
                raise ValueError("OSC int argument truncated")
            value = struct.unpack(">i", data[idx : idx + 4])[0]
            idx += 4
            args.append(value)
        elif tag == "f":
            if idx + 4 > len(data):
                raise ValueError("OSC float argument truncated")
            value = struct.unpack(">f", data[idx : idx + 4])[0]
            idx += 4
            args.append(value)
        elif tag == "s":
            value, idx = _decode_osc_string(data, idx)
            args.append(value)
        else:
            raise ValueError(f"Unsupported OSC type tag: {tag}")

    if idx != len(data):
        raise ValueError("OSC message has trailing bytes")
    return address, args


def format_arg(value: OscArg) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def describe_command(cmd: OscCommand) -> str:
    if not cmd.args:
        return cmd.address
    displayed_args = list(cmd.args)
    if cmd.address in PROTECTED_OSC_ADDRESSES:
        displayed_args[0] = "<redacted-auth-token>"
    return cmd.address + " " + " ".join(format_arg(arg) for arg in displayed_args)


def _try_parse_json(value: OscArg) -> object | None:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, OverflowError, RecursionError):
        return None


def _short_repr(value: object, max_len: int = 120) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, separators=(",", ":"))
    else:
        text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _optional_request_id(args: Sequence[OscArg], index: int) -> str | None:
    if len(args) <= index or args[index] in (None, ""):
        return None
    return str(args[index])


def _parse_error_ack(args: Sequence[OscArg]) -> tuple[str | None, list[OscArg]]:
    if len(args) < 3:
        return None, []
    marker = args[-2] if len(args) >= 4 else None
    correlation = args[-1]
    if (
        marker == "request_correlation"
        and isinstance(correlation, str)
        and correlation.startswith("req:")
    ):
        request_id = correlation[4:] or None
        return request_id, list(args[2:-2])
    return None, list(args[2:])


def parse_ack_event(address: str, args: Sequence[OscArg]) -> AckEvent:
    event = str(args[0]) if args else None
    request_id: str | None = None
    payload: dict[str, object] = {"args": list(args)}
    is_error = event == "error"

    if event == "pong":
        request_id = _optional_request_id(args, 1)
    elif event == "midi_cc" and len(args) >= 4:
        request_id = _optional_request_id(args, 4)
        payload = {"controller": args[1], "value": args[2], "channel": args[3]}
    elif event == "cc64" and len(args) >= 3:
        request_id = _optional_request_id(args, 3)
        payload = {"value": args[1], "channel": args[2]}
    elif event == "api_get" and len(args) >= 4:
        request_id = _optional_request_id(args, 4)
        payload = {
            "path": args[1],
            "property": args[2],
            "value": _try_parse_json(args[3]),
        }
    elif event == "api_set" and len(args) >= 4:
        request_id = _optional_request_id(args, 4)
        payload = {
            "path": args[1],
            "property": args[2],
            "result": _try_parse_json(args[3]),
        }
    elif event == "api_call" and len(args) >= 4:
        request_id = _optional_request_id(args, 4)
        payload = {
            "path": args[1],
            "method": args[2],
            "result": _try_parse_json(args[3]),
        }
    elif event == "api_children" and len(args) >= 4:
        request_id = _optional_request_id(args, 4)
        payload = {
            "path": args[1],
            "child_name": args[2],
            "children": _try_parse_json(args[3]),
        }
    elif event == "api_describe" and len(args) >= 3:
        request_id = _optional_request_id(args, 3)
        payload = {"path": args[1], "description": _try_parse_json(args[2])}
    elif event == "api_observe" and len(args) >= 5:
        request_id = _optional_request_id(args, 5)
        payload = {
            "observer_id": args[1],
            "path": args[2],
            "property": args[3],
            "snapshot": _try_parse_json(args[4]),
        }
    elif event == "api_unobserve" and len(args) >= 3:
        request_id = _optional_request_id(args, 3)
        payload = {"observer_id": args[1], "result": _try_parse_json(args[2])}
    elif event == "api_observers" and len(args) >= 2:
        request_id = _optional_request_id(args, 2)
        payload = {"observers": _try_parse_json(args[1])}
    elif event == "api_clear_observers" and len(args) >= 2:
        request_id = _optional_request_id(args, 2)
        payload = {"result": _try_parse_json(args[1])}
    elif event == "api_session_context" and len(args) >= 2:
        request_id = _optional_request_id(args, 2)
        payload = {"context": _try_parse_json(args[1])}
    elif event == "api_theory_status" and len(args) >= 2:
        request_id = _optional_request_id(args, 2)
        payload = {"status": _try_parse_json(args[1])}
    elif event == "api_tuning_status" and len(args) >= 2:
        request_id = _optional_request_id(args, 2)
        payload = {"status": _try_parse_json(args[1])}
    elif event == "api_device_list" and len(args) >= 3:
        request_id = _optional_request_id(args, 3)
        payload = {"target": args[1], "devices": _try_parse_json(args[2])}
    elif event == "api_device_parameters" and len(args) >= 3:
        request_id = _optional_request_id(args, 3)
        payload = {"device_path": args[1], "parameters": _try_parse_json(args[2])}
    elif event == "api_parameter_set" and len(args) >= 3:
        request_id = _optional_request_id(args, 3)
        payload = {"parameter_path": args[1], "parameter": _try_parse_json(args[2])}
    elif event == "api_mixer_status" and len(args) >= 3:
        request_id = _optional_request_id(args, 3)
        payload = {"track_path": args[1], "mixer": _try_parse_json(args[2])}
    elif event == "api_insert_device" and len(args) >= 4:
        request_id = _optional_request_id(args, 4)
        payload = {
            "target_path": args[1],
            "device_name": args[2],
            "result": _try_parse_json(args[3]),
        }
    elif event == "api_insert_chain" and len(args) >= 3:
        request_id = _optional_request_id(args, 3)
        payload = {"rack_path": args[1], "result": _try_parse_json(args[2])}
    elif event == "api_drum_chain_in_note" and len(args) >= 3:
        request_id = _optional_request_id(args, 3)
        payload = {"chain_path": args[1], "chain": _try_parse_json(args[2])}
    elif (
        event == "api_session_clip_inspect" or event in ARRANGEMENT_INSPECTION_EVENTS
    ) and len(args) >= 3:
        request_id = _optional_request_id(args, 2)
        payload = {"fragment": _try_parse_json(args[1])}
    elif event == "api_event" and len(args) >= 3:
        event_payload = _try_parse_json(args[2])
        payload = {"observer_id": args[1], "event_payload": event_payload}
        if isinstance(event_payload, dict):
            payload.update(
                {
                    "path": event_payload.get("current_path")
                    or event_payload.get("requested_path"),
                    "property": event_payload.get("property"),
                    "value": event_payload.get("value"),
                    "event_count": event_payload.get("event_count"),
                    "dropped_events": event_payload.get("dropped_events"),
                    "timestamp_ms": event_payload.get("timestamp_ms"),
                }
            )
    elif event == "api_observe_event" and len(args) >= 5:
        event_payload = _try_parse_json(args[4])
        payload = {
            "observer_id": args[1],
            "path": args[2],
            "property": args[3],
            "event_payload": event_payload,
        }
        if len(args) >= 6:
            payload["event_count"] = args[5]
        if len(args) >= 7:
            payload["timestamp_ms"] = args[6]
    elif event == "error" and len(args) >= 2:
        request_id, details = _parse_error_ack(args)
        payload = {"code": args[1], "details": details}

    return AckEvent(
        address=address,
        event=event,
        request_id=request_id,
        payload=payload,
        is_error=is_error,
    )


def _rpc_ack_summary(args: Sequence[OscArg]) -> str | None:
    if not args:
        return None
    event = str(args[0])

    def _req_suffix(request_id: OscArg | None) -> str:
        return "" if request_id in (None, "") else f" req={request_id}"

    if event == "midi_cc" and len(args) >= 4:
        controller, value, channel = args[1], args[2], args[3]
        request_id = args[4] if len(args) >= 5 else None
        return (
            f"midi_cc ctrl={controller} value={value} ch={channel}"
            f"{_req_suffix(request_id)}"
        )

    if event == "cc64" and len(args) >= 3:
        value, channel = args[1], args[2]
        request_id = args[3] if len(args) >= 4 else None
        return f"cc64 value={value} ch={channel}{_req_suffix(request_id)}"

    if event == "api_get" and len(args) >= 4:
        path, prop, value = args[1], args[2], args[3]
        request_id = args[4] if len(args) >= 5 else None
        parsed = _try_parse_json(value)
        value_text = _short_repr(parsed if parsed is not None else value)
        return f"api_get {path} {prop} -> {value_text}{_req_suffix(request_id)}"

    if event == "api_set" and len(args) >= 4:
        path, prop, result = args[1], args[2], args[3]
        request_id = args[4] if len(args) >= 5 else None
        parsed = _try_parse_json(result)
        result_text = _short_repr(parsed if parsed is not None else result)
        return f"api_set {path} {prop} -> {result_text}{_req_suffix(request_id)}"

    if event == "api_call" and len(args) >= 4:
        path, method, result = args[1], args[2], args[3]
        request_id = args[4] if len(args) >= 5 else None
        parsed = _try_parse_json(result)
        result_text = _short_repr(parsed if parsed is not None else result)
        return f"api_call {path} {method} -> {result_text}{_req_suffix(request_id)}"

    if event == "api_children" and len(args) >= 4:
        path, child_name, children_json = args[1], args[2], args[3]
        request_id = args[4] if len(args) >= 5 else None
        parsed = _try_parse_json(children_json)
        count = len(parsed) if isinstance(parsed, list) else "?"
        preview = ""
        if isinstance(parsed, list) and parsed:
            names = [
                str(item.get("name", item.get("path")))
                if isinstance(item, dict) else "<malformed child>"
                for item in parsed[:3]
            ]
            preview = f" first={names}"
        return (
            f"api_children {path} {child_name} count={count}{preview}"
            f"{_req_suffix(request_id)}"
        )

    if event == "api_describe" and len(args) >= 3:
        path, describe_json = args[1], args[2]
        request_id = args[3] if len(args) >= 4 else None
        parsed = _try_parse_json(describe_json)
        if isinstance(parsed, dict):
            core = {
                "id": parsed.get("id"),
                "name": parsed.get("name"),
                "type": parsed.get("type"),
            }
            core = {k: v for k, v in core.items() if v not in (None, "")}
            core_text = _short_repr(core) if core else _short_repr(parsed)
        else:
            core_text = _short_repr(describe_json)
        return f"api_describe {path} -> {core_text}{_req_suffix(request_id)}"

    if event == "api_observe" and len(args) >= 5:
        observer_id, path, property_name, observe_json = args[1], args[2], args[3], args[4]
        request_id = args[5] if len(args) >= 6 else None
        parsed = _try_parse_json(observe_json)
        snapshot_text = _short_repr(parsed if parsed is not None else observe_json)
        return (
            f"api_observe {observer_id} {path} {property_name} -> {snapshot_text}"
            f"{_req_suffix(request_id)}"
        )

    if event == "api_unobserve" and len(args) >= 3:
        observer_id, result_json = args[1], args[2]
        request_id = args[3] if len(args) >= 4 else None
        parsed = _try_parse_json(result_json)
        result_text = _short_repr(parsed if parsed is not None else result_json)
        return f"api_unobserve {observer_id} -> {result_text}{_req_suffix(request_id)}"

    if event == "api_observers" and len(args) >= 2:
        observers_json = args[1]
        request_id = args[2] if len(args) >= 3 else None
        parsed = _try_parse_json(observers_json)
        count = len(parsed) if isinstance(parsed, list) else "?"
        return f"api_observers count={count}{_req_suffix(request_id)}"

    if event == "api_clear_observers" and len(args) >= 2:
        result_json = args[1]
        request_id = args[2] if len(args) >= 3 else None
        parsed = _try_parse_json(result_json)
        result_text = _short_repr(parsed if parsed is not None else result_json)
        return f"api_clear_observers -> {result_text}{_req_suffix(request_id)}"

    if event == "api_session_context" and len(args) >= 2:
        request_id = args[2] if len(args) >= 3 else None
        parsed = _try_parse_json(args[1])
        counts = parsed.get("counts") if isinstance(parsed, dict) else None
        return f"api_session_context -> {_short_repr(counts if counts else parsed)}{_req_suffix(request_id)}"

    if event == "api_theory_status" and len(args) >= 2:
        request_id = args[2] if len(args) >= 3 else None
        parsed = _try_parse_json(args[1])
        theory = parsed.get("theory") if isinstance(parsed, dict) else None
        return f"api_theory_status -> {_short_repr(theory if theory else parsed)}{_req_suffix(request_id)}"

    if event == "api_tuning_status" and len(args) >= 2:
        request_id = args[2] if len(args) >= 3 else None
        parsed = _try_parse_json(args[1])
        tuning = parsed.get("tuning") if isinstance(parsed, dict) else None
        return f"api_tuning_status -> {_short_repr(tuning if tuning else parsed)}{_req_suffix(request_id)}"

    if event == "api_session_clip_inspect" and len(args) >= 3:
        request_id = args[2]
        parsed = _try_parse_json(args[1])
        if not isinstance(parsed, dict):
            return f"api_session_clip_inspect malformed{_req_suffix(request_id)}"
        transfer = parsed.get("transfer")
        data = parsed.get("data")
        if not isinstance(transfer, dict) or not isinstance(data, dict):
            return f"api_session_clip_inspect malformed{_req_suffix(request_id)}"
        fragment_index = transfer.get("fragment_index")
        fragment_count = transfer.get("fragment_count")
        kind = transfer.get("fragment_kind", "?")
        detail = ""
        if kind == "complete":
            detail = (
                f" devices={data.get('device_count', '?')}"
                f" notes={data.get('note_count', '?')}"
            )
        elif kind == "context":
            summary = data.get("summary")
            note_count = summary.get("note_count", "?") if isinstance(summary, dict) else "?"
            detail = f" notes={note_count}"
        elif kind == "device_page":
            detail = (
                f" devices={data.get('device_offset', '?')}+"
                f"{data.get('device_count', '?')}/{data.get('device_total', '?')}"
            )
        elif kind == "note_page":
            detail = (
                f" notes={data.get('note_offset', '?')}+"
                f"{data.get('note_count', '?')}/{data.get('note_total', '?')}"
            )
        index_text = (
            int(fragment_index) + 1
            if isinstance(fragment_index, int) and not isinstance(fragment_index, bool)
            else "?"
        )
        return (
            f"api_session_clip_inspect {kind} fragment={index_text}/{fragment_count}"
            f"{detail}{_req_suffix(request_id)}"
        )

    if event in ARRANGEMENT_INSPECTION_EVENTS and len(args) >= 3:
        request_id = args[2]
        parsed = _try_parse_json(args[1])
        transfer = parsed.get("transfer") if isinstance(parsed, dict) else None
        correlation = parsed.get("correlation") if isinstance(parsed, dict) else None
        if not isinstance(transfer, dict) or not isinstance(correlation, dict):
            return f"{event} malformed{_req_suffix(request_id)}"
        index = transfer.get("fragment_index")
        index_text = (
            int(index) + 1
            if isinstance(index, int) and not isinstance(index, bool)
            else "?"
        )
        return (
            f"{event} scope={correlation.get('scope', '?')} "
            f"fragment={index_text}/{transfer.get('fragment_count', '?')}"
            f"{_req_suffix(request_id)}"
        )

    if event in {
        "api_device_list",
        "api_device_parameters",
        "api_parameter_set",
        "api_mixer_status",
        "api_insert_device",
        "api_insert_chain",
        "api_drum_chain_in_note",
    }:
        request_id = None
        detail_args = list(args[1:])
        if event == "api_insert_device":
            if len(args) >= 5:
                request_id = args[4]
                detail_args = list(args[1:4])
        elif len(args) >= 4:
            request_id = args[3]
            detail_args = list(args[1:3])
        details = " ".join(str(a) for a in detail_args)
        return f"{event} {details}{_req_suffix(request_id)}"

    if event == "api_event" and len(args) >= 3:
        observer_id = args[1]
        payload_json = args[2]
        parsed = _try_parse_json(payload_json)
        path = "?"
        property_name = "?"
        event_index = None
        event_ms = None
        if isinstance(parsed, dict):
            path = parsed.get("current_path") or parsed.get("requested_path") or "?"
            property_name = parsed.get("property") or "?"
            event_index = parsed.get("event_count")
            event_ms = parsed.get("timestamp_ms")
        parsed = _try_parse_json(payload_json)
        payload_text = _short_repr(parsed if parsed is not None else payload_json)
        suffix = ""
        if event_index is not None:
            suffix += f" event={event_index}"
        if event_ms is not None:
            suffix += f" at={event_ms}"
        return f"api_event {observer_id} {path} {property_name} -> {payload_text}{suffix}"

    if event == "api_observe_event" and len(args) >= 5:
        observer_id = args[1]
        path = args[2]
        property_name = args[3]
        payload_json = args[4]
        parsed = _try_parse_json(payload_json)
        event_index = args[5] if len(args) >= 6 else None
        event_ms = args[6] if len(args) >= 7 else None
        payload_text = _short_repr(parsed if parsed is not None else payload_json)
        suffix = ""
        if event_index is not None:
            suffix += f" event={event_index}"
        if event_ms is not None:
            suffix += f" at={event_ms}"
        return f"api_observe_event {observer_id} {path} {property_name} -> {payload_text}{suffix}"

    if event == "error" and len(args) >= 2 and str(args[1]).startswith("api_"):
        request_id, details = _parse_error_ack(args)
        detail = " ".join(str(a) for a in [args[1], *details])
        return f"api_error {detail}{_req_suffix(request_id)}"

    return None


def summarize_ack(address: str, args: Sequence[OscArg]) -> List[str]:
    if address == "/ack":
        summary = _rpc_ack_summary(args)
        if summary:
            if args and (
                args[0] == "api_session_clip_inspect"
                or args[0] in ARRANGEMENT_INSPECTION_EVENTS
            ):
                return [f"ack:  {summary}"]
            suffix = "" if not args else " " + " ".join(format_arg(a) for a in args)
            lines = [f"ack:  {address}{suffix}"]
            lines.append(f"ack:  {summary}")
            return lines

    suffix = "" if not args else " " + " ".join(format_arg(a) for a in args)
    return [f"ack:  {address}{suffix}"]


def build_commands(cfg: BridgeConfig) -> List[OscCommand]:
    commands: List[OscCommand] = []

    bounded_track_counts = {
        "add_midi_tracks": cfg.add_midi_tracks,
        "add_audio_tracks": cfg.add_audio_tracks,
        "delete_midi_tracks": cfg.delete_midi_tracks,
        "delete_audio_tracks": cfg.delete_audio_tracks,
    }
    for name, count in bounded_track_counts.items():
        if count > MAX_TRACKS_PER_COMMAND:
            raise ValueError(
                f"{name} may not exceed {MAX_TRACKS_PER_COMMAND} per command"
            )
    if (
        cfg.ensure_midi_tracks is not None
        and cfg.ensure_midi_tracks > MAX_TRACK_TARGET
    ):
        raise ValueError(
            f"ensure_midi_tracks target may not exceed {MAX_TRACK_TARGET}"
        )

    if cfg.ping_first:
        commands.append(OscCommand("/ping"))

    def _with_request_id(args: List[OscArg], request_id: str | None) -> Tuple[OscArg, ...]:
        if request_id is None:
            return tuple(args)
        return tuple(args + [request_id])

    def _protected(
        args: Sequence[OscArg],
        request_id: str | None = None,
    ) -> Tuple[OscArg, ...]:
        return authenticated_args(
            cfg.auth_token,
            _with_request_id(list(args), request_id),
        )

    # Additive LiveAPI RPC preflight surface.
    for request_id in cfg.api_pings:
        commands.append(OscCommand("/api/ping", _with_request_id([], request_id)))
    for path, prop, request_id in cfg.api_gets:
        commands.append(OscCommand("/api/get", _with_request_id([path, prop], request_id)))
    for path, prop, value_json, request_id in cfg.api_sets:
        commands.append(
            OscCommand("/api/set", _protected([path, prop, value_json], request_id))
        )
    for path, method, args_json, request_id in cfg.api_calls:
        commands.append(
            OscCommand("/api/call", _protected([path, method, args_json], request_id))
        )
    for path, child_name, request_id in cfg.api_children:
        commands.append(
            OscCommand("/api/children", _with_request_id([path, child_name], request_id))
        )
    for path, request_id in cfg.api_describes:
        commands.append(OscCommand("/api/describe", _with_request_id([path], request_id)))
    for path, property_name, options_json, request_id in cfg.api_observes:
        commands.append(
            OscCommand(
                "/api_observe",
                _protected([path, property_name, options_json], request_id),
            )
        )
    for observer_id, request_id in cfg.api_unobserves:
        commands.append(
            OscCommand("/api_unobserve", _protected([observer_id], request_id))
        )
    for request_id in cfg.api_observers:
        commands.append(OscCommand("/api_observers", _with_request_id([], request_id)))
    for request_id in cfg.api_clear_observers:
        commands.append(OscCommand("/api_clear_observers", _protected([], request_id)))
    for request_id in cfg.api_session_contexts:
        commands.append(OscCommand("/api/session_context", _with_request_id([], request_id)))
    for request_id in cfg.api_theory_statuses:
        commands.append(OscCommand("/api/theory_status", _with_request_id([], request_id)))
    for request_id in cfg.api_tuning_statuses:
        commands.append(OscCommand("/api/tuning_status", _with_request_id([], request_id)))
    for track_ref, request_id in cfg.api_device_lists:
        commands.append(OscCommand("/api/device_list", _with_request_id([track_ref], request_id)))
    for device_path, request_id in cfg.api_device_parameters:
        commands.append(OscCommand("/api/device_parameters", _with_request_id([device_path], request_id)))
    for parameter_path, value_json, request_id in cfg.api_parameter_sets:
        commands.append(
            OscCommand(
                "/api/parameter_set",
                _protected([parameter_path, value_json], request_id),
            )
        )
    for track_ref, request_id in cfg.api_mixer_statuses:
        commands.append(OscCommand("/api/mixer_status", _with_request_id([track_ref], request_id)))
    for target_path, device_name, target_index, request_id in cfg.api_insert_devices:
        commands.append(
            OscCommand(
                "/api/insert_device",
                _protected([target_path, device_name, target_index], request_id),
            )
        )
    for rack_path, target_index, request_id in cfg.api_insert_chains:
        commands.append(
            OscCommand(
                "/api/insert_chain",
                _protected([rack_path, target_index], request_id),
            )
        )
    for chain_path, note, request_id in cfg.api_drum_chain_in_notes:
        commands.append(
            OscCommand(
                "/api/drum_chain_in_note",
                _protected([chain_path, note], request_id),
            )
        )
    for track_index, slot_index, request_id in cfg.api_session_clip_inspects:
        commands.append(
            OscCommand(
                "/api/session_clip_inspect",
                (track_index, slot_index, 1, request_id),
            )
        )
    for request_id in cfg.api_arrangement_project_inspects:
        commands.append(
            OscCommand("/api/arrangement_project_inspect", (1, request_id))
        )
    for track_index, request_id in cfg.api_arrangement_track_inspects:
        commands.append(
            OscCommand("/api/arrangement_track_inspect", (track_index, 1, request_id))
        )
    for track_index, clip_index, include_notes, request_id in cfg.api_arrangement_clip_inspects:
        commands.append(
            OscCommand(
                "/api/arrangement_clip_inspect",
                (track_index, clip_index, int(include_notes), 1, request_id),
            )
        )

    if cfg.status:
        commands.append(OscCommand("/status"))

    if cfg.delete_audio_tracks > 0:
        commands.append(
            OscCommand(
                "/delete_audio_tracks",
                authenticated_args(cfg.auth_token, (cfg.delete_audio_tracks,)),
            )
        )

    if cfg.delete_midi_tracks > 0:
        commands.append(
            OscCommand(
                "/delete_midi_tracks",
                authenticated_args(cfg.auth_token, (cfg.delete_midi_tracks,)),
            )
        )

    if cfg.tempo is not None:
        commands.append(
            OscCommand("/tempo", authenticated_args(cfg.auth_token, (cfg.tempo,)))
        )

    if cfg.sig_num is not None:
        commands.append(
            OscCommand("/sig_num", authenticated_args(cfg.auth_token, (cfg.sig_num,)))
        )

    if cfg.sig_den is not None:
        commands.append(
            OscCommand("/sig_den", authenticated_args(cfg.auth_token, (cfg.sig_den,)))
        )

    for _ in range(cfg.create_midi_tracks):
        commands.append(
            OscCommand("/create_midi_track", authenticated_args(cfg.auth_token))
        )

    if cfg.add_midi_tracks > 0:
        commands.append(
            OscCommand(
                "/add_midi_tracks",
                authenticated_args(
                    cfg.auth_token,
                    (cfg.add_midi_tracks, cfg.midi_name),
                ),
            )
        )

    for _ in range(cfg.create_audio_tracks):
        commands.append(
            OscCommand("/create_audio_track", authenticated_args(cfg.auth_token))
        )

    if cfg.add_audio_tracks > 0:
        commands.append(
            OscCommand(
                "/add_audio_tracks",
                authenticated_args(
                    cfg.auth_token,
                    (cfg.add_audio_tracks, cfg.audio_prefix),
                ),
            )
        )

    if (
        cfg.session_clip_track_index is not None
        and cfg.session_clip_slot_index is not None
        and cfg.session_clip_length is not None
        and cfg.session_clip_notes_json is not None
    ):
        clip_name = "" if cfg.session_clip_name is None else cfg.session_clip_name
        commands.append(
            OscCommand(
                "/set_session_clip_notes",
                authenticated_args(
                    cfg.auth_token,
                    (
                        cfg.session_clip_track_index,
                        cfg.session_clip_slot_index,
                        cfg.session_clip_length,
                        cfg.session_clip_notes_json,
                        clip_name,
                    ),
                ),
            )
        )

    if (
        cfg.append_session_clip_track_index is not None
        and cfg.append_session_clip_slot_index is not None
        and cfg.append_session_clip_notes_json is not None
    ):
        commands.append(
            OscCommand(
                "/append_session_clip_notes",
                authenticated_args(
                    cfg.auth_token,
                    (
                        cfg.append_session_clip_track_index,
                        cfg.append_session_clip_slot_index,
                        cfg.append_session_clip_notes_json,
                    ),
                ),
            )
        )

    if (
        cfg.inspect_session_clip_track_index is not None
        and cfg.inspect_session_clip_slot_index is not None
    ):
        commands.append(
            OscCommand(
                "/inspect_session_clip_notes",
                (cfg.inspect_session_clip_track_index, cfg.inspect_session_clip_slot_index),
            )
        )

    if cfg.rename_track_index is not None and cfg.rename_track_name is not None:
        commands.append(
            OscCommand(
                "/rename_track",
                authenticated_args(
                    cfg.auth_token,
                    (cfg.rename_track_index, cfg.rename_track_name),
                ),
            )
        )

    if cfg.ensure_midi_tracks is not None:
        commands.append(
            OscCommand(
                "/ensure_midi_tracks",
                authenticated_args(cfg.auth_token, (cfg.ensure_midi_tracks,)),
            )
        )

    for controller, value, channel in cfg.midi_ccs:
        commands.append(
            OscCommand(
                "/midi_cc",
                authenticated_args(cfg.auth_token, (controller, value, channel)),
            )
        )

    for value, channel in cfg.cc64s:
        commands.append(
            OscCommand(
                "/cc64",
                authenticated_args(cfg.auth_token, (value, channel)),
            )
        )

    return commands


def open_ack_socket(cfg: BridgeConfig) -> socket.socket | None:
    if not cfg.expect_ack or cfg.dry_run:
        return None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((cfg.host, cfg.ack_port))
    except OSError as exc:
        print(
            f"warning: could not bind ack socket on {cfg.host}:{cfg.ack_port}: {exc}",
            file=sys.stderr,
        )
        sock.close()
        return None

    sock.setblocking(False)
    return sock


def _finite_wait_seconds(value: float, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _bounded_packet_diagnostic(exc: Exception, packet: bytes) -> str:
    sample = repr(packet[:64])
    suffix = "" if len(packet) <= 64 else f"... ({len(packet)} bytes total)"
    diagnostic = f"{exc}: {sample}{suffix}"
    if len(diagnostic) <= ACK_MAX_DIAGNOSTIC_CHARS:
        return diagnostic
    return diagnostic[: ACK_MAX_DIAGNOSTIC_CHARS - 3] + "..."


def _sender_matches(
    sender: object,
    expected_sender_host: str | None,
) -> bool:
    if expected_sender_host is None:
        return True
    if expected_sender_host.strip().lower() == "localhost":
        expected_sender_host = DEFAULT_HOST
    return (
        isinstance(sender, tuple)
        and len(sender) >= 1
        and sender[0] == expected_sender_host
    )


_INVALID_ACK_JSON = object()
_ACK_JSON_FIELDS = {
    "api_get": (3, None),
    "api_set": (3, dict),
    "api_call": (3, None),
    "api_children": (3, list),
    "api_describe": (2, dict),
    "api_observe": (4, dict),
    "api_unobserve": (2, dict),
    "api_observers": (1, list),
    "api_clear_observers": (1, dict),
    "api_session_context": (1, dict),
    "api_theory_status": (1, dict),
    "api_tuning_status": (1, dict),
    "api_device_list": (2, dict),
    "api_device_parameters": (2, dict),
    "api_parameter_set": (2, dict),
    "api_mixer_status": (2, dict),
    "api_insert_device": (3, dict),
    "api_insert_chain": (2, dict),
    "api_drum_chain_in_note": (2, dict),
}


def _ack_json(value: object) -> object:
    if not isinstance(value, str):
        return _INVALID_ACK_JSON

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite JSON number")
        return number

    try:
        return json.loads(value, parse_constant=reject_constant, parse_float=finite_float)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return _INVALID_ACK_JSON


def _ack_number(value: object, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and (minimum is None or value >= minimum)
    except (OverflowError, ValueError):
        return False


def _ack_integer(value: object, *, minimum: int = 0) -> bool:
    return _ack_number(value, minimum=minimum) and int(value) == value


def _normalized_live_path(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) >= 2 and text[0] in {"'", '"'} and text[-1] == text[0]:
        text = text[1:-1].strip()
    return " ".join(text.split())


def _ack_path_matches(actual: object, requested: object) -> bool:
    actual_path = _normalized_live_path(actual)
    requested_path = _normalized_live_path(requested)
    if not actual_path or not requested_path:
        return False
    if actual_path == requested_path:
        return True
    tokens = requested_path.split()
    resolving_alias = (
        (len(tokens) == 2 and tokens[0] == "id" and tokens[1].isdigit())
        or tokens[0] == "this_device"
        or "canonical_parent" in tokens
        or "view" in tokens
        or any(token.startswith("selected_") for token in tokens)
    )
    if not resolving_alias:
        return False
    # LiveAPI resolves these aliases before echoing its path. The response is
    # correlated, but the wire contract does not prove the alias's object ID.
    resolved = actual_path.split()
    return resolved[0] in {"live_set", "live_app"} and all(
        token.isidentifier() or token.isdigit() for token in resolved
    )


def _track_reference_path(value: object) -> str:
    text = str(value).strip()
    if text in {"", "default"}:
        return "live_set tracks 0"
    if text == "master":
        return "live_set master_track"
    parts = text.replace(":", " ").split()
    if len(parts) == 2 and parts[0] == "return" and parts[1].isdigit():
        return f"live_set return_tracks {int(parts[1])}"
    if text.isdigit():
        return f"live_set tracks {int(text)}"
    return text


def _command_ack_is_valid(command: OscCommand, address: str, args: Sequence[OscArg]) -> bool:
    expected_event, request_id = _command_ack_expectation(command)
    event = parse_ack_event(address, args)
    if (
        address != "/ack" or event.is_error
        or event.event != expected_event or event.request_id != request_id
    ):
        return False
    supplied = command.args[1:] if command.address in PROTECTED_OSC_ADDRESSES else command.args
    layout = _ACK_JSON_FIELDS.get(expected_event)
    if layout is not None:
        json_index, payload_type = layout
        if len(args) != json_index + 1 + int(request_id is not None):
            return False
        payload = _ack_json(args[json_index])
        if payload is _INVALID_ACK_JSON or (
            payload_type is not None and not isinstance(payload, payload_type)
        ):
            return False
        if expected_event in {"api_children", "api_observers"} and not all(
            isinstance(item, dict) for item in payload
        ):
            return False
        if expected_event in {"api_set", "api_insert_device", "api_insert_chain"}:
            if payload.get("ok") is not True:
                return False
        if expected_event in {"api_get", "api_set", "api_call", "api_children"}:
            return (
                len(supplied) >= 2 and _ack_path_matches(args[1], supplied[0])
                and isinstance(args[2], str) and args[2] == str(supplied[1]).strip()
            )
        if expected_event == "api_observe":
            return (
                len(supplied) >= 2 and isinstance(args[1], str) and bool(args[1])
                and _ack_path_matches(args[2], supplied[0])
                and args[3] == str(supplied[1]).strip()
            )
        if expected_event == "api_unobserve":
            return bool(supplied) and args[1] == str(supplied[0]).strip() and payload.get("removed") is True
        if expected_event == "api_device_list":
            return bool(supplied) and args[1] == (str(supplied[0]).strip() or "all")
        if expected_event == "api_mixer_status":
            return bool(supplied) and _ack_path_matches(args[1], _track_reference_path(supplied[0]))
        if expected_event in {
            "api_describe", "api_device_parameters", "api_parameter_set",
            "api_insert_device", "api_insert_chain", "api_drum_chain_in_note",
        }:
            if not supplied or not _ack_path_matches(args[1], supplied[0]):
                return False
            if expected_event == "api_insert_device":
                return len(supplied) >= 2 and args[2] == str(supplied[1]).strip()
        return True

    if expected_event == "pong":
        return len(args) == 1 + int(request_id is not None)
    if expected_event == "status":
        return (
            len(args) >= 5 and all(_ack_integer(value) for value in args[1:5])
            and args[2] + args[3] <= args[1]
        )
    if expected_event in {"tempo", "sig_num", "sig_den"}:
        return (
            len(args) == 2 and bool(supplied) and _ack_number(args[1], minimum=0)
            and _ack_number(supplied[0])
            and math.isclose(args[1], supplied[0], rel_tol=1e-6, abs_tol=1e-6)
        )
    if expected_event in {"create_midi_track", "create_audio_track"}:
        return len(args) == 2 and args[1] == -1
    if expected_event == "track_renamed":
        return (
            len(args) == 3 and len(supplied) >= 2 and _ack_integer(args[1])
            and args[1] == supplied[0]
            and args[2] == (str(supplied[1]).strip() or f"Track {supplied[0]}")
        )
    if expected_event in {"midi_cc", "cc64"}:
        count = 3 if expected_event == "midi_cc" else 2
        return len(args) == count + 1 + int(request_id is not None) and list(args[1:count + 1]) == list(supplied[:count])
    if expected_event in {"add_midi_tracks", "add_audio_tracks"}:
        return (
            len(args) == 5 and bool(supplied) and args[1] == supplied[0]
            and isinstance(args[2], str) and all(_ack_integer(args[i]) for i in (1, 3, 4))
            and args[3] == args[1] and args[4] >= args[3]
        )
    if expected_event in {"delete_midi_tracks", "delete_audio_tracks"}:
        return (
            len(args) == 4 and bool(supplied) and args[1] == supplied[0]
            and all(_ack_integer(value) for value in args[1:]) and args[2] <= args[1]
        )
    if expected_event == "ensure_midi_tracks":
        return (
            len(args) == 5 and bool(supplied) and args[1] == supplied[0]
            and all(_ack_integer(value) for value in args[1:])
            and args[3] == max(0, args[1] - args[2]) and args[4] >= args[2]
        )
    if expected_event in {"set_session_clip_notes", "append_session_clip_notes", "inspect_session_clip_notes"}:
        minimum_length = {"set_session_clip_notes": 7, "append_session_clip_notes": 5, "inspect_session_clip_notes": 8}[expected_event]
        if len(args) != minimum_length or len(supplied) < 2 or not all(_ack_integer(value) for value in args[1:3]):
            return False
        if list(args[1:3]) != list(supplied[:2]):
            return False
        if expected_event == "inspect_session_clip_notes":
            payload = _ack_json(args[7])
            return (
                _ack_integer(args[3]) and _ack_number(args[6], minimum=0)
                and isinstance(payload, dict) and isinstance(payload.get("notes"), list)
                and len(payload["notes"]) == args[3]
            )
        if expected_event == "set_session_clip_notes":
            if len(supplied) < 5:
                return False
            supplied_notes = _ack_json(supplied[3])
            notes = supplied_notes.get("notes") if isinstance(supplied_notes, dict) else supplied_notes
            return (
                _ack_number(args[3], minimum=0) and _ack_number(supplied[2], minimum=0)
                and math.isclose(args[3], supplied[2], rel_tol=1e-6, abs_tol=1e-6)
                and _ack_integer(args[4]) and isinstance(notes, list) and args[4] == len(notes)
                and _ack_integer(args[5]) and args[6] == str(supplied[4]).strip()
            )
        if len(supplied) < 3:
            return False
        supplied_notes = _ack_json(supplied[2])
        notes = supplied_notes.get("notes") if isinstance(supplied_notes, dict) else supplied_notes
        return (
            _ack_integer(args[3]) and isinstance(notes, list) and args[3] == len(notes)
            and _ack_integer(args[4])
        )
    return False


def validate_command_acks(
    command: OscCommand,
    acks: Sequence[Tuple[str, List[OscArg]]],
) -> AckEvent:
    """Require a complete matching success and reject a correlated error."""
    expected_event, request_id = _command_ack_expectation(command)
    matched: AckEvent | None = None
    for address, args in acks:
        event = parse_ack_event(address, args)
        if address != "/ack" or event.request_id != request_id:
            continue
        if event.is_error:
            raise BridgeAcknowledgementError(
                f"bridge rejected {command.address} (request {request_id}): "
                f"{event.payload.get('code', 'unknown_error')}"
            )
        if _command_ack_is_valid(command, address, args):
            matched = event
    if matched is None:
        raise BridgeAcknowledgementError(
            f"did not receive the {expected_event} acknowledgement with a complete "
            f"matching payload for {command.address} (request {request_id})"
        )
    return matched


def wait_for_acks(
    sock: socket.socket,
    timeout_s: float,
    quiet_window_s: float = 0.05,
    expected_sender_host: str | None = None,
    expected_request_id: str | None = None,
    expected_event: str | None = None,
    expected_commands: Sequence[OscCommand] = (),
) -> List[Tuple[str, List[OscArg]]]:
    timeout = _finite_wait_seconds(timeout_s, "timeout_s")
    quiet_window = _finite_wait_seconds(quiet_window_s, "quiet_window_s")
    if timeout <= 0:
        return []
    if len(expected_commands) > ACK_MAX_BATCH_COMMANDS:
        raise ValueError("ACK batch exceeds the pending-command limit")
    expected = {_command_ack_expectation(command): command for command in expected_commands}
    expected_ids = {key[1] for key in expected}
    if len(expected) != len(expected_commands) or (
        len(expected_commands) > 1
        and (None in expected_ids or len(expected_ids) != len(expected_commands))
    ):
        raise ValueError("ACK batch contains ambiguous command correlations")
    matched_commands: set[tuple[str, str | None]] = set()
    retained_byte_limit = ACK_WAIT_MAX_RETAINED_BYTES * max(1, len(expected))

    deadline = time.monotonic() + timeout
    received: List[Tuple[str, List[OscArg]]] = []
    packets_processed = 0
    retained_bytes = 0
    received_expected_event = False
    quiet_window = max(0.0, quiet_window)

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        # Wait up to the full timeout for the expected command response; use
        # the shorter quiet window only after its ACK or matching error arrives.
        wait_timeout = remaining
        if (
            received
            and quiet_window > 0.0
            and (
                (len(matched_commands) == len(expected) if expected else expected_event is None)
                or received_expected_event
            )
        ):
            wait_timeout = min(wait_timeout, quiet_window)

        readable, _, _ = select.select([sock], [], [], wait_timeout)
        if not readable:
            if received:
                break
            break

        packets_in_batch = 0
        while (
            packets_in_batch < ACK_MAX_PACKETS_PER_SELECT
            and packets_processed < ACK_WAIT_MAX_PACKETS
        ):
            if time.monotonic() >= deadline:
                return received
            try:
                packet, sender = sock.recvfrom(ACK_PACKET_BUDGET_BYTES + 1)
            except BlockingIOError:
                break
            except OSError:
                return received
            packets_in_batch += 1
            packets_processed += 1
            if time.monotonic() >= deadline:
                return received
            if not _sender_matches(sender, expected_sender_host):
                continue

            retained_item: Tuple[str, List[OscArg]]
            if len(packet) > ACK_PACKET_BUDGET_BYTES:
                retained_item = (
                    "<unparsed>",
                    [
                        "OSC packet exceeds "
                        f"{ACK_PACKET_BUDGET_BYTES}-byte budget: "
                        f"{len(packet)} bytes"
                    ],
                )
            else:
                try:
                    retained_item = decode_osc_message(packet)
                except Exception as exc:  # noqa: BLE001 - best-effort debug output
                    retained_item = (
                        "<unparsed>",
                        [_bounded_packet_diagnostic(exc, packet)],
                    )

            event: AckEvent | None = None
            if expected_request_id is not None or expected_event is not None or expected:
                event = parse_ack_event(*retained_item)
                if (
                    expected_request_id is not None
                    and event.request_id != expected_request_id
                ):
                    continue
                if expected:
                    if event.address != "/ack" or event.request_id not in expected_ids:
                        continue
                    if not event.is_error and (event.event, event.request_id) not in expected:
                        continue

            if (
                event is not None and event.address == "/ack" and event.is_error
                and (expected or event.request_id == expected_request_id)
            ):
                # A definitive failure must survive duplicate successes filling
                # either retention cap. One UDP packet fits the per-command cap.
                if (
                    len(received) >= ACK_WAIT_MAX_RETAINED_ACKS
                    or retained_bytes + len(packet) > retained_byte_limit
                ):
                    return [retained_item]
                return received + [retained_item]

            if (
                len(received) < ACK_WAIT_MAX_RETAINED_ACKS
                and retained_bytes + len(packet) <= retained_byte_limit
            ):
                received.append(retained_item)
                retained_bytes += len(packet)
                if expected and event is not None and event.address == "/ack":
                    key = (event.event, event.request_id)
                    command = expected.get(key)
                    if command is not None and _command_ack_is_valid(command, *retained_item):
                        matched_commands.add(key)
                elif (
                    expected_event is not None
                    and event is not None
                    and event.address == "/ack"
                    and event.request_id == expected_request_id
                    and (event.is_error or event.event == expected_event)
                ):
                    received_expected_event = True

        if packets_processed >= ACK_WAIT_MAX_PACKETS:
            break

    return received


def wait_for_session_clip_inspection_acks(
    sock: socket.socket,
    timeout_s: float,
    request_id: str,
    *,
    expected_sender_host: str | None = None,
) -> List[Tuple[str, List[OscArg]]]:
    """Wait for a complete inspection, correlated error, or the full timeout."""
    timeout = _finite_wait_seconds(timeout_s, "timeout_s")
    if timeout <= 0:
        return []

    deadline = time.monotonic() + timeout
    received: List[Tuple[str, List[OscArg]]] = []
    assembler = SessionClipInspectionAssembler()
    correlated_ack_count = 0
    unrelated_ack_count = 0

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        readable, _, _ = select.select([sock], [], [], remaining)
        if not readable:
            break

        packets_processed = 0
        while packets_processed < SESSION_CLIP_INSPECTION_MAX_PACKETS_PER_SELECT:
            if time.monotonic() >= deadline:
                return received
            try:
                packet, sender = sock.recvfrom(
                    SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES + 1
                )
            except BlockingIOError:
                break
            except OSError:
                return received
            packets_processed += 1
            if time.monotonic() >= deadline:
                return received
            if not _sender_matches(sender, expected_sender_host):
                continue

            if len(packet) > SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES:
                if unrelated_ack_count < SESSION_CLIP_INSPECTION_MAX_UNRELATED_ACKS:
                    received.append(
                        (
                            "<unparsed>",
                            [
                                "OSC packet exceeds session clip inspection "
                                f"budget: {len(packet)} bytes"
                            ],
                        )
                    )
                    unrelated_ack_count += 1
                continue

            try:
                address, args = decode_osc_message(packet)
            except Exception as exc:  # noqa: BLE001 - best-effort debug output
                if unrelated_ack_count < SESSION_CLIP_INSPECTION_MAX_UNRELATED_ACKS:
                    received.append(
                        (
                            "<unparsed>",
                            [_bounded_packet_diagnostic(exc, packet)],
                        )
                    )
                    unrelated_ack_count += 1
                continue

            event = parse_ack_event(address, args)
            correlated_error = (
                address == "/ack" and event.is_error and event.request_id == request_id
            )
            correlated_fragment = (
                address == "/ack" and event.event == "api_session_clip_inspect"
                and event.request_id == request_id
            )
            if correlated_error or correlated_fragment:
                if (
                    correlated_ack_count
                    < SESSION_CLIP_INSPECTION_MAX_CORRELATED_ACKS
                ):
                    received.append((address, args))
                    correlated_ack_count += 1
            elif unrelated_ack_count < SESSION_CLIP_INSPECTION_MAX_UNRELATED_ACKS:
                received.append((address, args))
                unrelated_ack_count += 1

            if correlated_error:
                return received
            if correlated_fragment:
                if assembler.add_event(event) is not None:
                    return received

    return received


def wait_for_arrangement_inspection_acks(
    sock: socket.socket,
    timeout_s: float,
    request_id: str,
    expected_event: str,
    *,
    expected_sender_host: str | None = None,
) -> List[Tuple[str, List[OscArg]]]:
    """Wait for complete bounded Arrangement facts or a correlated failure."""
    if expected_event not in ARRANGEMENT_INSPECTION_EVENTS:
        raise ArrangementInspectionAssemblyError("malformed fragment event")
    timeout = _finite_wait_seconds(timeout_s, "timeout_s")
    if timeout <= 0:
        return []

    deadline = time.monotonic() + timeout
    received: List[Tuple[str, List[OscArg]]] = []
    assembler = ArrangementInspectionAssembler()
    correlated_ack_count = 0
    unrelated_ack_count = 0

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        readable, _, _ = select.select([sock], [], [], remaining)
        if not readable:
            break

        packets_processed = 0
        while packets_processed < SESSION_CLIP_INSPECTION_MAX_PACKETS_PER_SELECT:
            if time.monotonic() >= deadline:
                return received
            try:
                packet, sender = sock.recvfrom(
                    ARRANGEMENT_INSPECTION_PACKET_BUDGET_BYTES + 1
                )
            except (BlockingIOError, OSError):
                break
            packets_processed += 1
            if time.monotonic() >= deadline:
                return received
            if not _sender_matches(sender, expected_sender_host):
                continue
            if len(packet) > ARRANGEMENT_INSPECTION_PACKET_BUDGET_BYTES:
                if unrelated_ack_count < SESSION_CLIP_INSPECTION_MAX_UNRELATED_ACKS:
                    received.append(
                        (
                            "<unparsed>",
                            [
                                "OSC packet exceeds Arrangement inspection "
                                f"budget: {len(packet)} bytes"
                            ],
                        )
                    )
                    unrelated_ack_count += 1
                continue
            try:
                address, args = decode_osc_message(packet)
            except Exception as exc:  # noqa: BLE001 - bounded diagnostic only
                if unrelated_ack_count < SESSION_CLIP_INSPECTION_MAX_UNRELATED_ACKS:
                    received.append(
                        ("<unparsed>", [_bounded_packet_diagnostic(exc, packet)])
                    )
                    unrelated_ack_count += 1
                continue

            event = parse_ack_event(address, args)
            correlated_error = (
                address == "/ack" and event.is_error and event.request_id == request_id
            )
            correlated_fragment = (
                address == "/ack" and event.event == expected_event
                and event.request_id == request_id
            )
            if correlated_error or correlated_fragment:
                if correlated_ack_count < ARRANGEMENT_INSPECTION_MAX_FRAGMENTS + 1:
                    received.append((address, args))
                    correlated_ack_count += 1
            elif unrelated_ack_count < SESSION_CLIP_INSPECTION_MAX_UNRELATED_ACKS:
                received.append((address, args))
                unrelated_ack_count += 1
            if correlated_error:
                return received
            if correlated_fragment and assembler.add_event(event) is not None:
                return received

    return received


def _drain_acks_nonblocking(sock: socket.socket) -> List[Tuple[str, List[OscArg]]]:
    drained: List[Tuple[str, List[OscArg]]] = []
    for _packet_index in range(ACK_DRAIN_MAX_PACKETS):
        try:
            packet, _addr = sock.recvfrom(
                SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES + 1
            )
        except BlockingIOError:
            break
        except OSError:
            break
        if len(packet) > SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES:
            drained.append(
                (
                    "<unparsed>",
                    [
                        "OSC packet exceeds "
                        f"{SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES}-byte "
                        f"budget: {len(packet)} bytes"
                    ],
                )
            )
            continue
        try:
            address, args = decode_osc_message(packet)
            drained.append((address, args))
        except Exception as exc:  # noqa: BLE001
            drained.append(
                ("<unparsed>", [_bounded_packet_diagnostic(exc, packet)])
            )
    return drained


def _collect_and_print_acks(
    ack_sock: socket.socket,
    timeout_s: float,
    durations_ms: List[float],
    ack_counts: List[int],
    *,
    expected_event: str | None = None,
    expected_request_id: str | None = None,
    expected_sender_host: str | None = None,
    command_address: str | None = None,
    command: OscCommand | None = None,
) -> None:
    t0 = time.perf_counter()
    acks = wait_for_acks(
        ack_sock,
        timeout_s,
        expected_sender_host=expected_sender_host,
        expected_request_id=expected_request_id,
        expected_event=expected_event,
        expected_commands=() if command is None else (command,),
    )
    durations_ms.append((time.perf_counter() - t0) * 1000.0)
    ack_counts.append(len(acks))

    if not acks:
        target = command_address or "the command"
        correlation = (
            "" if expected_request_id is None else f" (request {expected_request_id})"
        )
        raise BridgeAcknowledgementError(
            f"no acknowledgement received for {target}{correlation} "
            f"within {timeout_s:.2f}s; bridge may not be loaded"
        )

    matching_ack = False
    matching_error: AckEvent | None = None
    for address, args in acks:
        for line in summarize_ack(address, args):
            print(line)
        if address != "/ack":
            continue
        event = parse_ack_event(address, args)
        if event.request_id != expected_request_id:
            continue
        if event.is_error:
            matching_error = event
        elif expected_event is None or event.event == expected_event:
            matching_ack = True

    if command is not None:
        validate_command_acks(command, acks)
        return

    target = command_address or "the command"
    correlation = (
        "" if expected_request_id is None else f" (request {expected_request_id})"
    )
    if matching_error is not None:
        code = matching_error.payload.get("code", "unknown_error")
        raise BridgeAcknowledgementError(
            f"bridge rejected {target}{correlation}: {code}"
        )
    if not matching_ack:
        event_description = (
            "an acknowledgement"
            if expected_event is None
            else f"the {expected_event} acknowledgement"
        )
        raise BridgeAcknowledgementError(
            f"did not receive {event_description} for {target}{correlation} "
            f"within {timeout_s:.2f}s"
        )


def _collect_and_print_ack_batch(
    ack_sock: socket.socket,
    cfg: BridgeConfig,
    commands: Sequence[OscCommand],
    durations_ms: List[float],
    ack_counts: List[int],
) -> None:
    started = time.perf_counter()
    acks = wait_for_acks(
        ack_sock,
        cfg.ack_timeout_s,
        expected_sender_host=cfg.host,
        expected_commands=commands,
    )
    durations_ms.append((time.perf_counter() - started) * 1000.0)
    ack_counts.append(len(acks))
    for address, args in acks:
        for line in summarize_ack(address, args):
            print(line)
    for command in commands:
        validate_command_acks(command, acks)


def _validate_inspection_target(
    command: OscCommand,
    inspection: dict[str, object],
) -> None:
    """Compare a schema-validated inspection with the actual requested target."""
    correlation = inspection["correlation"]
    assert isinstance(correlation, dict)
    if command.address == "/api/session_clip_inspect":
        expected = {
            "track_index": command.args[0],
            "slot_index": command.args[1],
            "request_id": command.args[3],
        }
        data = inspection
        clip_path = f"live_set tracks {command.args[0]} clip_slots {command.args[1]} clip"
    else:
        scope = command.address.removeprefix("/api/arrangement_").removesuffix("_inspect")
        expected = {
            "scope": scope,
            "track_index": None if scope == "project" else command.args[0],
            "clip_index": command.args[1] if scope == "clip" else None,
            "request_id": command.args[-1],
        }
        data = inspection["data"]
        assert isinstance(data, dict)
        privacy = data["privacy"]
        assert isinstance(privacy, dict)
        requested_notes = scope == "clip" and command.args[2] == 1
        if privacy["notes_requested"] != requested_notes:
            raise BridgeAcknowledgementError(
                f"{command.address} acknowledgement does not match requested note access"
            )
        clip_path = f"live_set tracks {command.args[0]} arrangement_clips {command.args[1]}"
    if any(correlation.get(key) != value for key, value in expected.items()):
        raise BridgeAcknowledgementError(
            f"{command.address} acknowledgement does not match the requested target"
        )
    if expected["track_index"] is not None:
        track = data["track"]
        assert isinstance(track, dict)
        if not _ack_path_matches(track["path"], f"live_set tracks {expected['track_index']}"):
            raise BridgeAcknowledgementError(
                f"{command.address} acknowledgement has the wrong track path"
            )
    if "clip" in data:
        clip = data["clip"]
        assert isinstance(clip, dict)
        if not _ack_path_matches(clip["path"], clip_path):
            raise BridgeAcknowledgementError(
                f"{command.address} acknowledgement has the wrong clip path"
            )


def _collect_and_print_session_clip_inspection_acks(
    ack_sock: socket.socket,
    timeout_s: float,
    request_id: str,
    durations_ms: List[float],
    ack_counts: List[int],
    *,
    expected_sender_host: str | None = None,
    command: OscCommand | None = None,
) -> None:
    t0 = time.perf_counter()
    acks = wait_for_session_clip_inspection_acks(
        ack_sock,
        timeout_s,
        request_id,
        expected_sender_host=expected_sender_host,
    )
    durations_ms.append((time.perf_counter() - t0) * 1000.0)
    ack_counts.append(len(acks))

    if not acks:
        raise BridgeAcknowledgementError(
            "no acknowledgement received for /api/session_clip_inspect "
            f"(request {request_id}) within {timeout_s:.2f}s; "
            "bridge may not be loaded"
        )

    assembler = SessionClipInspectionAssembler()
    inspection_completed = False
    for address, args in acks:
        for line in summarize_ack(address, args):
            print(line)
        if address != "/ack":
            continue
        event = parse_ack_event(address, args)
        if event.request_id != request_id:
            continue
        if event.is_error:
            code = event.payload.get("code", "unknown_error")
            raise BridgeAcknowledgementError(
                "bridge rejected /api/session_clip_inspect "
                f"(request {request_id}): {code}"
            )
        if event.event == "api_session_clip_inspect":
            inspection = assembler.add_event(event)
            if inspection is not None:
                if command is not None:
                    _validate_inspection_target(command, inspection)
                inspection_completed = True

    if not inspection_completed:
        raise BridgeAcknowledgementError(
            "did not receive a complete /api/session_clip_inspect "
            f"acknowledgement (request {request_id}) within {timeout_s:.2f}s"
        )


def _collect_and_print_arrangement_inspection_acks(
    ack_sock: socket.socket,
    timeout_s: float,
    request_id: str,
    expected_event: str,
    durations_ms: List[float],
    ack_counts: List[int],
    *,
    expected_sender_host: str | None = None,
    command: OscCommand | None = None,
) -> None:
    started = time.perf_counter()
    acks = wait_for_arrangement_inspection_acks(
        ack_sock,
        timeout_s,
        request_id,
        expected_event,
        expected_sender_host=expected_sender_host,
    )
    durations_ms.append((time.perf_counter() - started) * 1000.0)
    ack_counts.append(len(acks))
    command_address = "/api/" + expected_event[len("api_"):]
    if not acks:
        raise BridgeAcknowledgementError(
            f"no acknowledgement received for {command_address} "
            f"(request {request_id}) within {timeout_s:.2f}s; "
            "bridge may not be loaded"
        )

    assembler = ArrangementInspectionAssembler()
    completed = False
    for address, args in acks:
        for line in summarize_ack(address, args):
            print(line)
        if address != "/ack":
            continue
        event = parse_ack_event(address, args)
        if event.request_id != request_id:
            continue
        if event.is_error:
            code = event.payload.get("code", "unknown_error")
            raise BridgeAcknowledgementError(
                f"bridge rejected {command_address} "
                f"(request {request_id}): {code}"
            )
        if event.event == expected_event:
            inspection = assembler.add_event(event)
            if inspection is not None:
                if command is not None:
                    _validate_inspection_target(command, inspection)
                completed = True
    if not completed:
        raise BridgeAcknowledgementError(
            f"did not receive a complete {command_address} "
            f"acknowledgement (request {request_id}) within {timeout_s:.2f}s"
        )


_ACK_REQUEST_ARGUMENT_COUNTS = {
    "/api/ping": 0,
    "/api/get": 2,
    "/api/set": 4,
    "/api/call": 4,
    "/api/children": 2,
    "/api/describe": 1,
    "/api_observe": 4,
    "/api_unobserve": 2,
    "/api_observers": 0,
    "/api_clear_observers": 1,
    "/api/session_context": 0,
    "/api/theory_status": 0,
    "/api/tuning_status": 0,
    "/api/device_list": 1,
    "/api/device_parameters": 1,
    "/api/parameter_set": 3,
    "/api/mixer_status": 1,
    "/api/insert_device": 4,
    "/api/insert_chain": 3,
    "/api/drum_chain_in_note": 3,
    "/api/arrangement_project_inspect": 1,
    "/api/arrangement_track_inspect": 2,
    "/api/arrangement_clip_inspect": 4,
    "/midi_cc": 4,
    "/cc64": 3,
}


def _command_ack_expectation(command: OscCommand) -> tuple[str, str | None]:
    if command.address in {"/ping", "/api/ping"}:
        expected_event = "pong"
    elif command.address == "/rename_track":
        expected_event = "track_renamed"
    else:
        expected_event = command.address.lstrip("/").replace("/", "_")

    request_argument_index = _ACK_REQUEST_ARGUMENT_COUNTS.get(command.address)
    if request_argument_index is None or len(command.args) <= request_argument_index:
        return expected_event, None
    request_id = command.args[request_argument_index]
    if isinstance(request_id, str) and request_id:
        return expected_event, request_id
    return expected_event, None


def _ack_command_batches(
    cfg: BridgeConfig,
    commands: Sequence[OscCommand],
) -> Iterable[list[OscCommand]]:
    limit = ACK_MAX_BATCH_COMMANDS
    if cfg.ack_mode == "flush_interval":
        limit = min(limit, max(1, cfg.ack_flush_interval))
    pending: list[OscCommand] = []
    request_ids: set[str] = set()
    for command in commands:
        event, request_id = _command_ack_expectation(command)
        serial = (
            not cfg.expect_ack or cfg.ack_mode == "per_command" or request_id is None
            or command.address == "/api/session_clip_inspect"
            or event in ARRANGEMENT_INSPECTION_EVENTS
        )
        if pending and (serial or request_id in request_ids):
            yield pending
            pending = []
            request_ids.clear()
        if serial:
            yield [command]
            continue
        assert request_id is not None
        pending.append(command)
        request_ids.add(request_id)
        if len(pending) == limit:
            yield pending
            pending = []
            request_ids.clear()
    if pending:
        yield pending


def send_commands(cfg: BridgeConfig, commands: Sequence[OscCommand]) -> SendMetrics:
    delay_s = cfg.delay_ms / 1000.0

    if cfg.dry_run:
        print(f"Target: udp://{cfg.host}:{cfg.port}")
        for cmd in commands:
            print(f"-> {describe_command(cmd)}")
        return SendMetrics(
            command_count=len(commands),
            send_durations_ms=(),
            ack_wait_durations_ms=(),
            acks_per_command=(),
            elapsed_ms=0.0,
        )

    ack_sock = open_ack_socket(cfg)
    if cfg.expect_ack and ack_sock is None:
        raise BridgeAcknowledgementError(
            "acknowledgements were requested, but the ack socket could not "
            f"bind on {cfg.host}:{cfg.ack_port}; no commands were sent"
        )
    send_durations_ms: List[float] = []
    ack_wait_durations_ms: List[float] = []
    ack_counts: List[int] = []

    t_all = time.perf_counter()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            print(f"Target: udp://{cfg.host}:{cfg.port}")
            if ack_sock is not None:
                print(
                    f"Ack:    udp://{cfg.host}:{cfg.ack_port} "
                    f"(timeout {cfg.ack_timeout_s:.2f}s)"
                )

            sent_count = 0
            for batch in _ack_command_batches(cfg, commands):
                if ack_sock is not None:
                    _drain_acks_nonblocking(ack_sock)

                payloads = [encode_osc_message(cmd.address, cmd.args) for cmd in batch]
                for index, (cmd, payload) in enumerate(zip(batch, payloads)):
                    t_send = time.perf_counter()
                    sock.sendto(payload, (cfg.host, cfg.port))
                    send_durations_ms.append((time.perf_counter() - t_send) * 1000.0)
                    sent_count += 1
                    print(f"sent: {describe_command(cmd)}")
                    if delay_s > 0 and index < len(batch) - 1:
                        time.sleep(delay_s)

                if ack_sock is not None:
                    cmd = batch[0]
                    if len(batch) > 1:
                        _collect_and_print_ack_batch(
                            ack_sock, cfg, batch, ack_wait_durations_ms, ack_counts
                        )
                    elif cmd.address == "/api/session_clip_inspect":
                        request_id = str(cmd.args[3])
                        _collect_and_print_session_clip_inspection_acks(
                            ack_sock,
                            cfg.ack_timeout_s,
                            request_id,
                            ack_wait_durations_ms,
                            ack_counts,
                            expected_sender_host=cfg.host,
                            command=cmd,
                        )
                    elif cmd.address in {
                        "/api/arrangement_project_inspect",
                        "/api/arrangement_track_inspect",
                        "/api/arrangement_clip_inspect",
                    }:
                        expected_event, request_id = _command_ack_expectation(cmd)
                        assert request_id is not None
                        _collect_and_print_arrangement_inspection_acks(
                            ack_sock,
                            cfg.ack_timeout_s,
                            request_id,
                            expected_event,
                            ack_wait_durations_ms,
                            ack_counts,
                            expected_sender_host=cfg.host,
                            command=cmd,
                        )
                    else:
                        expected_event, request_id = _command_ack_expectation(cmd)
                        _collect_and_print_acks(
                            ack_sock,
                            cfg.ack_timeout_s,
                            ack_wait_durations_ms,
                            ack_counts,
                            expected_event=expected_event,
                            expected_request_id=request_id,
                            expected_sender_host=cfg.host,
                            command_address=cmd.address,
                            command=cmd,
                        )

                if delay_s > 0 and sent_count < len(commands):
                    time.sleep(delay_s)
    finally:
        if ack_sock is not None:
            ack_sock.close()

    metrics = SendMetrics(
        command_count=len(commands),
        send_durations_ms=tuple(send_durations_ms),
        ack_wait_durations_ms=tuple(ack_wait_durations_ms),
        acks_per_command=tuple(ack_counts),
        elapsed_ms=(time.perf_counter() - t_all) * 1000.0,
    )

    if cfg.report_metrics:
        for line in _summarize_metrics(metrics):
            print(line)

    return metrics


def listen_for_events(cfg: BridgeConfig) -> int:
    if cfg.dry_run:
        print("listen: skipped in dry-run")
        return 0

    ack_sock = open_ack_socket(cfg)
    if ack_sock is None:
        print(
            f"error: could not listen on {cfg.host}:{cfg.ack_port}",
            file=sys.stderr,
        )
        return -1

    max_events = max(0, int(cfg.listen_max_events))
    listen_timeout = _finite_wait_seconds(
        cfg.listen_timeout_s,
        "listen_timeout_s",
    )
    if listen_timeout < 0:
        raise ValueError("listen_timeout_s must be >= 0")
    deadline = (
        None
        if listen_timeout == 0
        else time.monotonic() + listen_timeout
    )
    event_count = 0
    print(f"Listening: udp://{cfg.host}:{cfg.ack_port}")

    try:
        while True:
            if max_events > 0 and event_count >= max_events:
                break

            wait_timeout = 0.25
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                wait_timeout = min(wait_timeout, remaining)

            readable, _, _ = select.select([ack_sock], [], [], wait_timeout)
            if not readable:
                continue

            packets_in_batch = 0
            while packets_in_batch < ACK_MAX_PACKETS_PER_SELECT:
                if deadline is not None and time.monotonic() >= deadline:
                    return event_count
                try:
                    packet, _addr = ack_sock.recvfrom(
                        ACK_PACKET_BUDGET_BYTES + 1
                    )
                except BlockingIOError:
                    break
                except OSError:
                    return event_count
                packets_in_batch += 1
                if deadline is not None and time.monotonic() >= deadline:
                    return event_count

                if len(packet) > ACK_PACKET_BUDGET_BYTES:
                    address, args = "<unparsed>", [
                        "OSC packet exceeds "
                        f"{ACK_PACKET_BUDGET_BYTES}-byte budget: "
                        f"{len(packet)} bytes"
                    ]
                else:
                    try:
                        address, args = decode_osc_message(packet)
                    except Exception as exc:  # noqa: BLE001
                        address, args = "<unparsed>", [
                            _bounded_packet_diagnostic(exc, packet)
                        ]
                for line in summarize_ack(address, args):
                    print(line)
                event_count += 1
                if max_events > 0 and event_count >= max_events:
                    break
    finally:
        ack_sock.close()

    return event_count


def main(argv: Iterable[str]) -> int:
    cfg = parse_args(argv)

    try:
        commands = build_commands(cfg)
        if not commands and not cfg.listen:
            print("No commands to send. Use --help for options.", file=sys.stderr)
            return 2
        if commands:
            send_commands(cfg, commands)
        if cfg.listen:
            if listen_for_events(cfg) < 0:
                return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level CLI error handler
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
