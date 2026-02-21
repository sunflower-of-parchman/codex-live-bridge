#!/usr/bin/env python3
"""Analyze an existing Ableton Live set through the UDP bridge."""

from __future__ import annotations

import argparse
import json
import math
import secrets
import socket
import statistics
import sys
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import ableton_udp_bridge as bridge


DEFAULT_HOST = bridge.DEFAULT_HOST
DEFAULT_PORT = bridge.DEFAULT_PORT
DEFAULT_ACK_PORT = bridge.DEFAULT_ACK_PORT
DEFAULT_ACK_TIMEOUT_S = 1.0

DEFAULT_TRACK_QUERY = "piano"
DEFAULT_HAND_SPLIT_PITCH = 60
DEFAULT_TOPOLOGY_DEPTH = 2
DEFAULT_MAX_CHILDREN = 32
DEFAULT_MAX_SESSION_SLOTS = 24
DEFAULT_MAX_ARRANGEMENT_CLIPS = 24
DEFAULT_OUTPUT_DIR = Path("memory/analysis/live_sets")


OscAck = tuple[str, list[bridge.OscArg]]


@dataclass(frozen=True)
class AnalyzeConfig:
    host: str
    port: int
    ack_port: int
    ack_timeout_s: float
    track_query: str
    track_index: int | None
    clip_scope: str
    include_note_events: bool
    hand_split_pitch: int
    topology_depth: int
    topology_max_children: int
    max_session_slots: int
    max_arrangement_clips: int
    output_dir: Path
    write_markdown: bool
    dry_run: bool


@dataclass(frozen=True)
class TrackRef:
    index: int
    path: str
    name: str
    type: str


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_args(argv: Iterable[str]) -> AnalyzeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bridge host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bridge command port")
    parser.add_argument("--ack-port", type=int, default=DEFAULT_ACK_PORT, help="Bridge ack port")
    parser.add_argument(
        "--ack-timeout",
        type=_positive_float,
        default=DEFAULT_ACK_TIMEOUT_S,
        help=f"Ack wait timeout in seconds (default: {DEFAULT_ACK_TIMEOUT_S})",
    )
    parser.add_argument(
        "--track-query",
        default=DEFAULT_TRACK_QUERY,
        help=f"Case-insensitive track name match (default: {DEFAULT_TRACK_QUERY})",
    )
    parser.add_argument(
        "--track-index",
        type=_non_negative_int,
        default=None,
        help="Optional exact track index to analyze",
    )
    parser.add_argument(
        "--clip-scope",
        choices=("arrangement", "session", "both"),
        default="both",
        help="Which clip surfaces to inspect (default: both)",
    )
    parser.add_argument(
        "--include-note-events",
        action="store_true",
        help="Persist extracted note events into the JSON artifact",
    )
    parser.add_argument(
        "--hand-split-pitch",
        type=int,
        default=DEFAULT_HAND_SPLIT_PITCH,
        help=f"Split pitch for left/right hand analysis (default: {DEFAULT_HAND_SPLIT_PITCH})",
    )
    parser.add_argument(
        "--topology-depth",
        type=_non_negative_int,
        default=DEFAULT_TOPOLOGY_DEPTH,
        help=f"LiveAPI topology recursion depth (default: {DEFAULT_TOPOLOGY_DEPTH})",
    )
    parser.add_argument(
        "--topology-max-children",
        type=_positive_int,
        default=DEFAULT_MAX_CHILDREN,
        help=f"Max children to traverse per node (default: {DEFAULT_MAX_CHILDREN})",
    )
    parser.add_argument(
        "--max-session-slots",
        type=_positive_int,
        default=DEFAULT_MAX_SESSION_SLOTS,
        help=f"Max clip slots inspected per track (default: {DEFAULT_MAX_SESSION_SLOTS})",
    )
    parser.add_argument(
        "--max-arrangement-clips",
        type=_positive_int,
        default=DEFAULT_MAX_ARRANGEMENT_CLIPS,
        help=f"Max arrangement clips inspected per track (default: {DEFAULT_MAX_ARRANGEMENT_CLIPS})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip writing markdown summary output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected tracks and exit before extraction",
    )
    ns = parser.parse_args(list(argv))

    return AnalyzeConfig(
        host=str(ns.host),
        port=int(ns.port),
        ack_port=int(ns.ack_port),
        ack_timeout_s=float(ns.ack_timeout),
        track_query=str(ns.track_query).strip(),
        track_index=None if ns.track_index is None else int(ns.track_index),
        clip_scope=str(ns.clip_scope),
        include_note_events=bool(ns.include_note_events),
        hand_split_pitch=int(ns.hand_split_pitch),
        topology_depth=int(ns.topology_depth),
        topology_max_children=int(ns.topology_max_children),
        max_session_slots=int(ns.max_session_slots),
        max_arrangement_clips=int(ns.max_arrangement_clips),
        output_dir=Path(str(ns.output_dir)),
        write_markdown=not bool(ns.no_markdown),
        dry_run=bool(ns.dry_run),
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _new_run_id(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(4)


def _request_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(3)}"


def _decode_jsonish(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, str):
        inner = parsed.strip()
        if inner.startswith("{") or inner.startswith("["):
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                return parsed
    return parsed


def _scalar(value: object | None) -> object | None:
    if isinstance(value, list):
        if not value:
            return None
        return value[-1]
    return value


def _as_float(value: object | None, fallback: float | None = None) -> float | None:
    target = _scalar(value)
    if target is None:
        return fallback
    try:
        return float(target)
    except (TypeError, ValueError):
        return fallback


def _as_int(value: object | None, fallback: int | None = None) -> int | None:
    target = _scalar(value)
    if target is None:
        return fallback
    try:
        return int(float(target))
    except (TypeError, ValueError):
        return fallback


def _as_str(value: object | None, fallback: str = "") -> str:
    target = _scalar(value)
    if target is None:
        return fallback
    text = str(target).strip()
    return text if text else fallback


class LiveBridgeClient:
    def __init__(self, cfg: AnalyzeConfig):
        self._cfg = cfg
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._ack_sock = bridge.open_ack_socket(cfg.ack_port)

    def close(self) -> None:
        try:
            self._ack_sock.close()
        finally:
            self._sock.close()

    def _send(self, command: bridge.OscCommand) -> list[OscAck]:
        payload = bridge.encode_osc_message(command.address, command.args)
        self._sock.sendto(payload, (self._cfg.host, self._cfg.port))
        return bridge.wait_for_acks(self._ack_sock, self._cfg.ack_timeout_s)

    def ping(self) -> None:
        command = bridge.OscCommand("/ping")
        self._send(command)

    def _extract_event_args(
        self,
        *,
        acks: Sequence[OscAck],
        event_name: str,
        request_id: str,
    ) -> list[bridge.OscArg]:
        for address, args in acks:
            if address != "/ack" or not args:
                continue
            head = str(args[0])
            if head == "error" and len(args) >= 2 and str(args[-1]) == request_id:
                detail = " ".join(str(a) for a in args[1:-1])
                raise RuntimeError(f"bridge error for {request_id}: {detail}")
            if head == event_name and str(args[-1]) == request_id:
                return list(args)
        raise RuntimeError(f"missing ack event '{event_name}' for request_id={request_id}")

    def api_get(self, path: str, prop: str) -> object:
        request_id = _request_id("get")
        command = bridge.OscCommand("/api/get", (path, prop, request_id))
        acks = self._send(command)
        args = self._extract_event_args(acks=acks, event_name="api_get", request_id=request_id)
        return _decode_jsonish(args[3]) if len(args) >= 4 else None

    def api_call(self, path: str, method: str, args_payload: object) -> object:
        request_id = _request_id("call")
        command = bridge.OscCommand(
            "/api/call",
            (path, method, json.dumps(args_payload), request_id),
        )
        acks = self._send(command)
        args = self._extract_event_args(acks=acks, event_name="api_call", request_id=request_id)
        return _decode_jsonish(args[3]) if len(args) >= 4 else None

    def api_children(self, path: str, child_name: str) -> list[dict[str, Any]]:
        request_id = _request_id("children")
        command = bridge.OscCommand("/api/children", (path, child_name, request_id))
        acks = self._send(command)
        args = self._extract_event_args(acks=acks, event_name="api_children", request_id=request_id)
        if len(args) < 4:
            return []
        payload = _decode_jsonish(args[3])
        if not isinstance(payload, list):
            return []
        output: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, Mapping):
                output.append(dict(item))
        return output

    def api_describe(self, path: str) -> dict[str, Any]:
        request_id = _request_id("describe")
        command = bridge.OscCommand("/api/describe", (path, request_id))
        acks = self._send(command)
        args = self._extract_event_args(acks=acks, event_name="api_describe", request_id=request_id)
        if len(args) < 3:
            return {}
        payload = _decode_jsonish(args[2])
        return dict(payload) if isinstance(payload, Mapping) else {}


def _discover_tracks(client: LiveBridgeClient) -> list[TrackRef]:
    tracks = client.api_children("live_set", "tracks")
    output: list[TrackRef] = []
    for item in tracks:
        index = _as_int(item.get("index"), fallback=None)
        path = _as_str(item.get("path"), fallback="")
        if index is None or not path:
            continue
        name = _as_str(item.get("name"), fallback=f"Track {index}")
        type_name = _as_str(item.get("type"), fallback="")
        output.append(TrackRef(index=int(index), path=path, name=name, type=type_name))
    return output


def _select_tracks(tracks: Sequence[TrackRef], cfg: AnalyzeConfig) -> list[TrackRef]:
    if cfg.track_index is not None:
        selected = [track for track in tracks if int(track.index) == int(cfg.track_index)]
        if selected:
            return selected
        raise RuntimeError(f"track index not found: {cfg.track_index}")

    query = cfg.track_query.strip().lower()
    if not query:
        return list(tracks)
    selected = [track for track in tracks if query in track.name.lower()]
    if selected:
        return selected
    raise RuntimeError(f"no track matched query: {cfg.track_query}")


def _extract_notes_payload(payload: object) -> list[dict[str, Any]]:
    target = payload
    if isinstance(target, str):
        target = _decode_jsonish(target)
    if isinstance(target, Mapping):
        notes = target.get("notes")
        if isinstance(notes, list):
            output: list[dict[str, Any]] = []
            for note in notes:
                if isinstance(note, Mapping):
                    output.append(dict(note))
            return output
    if isinstance(target, list):
        output = []
        for note in target:
            if isinstance(note, Mapping):
                output.append(dict(note))
        return output
    return []


def _normalize_note_event(
    *,
    note: Mapping[str, Any],
    clip_global_start: float,
    clip_source: str,
    track_name: str,
    track_index: int,
    clip_path: str,
    clip_name: str,
) -> dict[str, Any]:
    start_time = _as_float(note.get("start_time"), 0.0) or 0.0
    duration = max(0.0, _as_float(note.get("duration"), 0.0) or 0.0)
    pitch = int(_as_int(note.get("pitch"), 0) or 0)
    velocity = int(_as_int(note.get("velocity"), 0) or 0)
    mute = bool(_as_int(note.get("mute"), 0) or 0)
    event = {
        "track_name": track_name,
        "track_index": int(track_index),
        "clip_source": clip_source,
        "clip_path": clip_path,
        "clip_name": clip_name,
        "note_id": _as_int(note.get("note_id"), None),
        "pitch": pitch,
        "velocity": velocity,
        "duration": duration,
        "start_time": start_time,
        "start_time_global": float(clip_global_start) + float(start_time),
        "end_time_global": float(clip_global_start) + float(start_time) + float(duration),
        "mute": mute,
    }
    return event


def _extract_arrangement_clips(
    *,
    client: LiveBridgeClient,
    track: TrackRef,
    cfg: AnalyzeConfig,
) -> list[dict[str, Any]]:
    clips = client.api_children(track.path, "arrangement_clips")
    output: list[dict[str, Any]] = []
    for item in clips[: cfg.max_arrangement_clips]:
        clip_path = _as_str(item.get("path"), "")
        clip_id = _as_int(item.get("id"), 0) or 0
        if not clip_path or clip_id <= 0:
            continue
        clip_name = _as_str(client.api_get(clip_path, "name"), fallback=f"Arrangement Clip {clip_id}")
        clip_length = _as_float(client.api_get(clip_path, "length"), fallback=0.0) or 0.0
        clip_start = _as_float(client.api_get(clip_path, "start_time"), fallback=0.0) or 0.0
        raw = client.api_call(clip_path, "get_all_notes_extended", [])
        notes = _extract_notes_payload(raw)
        events = [
            _normalize_note_event(
                note=note,
                clip_global_start=clip_start,
                clip_source="arrangement",
                track_name=track.name,
                track_index=track.index,
                clip_path=clip_path,
                clip_name=clip_name,
            )
            for note in notes
        ]
        output.append(
            {
                "track_name": track.name,
                "track_index": track.index,
                "clip_source": "arrangement",
                "clip_path": clip_path,
                "clip_name": clip_name,
                "clip_length": round(float(clip_length), 6),
                "clip_start_time": round(float(clip_start), 6),
                "note_count": len(events),
                "notes": events,
            }
        )
    return output


def _extract_session_clips(
    *,
    client: LiveBridgeClient,
    track: TrackRef,
    cfg: AnalyzeConfig,
) -> list[dict[str, Any]]:
    slots = client.api_children(track.path, "clip_slots")
    output: list[dict[str, Any]] = []
    for slot in slots[: cfg.max_session_slots]:
        slot_path = _as_str(slot.get("path"), "")
        slot_index = _as_int(slot.get("index"), -1)
        if not slot_path or slot_index is None or slot_index < 0:
            continue
        has_clip = _as_int(client.api_get(slot_path, "has_clip"), 0) or 0
        if has_clip != 1:
            continue
        clip_path = f"{slot_path} clip"
        clip_name = _as_str(client.api_get(clip_path, "name"), fallback=f"Slot {slot_index}")
        clip_length = _as_float(client.api_get(clip_path, "length"), fallback=0.0) or 0.0
        raw = client.api_call(clip_path, "get_all_notes_extended", [])
        notes = _extract_notes_payload(raw)
        events = [
            _normalize_note_event(
                note=note,
                clip_global_start=0.0,
                clip_source="session",
                track_name=track.name,
                track_index=track.index,
                clip_path=clip_path,
                clip_name=clip_name,
            )
            for note in notes
        ]
        output.append(
            {
                "track_name": track.name,
                "track_index": track.index,
                "clip_source": "session",
                "slot_index": int(slot_index),
                "clip_path": clip_path,
                "clip_name": clip_name,
                "clip_length": round(float(clip_length), 6),
                "clip_start_time": 0.0,
                "note_count": len(events),
                "notes": events,
            }
        )
    return output


def _round4(value: float) -> float:
    return round(float(value), 4)


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / float(len(values)))


def _safe_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(float(v) for v in values))


def _coverage_ratio(intervals: Sequence[tuple[float, float]], total_length: float) -> float:
    if total_length <= 0.0:
        return 0.0
    spans = [(max(0.0, s), max(0.0, e)) for s, e in intervals if e > s]
    if not spans:
        return 0.0
    spans.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[float, float]] = []
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start <= cur_end + 1e-6:
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    covered = sum(max(0.0, e - s) for s, e in merged)
    return max(0.0, min(1.0, covered / float(total_length)))


def _intersect_coverage_ratio(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
    total_length: float,
) -> float:
    if total_length <= 0.0 or not left or not right:
        return 0.0
    left_spans = sorted((s, e) for s, e in left if e > s)
    right_spans = sorted((s, e) for s, e in right if e > s)
    if not left_spans or not right_spans:
        return 0.0
    i = 0
    j = 0
    overlap = 0.0
    while i < len(left_spans) and j < len(right_spans):
        l0, l1 = left_spans[i]
        r0, r1 = right_spans[j]
        start = max(l0, r0)
        end = min(l1, r1)
        if end > start:
            overlap += float(end - start)
        if l1 < r1:
            i += 1
        else:
            j += 1
    return max(0.0, min(1.0, overlap / float(total_length)))


def _group_onsets(events: Sequence[Mapping[str, Any]]) -> dict[float, list[Mapping[str, Any]]]:
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for event in events:
        onset = round(float(event.get("start_time_global", 0.0)), 3)
        grouped.setdefault(onset, []).append(event)
    return grouped


def _build_harmony_metrics(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "enabled": False,
            "status": "no_notes",
        }

    pitches = [int(event.get("pitch", 0)) for event in events]
    pitch_classes = [pitch % 12 for pitch in pitches]
    pc_counts = Counter(pitch_classes)
    total_notes = len(events)
    onset_map = _group_onsets(events)
    onset_count = len(onset_map)

    chord_onsets = 0
    voicing_shapes: Counter[str] = Counter()
    bass_motion_steps: list[int] = []
    prev_bass: int | None = None

    for onset in sorted(onset_map.keys()):
        notes = onset_map[onset]
        unique_pitches = sorted({int(n.get("pitch", 0)) for n in notes})
        if len(unique_pitches) >= 2:
            chord_onsets += 1
            bass = unique_pitches[0]
            intervals = [pitch - bass for pitch in unique_pitches]
            shape = "-".join(str(interval) for interval in intervals)
            voicing_shapes[shape] += 1
            if prev_bass is not None:
                bass_motion_steps.append(abs(int(bass) - int(prev_bass)))
            prev_bass = bass

    top_voicings = []
    for shape, count in voicing_shapes.most_common(8):
        top_voicings.append(
            {
                "shape": shape,
                "count": int(count),
                "ratio": _round4(float(count) / float(max(1, chord_onsets))),
            }
        )

    return {
        "enabled": True,
        "status": "ok",
        "total_notes": int(total_notes),
        "onset_count": int(onset_count),
        "unique_pitch_class_count": int(len(pc_counts)),
        "pitch_class_histogram": {
            str(pc): {
                "count": int(count),
                "ratio": _round4(float(count) / float(max(1, total_notes))),
            }
            for pc, count in sorted(pc_counts.items(), key=lambda item: item[0])
        },
        "chord_onset_ratio": _round4(float(chord_onsets) / float(max(1, onset_count))),
        "avg_bass_motion_semitones": _round4(_safe_mean([float(step) for step in bass_motion_steps])),
        "top_voicing_shapes": top_voicings,
    }


def _build_velocity_metrics(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "enabled": False,
            "status": "no_notes",
        }

    velocities = [float(int(event.get("velocity", 0))) for event in events]
    pitches = [int(event.get("pitch", 0)) for event in events]

    def _bucket_values(min_pitch: int, max_pitch: int) -> list[float]:
        selected = []
        for velocity, pitch in zip(velocities, pitches):
            if min_pitch <= pitch <= max_pitch:
                selected.append(float(velocity))
        return selected

    low = _bucket_values(0, 54)
    mid = _bucket_values(55, 72)
    high = _bucket_values(73, 127)
    p10 = _round4(float(statistics.quantiles(velocities, n=10)[0])) if len(velocities) >= 10 else _round4(min(velocities))
    p90 = _round4(float(statistics.quantiles(velocities, n=10)[-1])) if len(velocities) >= 10 else _round4(max(velocities))

    def _summary(values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        return {
            "mean": _round4(_safe_mean(values)),
            "std": _round4(_safe_std(values)),
            "min": _round4(min(values)),
            "max": _round4(max(values)),
        }

    return {
        "enabled": True,
        "status": "ok",
        "total_notes": int(len(velocities)),
        "overall": {
            "mean": _round4(_safe_mean(velocities)),
            "median": _round4(float(statistics.median(velocities))),
            "std": _round4(_safe_std(velocities)),
            "p10": p10,
            "p90": p90,
            "min": _round4(min(velocities)),
            "max": _round4(max(velocities)),
        },
        "register_buckets": {
            "low_0_54": _summary(low),
            "mid_55_72": _summary(mid),
            "high_73_127": _summary(high),
        },
    }


def _build_piano_choreography_metrics(
    events: Sequence[Mapping[str, Any]],
    *,
    split_pitch: int,
) -> dict[str, Any]:
    if not events:
        return {
            "enabled": False,
            "status": "no_notes",
        }

    left = [event for event in events if int(event.get("pitch", 0)) <= int(split_pitch)]
    right = [event for event in events if int(event.get("pitch", 0)) > int(split_pitch)]
    total = len(events)

    if not left or not right:
        return {
            "enabled": True,
            "status": "single_hand_only",
            "split_pitch": int(split_pitch),
            "total_notes": int(total),
            "left_note_ratio": _round4(float(len(left)) / float(max(1, total))),
            "right_note_ratio": _round4(float(len(right)) / float(max(1, total))),
        }

    left_onsets = sorted({round(float(event.get("start_time_global", 0.0)), 3) for event in left})
    right_onsets = sorted({round(float(event.get("start_time_global", 0.0)), 3) for event in right})
    left_intervals = [
        (float(event.get("start_time_global", 0.0)), float(event.get("end_time_global", 0.0)))
        for event in left
    ]
    right_intervals = [
        (float(event.get("start_time_global", 0.0)), float(event.get("end_time_global", 0.0)))
        for event in right
    ]

    total_length = max(
        1e-6,
        max(float(event.get("end_time_global", 0.0)) for event in events)
        - min(float(event.get("start_time_global", 0.0)) for event in events),
    )

    left_coverage = _coverage_ratio(left_intervals, total_length)
    right_coverage = _coverage_ratio(right_intervals, total_length)
    overlap_coverage = _intersect_coverage_ratio(left_intervals, right_intervals, total_length)

    call_response_count = 0
    call_response_lags: list[float] = []
    left_queue = deque(left_onsets)
    for right_onset in right_onsets:
        while len(left_queue) >= 2 and float(left_queue[1]) <= float(right_onset):
            left_queue.popleft()
        if not left_queue:
            continue
        lag = float(right_onset) - float(left_queue[0])
        if 0.0 <= lag <= 1.0:
            call_response_count += 1
            call_response_lags.append(float(lag))

    simultaneous_onsets = len(set(left_onsets).intersection(set(right_onsets)))
    independence_ratio = max(0.0, min(1.0, 1.0 - (float(simultaneous_onsets) / float(max(1, len(set(left_onsets + right_onsets)))))))

    left_durations = [float(event.get("duration", 0.0)) for event in left]
    right_durations = [float(event.get("duration", 0.0)) for event in right]
    left_sustain_ratio = _safe_mean([1.0 if d >= 1.0 else 0.0 for d in left_durations])
    right_short_ratio = _safe_mean([1.0 if d <= 0.5 else 0.0 for d in right_durations])
    right_long_ratio = _safe_mean([1.0 if d >= 2.0 else 0.0 for d in right_durations])

    left_velocities = [float(int(event.get("velocity", 0))) for event in left]
    right_velocities = [float(int(event.get("velocity", 0))) for event in right]

    hand_spread = _safe_mean([float(int(event.get("pitch", 0))) for event in right]) - _safe_mean(
        [float(int(event.get("pitch", 0))) for event in left]
    )
    choreography_score = _safe_mean(
        [
            min(1.0, left_sustain_ratio / 0.6),
            min(1.0, right_short_ratio / 0.2),
            min(1.0, right_long_ratio / 0.1),
            min(1.0, call_response_count / float(max(1, len(right_onsets)))),
            min(1.0, overlap_coverage / 0.3),
            min(1.0, independence_ratio / 0.5),
        ]
    )

    flags: list[str] = []
    if left_sustain_ratio < 0.45:
        flags.append("left_hand_sustain_low")
    if right_short_ratio < 0.12:
        flags.append("right_hand_short_note_activity_low")
    if right_long_ratio < 0.05:
        flags.append("right_hand_long_tone_variety_low")
    if overlap_coverage < 0.06:
        flags.append("hands_overlap_low")
    if call_response_count == 0:
        flags.append("hand_call_response_absent")
    if hand_spread < 10.0:
        flags.append("hand_register_separation_low")

    return {
        "enabled": True,
        "status": "ok",
        "split_pitch": int(split_pitch),
        "total_notes": int(total),
        "left_note_ratio": _round4(float(len(left)) / float(max(1, total))),
        "right_note_ratio": _round4(float(len(right)) / float(max(1, total))),
        "left_sustain_ratio": _round4(left_sustain_ratio),
        "right_short_duration_ratio": _round4(right_short_ratio),
        "right_long_duration_ratio": _round4(right_long_ratio),
        "left_coverage_ratio": _round4(left_coverage),
        "right_coverage_ratio": _round4(right_coverage),
        "overlap_coverage_ratio": _round4(overlap_coverage),
        "simultaneous_onset_ratio": _round4(
            float(simultaneous_onsets) / float(max(1, len(set(left_onsets + right_onsets))))
        ),
        "independence_ratio": _round4(independence_ratio),
        "call_response_ratio": _round4(float(call_response_count) / float(max(1, len(right_onsets)))),
        "call_response_mean_lag_beats": _round4(_safe_mean(call_response_lags)),
        "left_velocity_mean": _round4(_safe_mean(left_velocities)),
        "right_velocity_mean": _round4(_safe_mean(right_velocities)),
        "hand_register_spread_semitones": _round4(hand_spread),
        "choreography_score": _round4(choreography_score),
        "flags": flags,
    }


def _build_topology_snapshot(
    *,
    client: LiveBridgeClient,
    root_path: str,
    max_depth: int,
    max_children_per_branch: int,
) -> dict[str, Any]:
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(root_path, 0)])
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    while queue:
        path, depth = queue.popleft()
        if path in visited:
            continue
        visited.add(path)
        describe = client.api_describe(path)
        node = {
            "path": path,
            "depth": int(depth),
            "id": _as_int(describe.get("id"), 0) or 0,
            "name": _as_str(describe.get("name"), ""),
            "type": _as_str(describe.get("type"), ""),
            "children": list(describe.get("children", []))
            if isinstance(describe.get("children"), list)
            else [],
            "property_count": len(describe.get("properties", []))
            if isinstance(describe.get("properties"), list)
            else 0,
            "function_count": len(describe.get("functions", []))
            if isinstance(describe.get("functions"), list)
            else 0,
        }
        nodes.append(node)

        if depth >= max_depth:
            continue
        child_keys = node["children"] if isinstance(node["children"], list) else []
        for child_key in child_keys:
            child_name = str(child_key).strip()
            if not child_name:
                continue
            children = client.api_children(path, child_name)[:max_children_per_branch]
            for child in children:
                child_path = _as_str(child.get("path"), "")
                if not child_path:
                    continue
                edges.append(
                    {
                        "from": path,
                        "child_name": child_name,
                        "to": child_path,
                        "index": _as_int(child.get("index"), 0) or 0,
                        "id": _as_int(child.get("id"), 0) or 0,
                        "name": _as_str(child.get("name"), ""),
                        "type": _as_str(child.get("type"), ""),
                    }
                )
                if child_path not in visited:
                    queue.append((child_path, depth + 1))

    return {
        "root_path": root_path,
        "max_depth": int(max_depth),
        "max_children_per_branch": int(max_children_per_branch),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def _flatten_clip_events(clips: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for clip in clips:
        notes = clip.get("notes")
        if not isinstance(notes, list):
            continue
        for event in notes:
            if isinstance(event, Mapping):
                events.append(dict(event))
    events.sort(key=lambda item: (float(item.get("start_time_global", 0.0)), int(item.get("pitch", 0))))
    return events


def _summarize_clip_counts(clips: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    summary = {"arrangement": 0, "session": 0}
    for clip in clips:
        source = str(clip.get("clip_source", "")).strip().lower()
        if source == "arrangement":
            summary["arrangement"] += 1
        elif source == "session":
            summary["session"] += 1
    return summary


def _artifact_paths(output_dir: Path, run_id: str, created_at: datetime) -> tuple[Path, Path]:
    day_dir = output_dir / created_at.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{run_id}.json", day_dir / f"{run_id}.md"


def _render_markdown_summary(artifact: Mapping[str, Any]) -> str:
    run = artifact.get("run", {}) if isinstance(artifact.get("run"), Mapping) else {}
    extraction = artifact.get("extraction", {}) if isinstance(artifact.get("extraction"), Mapping) else {}
    analysis = artifact.get("analysis", {}) if isinstance(artifact.get("analysis"), Mapping) else {}
    lines: list[str] = []
    lines.append("# Live Set Analysis")
    lines.append("")
    lines.append(f"- run_id: `{run.get('run_id', '')}`")
    lines.append(f"- created_at: `{run.get('created_at', '')}`")
    lines.append(f"- track_query: `{run.get('track_query', '')}`")
    lines.append(f"- selected_tracks: `{run.get('selected_track_count', 0)}`")
    lines.append(f"- total_clips: `{extraction.get('clip_count', 0)}`")
    lines.append(f"- total_notes: `{extraction.get('note_count', 0)}`")
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    for track_name, track_report in (analysis.get("by_track", {}) or {}).items():
        if not isinstance(track_report, Mapping):
            continue
        lines.append(f"### {track_name}")
        lines.append("")
        harmony = track_report.get("harmony", {})
        velocity = track_report.get("velocity", {})
        choreography = track_report.get("piano_choreography", {})
        if isinstance(harmony, Mapping):
            lines.append(f"- chord_onset_ratio: `{harmony.get('chord_onset_ratio', 0)}`")
            lines.append(f"- unique_pitch_class_count: `{harmony.get('unique_pitch_class_count', 0)}`")
        if isinstance(velocity, Mapping):
            overall = velocity.get("overall", {})
            if isinstance(overall, Mapping):
                lines.append(f"- velocity_mean: `{overall.get('mean', 0)}`")
                lines.append(f"- velocity_p90: `{overall.get('p90', 0)}`")
        if isinstance(choreography, Mapping):
            lines.append(f"- choreography_score: `{choreography.get('choreography_score', 0)}`")
            lines.append(f"- left_note_ratio: `{choreography.get('left_note_ratio', 0)}`")
            lines.append(f"- right_note_ratio: `{choreography.get('right_note_ratio', 0)}`")
            flags = choreography.get("flags", [])
            if isinstance(flags, list) and flags:
                lines.append(f"- flags: `{', '.join(str(flag) for flag in flags)}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _compose_artifact(
    *,
    cfg: AnalyzeConfig,
    run_id: str,
    created_at: datetime,
    live_context: Mapping[str, Any],
    topology: Mapping[str, Any],
    selected_tracks: Sequence[TrackRef],
    clips_by_track: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    analysis_by_track: dict[str, Any] = {}
    extraction_summary = {
        "clip_count": 0,
        "note_count": 0,
        "clip_source_counts": {"arrangement": 0, "session": 0},
        "by_track": {},
    }

    extraction_payload: dict[str, Any] = {}
    for track_name, clips in clips_by_track.items():
        events = _flatten_clip_events(clips)
        extraction_summary["clip_count"] += len(clips)
        extraction_summary["note_count"] += len(events)
        source_counts = _summarize_clip_counts(clips)
        extraction_summary["clip_source_counts"]["arrangement"] += source_counts["arrangement"]
        extraction_summary["clip_source_counts"]["session"] += source_counts["session"]
        extraction_summary["by_track"][track_name] = {
            "clip_count": len(clips),
            "note_count": len(events),
            "clip_source_counts": source_counts,
        }
        if cfg.include_note_events:
            extraction_payload[track_name] = {
                "clips": list(clips),
            }
        else:
            extraction_payload[track_name] = {
                "clips": [
                    {
                        key: value
                        for key, value in clip.items()
                        if key != "notes"
                    }
                    for clip in clips
                ],
            }
        analysis_by_track[track_name] = {
            "harmony": _build_harmony_metrics(events),
            "velocity": _build_velocity_metrics(events),
            "piano_choreography": _build_piano_choreography_metrics(
                events,
                split_pitch=cfg.hand_split_pitch,
            ),
        }

    return {
        "run": {
            "run_id": run_id,
            "created_at": _iso_utc(created_at),
            "track_query": cfg.track_query,
            "selected_track_count": len(selected_tracks),
            "selected_tracks": [
                {
                    "track_index": track.index,
                    "track_name": track.name,
                    "track_path": track.path,
                    "track_type": track.type,
                }
                for track in selected_tracks
            ],
            "clip_scope": cfg.clip_scope,
            "hand_split_pitch": cfg.hand_split_pitch,
        },
        "live_context": dict(live_context),
        "topology": dict(topology),
        "extraction": {
            **extraction_summary,
            "tracks": extraction_payload,
        },
        "analysis": {
            "by_track": analysis_by_track,
        },
    }


def run(cfg: AnalyzeConfig) -> int:
    created_at = _utc_now()
    run_id = _new_run_id(created_at)
    client = LiveBridgeClient(cfg)
    try:
        client.ping()
        tracks = _discover_tracks(client)
        selected_tracks = _select_tracks(tracks, cfg)
        print("info: selected tracks")
        for track in selected_tracks:
            print(f"  - [{track.index}] {track.name} ({track.path})")
        if cfg.dry_run:
            return 0

        tempo = _as_float(client.api_get("live_set", "tempo"), fallback=None)
        sig_num = _as_int(client.api_get("live_set", "signature_numerator"), fallback=None)
        sig_den = _as_int(client.api_get("live_set", "signature_denominator"), fallback=None)
        root_note = _as_int(client.api_get("live_set", "root_note"), fallback=None)
        scale_name = _as_str(client.api_get("live_set", "scale_name"), fallback="")

        topology = _build_topology_snapshot(
            client=client,
            root_path="live_set",
            max_depth=cfg.topology_depth,
            max_children_per_branch=cfg.topology_max_children,
        )

        clips_by_track: dict[str, list[dict[str, Any]]] = {}
        for track in selected_tracks:
            clips: list[dict[str, Any]] = []
            if cfg.clip_scope in {"arrangement", "both"}:
                clips.extend(_extract_arrangement_clips(client=client, track=track, cfg=cfg))
            if cfg.clip_scope in {"session", "both"}:
                clips.extend(_extract_session_clips(client=client, track=track, cfg=cfg))
            clips_by_track[track.name] = clips

        live_context = {
            "tempo": tempo,
            "signature_numerator": sig_num,
            "signature_denominator": sig_den,
            "root_note": root_note,
            "scale_name": scale_name,
        }
        artifact = _compose_artifact(
            cfg=cfg,
            run_id=run_id,
            created_at=created_at,
            live_context=live_context,
            topology=topology,
            selected_tracks=selected_tracks,
            clips_by_track=clips_by_track,
        )
        json_path, md_path = _artifact_paths(cfg.output_dir, run_id, created_at)
        json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"info: analysis artifact written to {json_path}")
        if cfg.write_markdown:
            md_path.write_text(_render_markdown_summary(artifact), encoding="utf-8")
            print(f"info: analysis summary written to {md_path}")
        return 0
    finally:
        client.close()


def main(argv: Iterable[str]) -> int:
    cfg = parse_args(argv)
    try:
        return run(cfg)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
