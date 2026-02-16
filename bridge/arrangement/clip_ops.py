from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import List, Sequence

import compose_kick_pattern as kick
from arrangement.base import (
    DEFAULT_NOTE_CHUNK_SIZE,
    DEFAULT_WRITE_CACHE_PATH,
    ArrangementClipRef,
    NoteDelta,
    Section,
    _repo_root,
    _req_id,
)


def _resolve_cache_path(raw_path: str | None) -> Path:
    if raw_path in (None, ""):
        return _repo_root() / DEFAULT_WRITE_CACHE_PATH
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return _repo_root() / path

def _normalize_note(note: dict) -> dict:
    return {
        "pitch": int(note.get("pitch", 0)),
        "start_time": float(round(float(note.get("start_time", 0.0)), 6)),
        "duration": float(round(float(note.get("duration", 0.0)), 6)),
        "velocity": int(note.get("velocity", 100)),
        "mute": int(note.get("mute", 0)),
    }

def _note_position_key(note: dict) -> tuple[int, float, float]:
    return (
        int(note.get("pitch", 0)),
        float(note.get("start_time", 0.0)),
        float(note.get("duration", 0.0)),
    )

def _stable_notes_payload(notes: Sequence[dict]) -> List[dict]:
    normalized = [_normalize_note(dict(note)) for note in notes]
    normalized.sort(
        key=lambda n: (
            int(n["pitch"]),
            float(n["start_time"]),
            float(n["duration"]),
            int(n["velocity"]),
            int(n["mute"]),
        )
    )
    return normalized

def _notes_payload_hash(notes: Sequence[dict]) -> str:
    payload = _stable_notes_payload(notes)
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _load_write_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    entries = raw.get("entries") if isinstance(raw, dict) else None
    if isinstance(entries, dict):
        loaded: dict[str, str] = {}
        for key, value in entries.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, dict):
                payload_hash = value.get("payload_hash")
                if isinstance(payload_hash, str) and payload_hash:
                    loaded[key] = payload_hash
            elif isinstance(value, str) and value:
                loaded[key] = value
        return loaded

    if isinstance(raw, dict):
        # Backward-compatible fallback for simple {cache_key: hash} stores.
        loaded = {}
        for key, value in raw.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                loaded[key] = value
        return loaded
    return {}

def _save_write_cache(path: Path, entries: dict[str, str]) -> None:
    payload = {
        "version": 1,
        "entries": {
            key: {"payload_hash": value}
            for key, value in sorted(entries.items(), key=lambda item: item[0])
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def _live_set_identity(
    sock: socket.socket,
    ack_sock: socket.socket,
    timeout_s: float,
) -> str:
    describe = kick._api_describe(
        sock,
        ack_sock,
        "live_set",
        _req_id("arr-live-set-describe"),
        timeout_s,
    )
    live_set_id = 0
    if isinstance(describe, dict):
        parsed = kick._as_int(describe.get("id"))
        if parsed is not None and parsed > 0:
            live_set_id = parsed

    name_value = kick._api_get(
        sock,
        ack_sock,
        "live_set",
        "name",
        _req_id("arr-live-set-name"),
        timeout_s,
    )
    name_scalar = kick._scalar(name_value)
    name_text = "" if name_scalar is None else str(name_scalar).strip()
    path_value = kick._api_get(
        sock,
        ack_sock,
        "live_set",
        "path",
        _req_id("arr-live-set-path"),
        timeout_s,
    )
    path_scalar = kick._scalar(path_value)
    path_text = "" if path_scalar is None else str(path_scalar).strip()
    if not path_text and isinstance(describe, dict):
        desc_path = describe.get("path")
        if isinstance(desc_path, str):
            path_text = desc_path.strip()

    tracks = kick._get_children(
        sock,
        ack_sock,
        "live_set",
        "tracks",
        _req_id("arr-live-set-tracks"),
        timeout_s,
    )
    normalized_names = []
    for item in tracks:
        raw_name = str(item.get("name", "")).strip().lower()
        if raw_name:
            normalized_names.append(raw_name)
    names_text = "|".join(sorted(normalized_names))
    track_sig = hashlib.sha256(names_text.encode("utf-8")).hexdigest()[:12] if names_text else "none"
    track_count = len(tracks)

    safe_name = name_text.replace("|", "/")
    safe_path = path_text.replace("|", "/")
    untitled = int(not name_text and not path_text)
    return (
        f"id:{live_set_id}|name:{safe_name}|path:{safe_path}|tracks:{track_count}|"
        f"track_sig:{track_sig}|untitled:{untitled}"
    )

def _write_cache_key(
    *,
    live_set_identity: str,
    track_path: str,
    section_start_beats: float,
    section_length_beats: float,
    bpm: float,
    sig_num: int,
    sig_den: int,
) -> str:
    return (
        f"{live_set_identity}|{track_path}|start:{section_start_beats:.6f}|"
        f"len:{section_length_beats:.6f}|sig:{sig_num}/{sig_den}|bpm:{float(bpm):g}"
    )

def _list_arrangement_clips(
    sock: socket.socket,
    ack_sock: socket.socket,
    track_path: str,
    timeout_s: float,
    request_prefix: str,
) -> List[ArrangementClipRef]:
    children = kick._get_children(
        sock,
        ack_sock,
        track_path,
        "arrangement_clips",
        _req_id(request_prefix, track_path),
        timeout_s,
    )

    refs: List[ArrangementClipRef] = []
    for idx, clip_info in enumerate(children):
        raw_path = clip_info.get("path")
        if not raw_path:
            continue
        clip_path = kick._sanitize_live_path(str(raw_path))
        start_value = kick._api_get(
            sock,
            ack_sock,
            clip_path,
            "start_time",
            _req_id(request_prefix, track_path, "start", idx),
            timeout_s,
        )
        end_value = kick._api_get(
            sock,
            ack_sock,
            clip_path,
            "end_time",
            _req_id(request_prefix, track_path, "end", idx),
            timeout_s,
        )
        start_time = kick._as_float(start_value)
        end_time = kick._as_float(end_value)
        if start_time is None or end_time is None:
            continue

        clip_id = kick._as_int(clip_info.get("id"))
        if clip_id is None or clip_id <= 0:
            clip_id = kick._as_int(
                kick._api_get(
                    sock,
                    ack_sock,
                    clip_path,
                    "id",
                    _req_id(request_prefix, track_path, "id", idx),
                    timeout_s,
                )
            )
        if clip_id is None or clip_id <= 0:
            clip_desc = kick._api_describe(
                sock,
                ack_sock,
                clip_path,
                _req_id(request_prefix, track_path, "describe", idx),
                timeout_s,
            )
            if isinstance(clip_desc, dict):
                clip_id = kick._as_int(clip_desc.get("id"))
        refs.append(
            ArrangementClipRef(
                path=clip_path,
                clip_id=clip_id if clip_id and clip_id > 0 else None,
                start_time=float(start_time),
                end_time=float(end_time),
            )
        )
    return refs

def _find_matching_arrangement_clip(
    refs: Sequence[ArrangementClipRef],
    section_start_beats: float,
    section_length_beats: float,
) -> ArrangementClipRef | None:
    expected_end = float(section_start_beats) + float(section_length_beats)
    epsilon = 1e-3
    matches: List[ArrangementClipRef] = []
    for ref in refs:
        if abs(ref.start_time - float(section_start_beats)) > epsilon:
            continue
        if abs(ref.end_time - expected_end) > epsilon:
            continue
        matches.append(ref)
    if not matches:
        return None
    matches.sort(
        key=lambda ref: (
            0 if ref.clip_id is not None else 1,
            -(int(ref.clip_id) if ref.clip_id is not None else 0),
        )
    )
    return matches[0]

def _delete_overlaps_from_refs(
    sock: socket.socket,
    ack_sock: socket.socket,
    track_path: str,
    refs: Sequence[ArrangementClipRef],
    start: float,
    end: float,
    timeout_s: float,
    preserve_clip_id: int | None,
    request_prefix: str,
) -> tuple[int, List[ArrangementClipRef]]:
    to_delete_ids: List[int] = []
    retained: List[ArrangementClipRef] = []
    for ref in refs:
        if preserve_clip_id is not None and ref.clip_id == preserve_clip_id:
            retained.append(ref)
            continue
        if not kick._overlaps(ref.start_time, ref.end_time, start, end):
            retained.append(ref)
            continue
        if ref.clip_id is None:
            retained.append(ref)
            continue
        to_delete_ids.append(int(ref.clip_id))

    deleted_ids = sorted(set(to_delete_ids))
    for delete_index, clip_id in enumerate(deleted_ids):
        kick._api_call(
            sock,
            ack_sock,
            track_path,
            "delete_clip",
            [clip_id],
            _req_id(request_prefix, track_path, delete_index),
            timeout_s,
        )

    return len(deleted_ids), retained

def _upsert_clip_ref(
    refs: Sequence[ArrangementClipRef],
    new_ref: ArrangementClipRef,
) -> List[ArrangementClipRef]:
    expected_start = float(new_ref.start_time)
    expected_end = float(new_ref.end_time)
    epsilon = 1e-3
    next_refs = [
        ref
        for ref in refs
        if not (
            abs(ref.start_time - expected_start) <= epsilon
            and abs(ref.end_time - expected_end) <= epsilon
        )
    ]
    next_refs.append(new_ref)
    return next_refs

def _delete_overlaps_except(
    sock: socket.socket,
    ack_sock: socket.socket,
    track_path: str,
    start: float,
    end: float,
    timeout_s: float,
    preserve_clip_id: int | None,
) -> int:
    refs = _list_arrangement_clips(sock, ack_sock, track_path, timeout_s, "arr-overlaps")
    deleted, _retained = _delete_overlaps_from_refs(
        sock,
        ack_sock,
        track_path,
        refs,
        start,
        end,
        timeout_s,
        preserve_clip_id=preserve_clip_id,
        request_prefix="arr-delete-overlap",
    )
    return deleted

def _extract_notes_from_result(result: object | None) -> List[dict] | None:
    if isinstance(result, dict):
        notes = result.get("notes")
        if isinstance(notes, list):
            return [dict(note) for note in notes if isinstance(note, dict)]
        return None
    if isinstance(result, list):
        if all(isinstance(item, dict) for item in result):
            return [dict(item) for item in result]
        for item in result:
            if isinstance(item, dict) and isinstance(item.get("notes"), list):
                return [dict(note) for note in item["notes"] if isinstance(note, dict)]
    return None

def _compute_note_delta(
    existing_notes: Sequence[dict],
    target_notes: Sequence[dict],
) -> NoteDelta:
    existing_by_pos: dict[tuple[int, float, float], List[dict]] = {}
    for existing in existing_notes:
        normalized = _normalize_note(existing)
        note_with_id = dict(normalized)
        note_with_id["note_id"] = kick._as_int(existing.get("note_id"))
        existing_by_pos.setdefault(_note_position_key(normalized), []).append(note_with_id)

    target_by_pos: dict[tuple[int, float, float], List[dict]] = {}
    for target in target_notes:
        normalized = _normalize_note(target)
        target_by_pos.setdefault(_note_position_key(normalized), []).append(normalized)

    remove_note_ids: List[int] = []
    modification_notes: List[dict] = []
    add_notes: List[dict] = []
    requires_full_replace = False

    all_positions = set(existing_by_pos.keys()) | set(target_by_pos.keys())
    for pos in sorted(all_positions):
        existing_group = existing_by_pos.get(pos, [])
        target_group = target_by_pos.get(pos, [])

        existing_group.sort(
            key=lambda note: (
                int(note.get("velocity", 100)),
                int(note.get("mute", 0)),
                int(note.get("note_id") or 0),
            )
        )
        target_group.sort(
            key=lambda note: (
                int(note.get("velocity", 100)),
                int(note.get("mute", 0)),
            )
        )

        shared = min(len(existing_group), len(target_group))
        for idx in range(shared):
            existing = existing_group[idx]
            target = target_group[idx]
            if (
                int(existing.get("velocity", 100)) == int(target.get("velocity", 100))
                and int(existing.get("mute", 0)) == int(target.get("mute", 0))
            ):
                continue
            note_id = kick._as_int(existing.get("note_id"))
            if note_id is None or note_id <= 0:
                requires_full_replace = True
                continue
            modification_notes.append(
                {
                    "note_id": int(note_id),
                    "pitch": int(target["pitch"]),
                    "start_time": float(target["start_time"]),
                    "duration": float(target["duration"]),
                    "velocity": int(target["velocity"]),
                    "mute": int(target["mute"]),
                }
            )

        if len(existing_group) > shared:
            for existing in existing_group[shared:]:
                note_id = kick._as_int(existing.get("note_id"))
                if note_id is None or note_id <= 0:
                    requires_full_replace = True
                    continue
                remove_note_ids.append(int(note_id))

        if len(target_group) > shared:
            for target in target_group[shared:]:
                add_notes.append(dict(target))

    return NoteDelta(
        remove_note_ids=tuple(sorted(set(remove_note_ids))),
        modification_notes=tuple(modification_notes),
        add_notes=tuple(add_notes),
        requires_full_replace=requires_full_replace,
    )

def _write_add_new_notes(
    sock: socket.socket,
    ack_sock: socket.socket,
    clip_path: str,
    track_path: str,
    section: Section,
    notes: Sequence[dict],
    timeout_s: float,
    note_chunk_size: int,
) -> None:
    chunk_size = int(note_chunk_size)
    # Keep explicit user overrides stable; adapt only when using the default.
    if chunk_size == DEFAULT_NOTE_CHUNK_SIZE:
        if len(notes) >= 2000:
            chunk_size = 120
        elif len(notes) >= 1000:
            chunk_size = 80
    note_chunks = kick._chunk_notes(notes, chunk_size=chunk_size)
    for idx, chunk in enumerate(note_chunks, start=1):
        kick._api_call(
            sock,
            ack_sock,
            clip_path,
            "add_new_notes",
            {"notes": chunk},
            _req_id("arr-add-notes", track_path, section.index, f"{idx}-of-{len(note_chunks)}"),
            max(timeout_s, 2.0),
        )

def _apply_delta_write(
    sock: socket.socket,
    ack_sock: socket.socket,
    clip_path: str,
    track_path: str,
    section: Section,
    section_length_beats: float,
    notes: Sequence[dict],
    timeout_s: float,
    note_chunk_size: int,
) -> bool:
    note_dump = kick._api_call(
        sock,
        ack_sock,
        clip_path,
        "get_notes_extended",
        {
            "from_pitch": 0,
            "pitch_span": 128,
            "from_time": 0.0,
            "time_span": float(section_length_beats),
            "return": ["note_id", "pitch", "start_time", "duration", "velocity", "mute"],
        },
        _req_id("arr-get-notes", track_path, section.index),
        max(timeout_s, 2.0),
    )
    existing_notes = _extract_notes_from_result(note_dump)
    if existing_notes is None:
        return False

    delta = _compute_note_delta(existing_notes, notes)
    if delta.requires_full_replace:
        return False

    if delta.remove_note_ids:
        id_chunks = [list(chunk) for chunk in kick._chunk_notes(delta.remove_note_ids, chunk_size=128)]
        for idx, note_ids in enumerate(id_chunks, start=1):
            kick._api_call(
                sock,
                ack_sock,
                clip_path,
                "remove_notes_extended",
                {"note_ids": note_ids},
                _req_id("arr-remove-notes", track_path, section.index, f"{idx}-of-{len(id_chunks)}"),
                max(timeout_s, 2.0),
            )

    if delta.modification_notes:
        mod_chunks = kick._chunk_notes(delta.modification_notes, chunk_size=note_chunk_size)
        for idx, chunk in enumerate(mod_chunks, start=1):
            kick._api_call(
                sock,
                ack_sock,
                clip_path,
                "apply_note_modifications",
                {"notes": chunk},
                _req_id("arr-mod-notes", track_path, section.index, f"{idx}-of-{len(mod_chunks)}"),
                max(timeout_s, 2.0),
            )

    if delta.add_notes:
        _write_add_new_notes(
            sock,
            ack_sock,
            clip_path,
            track_path,
            section,
            delta.add_notes,
            timeout_s,
            note_chunk_size,
        )

    return True

def _delete_overlaps(
    sock: socket.socket,
    ack_sock: socket.socket,
    track_path: str,
    start: float,
    end: float,
    timeout_s: float,
) -> int:
    refs = _list_arrangement_clips(sock, ack_sock, track_path, timeout_s, "arr-initial")
    deleted, _retained = _delete_overlaps_from_refs(
        sock,
        ack_sock,
        track_path,
        refs,
        start,
        end,
        timeout_s,
        preserve_clip_id=None,
        request_prefix="arr-delete-clip",
    )
    return deleted

def _create_section_clip(
    sock: socket.socket,
    ack_sock: socket.socket,
    track_path: str,
    section: Section,
    section_start_beats: float,
    section_length_beats: float,
    clip_name: str,
    sig_num: int,
    sig_den: int,
    notes: Sequence[dict],
    timeout_s: float,
    note_chunk_size: int,
    groove_id: int | None,
    apply_groove: bool,
) -> ArrangementClipRef | None:
    arrangement_before = kick._get_children(
        sock,
        ack_sock,
        track_path,
        "arrangement_clips",
        _req_id("arr-before", track_path, section.index),
        timeout_s,
    )

    create_result = kick._api_call(
        sock,
        ack_sock,
        track_path,
        "create_midi_clip",
        [section_start_beats, section_length_beats],
        _req_id("arr-create-clip", track_path, section.index),
        timeout_s,
    )

    clip_id = kick._extract_id_from_call_result(create_result)
    if clip_id is not None and clip_id > 0:
        clip_path = f"id {clip_id}"
    else:
        arrangement_after = kick._get_children(
            sock,
            ack_sock,
            track_path,
            "arrangement_clips",
            _req_id("arr-after", track_path, section.index),
            timeout_s,
        )
        clip_path = kick._new_child_path(arrangement_before, arrangement_after)

    if not clip_path:
        print(
            "error: could not identify new arrangement clip; reload the device in Live",
            file=sys.stderr,
        )
        return None

    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "name",
        clip_name,
        _req_id("arr-clip-name", track_path, section.index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "signature_numerator",
        int(sig_num),
        _req_id("arr-clip-sig-num", track_path, section.index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "signature_denominator",
        int(sig_den),
        _req_id("arr-clip-sig-den", track_path, section.index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "loop_start",
        0.0,
        _req_id("arr-loop-start", track_path, section.index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "loop_end",
        section_length_beats,
        _req_id("arr-loop-end", track_path, section.index),
        timeout_s,
    )

    if notes:
        _write_add_new_notes(
            sock,
            ack_sock,
            clip_path,
            track_path,
            section,
            notes,
            timeout_s,
            note_chunk_size,
        )

    if apply_groove and groove_id:
        kick._api_set(
            sock,
            ack_sock,
            clip_path,
            "groove",
            ["id", groove_id],
            _req_id("arr-clip-groove", track_path, section.index),
            timeout_s,
        )

    created_clip_id: int | None = clip_id if clip_id is not None and clip_id > 0 else None
    if created_clip_id is None:
        created_clip_id = kick._as_int(
            kick._api_get(
                sock,
                ack_sock,
                clip_path,
                "id",
                _req_id("arr-created-clip-id", track_path, section.index),
                timeout_s,
            )
        )
    if created_clip_id is None or created_clip_id <= 0:
        clip_desc = kick._api_describe(
            sock,
            ack_sock,
            clip_path,
            _req_id("arr-created-clip-describe", track_path, section.index),
            timeout_s,
        )
        if isinstance(clip_desc, dict):
            created_clip_id = kick._as_int(clip_desc.get("id"))

    return ArrangementClipRef(
        path=clip_path,
        clip_id=created_clip_id if created_clip_id and created_clip_id > 0 else None,
        start_time=float(section_start_beats),
        end_time=float(section_start_beats) + float(section_length_beats),
    )

def _update_existing_section_clip_delta(
    sock: socket.socket,
    ack_sock: socket.socket,
    clip_path: str,
    track_path: str,
    section: Section,
    section_length_beats: float,
    clip_name: str,
    sig_num: int,
    sig_den: int,
    notes: Sequence[dict],
    timeout_s: float,
    note_chunk_size: int,
    groove_id: int | None,
    apply_groove: bool,
) -> bool:
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "name",
        clip_name,
        _req_id("arr-clip-name", track_path, section.index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "signature_numerator",
        int(sig_num),
        _req_id("arr-clip-sig-num", track_path, section.index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "signature_denominator",
        int(sig_den),
        _req_id("arr-clip-sig-den", track_path, section.index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "loop_start",
        0.0,
        _req_id("arr-loop-start", track_path, section.index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "loop_end",
        section_length_beats,
        _req_id("arr-loop-end", track_path, section.index),
        timeout_s,
    )

    if not notes:
        kick._api_call(
            sock,
            ack_sock,
            clip_path,
            "remove_notes_extended",
            {
                "from_pitch": 0,
                "pitch_span": 128,
                "from_time": 0.0,
                "time_span": float(section_length_beats),
            },
            _req_id("arr-clear-notes", track_path, section.index),
            max(timeout_s, 2.0),
        )
    else:
        applied = _apply_delta_write(
            sock,
            ack_sock,
            clip_path,
            track_path,
            section,
            section_length_beats,
            notes,
            timeout_s,
            note_chunk_size,
        )
        if not applied:
            return False

    if apply_groove and groove_id:
        kick._api_set(
            sock,
            ack_sock,
            clip_path,
            "groove",
            ["id", groove_id],
            _req_id("arr-clip-groove", track_path, section.index),
            timeout_s,
        )

    return True
