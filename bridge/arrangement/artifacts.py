from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import compose_kick_pattern as kick
from arrangement.base import (
    DEFAULT_ARCHIVE_DIR,
    DEFAULT_COMPOSITION_PRINT_DIR,
    Section,
    _clamp_ratio,
    _repo_root,
    _req_id,
)


def _utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def _resolve_composition_print_dir(raw_path: str | None) -> Path:
    if raw_path in (None, ""):
        return _repo_root() / DEFAULT_COMPOSITION_PRINT_DIR
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return _repo_root() / path

def _serialize_section(section: Section) -> dict[str, Any]:
    return {
        "index": int(section.index),
        "start_bar": int(section.start_bar),
        "bar_count": int(section.bar_count),
        "label": str(section.label),
        "kick_on": bool(section.kick_on),
        "rim_on": bool(section.rim_on),
        "hat_on": bool(section.hat_on),
        "piano_mode": str(section.piano_mode),
        "kick_keep_ratio": float(section.kick_keep_ratio),
        "rim_keep_ratio": float(section.rim_keep_ratio),
        "hat_keep_ratio": float(section.hat_keep_ratio),
        "hat_density": str(section.hat_density),
    }

def _deserialize_section(payload: Mapping[str, Any]) -> Section:
    return Section(
        index=int(payload.get("index", 0)),
        start_bar=int(payload.get("start_bar", 0)),
        bar_count=max(1, int(payload.get("bar_count", 1))),
        label=str(payload.get("label", "unknown")),
        kick_on=bool(payload.get("kick_on", False)),
        rim_on=bool(payload.get("rim_on", False)),
        hat_on=bool(payload.get("hat_on", False)),
        piano_mode=str(payload.get("piano_mode", "chords")),  # type: ignore[arg-type]
        kick_keep_ratio=_clamp_ratio(float(payload.get("kick_keep_ratio", 0.0))),
        rim_keep_ratio=_clamp_ratio(float(payload.get("rim_keep_ratio", 0.0))),
        hat_keep_ratio=_clamp_ratio(float(payload.get("hat_keep_ratio", 0.0))),
        hat_density=str(payload.get("hat_density", "quarter")),  # type: ignore[arg-type]
    )

def _serialize_arranged_by_track(
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for track_name, section_payloads in arranged_by_track.items():
        entries: list[dict[str, Any]] = []
        for section, notes in section_payloads:
            entries.append(
                {
                    "section_index": int(section.index),
                    "notes": list(notes),
                }
            )
        payload[str(track_name)] = entries
    return payload

def _deserialize_arranged_by_track(
    payload: Mapping[str, Any],
    sections: Sequence[Section],
) -> dict[str, List[tuple[Section, List[dict[str, Any]]]]]:
    section_by_index = {int(section.index): section for section in sections}
    ordered_indices = [int(section.index) for section in sections]
    arranged: dict[str, List[tuple[Section, List[dict[str, Any]]]]] = {}
    for track_name, entries_raw in payload.items():
        if not isinstance(entries_raw, list):
            raise ValueError(f"invalid arranged_by_track entry for '{track_name}'")
        notes_by_index: dict[int, List[dict[str, Any]]] = {idx: [] for idx in ordered_indices}
        for entry in entries_raw:
            if not isinstance(entry, dict):
                continue
            section_index = int(entry.get("section_index", -1))
            if section_index not in section_by_index:
                raise ValueError(
                    f"composition print references unknown section index {section_index} for '{track_name}'"
                )
            notes_raw = entry.get("notes", [])
            if not isinstance(notes_raw, list):
                raise ValueError(
                    f"composition print section notes are not a list for '{track_name}' section {section_index}"
                )
            notes_clean: List[dict[str, Any]] = []
            for note in notes_raw:
                if isinstance(note, dict):
                    notes_clean.append(dict(note))
            notes_by_index[section_index] = notes_clean
        arranged[str(track_name)] = [
            (section_by_index[idx], list(notes_by_index.get(idx, []))) for idx in ordered_indices
        ]
    return arranged

def _persist_composition_print(
    *,
    run_label: str,
    mood: str,
    key_name: str,
    bpm: float,
    sig_num: int,
    sig_den: int,
    minutes: float,
    bars: int,
    section_bars: int,
    start_beats: float,
    section_profile_family: str | None,
    registry_path: Path,
    track_naming_mode: str,
    sections: Sequence[Section],
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
    output_dir: Path,
    source_print_path: str | None = None,
    multi_pass_report: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = output_dir / date_slug / f"{run_label}-{_utc_timestamp_slug()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_label": str(run_label),
        "source_print_path": source_print_path,
        "composition": {
            "mood": str(mood),
            "key_name": str(key_name),
            "bpm": float(bpm),
            "sig_num": int(sig_num),
            "sig_den": int(sig_den),
            "minutes": float(minutes),
            "bars": int(bars),
            "section_bars": int(section_bars),
            "start_beats": float(start_beats),
        },
        "section_profile_family": str(section_profile_family or "").strip() or "legacy_arc",
        "registry_path": str(registry_path),
        "track_naming_mode": str(track_naming_mode),
        "sections": [_serialize_section(section) for section in sections],
        "arranged_by_track": _serialize_arranged_by_track(arranged_by_track),
        "multi_pass_report": [dict(entry) for entry in (multi_pass_report or ())],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out_path

def _load_composition_print(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"composition print not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid composition print payload: {path}")
    sections_raw = raw.get("sections")
    arranged_raw = raw.get("arranged_by_track")
    if not isinstance(sections_raw, list) or not sections_raw:
        raise ValueError(f"composition print missing sections: {path}")
    if not isinstance(arranged_raw, dict):
        raise ValueError(f"composition print missing arranged_by_track: {path}")
    sections = [_deserialize_section(section_raw) for section_raw in sections_raw if isinstance(section_raw, dict)]
    sections.sort(key=lambda section: int(section.index))
    arranged_by_track = _deserialize_arranged_by_track(arranged_raw, sections)
    composition = raw.get("composition")
    if not isinstance(composition, dict):
        composition = {}
    return {
        "run_label": str(raw.get("run_label", "")),
        "composition": composition,
        "section_profile_family": str(raw.get("section_profile_family", "")).strip() or "legacy_arc",
        "sections": sections,
        "arranged_by_track": arranged_by_track,
        "source_path": str(path),
    }

def _resolve_archive_dir(raw_path: str | None) -> Path:
    if raw_path in (None, ""):
        return _repo_root() / DEFAULT_ARCHIVE_DIR
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return _repo_root() / path

def _archive_live_set(
    sock: socket.socket,
    ack_sock: socket.socket,
    timeout_s: float,
    archive_path: Path,
) -> tuple[bool, str]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_calls = [
        ("save", [str(archive_path)]),
        ("save_as", [str(archive_path)]),
        ("save", []),
    ]
    last_error = "no save method succeeded"
    for method_name, args in candidate_calls:
        try:
            kick._api_call(
                sock,
                ack_sock,
                "live_set",
                method_name,
                args,
                _req_id("arr-save", method_name),
                max(2.0, timeout_s),
            )
        except Exception as exc:  # noqa: BLE001 - keep trying candidate save methods
            last_error = f"{method_name} failed: {exc}"
            continue
        if archive_path.exists():
            return True, f"saved via {method_name}"
        if method_name == "save":
            # If Live saved the currently loaded set instead of explicit path,
            # copy that saved set into the requested archive location when possible.
            path_value = kick._api_get(
                sock,
                ack_sock,
                "live_set",
                "path",
                _req_id("arr-save-path"),
                timeout_s,
            )
            saved_path = str(kick._scalar(path_value) or "").strip()
            if saved_path:
                src = Path(saved_path)
                if src.exists():
                    try:
                        archive_path.write_bytes(src.read_bytes())
                        return True, "saved then copied from live_set.path"
                    except Exception as exc:  # noqa: BLE001
                        last_error = f"save copy failed: {exc}"
                        continue
        last_error = f"{method_name} did not create expected archive path"
    ui_ok, ui_message = _try_ui_save_live_set(archive_path)
    if ui_ok and archive_path.exists():
        return True, ui_message
    if ui_ok:
        return True, f"{ui_message}; archive path resolved indirectly"
    last_error = f"{last_error}; {ui_message}"
    return False, last_error


def _detect_running_live_app_name() -> str:
    default = "Ableton Live 12 Suite"
    try:
        probe = subprocess.run(
            ["ps", "-A", "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return default
    for raw_line in probe.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.search(r"/([^/]+)\.app/Contents/MacOS/Live(?:\s|$)", line)
        if match:
            return match.group(1).strip() or default
    return default


def _locate_ui_saved_als(out_dir: Path, base_name: str) -> Path | None:
    candidates = (
        out_dir / f"{base_name}.als",
        out_dir / f"{base_name} Project" / f"{base_name}.als",
        out_dir / f"{base_name}.als Project" / f"{base_name}.als.als",
        out_dir / f"{base_name}.als Project" / f"{base_name}.als",
        out_dir / f"{base_name} Project" / f"{base_name}.als.als",
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    wildcard_candidates = sorted(
        out_dir.glob(f"{base_name}* Project/{base_name}*.als*"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for candidate in wildcard_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _try_ui_save_live_set(archive_path: Path) -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "ui save fallback unavailable (non-macOS)"

    out_dir = archive_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = archive_path.stem if archive_path.suffix.lower() == ".als" else archive_path.name
    app_name = _detect_running_live_app_name()
    applescript = (
        "on run argv\n"
        "set appName to item 1 of argv\n"
        "set outDir to item 2 of argv\n"
        "set fileName to item 3 of argv\n"
        "tell application appName to activate\n"
        "delay 0.4\n"
        "tell application \"System Events\"\n"
        "keystroke \"S\" using {command down, shift down}\n"
        "delay 0.7\n"
        "keystroke \"G\" using {command down, shift down}\n"
        "delay 0.3\n"
        "keystroke outDir\n"
        "key code 36\n"
        "delay 0.5\n"
        "keystroke fileName\n"
        "key code 36\n"
        "end tell\n"
        "end run\n"
    )
    try:
        subprocess.run(
            ["osascript", "-e", applescript, app_name, str(out_dir), base_name],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"ui save fallback failed to execute: {exc}"

    deadline = time.time() + 6.0
    resolved: Path | None = None
    while time.time() < deadline:
        resolved = _locate_ui_saved_als(out_dir, base_name)
        if resolved is not None:
            break
        time.sleep(0.3)
    if resolved is None:
        return False, "ui save fallback did not produce a detectable .als file"

    if resolved != archive_path:
        try:
            archive_path.write_bytes(resolved.read_bytes())
        except Exception as exc:  # noqa: BLE001
            return False, f"ui save fallback copy failed: {exc}"
    return True, f"saved via ui fallback ({resolved})"
