#!/usr/bin/env python3
"""Composition eval artifacts: cache runs, measure novelty, and prompt reflection."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_RELATIVE_LOG_DIR = Path("memory/evals/compositions")
DEFAULT_RELATIVE_INDEX_PATH = Path("memory/evals/composition_index.json")
MAX_INDEX_ENTRIES = 300


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso_utc(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _safe_bpm_text(value: float) -> str:
    return f"{float(value):g}".replace(".", "_")


def _safe_token(value: str, fallback: str = "untitled") -> str:
    raw = str(value).strip().lower()
    if not raw:
        return fallback
    token = re.sub(r"[^a-z0-9]+", "_", raw)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or fallback


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo_root(path: Path | None = None) -> Path:
    if path is not None:
        return path
    return Path(__file__).resolve().parents[1]


def _normalize_seq(values: Sequence[Any]) -> list[str]:
    return [str(v) for v in values]


def _sequence_similarity(left: Sequence[Any], right: Sequence[Any]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    max_len = max(len(left), len(right))
    min_len = min(len(left), len(right))
    matches = sum(1 for idx in range(min_len) if left[idx] == right[idx])
    return matches / float(max_len)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    size = max(len(left), len(right))
    if size == 0:
        return 1.0

    lvals = [0.0] * size
    rvals = [0.0] * size
    for idx, value in enumerate(left):
        lvals[idx] = float(value)
    for idx, value in enumerate(right):
        rvals[idx] = float(value)

    lmag = math.sqrt(sum(v * v for v in lvals))
    rmag = math.sqrt(sum(v * v for v in rvals))
    if lmag == 0.0 and rmag == 0.0:
        return 1.0
    if lmag == 0.0 or rmag == 0.0:
        return 0.0

    dot = sum(a * b for a, b in zip(lvals, rvals))
    return max(0.0, min(1.0, dot / (lmag * rmag)))


def _section_count_paths(
    section_count: int,
    arranged_by_track: Mapping[str, Sequence[tuple[Any, Sequence[dict[str, Any]]]]],
) -> dict[str, list[int]]:
    paths: dict[str, list[int]] = {}
    for track_name, payloads in arranged_by_track.items():
        counts = [0] * section_count
        for section, notes in payloads:
            index = int(getattr(section, "index", -1))
            if 0 <= index < section_count:
                counts[index] = len(notes)
        paths[str(track_name)] = counts
    return paths


def _track_totals(paths: Mapping[str, Sequence[int]]) -> dict[str, int]:
    return {track: int(sum(counts)) for track, counts in paths.items()}


def _find_track_key(mapping: Mapping[str, Any], wanted: str) -> str | None:
    target = str(wanted).strip().lower()
    if not target:
        return None
    for key in mapping.keys():
        if str(key).strip().lower() == target:
            return str(key)
    return None


def _flatten_track_notes(
    arranged_by_track: Mapping[str, Sequence[tuple[Any, Sequence[dict[str, Any]]]]],
    track_name: str,
) -> list[dict[str, float | int]]:
    payloads = arranged_by_track.get(track_name, ())
    events: list[dict[str, float | int]] = []
    for section, notes in payloads:
        section_idx = int(getattr(section, "index", 0))
        for note in notes:
            events.append(
                {
                    "section_index": section_idx,
                    "start_time": float(note.get("start_time", 0.0)),
                    "duration": float(note.get("duration", 0.0)),
                    "pitch": int(note.get("pitch", 0)),
                    "velocity": int(note.get("velocity", 0)),
                }
            )
    events.sort(key=lambda n: (int(n["section_index"]), float(n["start_time"]), int(n["pitch"])))
    return events


def _compute_instrument_identity_metrics(
    arranged_by_track: Mapping[str, Sequence[tuple[Any, Sequence[dict[str, Any]]]]],
    contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        return {"enabled": False, "status": "no_contract"}

    marimba_name = str(contract.get("track_name", "Marimba"))
    pair_name = str(contract.get("pair_track_name", "Vibraphone"))
    marimba_key = _find_track_key(arranged_by_track, marimba_name)
    pair_key = _find_track_key(arranged_by_track, pair_name)
    if marimba_key is None:
        return {"enabled": False, "status": "marimba_track_missing", "track_name": marimba_name}

    constraints = contract.get("constraints", {})
    if not isinstance(constraints, Mapping):
        constraints = {}
    pair_rules = contract.get("pair_rules", {})
    if not isinstance(pair_rules, Mapping):
        pair_rules = {}
    attack_rule = pair_rules.get("attack_answer", {})
    if not isinstance(attack_rule, Mapping):
        attack_rule = {}

    midi_min = int(constraints.get("midi_min_pitch", 0) or 0)
    midi_max = int(constraints.get("midi_max_pitch", 127) or 127)
    max_leap = max(1, int(constraints.get("max_leap_semitones", 12) or 12))
    overlap_tol = max(0.0, float(attack_rule.get("min_start_separation_beats", 0.25)))
    marimba_max_dur = max(0.05, float(attack_rule.get("marimba_max_duration_beats", 0.55)))
    vib_min_dur = max(0.05, float(attack_rule.get("vibraphone_min_duration_beats", 0.7)))

    marimba_events = _flatten_track_notes(arranged_by_track, marimba_key)
    pair_events = _flatten_track_notes(arranged_by_track, pair_key) if pair_key else []
    if not marimba_events:
        return {
            "enabled": True,
            "status": "marimba_has_no_notes",
            "marimba_track": marimba_key,
            "pair_track": pair_key,
        }

    in_range = sum(1 for e in marimba_events if midi_min <= int(e["pitch"]) <= midi_max)
    range_ratio = float(in_range) / float(len(marimba_events))

    leaps_ok = 0
    leap_total = 0
    prev_pitch: int | None = None
    for event in marimba_events:
        pitch = int(event["pitch"])
        if prev_pitch is not None:
            leap_total += 1
            if abs(pitch - prev_pitch) <= max_leap:
                leaps_ok += 1
        prev_pitch = pitch
    leap_ratio = 1.0 if leap_total == 0 else float(leaps_ok) / float(leap_total)

    marimba_avg_duration = _safe_mean([float(e["duration"]) for e in marimba_events])
    marimba_attack_ratio = _safe_mean(
        [1.0 if float(e["duration"]) <= marimba_max_dur else 0.0 for e in marimba_events]
    )

    pair_overlap_ratio = 0.0
    pair_answer_ratio = 0.0
    pair_avg_duration = 0.0
    pair_sustain_ratio = 0.0
    if pair_events:
        overlaps = 0
        answers = 0
        for pe in pair_events:
            same_section_marimba = [
                me
                for me in marimba_events
                if int(me["section_index"]) == int(pe["section_index"])
            ]
            if not same_section_marimba:
                continue
            pstart = float(pe["start_time"])
            if any(abs(pstart - float(me["start_time"])) <= overlap_tol for me in same_section_marimba):
                overlaps += 1
            if any(
                (pstart > float(me["start_time"]))
                and (pstart <= float(me["start_time"]) + max(0.25, overlap_tol * 2.0))
                for me in same_section_marimba
            ):
                answers += 1
        pair_overlap_ratio = float(overlaps) / float(max(1, len(pair_events)))
        pair_answer_ratio = float(answers) / float(max(1, len(pair_events)))
        pair_avg_duration = _safe_mean([float(e["duration"]) for e in pair_events])
        pair_sustain_ratio = _safe_mean(
            [1.0 if float(e["duration"]) >= vib_min_dur else 0.0 for e in pair_events]
        )

    flags: list[str] = []
    if range_ratio < float(contract.get("eval", {}).get("target_range_adherence_min_ratio", 0.98) if isinstance(contract.get("eval"), Mapping) else 0.98):
        flags.append("marimba_range_adherence_below_target")
    if leap_ratio < float(contract.get("eval", {}).get("target_leap_adherence_min_ratio", 0.8) if isinstance(contract.get("eval"), Mapping) else 0.8):
        flags.append("marimba_leap_discipline_below_target")
    if pair_events:
        overlap_target = float(contract.get("eval", {}).get("target_pair_overlap_max_ratio", 0.34) if isinstance(contract.get("eval"), Mapping) else 0.34)
        if pair_overlap_ratio > overlap_target:
            flags.append("marimba_vibraphone_overlap_above_target")
        if pair_answer_ratio < 0.2:
            flags.append("marimba_vibraphone_answer_ratio_low")

    return {
        "enabled": True,
        "status": "ok",
        "marimba_track": marimba_key,
        "pair_track": pair_key,
        "marimba": {
            "note_count": len(marimba_events),
            "range_adherence_ratio": round(range_ratio, 4),
            "leap_adherence_ratio": round(leap_ratio, 4),
            "avg_duration_beats": round(float(marimba_avg_duration), 4),
            "attack_duration_ratio": round(float(marimba_attack_ratio), 4),
        },
        "pair": {
            "note_count": len(pair_events),
            "overlap_ratio": round(float(pair_overlap_ratio), 4),
            "answer_ratio": round(float(pair_answer_ratio), 4),
            "avg_duration_beats": round(float(pair_avg_duration), 4),
            "sustain_ratio": round(float(pair_sustain_ratio), 4),
        },
        "flags": flags,
    }


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values)) / float(len(values))


def _merit_rubric(
    *,
    note_count_paths: Mapping[str, Sequence[int]],
    similarity: float | None,
    repetition_flags: Sequence[str],
) -> dict[str, float]:
    if not note_count_paths:
        return {
            "pulse_clarity_proxy": 0.0,
            "form_contrast_proxy": 0.0,
            "instrument_diversity_proxy": 0.0,
            "repetition_risk": 1.0 if similarity is not None and similarity >= 0.9 else 0.0,
        }

    track_names = list(note_count_paths.keys())
    section_count = max(len(counts) for counts in note_count_paths.values())
    if section_count <= 0:
        section_count = 1

    per_section_totals = [0.0] * section_count
    per_section_active_tracks = [0] * section_count
    anchor_coverage: list[float] = []
    for idx in range(section_count):
        section_total = 0.0
        section_active = 0
        anchor_active = 0
        anchor_candidates = 0
        for track_name, counts in note_count_paths.items():
            value = float(counts[idx]) if idx < len(counts) else 0.0
            section_total += value
            if value > 0:
                section_active += 1
            lower = str(track_name).lower()
            if any(token in lower for token in ("kick", "rim", "hat")):
                anchor_candidates += 1
                if value > 0:
                    anchor_active += 1
        per_section_totals[idx] = section_total
        per_section_active_tracks[idx] = section_active
        if anchor_candidates > 0:
            anchor_coverage.append(float(anchor_active) / float(anchor_candidates))

    mean_total = _safe_mean(per_section_totals)
    variance = _safe_mean([(value - mean_total) ** 2 for value in per_section_totals])
    std_total = math.sqrt(max(0.0, variance))
    form_contrast = 0.0 if mean_total <= 0.0 else min(1.0, std_total / (mean_total + 1e-6))
    instrument_diversity = min(1.0, _safe_mean([float(v) / float(max(1, len(track_names))) for v in per_section_active_tracks]))
    pulse_clarity = min(1.0, _safe_mean(anchor_coverage)) if anchor_coverage else 0.0

    repetition_risk = 0.0 if similarity is None else float(similarity)
    if repetition_flags:
        repetition_risk = min(1.0, repetition_risk + 0.1 * float(len(repetition_flags)))

    return {
        "pulse_clarity_proxy": round(pulse_clarity, 4),
        "form_contrast_proxy": round(form_contrast, 4),
        "instrument_diversity_proxy": round(instrument_diversity, 4),
        "repetition_risk": round(min(1.0, repetition_risk), 4),
    }


def _build_fingerprints(
    sig_num: int,
    sig_den: int,
    bpm: float,
    sections: Sequence[Any],
    note_count_paths: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    form_labels = [str(getattr(section, "label", "")) for section in sections]
    hat_density_path = [str(getattr(section, "hat_density", "")) for section in sections]
    piano_mode_path = [str(getattr(section, "piano_mode", "")) for section in sections]

    payload = {
        "meter_bpm": f"{sig_num}/{sig_den}@{float(bpm):g}",
        "form_labels": form_labels,
        "hat_density_path": hat_density_path,
        "piano_mode_path": piano_mode_path,
        "note_count_paths": {track: list(counts) for track, counts in note_count_paths.items()},
    }
    payload["fingerprint_hash"] = _sha256_text(_stable_json(payload))
    return payload


def _load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def _load_recent_artifacts(repo_root: Path, limit: int = 50) -> list[dict[str, Any]]:
    index_path = repo_root / DEFAULT_RELATIVE_INDEX_PATH
    index_data = _load_json(index_path, {"version": 1, "entries": []})
    entries = index_data.get("entries") if isinstance(index_data, dict) else []
    if not isinstance(entries, list):
        return []

    artifacts: list[dict[str, Any]] = []
    for entry in entries[:limit]:
        if not isinstance(entry, dict):
            continue
        rel = entry.get("artifact_path")
        if not isinstance(rel, str) or not rel:
            continue
        artifact_path = repo_root / rel
        artifact = _load_json(artifact_path, None)
        if isinstance(artifact, dict):
            artifacts.append(artifact)
    return artifacts


def _pick_reference_artifact(
    current_meter_bpm: str,
    recent_artifacts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for artifact in recent_artifacts:
        fp = artifact.get("fingerprints")
        if isinstance(fp, Mapping) and str(fp.get("meter_bpm", "")) == current_meter_bpm:
            return artifact
    return recent_artifacts[0] if recent_artifacts else None


def _compute_similarity(
    current_fp: Mapping[str, Any],
    reference_fp: Mapping[str, Any] | None,
) -> tuple[float | None, dict[str, float], list[str], str | None]:
    if reference_fp is None:
        return None, {}, [], None

    scores: dict[str, float] = {}
    scores["form_labels"] = _sequence_similarity(
        _normalize_seq(current_fp.get("form_labels", [])),
        _normalize_seq(reference_fp.get("form_labels", [])),
    )
    scores["hat_density_path"] = _sequence_similarity(
        _normalize_seq(current_fp.get("hat_density_path", [])),
        _normalize_seq(reference_fp.get("hat_density_path", [])),
    )
    scores["piano_mode_path"] = _sequence_similarity(
        _normalize_seq(current_fp.get("piano_mode_path", [])),
        _normalize_seq(reference_fp.get("piano_mode_path", [])),
    )

    current_paths = current_fp.get("note_count_paths", {})
    reference_paths = reference_fp.get("note_count_paths", {})
    if isinstance(current_paths, Mapping) and isinstance(reference_paths, Mapping):
        for track, current_counts in current_paths.items():
            if track not in reference_paths:
                continue
            ref_counts = reference_paths.get(track)
            if not isinstance(current_counts, Sequence) or isinstance(current_counts, (str, bytes)):
                continue
            if not isinstance(ref_counts, Sequence) or isinstance(ref_counts, (str, bytes)):
                continue
            key = f"note_shape:{track}"
            scores[key] = _cosine_similarity(
                [float(v) for v in current_counts],
                [float(v) for v in ref_counts],
            )

    if not scores:
        return None, {}, [], None

    overall = sum(scores.values()) / float(len(scores))
    flags: list[str] = []
    if overall >= 0.90:
        flags.append("overall_structure_highly_similar_to_recent_run")
    if scores.get("hat_density_path", 0.0) >= 0.99:
        flags.append("hat_density_trajectory_repeated")
    if scores.get("piano_mode_path", 0.0) >= 0.99:
        flags.append("piano_mode_trajectory_repeated")

    reference_run_id = reference_fp.get("fingerprint_hash") if isinstance(reference_fp, Mapping) else None
    return round(overall, 4), {k: round(v, 4) for k, v in scores.items()}, flags, str(reference_run_id) if reference_run_id else None


def _reflection_prompts(
    repetition_flags: Sequence[str],
    instrument_identity_flags: Sequence[str] | None = None,
) -> list[str]:
    prompts = [
        "What compositional risk did this run attempt that your previous run did not?",
        "How did the opening establish pulse, tonal gravity, and density floor?",
        "How did the ending resolve at least one active tension (rhythmic, harmonic, or textural)?",
        "What should remain invariant next run, and what should intentionally diverge?",
    ]

    if "hat_density_trajectory_repeated" in repetition_flags:
        prompts.append(
            "The hat density path repeated; choose a different valid escalation trajectory next run (without requiring sixteenth-note peak)."
        )
    if "piano_mode_trajectory_repeated" in repetition_flags:
        prompts.append(
            "The piano mode path repeated; choose an alternate mode trajectory while keeping form clarity."
        )
    if "overall_structure_highly_similar_to_recent_run" in repetition_flags:
        prompts.append(
            "Overall structure was highly similar to a recent run; alter one macro form decision while keeping musical intent."
        )
    identity_flags = [str(flag) for flag in (instrument_identity_flags or []) if str(flag).strip()]
    if "marimba_range_adherence_below_target" in identity_flags:
        prompts.append(
            "Marimba range adherence dropped; tighten register selection before adding more density."
        )
    if "marimba_leap_discipline_below_target" in identity_flags:
        prompts.append(
            "Marimba leaps were too wide; simplify contour to preserve playability."
        )
    if "marimba_vibraphone_overlap_above_target" in identity_flags:
        prompts.append(
            "Marimba and vibraphone overlapped heavily; increase onset separation and role contrast."
        )
    return prompts


def build_composition_artifact(
    *,
    mood: str,
    key_name: str,
    bpm: float,
    sig_num: int,
    sig_den: int,
    minutes: float,
    bars: int,
    section_bars: int,
    sections: Sequence[Any],
    arranged_by_track: Mapping[str, Sequence[tuple[Any, Sequence[dict[str, Any]]]]],
    created_clips_by_track: Mapping[str, int],
    status: str,
    run_metadata: Mapping[str, Any] | None = None,
    instrument_identity_contract: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    now = _utc_now()
    run_id = (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}"
        f"-{sig_num}_{sig_den}"
        f"-{_safe_bpm_text(bpm)}bpm"
        f"-{_safe_token(str(mood), fallback='mood')}"
        f"-{_sha256_text(str(now.timestamp()))[:8]}"
    )

    section_count = len(sections)
    note_count_paths = _section_count_paths(section_count, arranged_by_track)
    fingerprints = _build_fingerprints(sig_num, sig_den, bpm, sections, note_count_paths)

    recent = _load_recent_artifacts(root, limit=50)
    reference_artifact = _pick_reference_artifact(str(fingerprints["meter_bpm"]), recent)
    reference_fp = reference_artifact.get("fingerprints") if isinstance(reference_artifact, Mapping) else None
    if not isinstance(reference_fp, Mapping):
        reference_fp = None

    similarity, similarity_breakdown, repetition_flags, reference_fingerprint = _compute_similarity(
        fingerprints,
        reference_fp,
    )

    novelty_score = None if similarity is None else round(1.0 - similarity, 4)
    track_totals = _track_totals(note_count_paths)
    instrument_identity = _compute_instrument_identity_metrics(
        arranged_by_track=arranged_by_track,
        contract=instrument_identity_contract,
    )
    merit_rubric = _merit_rubric(
        note_count_paths=note_count_paths,
        similarity=similarity,
        repetition_flags=repetition_flags,
    )

    artifact = {
        "version": 1,
        "run_id": run_id,
        "timestamp_utc": _iso_utc(now),
        "status": str(status),
        "composition": {
            "mood": str(mood),
            "key_name": str(key_name),
            "tempo_bpm": float(bpm),
            "signature": f"{int(sig_num)}/{int(sig_den)}",
            "minutes": float(minutes),
            "bars": int(bars),
            "section_bars": int(section_bars),
        },
        "run": dict(run_metadata or {}),
        "section_strategy": [
            {
                "index": int(getattr(section, "index", idx)),
                "label": str(getattr(section, "label", "")),
                "piano_mode": str(getattr(section, "piano_mode", "")),
                "hat_density": str(getattr(section, "hat_density", "")),
                "kick_keep_ratio": float(getattr(section, "kick_keep_ratio", 0.0)),
                "rim_keep_ratio": float(getattr(section, "rim_keep_ratio", 0.0)),
                "hat_keep_ratio": float(getattr(section, "hat_keep_ratio", 0.0)),
            }
            for idx, section in enumerate(sections)
        ],
        "tracks": {
            str(track): {
                "section_note_counts": list(counts),
                "total_notes": int(track_totals.get(track, 0)),
                "created_section_clips": int(created_clips_by_track.get(track, 0)),
            }
            for track, counts in note_count_paths.items()
        },
        "fingerprints": fingerprints,
        "reflection": {
            "novelty_score": novelty_score,
            "similarity_to_reference": similarity,
            "similarity_breakdown": similarity_breakdown,
            "reference_fingerprint": reference_fingerprint,
            "repetition_flags": repetition_flags,
            "merit_rubric": merit_rubric,
            "instrument_identity": instrument_identity,
            "prompts": _reflection_prompts(
                repetition_flags,
                instrument_identity.get("flags", []) if isinstance(instrument_identity, Mapping) else [],
            ),
        },
    }
    return artifact


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def persist_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    relative_log_dir: Path = DEFAULT_RELATIVE_LOG_DIR,
    relative_index_path: Path = DEFAULT_RELATIVE_INDEX_PATH,
) -> Path:
    root = _repo_root(repo_root)

    timestamp = str(artifact.get("timestamp_utc", ""))
    date_part = timestamp[:10] if len(timestamp) >= 10 else "unknown-date"
    run_id = str(artifact.get("run_id", "unknown-run"))

    artifact_rel_path = relative_log_dir / date_part / f"{run_id}.json"
    artifact_abs_path = root / artifact_rel_path
    _write_json(artifact_abs_path, artifact)

    index_abs_path = root / relative_index_path
    index_data = _load_json(index_abs_path, {"version": 1, "entries": []})
    entries = index_data.get("entries") if isinstance(index_data, dict) else []
    if not isinstance(entries, list):
        entries = []

    entry = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "artifact_path": str(artifact_rel_path),
        "status": str(artifact.get("status", "unknown")),
        "mood": str(artifact.get("composition", {}).get("mood", "")),
        "meter_bpm": str(artifact.get("fingerprints", {}).get("meter_bpm", "")),
        "novelty_score": artifact.get("reflection", {}).get("novelty_score"),
    }

    existing = [item for item in entries if isinstance(item, dict) and item.get("run_id") != run_id]
    updated_entries = [entry] + existing
    updated_entries = updated_entries[:MAX_INDEX_ENTRIES]

    index_payload = {"version": 1, "entries": updated_entries}
    _write_json(index_abs_path, index_payload)

    return artifact_abs_path


def log_composition_run(
    *,
    mood: str,
    key_name: str,
    bpm: float,
    sig_num: int,
    sig_den: int,
    minutes: float,
    bars: int,
    section_bars: int,
    sections: Sequence[Any],
    arranged_by_track: Mapping[str, Sequence[tuple[Any, Sequence[dict[str, Any]]]]],
    created_clips_by_track: Mapping[str, int],
    status: str,
    run_metadata: Mapping[str, Any] | None = None,
    instrument_identity_contract: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
    relative_log_dir: Path = DEFAULT_RELATIVE_LOG_DIR,
) -> tuple[dict[str, Any], Path]:
    artifact = build_composition_artifact(
        mood=mood,
        key_name=key_name,
        bpm=bpm,
        sig_num=sig_num,
        sig_den=sig_den,
        minutes=minutes,
        bars=bars,
        section_bars=section_bars,
        sections=sections,
        arranged_by_track=arranged_by_track,
        created_clips_by_track=created_clips_by_track,
        status=status,
        run_metadata=run_metadata,
        instrument_identity_contract=instrument_identity_contract,
        repo_root=repo_root,
    )
    path = persist_artifact(
        artifact,
        repo_root=repo_root,
        relative_log_dir=relative_log_dir,
    )
    return artifact, path
