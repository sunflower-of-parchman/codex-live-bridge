#!/usr/bin/env python3
"""Eval-driven governance loop for memory updates.

This module turns repeated composition-eval signals into bounded memory updates.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


DEFAULT_POLICY_REL_PATH = Path("memory/eval_governance_policy.toml")
DEFAULT_INDEX_REL_PATH = Path("memory/evals/composition_index.json")
DEFAULT_STATE_REL_PATH = Path("memory/governance/state.json")
DEFAULT_ACTIVE_REL_PATH = Path("memory/governance/active.md")
DEFAULT_DEMOTION_ARCHIVE_REL_PATH = Path("memory/archive/demoted_guidance.md")


def _repo_root(path: Path | None = None) -> Path:
    if path is not None:
        return path
    return Path(__file__).resolve().parents[1]


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _iso_utc(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _parse_utc(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _safe_bpm_text(value: float) -> str:
    return f"{float(value):g}"


def _meter_bpm_key(meter: str | None, bpm: float | None) -> str | None:
    if meter in (None, "") or bpm is None:
        return None
    return f"{str(meter).strip()}@{_safe_bpm_text(float(bpm))}"


def _normalize_marker(text: str) -> str:
    raw = str(text).strip().lower()
    raw = re.sub(r"[^a-z0-9:_-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "unknown"


def _ensure_trailing_newline(text: str) -> str:
    if not text.endswith("\n"):
        return text + "\n"
    return text


def _append_marked_block(text: str, heading: str, lines: Sequence[str]) -> str:
    if not lines:
        return text
    out = text
    if not out.strip():
        out = ""
    out = _ensure_trailing_newline(out) if out else ""
    if heading not in out:
        if out and not out.endswith("\n\n"):
            out += "\n"
        out += f"## {heading}\n\n"
    elif not out.endswith("\n"):
        out += "\n"
    for line in lines:
        out += f"- {line}\n"
    return out


def load_policy(path: Path | None = None) -> dict[str, Any]:
    root = _repo_root()
    policy_path = path if path is not None else (root / DEFAULT_POLICY_REL_PATH)
    if not policy_path.is_absolute():
        policy_path = root / policy_path
    if not policy_path.exists():
        raise FileNotFoundError(f"policy file not found: {policy_path}")

    with policy_path.open("rb") as handle:
        payload = tomllib.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("policy TOML did not parse into a dictionary")

    defaults_raw = payload.get("defaults", {}) if isinstance(payload.get("defaults"), dict) else {}
    thresholds_raw = payload.get("thresholds", {}) if isinstance(payload.get("thresholds"), dict) else {}

    defaults = {
        "lookback": max(1, int(defaults_raw.get("lookback", 30))),
        "statuses": [str(v) for v in defaults_raw.get("statuses", ["success", "save_failed"]) if str(v).strip()],
        "session_capture_min_count": max(1, int(defaults_raw.get("session_capture_min_count", 1))),
        "promotion_repeat_threshold": max(1, int(defaults_raw.get("promotion_repeat_threshold", 3))),
        "stale_days": max(1, int(defaults_raw.get("stale_days", 21))),
        "max_promotions_per_apply": max(1, int(defaults_raw.get("max_promotions_per_apply", 3))),
        "max_session_updates_per_apply": max(1, int(defaults_raw.get("max_session_updates_per_apply", 6))),
        "max_active_items_per_file": max(1, int(defaults_raw.get("max_active_items_per_file", 5))),
    }

    thresholds = {
        "low_novelty_max": float(thresholds_raw.get("low_novelty_max", 0.2)),
        "high_repetition_risk_min": float(thresholds_raw.get("high_repetition_risk_min", 0.85)),
    }

    rules_raw = payload.get("rules", [])
    if not isinstance(rules_raw, list) or not rules_raw:
        raise ValueError("policy must define at least one [[rules]] entry")

    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw in enumerate(rules_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"policy rule at index {idx} is not a table")
        signal_id = str(raw.get("id", "")).strip()
        target = str(raw.get("target", "")).strip()
        session_text = str(raw.get("session_text", "")).strip()
        promotion_text = str(raw.get("promotion_text", "")).strip()
        kind = str(raw.get("kind", "derived_metric")).strip() or "derived_metric"
        if not signal_id:
            raise ValueError(f"policy rule at index {idx} is missing id")
        if signal_id in seen:
            raise ValueError(f"duplicate policy rule id: {signal_id}")
        if not target:
            raise ValueError(f"policy rule '{signal_id}' is missing target")
        if not session_text:
            raise ValueError(f"policy rule '{signal_id}' is missing session_text")
        if not promotion_text:
            raise ValueError(f"policy rule '{signal_id}' is missing promotion_text")
        seen.add(signal_id)
        rules.append(
            {
                "id": signal_id,
                "kind": kind,
                "target": target,
                "session_text": session_text,
                "promotion_text": promotion_text,
                "promote_after": max(
                    1,
                    int(raw.get("promote_after", defaults["promotion_repeat_threshold"])),
                ),
            }
        )

    return {
        "version": int(payload.get("version", 1)),
        "path": str(policy_path),
        "defaults": defaults,
        "thresholds": thresholds,
        "rules": rules,
    }


def load_recent_eval_artifacts(repo_root: Path, lookback: int) -> list[dict[str, Any]]:
    root = _repo_root(repo_root)
    index_path = root / DEFAULT_INDEX_REL_PATH
    index_payload = _load_json(index_path, {"version": 1, "entries": []})
    entries = index_payload.get("entries", []) if isinstance(index_payload, dict) else []

    artifacts: list[dict[str, Any]] = []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rel_path = entry.get("artifact_path")
            if not isinstance(rel_path, str) or not rel_path.strip():
                continue
            artifact_path = root / rel_path
            payload = _load_json(artifact_path, None)
            if not isinstance(payload, dict):
                continue
            payload = dict(payload)
            payload["_artifact_path"] = str(rel_path)
            if isinstance(entry.get("timestamp_utc"), str):
                payload.setdefault("timestamp_utc", entry.get("timestamp_utc"))
            artifacts.append(payload)

    if not artifacts:
        fallback_paths = sorted(
            (root / "memory/evals/compositions").glob("**/*.json"),
            key=lambda p: p.name,
            reverse=True,
        )
        for path in fallback_paths:
            payload = _load_json(path, None)
            if not isinstance(payload, dict):
                continue
            payload = dict(payload)
            payload["_artifact_path"] = str(path.relative_to(root)).replace("\\", "/")
            artifacts.append(payload)

    artifacts.sort(
        key=lambda artifact: (
            _parse_utc(str(artifact.get("timestamp_utc", ""))) or datetime.fromtimestamp(0, UTC),
            str(artifact.get("run_id", "")),
        ),
        reverse=True,
    )
    return artifacts[: max(1, int(lookback))]


def _filter_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    statuses: Sequence[str],
    meter: str | None = None,
    bpm: float | None = None,
    mood: str | None = None,
) -> list[dict[str, Any]]:
    normalized_statuses = {str(status).strip().lower() for status in statuses if str(status).strip()}
    meter_bpm = _meter_bpm_key(meter, bpm)
    mood_token = str(mood).strip().lower() if mood not in (None, "") else None

    out: list[dict[str, Any]] = []
    for artifact_raw in artifacts:
        artifact = dict(artifact_raw)
        status = str(artifact.get("status", "")).strip().lower()
        if normalized_statuses and status not in normalized_statuses:
            continue
        composition = artifact.get("composition", {}) if isinstance(artifact.get("composition"), dict) else {}
        fingerprints = artifact.get("fingerprints", {}) if isinstance(artifact.get("fingerprints"), dict) else {}

        if meter_bpm is not None and str(fingerprints.get("meter_bpm", "")) != meter_bpm:
            continue
        if mood_token is not None and str(composition.get("mood", "")).strip().lower() != mood_token:
            continue

        out.append(artifact)
    return out


def _artifact_signal_ids(artifact: Mapping[str, Any], thresholds: Mapping[str, Any]) -> set[str]:
    reflection = artifact.get("reflection", {}) if isinstance(artifact.get("reflection"), dict) else {}
    ids: set[str] = set()

    repetition_flags = reflection.get("repetition_flags", [])
    if isinstance(repetition_flags, Sequence) and not isinstance(repetition_flags, (str, bytes)):
        for flag in repetition_flags:
            token = str(flag).strip()
            if token:
                ids.add(token)

    identity = reflection.get("instrument_identity", {}) if isinstance(reflection.get("instrument_identity"), dict) else {}
    identity_flags = identity.get("flags", []) if isinstance(identity, Mapping) else []
    if isinstance(identity_flags, Sequence) and not isinstance(identity_flags, (str, bytes)):
        for flag in identity_flags:
            token = str(flag).strip()
            if token:
                ids.add(token)

    novelty_score = reflection.get("novelty_score")
    if isinstance(novelty_score, (int, float)) and float(novelty_score) <= float(thresholds.get("low_novelty_max", 0.2)):
        ids.add("low_novelty_score")

    merit = reflection.get("merit_rubric", {}) if isinstance(reflection.get("merit_rubric"), dict) else {}
    repetition_risk = merit.get("repetition_risk")
    if isinstance(repetition_risk, (int, float)) and float(repetition_risk) >= float(
        thresholds.get("high_repetition_risk_min", 0.85)
    ):
        ids.add("high_repetition_risk")

    return ids


def build_governance_signals(artifacts: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = policy.get("thresholds", {}) if isinstance(policy.get("thresholds"), Mapping) else {}
    rules = policy.get("rules", []) if isinstance(policy.get("rules"), Sequence) else []
    rule_ids = {str(rule.get("id", "")).strip() for rule in rules if isinstance(rule, Mapping)}

    signals: dict[str, dict[str, Any]] = {}
    considered_runs: list[str] = []

    for artifact in artifacts:
        run_id = str(artifact.get("run_id", "")).strip()
        if run_id:
            considered_runs.append(run_id)

        timestamp = str(artifact.get("timestamp_utc", "")).strip()
        composition = artifact.get("composition", {}) if isinstance(artifact.get("composition"), dict) else {}
        fingerprints = artifact.get("fingerprints", {}) if isinstance(artifact.get("fingerprints"), dict) else {}
        meter_bpm = str(fingerprints.get("meter_bpm", "")).strip()
        mood = str(composition.get("mood", "")).strip() or "unknown"

        for signal_id in sorted(_artifact_signal_ids(artifact, thresholds)):
            if rule_ids and signal_id not in rule_ids:
                continue
            payload = signals.setdefault(
                signal_id,
                {
                    "id": signal_id,
                    "count": 0,
                    "last_seen": "",
                    "meter_bpm_counts": {},
                    "mood_counts": {},
                    "run_ids": [],
                },
            )
            payload["count"] = int(payload.get("count", 0)) + 1
            if timestamp and str(payload.get("last_seen", "")) < timestamp:
                payload["last_seen"] = timestamp
            if meter_bpm:
                meter_counts = payload.get("meter_bpm_counts", {})
                meter_counts[meter_bpm] = int(meter_counts.get(meter_bpm, 0)) + 1
                payload["meter_bpm_counts"] = meter_counts
            mood_counts = payload.get("mood_counts", {})
            mood_counts[mood] = int(mood_counts.get(mood, 0)) + 1
            payload["mood_counts"] = mood_counts
            if run_id and run_id not in payload["run_ids"]:
                payload["run_ids"].append(run_id)

    return {
        "generated_at_utc": _iso_utc(_utc_now()),
        "artifact_count": len(artifacts),
        "considered_runs": considered_runs,
        "signals": {key: signals[key] for key in sorted(signals.keys())},
    }


def plan_memory_actions(signals: Mapping[str, Any], policy: Mapping[str, Any], date_text: str) -> dict[str, Any]:
    defaults = policy.get("defaults", {}) if isinstance(policy.get("defaults"), Mapping) else {}
    rules = policy.get("rules", []) if isinstance(policy.get("rules"), Sequence) else []
    by_id = signals.get("signals", {}) if isinstance(signals.get("signals"), Mapping) else {}

    session_updates: list[dict[str, Any]] = []
    promotion_candidates: list[dict[str, Any]] = []

    min_session_count = max(1, int(defaults.get("session_capture_min_count", 1)))

    for raw_rule in rules:
        if not isinstance(raw_rule, Mapping):
            continue
        rule = dict(raw_rule)
        signal_id = str(rule.get("id", "")).strip()
        if not signal_id:
            continue
        signal = by_id.get(signal_id)
        if not isinstance(signal, Mapping):
            continue

        count = int(signal.get("count", 0))
        if count <= 0:
            continue

        marker_base = _normalize_marker(signal_id)
        last_seen = str(signal.get("last_seen", ""))
        context_counts = signal.get("meter_bpm_counts", {}) if isinstance(signal.get("meter_bpm_counts"), Mapping) else {}
        dominant_context = ""
        if context_counts:
            dominant_context = sorted(
                ((str(key), int(value)) for key, value in context_counts.items()),
                key=lambda item: (-item[1], item[0]),
            )[0][0]

        if count >= min_session_count:
            session_updates.append(
                {
                    "id": signal_id,
                    "marker": f"gov-session:{date_text}:{marker_base}",
                    "target": f"memory/sessions/{date_text}.md",
                    "text": (
                        f"{rule.get('session_text', '')} "
                        f"(count={count}, last_seen={last_seen or 'n/a'}, context={dominant_context or 'mixed'})"
                    ).strip(),
                    "count": count,
                    "last_seen": last_seen,
                }
            )

        promote_after = max(1, int(rule.get("promote_after", defaults.get("promotion_repeat_threshold", 3))))
        if count >= promote_after:
            promotion_candidates.append(
                {
                    "id": signal_id,
                    "marker": f"gov:{marker_base}",
                    "target": str(rule.get("target", "")).strip(),
                    "text": str(rule.get("promotion_text", "")).strip(),
                    "count": count,
                    "last_seen": last_seen,
                }
            )

    promotion_candidates.sort(key=lambda item: (-int(item.get("count", 0)), str(item.get("id", ""))))
    session_updates.sort(key=lambda item: (-int(item.get("count", 0)), str(item.get("id", ""))))

    return {
        "date": date_text,
        "signals": signals,
        "policy": policy,
        "session_updates": session_updates,
        "promotion_candidates": promotion_candidates,
        "demotion_candidates": [],
    }


def _load_state(root: Path) -> dict[str, Any]:
    path = root / DEFAULT_STATE_REL_PATH
    state = _load_json(path, {"version": 1, "signals": {}})
    if not isinstance(state, dict):
        state = {"version": 1, "signals": {}}
    if not isinstance(state.get("signals"), dict):
        state["signals"] = {}
    state.setdefault("version", 1)
    return state


def _compute_demotion_candidates(
    *,
    signals: Mapping[str, Any],
    policy: Mapping[str, Any],
    state: Mapping[str, Any],
    now: datetime,
) -> list[dict[str, Any]]:
    defaults = policy.get("defaults", {}) if isinstance(policy.get("defaults"), Mapping) else {}
    stale_days = max(1, int(defaults.get("stale_days", 21)))
    rules = policy.get("rules", []) if isinstance(policy.get("rules"), Sequence) else []
    rules_by_id = {
        str(rule.get("id", "")).strip(): dict(rule)
        for rule in rules
        if isinstance(rule, Mapping) and str(rule.get("id", "")).strip()
    }

    signal_map = signals.get("signals", {}) if isinstance(signals.get("signals"), Mapping) else {}
    state_signals = state.get("signals", {}) if isinstance(state.get("signals"), Mapping) else {}

    candidates: list[dict[str, Any]] = []
    for signal_id, item_raw in state_signals.items():
        if not isinstance(item_raw, Mapping):
            continue
        if not bool(item_raw.get("promoted", False)):
            continue
        current = signal_map.get(signal_id)
        if isinstance(current, Mapping) and int(current.get("count", 0)) > 0:
            continue

        last_seen = str(item_raw.get("last_seen", "")).strip()
        last_seen_dt = _parse_utc(last_seen)
        if last_seen_dt is None:
            continue

        age_days = (now - last_seen_dt).days
        if age_days < stale_days:
            continue

        rule = rules_by_id.get(str(signal_id), {})
        marker = str(item_raw.get("marker", "")).strip() or f"gov:{_normalize_marker(str(signal_id))}"
        target = str(item_raw.get("target", "")).strip() or str(rule.get("target", "")).strip()
        if not target:
            continue

        candidates.append(
            {
                "id": str(signal_id),
                "marker": marker,
                "target": target,
                "reason": f"stale signal (last_seen={last_seen}, age_days={age_days})",
                "age_days": age_days,
                "last_seen": last_seen,
            }
        )

    candidates.sort(key=lambda item: (-int(item.get("age_days", 0)), str(item.get("id", ""))))
    return candidates


def _count_active_markers(text: str) -> int:
    return len(re.findall(r"\[gov:[^\]]+\]", text))


def _default_markdown_header(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip().title() or "Notes"
    return f"# {stem}\n\n"


def _render_active_guidance(state: Mapping[str, Any], rules_by_id: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "# Active Governance Guidance",
        "",
        "This file is generated by `python3 -m memory.eval_governance apply`.",
        "",
    ]

    signals = state.get("signals", {}) if isinstance(state.get("signals"), Mapping) else {}
    promoted: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for signal_id, item_raw in signals.items():
        if not isinstance(item_raw, Mapping):
            continue
        if not bool(item_raw.get("promoted", False)):
            continue
        rule = rules_by_id.get(str(signal_id), {})
        promoted.append((str(signal_id), item_raw, rule))

    promoted.sort(key=lambda item: item[0])
    if not promoted:
        lines.extend(
            [
                "No active promoted guidance.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.append("## Promoted Rules")
    lines.append("")
    for signal_id, item, rule in promoted:
        marker = str(item.get("marker", f"gov:{_normalize_marker(signal_id)}"))
        text = str(rule.get("promotion_text", "")).strip() or str(item.get("text", "")).strip()
        target = str(item.get("target", "")).strip() or str(rule.get("target", "")).strip()
        last_seen = str(item.get("last_seen", "")).strip() or "n/a"
        lines.append(f"- [{marker}] {text} (target={target}, last_seen={last_seen})")
    lines.append("")
    return "\n".join(lines)


def apply_memory_actions(actions: Mapping[str, Any], repo_root: Path, dry_run: bool) -> dict[str, Any]:
    root = _repo_root(repo_root)
    policy = actions.get("policy", {}) if isinstance(actions.get("policy"), Mapping) else {}
    defaults = policy.get("defaults", {}) if isinstance(policy.get("defaults"), Mapping) else {}
    rules = policy.get("rules", []) if isinstance(policy.get("rules"), Sequence) else []
    rules_by_id = {
        str(rule.get("id", "")).strip(): dict(rule)
        for rule in rules
        if isinstance(rule, Mapping) and str(rule.get("id", "")).strip()
    }

    date_text = str(actions.get("date", _utc_now().strftime("%Y-%m-%d"))).strip() or _utc_now().strftime("%Y-%m-%d")
    session_updates_raw = actions.get("session_updates", []) if isinstance(actions.get("session_updates"), Sequence) else []
    promotion_candidates_raw = (
        actions.get("promotion_candidates", []) if isinstance(actions.get("promotion_candidates"), Sequence) else []
    )
    demotion_candidates_raw = (
        actions.get("demotion_candidates", []) if isinstance(actions.get("demotion_candidates"), Sequence) else []
    )
    signals = actions.get("signals", {}) if isinstance(actions.get("signals"), Mapping) else {}

    max_promotions = max(1, int(defaults.get("max_promotions_per_apply", 3)))
    max_session_updates = max(1, int(defaults.get("max_session_updates_per_apply", 6)))
    max_active_items_per_file = max(1, int(defaults.get("max_active_items_per_file", 5)))

    state = _load_state(root)
    state_signals = state.get("signals", {}) if isinstance(state.get("signals"), Mapping) else {}

    mutated_texts: dict[Path, str] = {}
    source_cache: dict[Path, str] = {}

    def read_text(path: Path) -> str:
        if path in mutated_texts:
            return mutated_texts[path]
        if path in source_cache:
            return source_cache[path]
        source_cache[path] = _safe_read_text(path)
        return source_cache[path]

    def set_text(path: Path, text: str) -> None:
        if read_text(path) == text:
            return
        mutated_texts[path] = text

    session_applied = 0
    promotions_applied = 0
    demotions_applied = 0

    session_path = root / "memory/sessions" / f"{date_text}.md"
    session_existing = read_text(session_path)
    if not session_existing.strip():
        session_existing = f"# Session {date_text}\n\n"

    session_lines: list[str] = []
    for raw in session_updates_raw:
        if session_applied >= max_session_updates:
            break
        if not isinstance(raw, Mapping):
            continue
        marker = str(raw.get("marker", "")).strip()
        text = str(raw.get("text", "")).strip()
        if not marker or not text:
            continue
        if f"[{marker}]" in session_existing or any(f"[{marker}]" in line for line in session_lines):
            continue
        session_lines.append(f"[{marker}] {text}")
        session_applied += 1

    if session_lines:
        session_updated = _append_marked_block(session_existing, f"Governance {date_text}", session_lines)
        set_text(session_path, session_updated)

    promotion_candidates: list[dict[str, Any]] = []
    for raw in promotion_candidates_raw:
        if not isinstance(raw, Mapping):
            continue
        promotion_candidates.append(dict(raw))

    for item in promotion_candidates:
        if promotions_applied >= max_promotions:
            break
        marker = str(item.get("marker", "")).strip()
        target_rel = str(item.get("target", "")).strip()
        text = str(item.get("text", "")).strip()
        signal_id = str(item.get("id", "")).strip()
        if not marker or not target_rel or not text or not signal_id:
            continue
        target_path = root / target_rel
        target_existing = read_text(target_path)
        if not target_existing.strip():
            target_existing = _default_markdown_header(target_path)
        if f"[{marker}]" in target_existing:
            state_item = state_signals.setdefault(signal_id, {}) if isinstance(state_signals, dict) else {}
            if isinstance(state_item, dict):
                state_item.update(
                    {
                        "marker": marker,
                        "target": target_rel,
                        "promoted": True,
                    }
                )
            continue

        active_count = _count_active_markers(target_existing)
        if active_count >= max_active_items_per_file:
            continue

        updated_text = _append_marked_block(target_existing, "Governance Rules", [f"[{marker}] {text}"])
        set_text(target_path, updated_text)

        state_item = state_signals.setdefault(signal_id, {}) if isinstance(state_signals, dict) else {}
        if isinstance(state_item, dict):
            state_item.update(
                {
                    "marker": marker,
                    "target": target_rel,
                    "promoted": True,
                    "last_promoted_at": _iso_utc(_utc_now()),
                }
            )
        promotions_applied += 1

    archive_path = root / DEFAULT_DEMOTION_ARCHIVE_REL_PATH
    archive_lines: list[str] = []
    demotion_candidates: list[dict[str, Any]] = []
    for raw in demotion_candidates_raw:
        if isinstance(raw, Mapping):
            demotion_candidates.append(dict(raw))

    for item in demotion_candidates:
        marker = str(item.get("marker", "")).strip()
        target_rel = str(item.get("target", "")).strip()
        signal_id = str(item.get("id", "")).strip()
        reason = str(item.get("reason", "stale signal")).strip()
        if not marker or not target_rel or not signal_id:
            continue
        target_path = root / target_rel
        target_existing = read_text(target_path)
        if f"[{marker}]" not in target_existing:
            continue

        filtered_lines = [line for line in target_existing.splitlines() if f"[{marker}]" not in line]
        updated_text = "\n".join(filtered_lines).rstrip() + "\n"
        set_text(target_path, updated_text)

        demotion_marker = f"gov-demote:{date_text}:{_normalize_marker(signal_id)}"
        archive_lines.append(
            f"[{demotion_marker}] removed [{marker}] from `{target_rel}` ({reason})"
        )

        state_item = state_signals.setdefault(signal_id, {}) if isinstance(state_signals, dict) else {}
        if isinstance(state_item, dict):
            state_item.update(
                {
                    "promoted": False,
                    "last_demoted_at": _iso_utc(_utc_now()),
                }
            )
        demotions_applied += 1

    if archive_lines:
        archive_existing = read_text(archive_path)
        if not archive_existing.strip():
            archive_existing = "# Demoted Guidance Archive\n\n"
        archive_updated = _append_marked_block(archive_existing, date_text, archive_lines)
        set_text(archive_path, archive_updated)

    signal_map = signals.get("signals", {}) if isinstance(signals.get("signals"), Mapping) else {}
    for signal_id, payload_raw in signal_map.items():
        if not isinstance(payload_raw, Mapping):
            continue
        payload = dict(payload_raw)
        state_item = state_signals.setdefault(str(signal_id), {}) if isinstance(state_signals, dict) else {}
        if not isinstance(state_item, dict):
            continue
        state_item["last_seen"] = str(payload.get("last_seen", ""))
        state_item["last_count"] = int(payload.get("count", 0))
        rule = rules_by_id.get(str(signal_id), {})
        if str(rule.get("target", "")).strip() and not str(state_item.get("target", "")).strip():
            state_item["target"] = str(rule.get("target", "")).strip()
        if str(state_item.get("marker", "")).strip() == "":
            state_item["marker"] = f"gov:{_normalize_marker(str(signal_id))}"

    state_path = root / DEFAULT_STATE_REL_PATH
    prior_state_text = _safe_read_text(state_path)
    prior_updated_at = str(state.get("updated_at_utc", "")).strip()
    if not prior_updated_at:
        prior_updated_at = _iso_utc(_utc_now())

    state_payload = {
        "version": 1,
        "updated_at_utc": prior_updated_at,
        "signals": state_signals,
    }
    state_text = json.dumps(state_payload, indent=2, sort_keys=True) + "\n"

    # Only advance updated_at_utc when state content has changed.
    if prior_state_text != state_text:
        state_payload["updated_at_utc"] = _iso_utc(_utc_now())
        state_text = json.dumps(state_payload, indent=2, sort_keys=True) + "\n"

    state_changed = prior_state_text != state_text

    active_path = root / DEFAULT_ACTIVE_REL_PATH
    active_text = _render_active_guidance(state_payload, rules_by_id)
    set_text(active_path, active_text)

    files_to_change = sorted(
        str(path.relative_to(root)).replace("\\", "/") for path in mutated_texts.keys()
    )
    state_rel = str(DEFAULT_STATE_REL_PATH).replace("\\", "/")
    if state_changed and state_rel not in files_to_change:
        files_to_change.append(state_rel)

    if not dry_run:
        for path in sorted(mutated_texts.keys()):
            _atomic_write_text(path, mutated_texts[path])
        if state_changed:
            _atomic_write_text(state_path, state_text)

    return {
        "date": date_text,
        "dry_run": bool(dry_run),
        "session_updates": session_applied,
        "promotions": promotions_applied,
        "demotions": demotions_applied,
        "files_to_change": sorted(files_to_change),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    summarize = sub.add_parser("summarize", help="Summarize governance signals from eval artifacts")
    summarize.add_argument("--repo-root", default=None, help="Optional repository root")
    summarize.add_argument("--policy-path", default=None, help="Optional policy file path")
    summarize.add_argument("--lookback", type=int, default=None, help="Eval artifact lookback window")
    summarize.add_argument("--meter", default=None, help="Filter by meter (for example 5/4)")
    summarize.add_argument("--bpm", type=float, default=None, help="Filter by BPM")
    summarize.add_argument("--mood", default=None, help="Filter by mood")

    apply = sub.add_parser("apply", help="Apply bounded memory updates from governance signals")
    apply.add_argument("--repo-root", default=None, help="Optional repository root")
    apply.add_argument("--policy-path", default=None, help="Optional policy file path")
    apply.add_argument("--lookback", type=int, default=None, help="Eval artifact lookback window")
    apply.add_argument("--meter", default=None, help="Optional meter filter")
    apply.add_argument("--bpm", type=float, default=None, help="Optional BPM filter")
    apply.add_argument("--mood", default=None, help="Optional mood filter")
    apply.add_argument("--date", default=None, help="Session date in YYYY-MM-DD (defaults to UTC today)")
    apply.add_argument("--dry-run", action="store_true", help="Plan updates without writing files")

    return parser


def _resolve_policy_path(arg_value: str | None, root: Path) -> Path | None:
    if arg_value in (None, ""):
        return None
    path = Path(str(arg_value))
    if path.is_absolute():
        return path
    return root / path


def _resolve_date_text(arg_value: str | None) -> str:
    if arg_value in (None, ""):
        return _utc_now().strftime("%Y-%m-%d")
    text = str(arg_value).strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        raise ValueError("--date must be YYYY-MM-DD")
    return text


def _prepare_signal_bundle(
    *,
    root: Path,
    policy: Mapping[str, Any],
    lookback: int | None,
    meter: str | None,
    bpm: float | None,
    mood: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    defaults = policy.get("defaults", {}) if isinstance(policy.get("defaults"), Mapping) else {}
    statuses = defaults.get("statuses", []) if isinstance(defaults.get("statuses"), Sequence) else []
    effective_lookback = max(1, int(lookback if lookback is not None else defaults.get("lookback", 30)))

    recent = load_recent_eval_artifacts(root, effective_lookback)
    filtered = _filter_artifacts(
        recent,
        statuses=[str(v) for v in statuses],
        meter=meter,
        bpm=bpm,
        mood=mood,
    )
    signals = build_governance_signals(filtered, policy)
    state = _load_state(root)
    demotions = _compute_demotion_candidates(signals=signals, policy=policy, state=state, now=_utc_now())
    return filtered, signals, state, demotions


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = _repo_root(None if args.repo_root in (None, "") else Path(args.repo_root))
    policy_path = _resolve_policy_path(args.policy_path, root)
    policy = load_policy(policy_path)

    filtered, signals, _state, demotions = _prepare_signal_bundle(
        root=root,
        policy=policy,
        lookback=args.lookback,
        meter=args.meter,
        bpm=args.bpm,
        mood=args.mood,
    )

    if args.command == "summarize":
        actions = plan_memory_actions(signals, policy, _utc_now().strftime("%Y-%m-%d"))
        print(f"lookback_runs={max(1, int(args.lookback if args.lookback is not None else policy['defaults']['lookback']))}")
        print(f"considered_runs={len(filtered)}")
        print(f"governance_signals={len(signals.get('signals', {}))}")
        print(f"promotion_candidates={len(actions.get('promotion_candidates', []))}")
        print(f"demotion_candidates={len(demotions)}")

        rules = policy.get("rules", []) if isinstance(policy.get("rules"), Sequence) else []
        rule_targets = {
            str(rule.get("id", "")).strip(): str(rule.get("target", "")).strip()
            for rule in rules
            if isinstance(rule, Mapping)
        }
        signal_map = signals.get("signals", {}) if isinstance(signals.get("signals"), Mapping) else {}
        for signal_id in sorted(signal_map.keys()):
            payload = signal_map.get(signal_id, {})
            if not isinstance(payload, Mapping):
                continue
            print(
                "signal="
                f"{signal_id} "
                f"count={int(payload.get('count', 0))} "
                f"last_seen={str(payload.get('last_seen', '')) or 'n/a'} "
                f"target={rule_targets.get(str(signal_id), 'n/a')}"
            )
        return 0

    if args.command == "apply":
        date_text = _resolve_date_text(args.date)
        actions = plan_memory_actions(signals, policy, date_text)
        actions["demotion_candidates"] = demotions

        result = apply_memory_actions(actions, root, bool(args.dry_run))
        print(f"session_updates={result['session_updates']}")
        print(f"promotions={result['promotions']}")
        print(f"demotions={result['demotions']}")
        print(f"files_to_change={','.join(result['files_to_change']) if result['files_to_change'] else '(none)'}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
