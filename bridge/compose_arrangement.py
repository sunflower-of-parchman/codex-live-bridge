#!/usr/bin/env python3
"""Compose arrangement clips via the UDP bridge."""

from __future__ import annotations

import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import ableton_udp_bridge as bridge
import compose_kick_pattern as kick
import composition_feedback_loop as feedback
import arrangement.artifacts as _artifacts
import arrangement.base as _base
import arrangement.clip_ops as _clip_ops
from arrangement.config import ArrangementConfig, parse_args
import arrangement.marimba as _marimba
import arrangement.multi_pass as _multi_pass


for _module in (_base, _clip_ops, _marimba, _artifacts, _multi_pass):
    for _name, _value in vars(_module).items():
        if _name.startswith("__"):
            continue
        globals()[_name] = _value


NOTE_TO_ROOT = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}


def _parse_live_key_name(text: str) -> tuple[int, str] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    parts = raw.split()
    if len(parts) < 2:
        return None
    root_raw = parts[0].strip().upper().replace("♯", "#").replace("♭", "B")
    quality_raw = parts[1].strip().lower()
    root_note = NOTE_TO_ROOT.get(root_raw)
    if root_note is None:
        return None
    if quality_raw.startswith("maj"):
        return int(root_note), "Major"
    if quality_raw.startswith("min"):
        return int(root_note), "Minor"
    return None


def _apply_live_key(
    sock: socket.socket,
    ack_sock: socket.socket,
    key_name: str,
    timeout_s: float,
) -> tuple[int, str] | None:
    parsed = _parse_live_key_name(key_name)
    if parsed is None:
        return None
    root_note, scale_name = parsed
    kick._api_set(
        sock,
        ack_sock,
        "live_set",
        "root_note",
        int(root_note),
        "arr-root-note",
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        "live_set",
        "scale_name",
        str(scale_name),
        "arr-scale-name",
        timeout_s,
    )
    return int(root_note), str(scale_name)


def _ensure_live_transport(
    sock: socket.socket,
    ack_sock: socket.socket,
    *,
    bpm: float,
    sig_num: int,
    sig_den: int,
    timeout_s: float,
    max_attempts: int = 4,
) -> tuple[float | None, int | None, int | None]:
    observed_tempo: float | None = None
    observed_sig_num: int | None = None
    observed_sig_den: int | None = None
    target_bpm = float(bpm)
    target_sig_num = int(sig_num)
    target_sig_den = int(sig_den)

    for attempt in range(1, max(1, int(max_attempts)) + 1):
        kick._api_set(
            sock,
            ack_sock,
            "live_set",
            "tempo",
            target_bpm,
            _req_id("arr-live-tempo", attempt),
            timeout_s,
        )
        kick._api_set(
            sock,
            ack_sock,
            "live_set",
            "signature_numerator",
            target_sig_num,
            _req_id("arr-live-sig-num", attempt),
            timeout_s,
        )
        kick._api_set(
            sock,
            ack_sock,
            "live_set",
            "signature_denominator",
            target_sig_den,
            _req_id("arr-live-sig-den", attempt),
            timeout_s,
        )

        observed_tempo = kick._as_float(
            kick._api_get(
                sock,
                ack_sock,
                "live_set",
                "tempo",
                _req_id("arr-live-tempo-observed", attempt),
                timeout_s,
            )
        )
        observed_sig_num = kick._as_int(
            kick._api_get(
                sock,
                ack_sock,
                "live_set",
                "signature_numerator",
                _req_id("arr-live-sig-num-observed", attempt),
                timeout_s,
            )
        )
        observed_sig_den = kick._as_int(
            kick._api_get(
                sock,
                ack_sock,
                "live_set",
                "signature_denominator",
                _req_id("arr-live-sig-den-observed", attempt),
                timeout_s,
            )
        )

        tempo_ok = observed_tempo is not None and abs(float(observed_tempo) - target_bpm) <= 1e-6
        sig_num_ok = observed_sig_num == target_sig_num
        sig_den_ok = observed_sig_den == target_sig_den
        if tempo_ok and sig_num_ok and sig_den_ok:
            return observed_tempo, observed_sig_num, observed_sig_den
    return observed_tempo, observed_sig_num, observed_sig_den


def _resolve_track_path(
    sock: socket.socket,
    ack_sock: socket.socket,
    track_name: str,
    timeout_s: float,
) -> str | None:
    tracks_before = kick._get_children(
        sock,
        ack_sock,
        "live_set",
        "tracks",
        _req_id("arr-tracks-before", track_name),
        timeout_s,
    )
    if not tracks_before:
        return None

    track_index = kick._find_track_index_by_name(tracks_before, track_name)
    if track_index is not None:
        track_path = f"live_set tracks {track_index}"
        kick._api_set(
            sock,
            ack_sock,
            track_path,
            "name",
            track_name,
            _req_id("arr-track-name", track_name),
            timeout_s,
        )
        return track_path

    kick._api_call(
        sock,
        ack_sock,
        "live_set",
        "create_midi_track",
        [-1],
        _req_id("arr-create-track", track_name),
        timeout_s,
    )
    tracks_after = kick._get_children(
        sock,
        ack_sock,
        "live_set",
        "tracks",
        _req_id("arr-tracks-after", track_name),
        timeout_s,
    )
    if len(tracks_after) <= len(tracks_before):
        return None

    track_index = len(tracks_after) - 1
    track_path = f"live_set tracks {track_index}"
    kick._api_set(
        sock,
        ack_sock,
        track_path,
        "name",
        track_name,
        _req_id("arr-track-name", track_name),
        timeout_s,
    )
    return track_path


def _print_multi_pass_reports(reports: Sequence[Mapping[str, Any]]) -> None:
    for report in reports:
        pass_index = int(report.get("pass_index", 0))
        pass_name = str(report.get("pass_name", "unknown"))
        before = int(report.get("note_count_before", 0))
        after = int(report.get("note_count_after", 0))
        changed_tracks = report.get("changed_tracks", [])
        if isinstance(changed_tracks, Sequence) and not isinstance(changed_tracks, (str, bytes)):
            changed_count = len(list(changed_tracks))
        else:
            changed_count = int(report.get("changed_track_count", 0))
        print(
            f"info: multi-pass[{pass_index}] {pass_name} "
            f"notes {before}->{after} changed_tracks={changed_count}"
        )


def _enforce_piano_key_lock(
    *,
    arranged_by_track: Mapping[str, Sequence[tuple[Section, Sequence[dict[str, Any]]]]],
    specs: Sequence[InstrumentSpec],
    key_name: str,
) -> tuple[dict[str, list[tuple[Section, list[dict[str, Any]]]]], list[dict[str, int | str]]]:
    allowed_pitch_classes = _key_pitch_classes(key_name)
    root_pitch_class = _key_root_pitch_class(key_name)
    fifth_pitch_class = None if root_pitch_class is None else int((int(root_pitch_class) + 7) % 12)
    spec_by_name = {str(spec.name): spec for spec in specs}
    updated: dict[str, list[tuple[Section, list[dict[str, Any]]]]] = {}
    summaries: list[dict[str, int | str]] = []

    def _nearest_pitch_for_pc(
        *,
        pitch: int,
        target_pitch_class: int,
        midi_min_pitch: int,
        midi_max_pitch: int,
        preferred_center: int,
    ) -> int:
        low = max(0, min(127, int(midi_min_pitch)))
        high = max(0, min(127, int(midi_max_pitch)))
        if low > high:
            low, high = high, low
        candidates = [value for value in range(low, high + 1) if value % 12 == int(target_pitch_class)]
        if not candidates:
            return int(max(low, min(high, int(pitch))))
        return int(
            min(
                candidates,
                key=lambda value: (
                    abs(value - int(preferred_center)),
                    abs(value - int(pitch)),
                    value,
                ),
            )
        )

    for track_name, payloads in arranged_by_track.items():
        token = str(track_name).strip().lower()
        if "piano" not in token:
            updated[str(track_name)] = [
                (section, [dict(note) for note in notes])
                for section, notes in payloads
            ]
            continue

        spec = spec_by_name.get(str(track_name))
        midi_min = 0 if spec is None else int(spec.midi_min_pitch)
        midi_max = 127 if spec is None else int(spec.midi_max_pitch)
        before_out = 0
        after_out = 0
        total_notes = 0
        constrained_payloads: list[tuple[Section, list[dict[str, Any]]]] = []
        for section, notes in payloads:
            note_list = [dict(note) for note in notes]
            constrained = _constrain_notes_to_key(
                note_list,
                key_name=key_name,
                midi_min_pitch=midi_min,
                midi_max_pitch=midi_max,
            )
            if root_pitch_class is not None:
                low_register_max = min(int(midi_max), 60)
                for note in constrained:
                    pitch_value = int(note.get("pitch", 0))
                    if pitch_value > low_register_max:
                        continue
                    start_time = float(note.get("start_time", 0.0))
                    beat_index = int(round(start_time))
                    bar_index = max(0, beat_index // 4)
                    use_fifth = (
                        fifth_pitch_class is not None
                        and fifth_pitch_class != root_pitch_class
                        and (bar_index % 4) == 2
                    )
                    target_pc = int(fifth_pitch_class) if use_fifth else int(root_pitch_class)
                    preferred_center = 47 if use_fifth else 40
                    note["pitch"] = int(
                        _nearest_pitch_for_pc(
                            pitch=pitch_value,
                            target_pitch_class=target_pc,
                            midi_min_pitch=midi_min,
                            midi_max_pitch=low_register_max,
                            preferred_center=preferred_center,
                        )
                    )
            if allowed_pitch_classes:
                before_out += sum(
                    1
                    for note in note_list
                    if int(note.get("pitch", 0)) % 12 not in allowed_pitch_classes
                )
                after_out += sum(
                    1
                    for note in constrained
                    if int(note.get("pitch", 0)) % 12 not in allowed_pitch_classes
                )
            total_notes += len(constrained)
            constrained_payloads.append((section, constrained))

        updated[str(track_name)] = constrained_payloads
        summaries.append(
            {
                "track_name": str(track_name),
                "total_notes": int(total_notes),
                "out_of_key_before": int(before_out),
                "out_of_key_after": int(after_out),
            }
        )

    return updated, summaries


def _maybe_print_memory_brief(
    *,
    cfg: ArrangementConfig,
    sig_num: int,
    sig_den: int,
    bpm: float,
    mood: str,
) -> None:
    if not cfg.memory_brief:
        return
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))
        from memory import retrieval as memory_retrieval

        brief = memory_retrieval.build_context_brief(
            meter=f"{int(sig_num)}/{int(sig_den)}",
            bpm=float(bpm),
            mood=str(mood),
            focus=cfg.focus,
            max_results=max(1, int(cfg.memory_brief_results)),
            repo_root=repo_root,
        )
        print("\nMemory context:")
        print(brief)
    except Exception as exc:  # noqa: BLE001 - retrieval should not break composition flow
        print(f"warning: failed to build memory brief: {exc}", file=sys.stderr)


def _merge_section_payloads_for_single_clip(
    section_payloads: Sequence[tuple[Section, Sequence[dict[str, Any]]]],
    beats_per_bar: float,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for section, notes in section_payloads:
        section_start_offset, _, _ = _section_bounds(section, beats_per_bar)
        for note in notes:
            copied = dict(note)
            copied["start_time"] = float(note.get("start_time", 0.0)) + float(section_start_offset)
            merged.append(copied)
    merged.sort(key=lambda n: (float(n.get("start_time", 0.0)), int(n.get("pitch", 0))))
    return merged


def _overwrite_existing_clip_notes(
    sock: socket.socket,
    ack_sock: socket.socket,
    clip_path: str,
    track_path: str,
    section_index: int,
    clip_length_beats: float,
    clip_name: str,
    sig_num: int,
    sig_den: int,
    notes: Sequence[dict[str, Any]],
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
        _req_id("arr-clip-name", track_path, section_index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "signature_numerator",
        int(sig_num),
        _req_id("arr-clip-sig-num", track_path, section_index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "signature_denominator",
        int(sig_den),
        _req_id("arr-clip-sig-den", track_path, section_index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "loop_start",
        0.0,
        _req_id("arr-loop-start", track_path, section_index),
        timeout_s,
    )
    kick._api_set(
        sock,
        ack_sock,
        clip_path,
        "loop_end",
        float(clip_length_beats),
        _req_id("arr-loop-end", track_path, section_index),
        timeout_s,
    )
    def _scan_note_ids(from_time: float, time_span: float, suffix: str) -> list[int]:
        note_dump = kick._api_call(
            sock,
            ack_sock,
            clip_path,
            "get_notes_extended",
            {
                "from_pitch": 0,
                "pitch_span": 128,
                "from_time": float(from_time),
                "time_span": float(time_span),
                "return": ["note_id", "pitch", "start_time", "duration", "velocity", "mute"],
            },
            _req_id("arr-clear-notes-scan", track_path, section_index, suffix),
            max(timeout_s, 2.0),
        )
        existing_notes = _extract_notes_from_result(note_dump) or []
        return [int(note.get("note_id", 0)) for note in existing_notes if int(note.get("note_id", 0)) > 0]

    note_ids = _scan_note_ids(0.0, float(clip_length_beats), "full")
    if not note_ids:
        scan_span = max(8.0, min(32.0, float(clip_length_beats)))
        cursor = 0.0
        window_index = 1
        total_windows = max(
            1,
            int((float(clip_length_beats) / scan_span) + (1 if float(clip_length_beats) % scan_span > 1e-6 else 0)),
        )
        collected_ids: set[int] = set()
        while cursor < float(clip_length_beats) - 1e-6:
            span = min(scan_span, float(clip_length_beats) - cursor)
            window_ids = _scan_note_ids(
                float(round(cursor, 6)),
                float(round(span, 6)),
                f"{window_index}-of-{total_windows}",
            )
            for note_id in window_ids:
                collected_ids.add(int(note_id))
            cursor += span
            window_index += 1
        note_ids = sorted(collected_ids)

    if not note_ids:
        return False

    id_chunks = [list(chunk) for chunk in kick._chunk_notes(note_ids, chunk_size=128)]
    for idx, note_ids_chunk in enumerate(id_chunks, start=1):
        kick._api_call(
            sock,
            ack_sock,
            clip_path,
            "remove_notes_extended",
            {"note_ids": note_ids_chunk},
            _req_id("arr-clear-note-ids", track_path, section_index, f"{idx}-of-{len(id_chunks)}"),
            max(timeout_s, 2.0),
        )
    if notes:
        # section.index is used only for request IDs by write helpers.
        dummy_section = Section(
            index=int(section_index),
            start_bar=0,
            bar_count=1,
            label="full",
            kick_on=False,
            rim_on=False,
            hat_on=False,
            piano_mode="chords",
            kick_keep_ratio=1.0,
            rim_keep_ratio=1.0,
            hat_keep_ratio=1.0,
            hat_density="quarter",
        )
        _write_add_new_notes(
            sock,
            ack_sock,
            clip_path,
            track_path,
            dummy_section,
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
            _req_id("arr-clip-groove", track_path, section_index),
            timeout_s,
        )
    return True


def run(cfg: ArrangementConfig) -> int:
    registry_path = _resolve_registry_path(cfg.instrument_registry_path)
    specs = _load_instrument_registry(registry_path)
    if not specs:
        print("error: instrument registry resolved to an empty list", file=sys.stderr)
        return 2
    try:
        marimba_identity = _load_marimba_identity(_resolve_marimba_identity_path(cfg.marimba_identity_path))
    except Exception as exc:  # noqa: BLE001 - config parse errors should fail fast
        print(f"error: failed to load marimba identity config: {exc}", file=sys.stderr)
        return 2

    live_track_names = _build_live_track_names(specs, cfg.track_naming_mode)
    run_mood = cfg.mood
    run_key_name = cfg.key_name
    run_bpm = float(cfg.bpm)
    run_sig_num = int(cfg.sig_num)
    run_sig_den = int(cfg.sig_den)
    run_start_beats = float(cfg.start_beats)
    run_section_bars = int(cfg.section_bars)
    run_label = _run_label(run_sig_num, run_sig_den, run_bpm, run_mood)
    source_print_path: str | None = None
    multi_pass_reports: list[dict[str, Any]] = []
    marimba_runtime_meta: dict[str, Any] = {"enabled": False, "status": "not_applied"}
    section_profile_family = "legacy_arc"
    force_active: list[str] = []
    if cfg.focus:
        force_active.append(cfg.focus)
    if cfg.pair:
        force_active.append(cfg.pair)

    if cfg.composition_print_input:
        input_path = Path(cfg.composition_print_input)
        if not input_path.is_absolute():
            input_path = _repo_root() / input_path
        loaded = _load_composition_print(input_path)
        source_print_path = str(input_path)
        composition = loaded.get("composition", {})
        if isinstance(composition, dict):
            run_mood = str(composition.get("mood", run_mood))
            run_key_name = str(composition.get("key_name", run_key_name))
            run_bpm = float(composition.get("bpm", run_bpm))
            run_sig_num = int(composition.get("sig_num", run_sig_num))
            run_sig_den = int(composition.get("sig_den", run_sig_den))
            run_start_beats = float(composition.get("start_beats", run_start_beats))
            run_section_bars = int(composition.get("section_bars", run_section_bars))
            selected_minutes = float(composition.get("minutes", cfg.minutes or cfg.minutes_min))
            bars = int(composition.get("bars", 0))
        else:
            selected_minutes = float(cfg.minutes or cfg.minutes_min)
            bars = 0
        sections = loaded["sections"]
        arranged_by_track = loaded["arranged_by_track"]
        loaded_family = str(loaded.get("section_profile_family", "")).strip().lower()
        if loaded_family in {"legacy_arc", "lift_release", "wave_train"}:
            section_profile_family = loaded_family
        run_label = str(loaded.get("run_label", "")).strip() or _run_label(
            run_sig_num,
            run_sig_den,
            run_bpm,
            run_mood,
        )
        beats_per_bar = _beats_per_bar(run_sig_num, run_sig_den)
        if bars <= 0:
            bars = max(1, sum(max(1, int(section.bar_count)) for section in sections))
        clip_length = float(bars) * float(beats_per_bar)
        beat_step = _beat_step(run_sig_den)
        print(f"info: loaded composition print from {input_path}")
        if force_active:
            # If replaying a print, focus/pair flags are informational.
            print(f"info: focus/pair requested during print replay: {', '.join(force_active)}")
        if cfg.multi_pass_enabled and cfg.multi_pass_on_replay:
            arranged_by_track, multi_pass_reports = run_multi_pass_pipeline(
                arranged_by_track=arranged_by_track,
                specs=specs,
                sections=sections,
                beats_per_bar=beats_per_bar,
                beat_step=beat_step,
                key_name=run_key_name,
                pass_count=cfg.composition_passes,
                seed=cfg.duration_seed,
            )
            _print_multi_pass_reports(multi_pass_reports)
        elif cfg.multi_pass_enabled and not cfg.multi_pass_on_replay:
            print("info: multi-pass is enabled but skipped for print replay (use --multi-pass-on-replay to apply)")
    else:
        beats_per_bar = _beats_per_bar(run_sig_num, run_sig_den)
        beat_step = _beat_step(run_sig_den)
        selected_minutes = _select_minutes(
            explicit_minutes=cfg.minutes,
            minutes_min=cfg.minutes_min,
            minutes_max=cfg.minutes_max,
            seed=cfg.duration_seed,
            bpm=run_bpm,
            sig_num=run_sig_num,
            sig_den=run_sig_den,
            mood=run_mood,
            key_name=run_key_name,
        )
        bars = _bars_for_minutes(run_bpm, beats_per_bar, selected_minutes)
        clip_length = float(bars) * float(beats_per_bar)
        section_profile_family = _select_section_profile_family(
            sig_num=run_sig_num,
            sig_den=run_sig_den,
            bpm=run_bpm,
            mood=run_mood,
            seed=cfg.duration_seed,
        )
        sections = _build_sections(
            bars,
            run_section_bars,
            profile_family=section_profile_family,
        )
        sources = _build_source_sections(
            sections=sections,
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            transpose_semitones=cfg.transpose_semitones,
            key_name=run_key_name,
        )
        activation_mask = _build_activation_mask(
            specs,
            sections,
            cfg.duration_seed,
            force_active_names=force_active or None,
        )
        arranged_by_track = _arrange_from_registry(
            specs=specs,
            sources=sources,
            sections=sections,
            activation_mask=activation_mask,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
        )
        if cfg.multi_pass_enabled:
            arranged_by_track, multi_pass_reports = run_multi_pass_pipeline(
                arranged_by_track=arranged_by_track,
                specs=specs,
                sections=sections,
                beats_per_bar=beats_per_bar,
                beat_step=beat_step,
                key_name=run_key_name,
                pass_count=cfg.composition_passes,
                seed=cfg.duration_seed,
            )
            _print_multi_pass_reports(multi_pass_reports)

    arranged_by_track, marimba_runtime_meta = _apply_marimba_identity(
        arranged_by_track=arranged_by_track,
        specs=specs,
        sections=sections,
        beats_per_bar=beats_per_bar,
        beat_step=beat_step,
        identity=marimba_identity,
        requested_strategy=cfg.marimba_strategy,
        requested_family=cfg.marimba_family,
        key_name=run_key_name,
        pair_mode=cfg.marimba_pair_mode,
        focus_track=cfg.focus,
        pair_track=cfg.pair,
        bpm=run_bpm,
    )
    if marimba_runtime_meta.get("enabled"):
        print(
            "info: marimba identity applied "
            f"(strategy_usage={marimba_runtime_meta.get('strategy_usage', {})}, "
            f"pair_mode={marimba_runtime_meta.get('resolved_pair_mode')})"
        )
    elif marimba_runtime_meta.get("status") not in {"identity_disabled_or_missing", "not_applied"}:
        print(f"warning: marimba identity not applied ({marimba_runtime_meta.get('status')})")

    arranged_by_track, piano_key_lock_summaries = _enforce_piano_key_lock(
        arranged_by_track=arranged_by_track,
        specs=specs,
        key_name=run_key_name,
    )
    for summary in piano_key_lock_summaries:
        print(
            "info: piano key lock "
            f"track={summary['track_name']} "
            f"out_of_key={summary['out_of_key_before']}->{summary['out_of_key_after']} "
            f"notes={summary['total_notes']}"
        )

    apply_groove_tracks = {spec.name for spec in specs if spec.apply_groove and spec.name in arranged_by_track}
    composition_print_path: Path | None = None
    if cfg.composition_print:
        try:
            print_dir = _resolve_composition_print_dir(cfg.composition_print_dir)
            composition_print_path = _persist_composition_print(
                run_label=run_label,
                mood=run_mood,
                key_name=run_key_name,
                bpm=run_bpm,
                sig_num=run_sig_num,
                sig_den=run_sig_den,
                minutes=selected_minutes,
                bars=bars,
                section_bars=run_section_bars,
                start_beats=run_start_beats,
                section_profile_family=section_profile_family,
                registry_path=registry_path,
                track_naming_mode=cfg.track_naming_mode,
                sections=sections,
                arranged_by_track=arranged_by_track,
                output_dir=print_dir,
                source_print_path=source_print_path,
                multi_pass_report=multi_pass_reports,
            )
            print(f"info: composition print saved to {composition_print_path}")
        except Exception as exc:  # noqa: BLE001 - print artifact should not break composition flow
            print(f"warning: failed to write composition print: {exc}", file=sys.stderr)

    _maybe_print_memory_brief(
        cfg=cfg,
        sig_num=run_sig_num,
        sig_den=run_sig_den,
        bpm=run_bpm,
        mood=run_mood,
    )

    if cfg.dry_run:
        print("Arrangement plan (dry run):")
        print(f"- mood/key:    {run_mood} / {run_key_name}")
        print(f"- tempo:       {run_bpm:g}")
        print(f"- signature:   {run_sig_num}/{run_sig_den}")
        print(f"- beats/bar:   {beats_per_bar:g}")
        print(
            f"- minutes:     {selected_minutes:g} "
            f"(range {cfg.minutes_min:g}-{cfg.minutes_max:g}, seed {cfg.duration_seed})"
        )
        print(f"- bars:        {bars}")
        print(f"- length:      {clip_length:g} beats")
        print(f"- sections:    {len(sections)} x ~{run_section_bars} bars")
        print(f"- instruments: {len(specs)} (registry: {registry_path})")
        print(f"- track names: {cfg.track_naming_mode}")
        print(f"- clip mode:   {cfg.clip_write_mode}")
        print(f"- form family: {section_profile_family}")
        print(f"- marimba family: {cfg.marimba_family}")
        print(f"- marimba strategy: {cfg.marimba_strategy}")
        print(f"- marimba pair: {cfg.marimba_pair_mode}")
        print(f"- multi-pass:  {'on' if cfg.multi_pass_enabled else 'off'} ({cfg.composition_passes} pass(es))")
        if cfg.composition_goal:
            print(f"- goal:        {cfg.composition_goal}")
        if cfg.focus:
            print(f"- focus:       {cfg.focus}")
        if cfg.pair:
            print(f"- pair:        {cfg.pair}")
        print(f"- run label:   {run_label}")
        for section in sections:
            start, end, length = _section_bounds(section, beats_per_bar)
            active_count = sum(
                1 for _track, payloads in arranged_by_track.items() if payloads[section.index][1]
            )
            print(
                f"  section {section.index:02d}: bars {section.start_bar + 1}-{section.start_bar + section.bar_count} "
                f"({length:g} beats) [{section.label}] active={active_count}"
            )
        print("\nDry run only. No OSC messages were sent.")
        if cfg.eval_log:
            try:
                relative_log_dir = Path(cfg.eval_log_dir) if cfg.eval_log_dir else feedback.DEFAULT_RELATIVE_LOG_DIR
                artifact, artifact_path = feedback.log_composition_run(
                    mood=run_mood,
                    key_name=run_key_name,
                    bpm=run_bpm,
                    sig_num=run_sig_num,
                    sig_den=run_sig_den,
                    minutes=selected_minutes,
                    bars=bars,
                    section_bars=run_section_bars,
                    sections=sections,
                    arranged_by_track=arranged_by_track,
                    created_clips_by_track={track: 0 for track in arranged_by_track.keys()},
                    status="dry_run",
                    relative_log_dir=relative_log_dir,
                    run_metadata={
                        "duration_seed": cfg.duration_seed,
                        "minutes_min": cfg.minutes_min,
                        "minutes_max": cfg.minutes_max,
                        "selected_minutes": selected_minutes,
                        "save_policy": cfg.save_policy,
                        "instrument_count": len(specs),
                        "run_label": run_label,
                        "section_profile_family": section_profile_family,
                        "marimba_runtime": marimba_runtime_meta,
                        "multi_pass_enabled": cfg.multi_pass_enabled,
                        "composition_passes": cfg.composition_passes,
                        "multi_pass_on_replay": cfg.multi_pass_on_replay,
                        "multi_pass_reports": multi_pass_reports,
                        "focus": cfg.focus,
                        "pair": cfg.pair,
                        "composition_goal": cfg.composition_goal,
                        "composition_print_source": source_print_path,
                        "composition_print_path": None
                        if composition_print_path is None
                        else str(composition_print_path),
                    },
                    instrument_identity_contract=None
                    if marimba_identity is None
                    else marimba_identity.payload,
                    human_feedback={
                        "mode": cfg.human_feedback_mode,
                        "text": cfg.human_feedback_text,
                    },
                )
                novelty = artifact.get("reflection", {}).get("novelty_score")
                print(f"info: eval artifact saved to {artifact_path}")
                print(f"info: novelty score {novelty if novelty is not None else 'n/a'}")
            except Exception as exc:  # noqa: BLE001 - logging should not break composition flow
                print(f"warning: failed to write eval artifact: {exc}", file=sys.stderr)
        return 0

    bridge_cfg = bridge.BridgeConfig(
        host=HOST,
        port=PORT,
        ack_port=ACK_PORT,
        ack_timeout_s=cfg.ack_timeout_s,
        expect_ack=True,
        ping_first=False,
        status=False,
        tempo=None,
        sig_num=None,
        sig_den=None,
        create_midi_tracks=0,
        add_midi_tracks=0,
        midi_name="MIDI",
        create_audio_tracks=0,
        add_audio_tracks=0,
        audio_prefix="Audio",
        delete_audio_tracks=0,
        delete_midi_tracks=0,
        rename_track_index=None,
        rename_track_name=None,
        session_clip_track_index=None,
        session_clip_slot_index=None,
        session_clip_length=None,
        session_clip_notes_json=None,
        session_clip_name=None,
        append_session_clip_track_index=None,
        append_session_clip_slot_index=None,
        append_session_clip_notes_json=None,
        inspect_session_clip_track_index=None,
        inspect_session_clip_slot_index=None,
        ensure_midi_tracks=None,
        midi_ccs=(),
        cc64s=(),
        api_pings=(),
        api_gets=(),
        api_sets=(),
        api_calls=(),
        api_children=(),
        api_describes=(),
        ack_mode=cfg.ack_mode,
        ack_flush_interval=cfg.ack_flush_interval,
        report_metrics=True,
        delay_ms=0,
        dry_run=False,
    )

    ack_sock = bridge.open_ack_socket(bridge_cfg)
    if ack_sock is None:
        print("error: failed to open ack socket", file=sys.stderr)
        return 1

    run_status = "success"
    save_result: dict[str, Any] = {
        "policy": cfg.save_policy,
        "status": "ephemeral",
        "path": None,
        "message": "intentional non-persistence",
    }

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        print(f"\nTarget: udp://{HOST}:{PORT}")
        print(f"Ack:    udp://{HOST}:{ACK_PORT} (timeout {cfg.ack_timeout_s:.2f}s)")

        # Prime the bridge and set tempo/signature explicitly for this run.
        for cmd in (
            bridge.OscCommand("/ping"),
            bridge.OscCommand("/status"),
            bridge.OscCommand("/tempo", (float(run_bpm),)),
            bridge.OscCommand("/sig_num", (int(run_sig_num),)),
            bridge.OscCommand("/sig_den", (int(run_sig_den),)),
        ):
            print(f"sent: {bridge.describe_command(cmd)}")
            _print_acks(_send_and_collect_acks(sock, ack_sock, cmd, cfg.ack_timeout_s))

        tempo_verified, sig_num_verified, sig_den_verified = _ensure_live_transport(
            sock=sock,
            ack_sock=ack_sock,
            bpm=run_bpm,
            sig_num=run_sig_num,
            sig_den=run_sig_den,
            timeout_s=cfg.ack_timeout_s,
        )
        if (
            tempo_verified is None
            or abs(float(tempo_verified) - float(run_bpm)) > 1e-6
            or sig_num_verified != int(run_sig_num)
            or sig_den_verified != int(run_sig_den)
        ):
            print(
                "warning: live_set transport did not fully verify; "
                f"observed tempo/signature {tempo_verified} {sig_num_verified}/{sig_den_verified}",
                file=sys.stderr,
            )

        key_applied = _apply_live_key(sock, ack_sock, run_key_name, cfg.ack_timeout_s)
        if key_applied is not None:
            root_note, scale_name = key_applied
            print(f"info: applied live key root_note={root_note} scale_name={scale_name}")
        else:
            print(f"warning: could not parse/apply key '{run_key_name}'")

        # Resolve track registry by name.
        track_paths: dict[str, str] = {}
        for instrument_name in arranged_by_track.keys():
            live_track_name = live_track_names.get(instrument_name, instrument_name)
            track_path = _resolve_track_path(sock, ack_sock, live_track_name, cfg.ack_timeout_s)
            if not track_path:
                print(
                    (
                        f"error: could not resolve track '{live_track_name}' "
                        f"(instrument '{instrument_name}'); reload the device in Live"
                    ),
                    file=sys.stderr,
                )
                ack_sock.close()
                return 2
            track_paths[instrument_name] = track_path

        arrangement_start = float(run_start_beats)
        arrangement_end = arrangement_start + clip_length

        print("\nArrangement plan:")
        print(f"- mood/key:    {run_mood} / {run_key_name}")
        print(f"- tempo:       {run_bpm:g}")
        print(f"- signature:   {run_sig_num}/{run_sig_den}")
        print(f"- beats/bar:   {beats_per_bar:g}")
        print(
            f"- minutes:     {selected_minutes:g} "
            f"(range {cfg.minutes_min:g}-{cfg.minutes_max:g}, seed {cfg.duration_seed})"
        )
        print(f"- bars:        {bars}")
        print(f"- length:      {clip_length:g} beats")
        print(f"- start/end:   {arrangement_start:g} -> {arrangement_end:g}")
        print(f"- sections:    {len(sections)} x ~{run_section_bars} bars")
        print(f"- instruments: {len(specs)} (registry: {registry_path})")
        print(f"- track names: {cfg.track_naming_mode}")
        print(f"- form family: {section_profile_family}")
        print(f"- marimba family: {cfg.marimba_family}")
        print(f"- marimba strategy: {cfg.marimba_strategy}")
        print(f"- marimba pair: {cfg.marimba_pair_mode}")
        print(f"- multi-pass:  {'on' if cfg.multi_pass_enabled else 'off'} ({cfg.composition_passes} pass(es))")
        if cfg.focus:
            print(f"- focus:       {cfg.focus}")
        if cfg.pair:
            print(f"- pair:        {cfg.pair}")
        print(f"- run label:   {run_label}")

        cache_entries: dict[str, str] = {}
        cache_path = _resolve_cache_path(cfg.cache_path)
        if cfg.cache_enabled:
            cache_entries = _load_write_cache(cache_path)
            print(f"info: write cache loaded from {cache_path}")
        live_set_id = _live_set_identity(sock, ack_sock, cfg.ack_timeout_s)

        if not cfg.cache_enabled and cfg.clip_write_mode != "single_clip":
            total_deleted = 0
            for track_path in track_paths.values():
                deleted = _delete_overlaps(
                    sock,
                    ack_sock,
                    track_path,
                    arrangement_start,
                    arrangement_end,
                    cfg.ack_timeout_s,
                )
                total_deleted += deleted
            if total_deleted > 0:
                print(f"info: deleted {total_deleted} overlapping arrangement clip(s)")

        groove_id = kick._find_groove_id_by_name(sock, ack_sock, cfg.groove_name, cfg.ack_timeout_s)
        if groove_id:
            kick._api_set(
                sock,
                ack_sock,
                "live_set",
                "groove_amount",
                1.0,
                "arr-groove-amount",
                cfg.ack_timeout_s,
            )
            print(f"info: groove '{cfg.groove_name}' resolved to id={groove_id}")
        else:
            print(f"warning: groove '{cfg.groove_name}' not found in groove pool")

        created_by_track: dict[str, int] = {}
        for track_name, section_payloads in arranged_by_track.items():
            track_path = track_paths[track_name]
            live_track_name = live_track_names.get(track_name, track_name)
            apply_groove_track = track_name in apply_groove_tracks
            track_refs = _list_arrangement_clips(
                sock,
                ack_sock,
                track_path,
                cfg.ack_timeout_s,
                request_prefix=f"arr-scan-{live_track_name}-initial",
            )
            created = 0
            updated = 0
            skipped = 0
            deleted = 0

            if cfg.clip_write_mode == "single_clip":
                merged_notes = _merge_section_payloads_for_single_clip(section_payloads, beats_per_bar)
                clip_name = f"{run_label} {live_track_name}"
                payload_hash = _notes_payload_hash(merged_notes)
                cache_key = _write_cache_key(
                    live_set_identity=live_set_id,
                    track_path=track_path,
                    section_start_beats=arrangement_start,
                    section_length_beats=clip_length,
                    bpm=run_bpm,
                    sig_num=run_sig_num,
                    sig_den=run_sig_den,
                )
                existing_clip = _find_matching_arrangement_clip(
                    track_refs,
                    arrangement_start,
                    clip_length,
                )
                cached_hash = cache_entries.get(cache_key) if cfg.cache_enabled else None
                if cfg.cache_enabled and cached_hash == payload_hash and existing_clip is not None:
                    skipped = 1
                    print(
                        f"info: skipped unchanged clip track={live_track_name} "
                        f"start={arrangement_start:g} len={clip_length:g}"
                    )
                elif existing_clip is not None:
                    overwrite_ok = _overwrite_existing_clip_notes(
                        sock,
                        ack_sock,
                        existing_clip.path,
                        track_path,
                        section_index=0,
                        clip_length_beats=clip_length,
                        clip_name=clip_name,
                        sig_num=run_sig_num,
                        sig_den=run_sig_den,
                        notes=merged_notes,
                        timeout_s=cfg.ack_timeout_s,
                        note_chunk_size=cfg.note_chunk_size,
                        groove_id=groove_id,
                        apply_groove=apply_groove_track,
                    )
                    if overwrite_ok:
                        updated = 1
                        if cfg.cache_enabled:
                            cache_entries[cache_key] = payload_hash
                    else:
                        deleted_count, track_refs = _delete_overlaps_from_refs(
                            sock,
                            ack_sock,
                            track_path,
                            track_refs,
                            arrangement_start,
                            arrangement_end,
                            cfg.ack_timeout_s,
                            preserve_clip_id=None,
                            request_prefix="arr-delete-overlap-full-single",
                        )
                        deleted += deleted_count
                        created_ref = _create_section_clip(
                            sock=sock,
                            ack_sock=ack_sock,
                            track_path=track_path,
                            section=sections[0],
                            section_start_beats=arrangement_start,
                            section_length_beats=clip_length,
                            clip_name=clip_name,
                            sig_num=run_sig_num,
                            sig_den=run_sig_den,
                            notes=merged_notes,
                            timeout_s=cfg.ack_timeout_s,
                            note_chunk_size=cfg.note_chunk_size,
                            groove_id=groove_id,
                            apply_groove=apply_groove_track,
                        )
                        if created_ref is not None:
                            created = 1
                            track_refs = _upsert_clip_ref(track_refs, created_ref)
                        if cfg.cache_enabled:
                            cache_entries[cache_key] = payload_hash
                else:
                    deleted_count, track_refs = _delete_overlaps_from_refs(
                        sock,
                        ack_sock,
                        track_path,
                        track_refs,
                        arrangement_start,
                        arrangement_end,
                        cfg.ack_timeout_s,
                        preserve_clip_id=None,
                        request_prefix="arr-delete-overlap-full-single",
                    )
                    deleted += deleted_count
                    created_ref = _create_section_clip(
                        sock=sock,
                        ack_sock=ack_sock,
                        track_path=track_path,
                        section=sections[0],
                        section_start_beats=arrangement_start,
                        section_length_beats=clip_length,
                        clip_name=clip_name,
                        sig_num=run_sig_num,
                        sig_den=run_sig_den,
                        notes=merged_notes,
                        timeout_s=cfg.ack_timeout_s,
                        note_chunk_size=cfg.note_chunk_size,
                        groove_id=groove_id,
                        apply_groove=apply_groove_track,
                    )
                    if created_ref is not None:
                        created = 1
                        track_refs = _upsert_clip_ref(track_refs, created_ref)
                    if cfg.cache_enabled:
                        cache_entries[cache_key] = payload_hash

                created_by_track[track_name] = created + updated
                print(
                    f"info: track '{live_track_name}' created={created} updated={updated} "
                    f"skipped={skipped} deleted={deleted}"
                )
                continue

            for section, notes in section_payloads:
                section_start_offset, _, section_length = _section_bounds(section, beats_per_bar)
                section_start_beats = arrangement_start + section_start_offset
                section_end_beats = section_start_beats + section_length
                clip_name = f"{run_label} {live_track_name} {section.label}"
                payload_hash = _notes_payload_hash(notes)
                cache_key = _write_cache_key(
                    live_set_identity=live_set_id,
                    track_path=track_path,
                    section_start_beats=section_start_beats,
                    section_length_beats=section_length,
                    bpm=run_bpm,
                    sig_num=run_sig_num,
                    sig_den=run_sig_den,
                )

                existing_clip = _find_matching_arrangement_clip(
                    track_refs,
                    section_start_beats,
                    section_length,
                )

                cached_hash = cache_entries.get(cache_key) if cfg.cache_enabled else None
                if cfg.cache_enabled and cached_hash == payload_hash:
                    if existing_clip is not None or not notes:
                        skipped += 1
                        print(
                            f"info: skipped unchanged clip track={live_track_name} "
                            f"start={section_start_beats:g} len={section_length:g}"
                        )
                        continue

                if cfg.write_strategy == "delta_update" and existing_clip is not None:
                    deleted_count, track_refs = _delete_overlaps_from_refs(
                        sock,
                        ack_sock,
                        track_path,
                        track_refs,
                        section_start_beats,
                        section_end_beats,
                        cfg.ack_timeout_s,
                        preserve_clip_id=existing_clip.clip_id,
                        request_prefix=f"arr-delete-overlap-delta-{section.index}",
                    )
                    deleted += deleted_count
                    ok = _update_existing_section_clip_delta(
                        sock=sock,
                        ack_sock=ack_sock,
                        clip_path=existing_clip.path,
                        track_path=track_path,
                        section=section,
                        section_length_beats=section_length,
                        clip_name=clip_name,
                        sig_num=run_sig_num,
                        sig_den=run_sig_den,
                        notes=notes,
                        timeout_s=cfg.ack_timeout_s,
                        note_chunk_size=cfg.note_chunk_size,
                        groove_id=groove_id,
                        apply_groove=apply_groove_track,
                    )
                    if ok:
                        updated += 1
                        if cfg.cache_enabled:
                            cache_entries[cache_key] = payload_hash
                        continue
                    print(
                        (
                            f"warning: delta update fallback to full replace on "
                            f"{live_track_name} section {section.index}"
                        )
                    )

                deleted_count, track_refs = _delete_overlaps_from_refs(
                    sock,
                    ack_sock,
                    track_path,
                    track_refs,
                    section_start_beats,
                    section_end_beats,
                    cfg.ack_timeout_s,
                    preserve_clip_id=None,
                    request_prefix=f"arr-delete-overlap-full-{section.index}",
                )
                deleted += deleted_count
                if not notes:
                    if cfg.cache_enabled:
                        cache_entries[cache_key] = payload_hash
                    continue

                created_ref = _create_section_clip(
                    sock=sock,
                    ack_sock=ack_sock,
                    track_path=track_path,
                    section=section,
                    section_start_beats=section_start_beats,
                    section_length_beats=section_length,
                    clip_name=clip_name,
                    sig_num=run_sig_num,
                    sig_den=run_sig_den,
                    notes=notes,
                    timeout_s=cfg.ack_timeout_s,
                    note_chunk_size=cfg.note_chunk_size,
                    groove_id=groove_id,
                    apply_groove=apply_groove_track,
                )
                if created_ref is not None:
                    created += 1
                    track_refs = _upsert_clip_ref(track_refs, created_ref)
                    if cfg.cache_enabled:
                        cache_entries[cache_key] = payload_hash

            created_by_track[track_name] = created + updated
            print(
                f"info: track '{live_track_name}' created={created} updated={updated} "
                f"skipped={skipped} deleted={deleted}"
            )

        if cfg.cache_enabled:
            _save_write_cache(cache_path, cache_entries)
            print(f"info: write cache updated at {cache_path}")

        if cfg.save_policy == "ephemeral":
            print("info: save policy is ephemeral (run intentionally not persisted as a Live Set file)")
        else:
            archive_dir = _resolve_archive_dir(cfg.archive_dir)
            date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            archive_path = archive_dir / date_slug / f"{run_label}.als"
            ok, save_message = _archive_live_set(
                sock,
                ack_sock,
                cfg.ack_timeout_s,
                archive_path,
            )
            if ok and archive_path.exists():
                save_result = {
                    "policy": cfg.save_policy,
                    "status": "archived",
                    "path": str(archive_path),
                    "message": save_message,
                }
                print(f"info: archived Live Set to {archive_path}")
            else:
                run_status = "save_failed"
                save_result = {
                    "policy": cfg.save_policy,
                    "status": "archive_failed",
                    "path": str(archive_path),
                    "message": save_message,
                }
                print(
                    f"error: archive save failed for {archive_path} ({save_message})",
                    file=sys.stderr,
                )

    ack_sock.close()

    if cfg.eval_log:
        try:
            relative_log_dir = Path(cfg.eval_log_dir) if cfg.eval_log_dir else feedback.DEFAULT_RELATIVE_LOG_DIR
            artifact, artifact_path = feedback.log_composition_run(
                mood=run_mood,
                key_name=run_key_name,
                bpm=run_bpm,
                sig_num=run_sig_num,
                sig_den=run_sig_den,
                minutes=selected_minutes,
                bars=bars,
                section_bars=run_section_bars,
                sections=sections,
                arranged_by_track=arranged_by_track,
                created_clips_by_track=created_by_track,
                status=run_status,
                relative_log_dir=relative_log_dir,
                run_metadata={
                    "duration_seed": cfg.duration_seed,
                    "minutes_min": cfg.minutes_min,
                    "minutes_max": cfg.minutes_max,
                    "selected_minutes": selected_minutes,
                    "run_label": run_label,
                    "section_profile_family": section_profile_family,
                    "save": save_result,
                    "instrument_count": len(specs),
                    "registry_path": str(registry_path),
                    "marimba_runtime": marimba_runtime_meta,
                    "multi_pass_enabled": cfg.multi_pass_enabled,
                    "composition_passes": cfg.composition_passes,
                    "multi_pass_on_replay": cfg.multi_pass_on_replay,
                    "multi_pass_reports": multi_pass_reports,
                    "focus": cfg.focus,
                    "pair": cfg.pair,
                    "composition_goal": cfg.composition_goal,
                    "composition_print_source": source_print_path,
                    "composition_print_path": None
                    if composition_print_path is None
                    else str(composition_print_path),
                },
                instrument_identity_contract=None
                if marimba_identity is None
                else marimba_identity.payload,
                human_feedback={
                    "mode": cfg.human_feedback_mode,
                    "text": cfg.human_feedback_text,
                },
            )
            novelty = artifact.get("reflection", {}).get("novelty_score")
            print(f"info: eval artifact saved to {artifact_path}")
            print(f"info: novelty score {novelty if novelty is not None else 'n/a'}")
            for flag in artifact.get("reflection", {}).get("repetition_flags", []):
                print(f"info: repetition flag: {flag}")
        except Exception as exc:  # noqa: BLE001 - logging should not break composition flow
            print(f"warning: failed to write eval artifact: {exc}", file=sys.stderr)

    if cfg.save_policy == "archive" and save_result.get("status") != "archived":
        return 3
    return 0

def main(argv: Iterable[str]) -> int:
    cfg = parse_args(argv)
    try:
        return run(cfg)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
