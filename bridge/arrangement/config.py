from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

from arrangement.base import (
    DEFAULT_ARCHIVE_DIR,
    DEFAULT_BPM,
    DEFAULT_COMPOSITION_PRINT_DIR,
    DEFAULT_CLIP_WRITE_MODE,
    DEFAULT_GROOVE_NAME,
    DEFAULT_HAT_TRACK,
    DEFAULT_INSTRUMENT_REGISTRY_PATH,
    DEFAULT_KEY,
    DEFAULT_KICK_TRACK,
    DEFAULT_MARIMBA_IDENTITY_PATH,
    DEFAULT_MINUTES_MAX,
    DEFAULT_MINUTES_MIN,
    DEFAULT_MOOD,
    DEFAULT_NOTE_CHUNK_SIZE,
    DEFAULT_PIANO_TRACK,
    DEFAULT_RIM_TRACK,
    DEFAULT_SAVE_POLICY,
    DEFAULT_SECTION_BARS,
    DEFAULT_SIG_DEN,
    DEFAULT_SIG_NUM,
    DEFAULT_TRACK_NAMING_MODE,
    DEFAULT_TRANSPOSE,
    DEFAULT_WRITE_CACHE_PATH,
    DEFAULT_WRITE_STRATEGY,
    MarimbaPairMode,
    SavePolicy,
    TrackNamingMode,
    ClipWriteMode,
    WriteStrategy,
)

DEFAULT_COMPOSITION_PASSES = 5


@dataclass(frozen=True)
class ArrangementConfig:
    minutes: float | None
    minutes_min: float
    minutes_max: float
    duration_seed: int
    bpm: float
    sig_num: int
    sig_den: int
    start_beats: float
    section_bars: int
    transpose_semitones: int
    mood: str
    key_name: str
    kick_track: str
    rim_track: str
    hat_track: str
    piano_track: str
    groove_name: str
    ack_timeout_s: float
    note_chunk_size: int
    ack_mode: str
    ack_flush_interval: int
    write_strategy: WriteStrategy
    cache_enabled: bool
    cache_path: str | None
    instrument_registry_path: str | None
    marimba_identity_path: str | None
    marimba_family: str
    marimba_strategy: str
    marimba_pair_mode: MarimbaPairMode
    focus: str | None
    pair: str | None
    track_naming_mode: TrackNamingMode
    clip_write_mode: ClipWriteMode
    save_policy: SavePolicy
    archive_dir: str | None
    composition_print: bool
    composition_print_dir: str | None
    composition_print_input: str | None
    composition_passes: int
    multi_pass_enabled: bool
    multi_pass_on_replay: bool
    eval_log: bool
    eval_log_dir: str | None
    memory_brief: bool
    memory_brief_results: int
    dry_run: bool

def parse_args(argv: Iterable[str]) -> ArrangementConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="Exact target minutes (overrides --minutes-min/--minutes-max)",
    )
    parser.add_argument(
        "--minutes-min",
        type=float,
        default=DEFAULT_MINUTES_MIN,
        help=f"Minimum target minutes when --minutes is not set (default: {DEFAULT_MINUTES_MIN})",
    )
    parser.add_argument(
        "--minutes-max",
        type=float,
        default=DEFAULT_MINUTES_MAX,
        help=f"Maximum target minutes when --minutes is not set (default: {DEFAULT_MINUTES_MAX})",
    )
    parser.add_argument(
        "--duration-seed",
        type=int,
        default=17,
        help="Deterministic seed used when selecting minutes in a range (default: 17)",
    )
    parser.add_argument("--bpm", type=float, default=DEFAULT_BPM, help="Tempo in BPM")
    parser.add_argument("--sig-num", type=int, default=DEFAULT_SIG_NUM, help="Time signature numerator")
    parser.add_argument("--sig-den", type=int, default=DEFAULT_SIG_DEN, help="Time signature denominator")
    parser.add_argument(
        "--start-beats",
        type=float,
        default=0.0,
        help="Arrangement start time in beats (default: 0)",
    )
    parser.add_argument(
        "--section-bars",
        type=int,
        default=DEFAULT_SECTION_BARS,
        help="Bars per arrangement section (default: 8)",
    )
    parser.add_argument(
        "--transpose-semitones",
        type=int,
        default=DEFAULT_TRANSPOSE,
        help="Transpose the piano palette by semitones (default: 2)",
    )
    parser.add_argument("--mood", default=DEFAULT_MOOD, help="Mood label for logging")
    parser.add_argument("--key-name", default=DEFAULT_KEY, help="Key label for logging")
    parser.add_argument("--kick-track", default=DEFAULT_KICK_TRACK, help="Kick track name")
    parser.add_argument("--rim-track", default=DEFAULT_RIM_TRACK, help="Rim track name")
    parser.add_argument("--hat-track", default=DEFAULT_HAT_TRACK, help="Hat track name")
    parser.add_argument("--piano-track", default=DEFAULT_PIANO_TRACK, help="Piano track name")
    parser.add_argument(
        "--groove-name",
        default=DEFAULT_GROOVE_NAME,
        help=f"Groove Pool name to apply to drums (default: {DEFAULT_GROOVE_NAME})",
    )
    parser.add_argument(
        "--ack-timeout",
        type=float,
        default=1.75,
        help="Ack wait timeout in seconds (default: 1.75)",
    )
    parser.add_argument(
        "--note-chunk-size",
        type=int,
        default=DEFAULT_NOTE_CHUNK_SIZE,
        help=f"Notes per add_new_notes call (default: {DEFAULT_NOTE_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--ack-mode",
        choices=("per_command", "flush_end", "flush_interval"),
        default="per_command",
        help="Bridge ACK handling mode (default: per_command)",
    )
    parser.add_argument(
        "--ack-flush-interval",
        type=int,
        default=10,
        help="ACK flush interval when --ack-mode=flush_interval (default: 10)",
    )
    parser.add_argument(
        "--write-strategy",
        choices=("full_replace", "delta_update"),
        default=DEFAULT_WRITE_STRATEGY,
        help="Clip write strategy (default: full_replace)",
    )
    parser.add_argument(
        "--no-write-cache",
        action="store_true",
        help="Disable unchanged clip skip cache",
    )
    parser.add_argument(
        "--write-cache-path",
        default=str(DEFAULT_WRITE_CACHE_PATH),
        help=f"Write cache artifact path (default: {DEFAULT_WRITE_CACHE_PATH})",
    )
    parser.add_argument(
        "--instrument-registry-path",
        default=str(DEFAULT_INSTRUMENT_REGISTRY_PATH),
        help=f"Instrument registry path (default: {DEFAULT_INSTRUMENT_REGISTRY_PATH})",
    )
    parser.add_argument(
        "--marimba-identity-path",
        default=str(DEFAULT_MARIMBA_IDENTITY_PATH),
        help=f"Marimba identity config path (default: {DEFAULT_MARIMBA_IDENTITY_PATH})",
    )
    parser.add_argument(
        "--marimba-family",
        choices=("auto", "legacy_sectional", "evolving_ostinato", "left_hand_ostinato_right_hand_melody"),
        default="auto",
        help="Marimba macro composition family (default: auto)",
    )
    parser.add_argument(
        "--marimba-strategy",
        choices=("auto", "ostinato_pulse", "broken_resonance", "chord_bloom", "lyrical_roll"),
        default="auto",
        help="Marimba strategy family (default: auto)",
    )
    parser.add_argument(
        "--marimba-pair-mode",
        choices=("auto", "off", "attack_answer"),
        default="auto",
        help="Marimba/vibraphone interaction mode (default: auto)",
    )
    parser.add_argument(
        "--focus",
        default=None,
        help="Optional focus instrument name for identity study (for example: Marimba)",
    )
    parser.add_argument(
        "--pair",
        default=None,
        help="Optional paired instrument name for identity study (for example: Vibraphone)",
    )
    parser.add_argument(
        "--track-naming-mode",
        choices=("slot", "registry"),
        default=DEFAULT_TRACK_NAMING_MODE,
        help="Live-facing track naming mode (default: slot)",
    )
    parser.add_argument(
        "--clip-write-mode",
        choices=("section_clips", "single_clip"),
        default=DEFAULT_CLIP_WRITE_MODE,
        help="Arrangement clip write mode (default: section_clips)",
    )
    parser.add_argument(
        "--save-policy",
        choices=("ephemeral", "archive"),
        default=DEFAULT_SAVE_POLICY,
        help="Live set save policy at end of run (default: ephemeral)",
    )
    parser.add_argument(
        "--archive-dir",
        default=str(DEFAULT_ARCHIVE_DIR),
        help=f"Archive directory used when --save-policy=archive (default: {DEFAULT_ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--no-composition-print",
        action="store_true",
        help="Disable full composition print artifact logging for this run",
    )
    parser.add_argument(
        "--composition-print-dir",
        default=str(DEFAULT_COMPOSITION_PRINT_DIR),
        help=f"Composition print output dir (default: {DEFAULT_COMPOSITION_PRINT_DIR})",
    )
    parser.add_argument(
        "--composition-print-input",
        default=None,
        help="Optional composition print JSON to replay instead of generating new notes",
    )
    parser.add_argument(
        "--composition-passes",
        type=int,
        default=DEFAULT_COMPOSITION_PASSES,
        help=f"Number of composition passes to run when multi-pass is enabled (default: {DEFAULT_COMPOSITION_PASSES})",
    )
    parser.add_argument(
        "--no-multi-pass",
        action="store_true",
        help="Disable the multi-pass composition pipeline",
    )
    parser.add_argument(
        "--multi-pass-on-replay",
        action="store_true",
        help="Apply multi-pass transforms when replaying --composition-print-input",
    )
    parser.add_argument(
        "--eval-log-dir",
        default=None,
        help="Optional relative path for composition eval artifacts (default: memory/evals/compositions)",
    )
    parser.add_argument(
        "--no-eval-log",
        action="store_true",
        help="Disable composition eval artifact logging for this run",
    )
    parser.add_argument(
        "--memory-brief",
        action="store_true",
        help="Print a retrieval brief from local memory/eval context before composing",
    )
    parser.add_argument(
        "--memory-brief-results",
        type=int,
        default=6,
        help="How many retrieval hits to include in --memory-brief output (default: 6)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without sending OSC messages",
    )

    ns = parser.parse_args(list(argv))

    if ns.minutes is not None and ns.minutes <= 0:
        parser.error("--minutes must be > 0")
    if ns.minutes_min <= 0:
        parser.error("--minutes-min must be > 0")
    if ns.minutes_max <= 0:
        parser.error("--minutes-max must be > 0")
    if ns.minutes_max < ns.minutes_min:
        parser.error("--minutes-max must be >= --minutes-min")
    if ns.bpm <= 0:
        parser.error("--bpm must be > 0")
    if ns.sig_num <= 0:
        parser.error("--sig-num must be > 0")
    if ns.sig_den <= 0:
        parser.error("--sig-den must be > 0")
    if ns.section_bars <= 0:
        parser.error("--section-bars must be > 0")
    if ns.ack_timeout <= 0:
        parser.error("--ack-timeout must be > 0")
    if ns.note_chunk_size <= 0:
        parser.error("--note-chunk-size must be > 0")
    if ns.ack_flush_interval <= 0:
        parser.error("--ack-flush-interval must be > 0")
    if ns.composition_passes <= 0:
        parser.error("--composition-passes must be > 0")
    if ns.memory_brief_results <= 0:
        parser.error("--memory-brief-results must be > 0")

    return ArrangementConfig(
        minutes=None if ns.minutes is None else float(ns.minutes),
        minutes_min=float(ns.minutes_min),
        minutes_max=float(ns.minutes_max),
        duration_seed=int(ns.duration_seed),
        bpm=float(ns.bpm),
        sig_num=int(ns.sig_num),
        sig_den=int(ns.sig_den),
        start_beats=float(ns.start_beats),
        section_bars=int(ns.section_bars),
        transpose_semitones=int(ns.transpose_semitones),
        mood=str(ns.mood),
        key_name=str(ns.key_name),
        kick_track=str(ns.kick_track),
        rim_track=str(ns.rim_track),
        hat_track=str(ns.hat_track),
        piano_track=str(ns.piano_track),
        groove_name=str(ns.groove_name),
        ack_timeout_s=float(ns.ack_timeout),
        note_chunk_size=int(ns.note_chunk_size),
        ack_mode=str(ns.ack_mode),
        ack_flush_interval=int(ns.ack_flush_interval),
        write_strategy=str(ns.write_strategy),  # type: ignore[arg-type]
        cache_enabled=not bool(ns.no_write_cache),
        cache_path=None if ns.write_cache_path in (None, "") else str(ns.write_cache_path),
        instrument_registry_path=None
        if ns.instrument_registry_path in (None, "")
        else str(ns.instrument_registry_path),
        marimba_identity_path=None
        if ns.marimba_identity_path in (None, "")
        else str(ns.marimba_identity_path),
        marimba_family=str(ns.marimba_family),
        marimba_strategy=str(ns.marimba_strategy),
        marimba_pair_mode=str(ns.marimba_pair_mode),  # type: ignore[arg-type]
        focus=None if ns.focus in (None, "") else str(ns.focus),
        pair=None if ns.pair in (None, "") else str(ns.pair),
        track_naming_mode=str(ns.track_naming_mode),  # type: ignore[arg-type]
        clip_write_mode=str(ns.clip_write_mode),  # type: ignore[arg-type]
        save_policy=str(ns.save_policy),  # type: ignore[arg-type]
        archive_dir=None if ns.archive_dir in (None, "") else str(ns.archive_dir),
        composition_print=not bool(ns.no_composition_print),
        composition_print_dir=None
        if ns.composition_print_dir in (None, "")
        else str(ns.composition_print_dir),
        composition_print_input=None
        if ns.composition_print_input in (None, "")
        else str(ns.composition_print_input),
        composition_passes=int(ns.composition_passes),
        multi_pass_enabled=not bool(ns.no_multi_pass),
        multi_pass_on_replay=bool(ns.multi_pass_on_replay),
        eval_log=not bool(ns.no_eval_log),
        eval_log_dir=None if ns.eval_log_dir in (None, "") else str(ns.eval_log_dir),
        memory_brief=bool(ns.memory_brief),
        memory_brief_results=int(ns.memory_brief_results),
        dry_run=bool(ns.dry_run),
    )
