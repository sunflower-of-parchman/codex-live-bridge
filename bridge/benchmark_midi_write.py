#!/usr/bin/env python3
"""Deterministic MIDI-write benchmark harness for arrangement clip strategies."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence

import compose_kick_pattern as kick
import compose_arrangement as arrangement


DEFAULT_ITERATIONS = 7
DEFAULT_ACK_TIMEOUT = 2.0
DEFAULT_OUT_PATH = Path("bridge/benchmarks/latest_midi_write.json")


@dataclass(frozen=True)
class CaseResult:
    name: str
    note_count: int
    command_count: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    failures: int


@dataclass(frozen=True)
class IterationSample:
    elapsed_ms: float
    command_count: int



def _pct(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (max(0.0, min(100.0, float(pct))) / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac



def _build_notes(count: int) -> list[dict]:
    notes: list[dict] = []
    for idx in range(count):
        quarter_index = idx % 256
        start_time = round(quarter_index * 0.25, 6)
        notes.append(
            {
                "pitch": 36 + (idx % 8),
                "start_time": float(start_time),
                "duration": 0.25,
                "velocity": 84 + (idx % 32),
                "mute": 0,
            }
        )
    notes.sort(key=lambda n: (n["start_time"], n["pitch"], n["velocity"]))
    return notes



def _mutate_notes(base: Sequence[dict]) -> list[dict]:
    mutated: list[dict] = []
    for idx, note in enumerate(base):
        if idx % 13 == 0:
            # Remove some notes.
            continue
        updated = dict(note)
        if idx % 7 == 0:
            updated["velocity"] = max(1, min(127, int(updated["velocity"]) + 9))
        if idx % 19 == 0:
            updated["mute"] = 1
        mutated.append(updated)

    extra_count = max(1, len(base) // 20)
    for extra_idx in range(extra_count):
        mutated.append(
            {
                "pitch": 48 + (extra_idx % 5),
                "start_time": round((extra_idx % 128) * 0.5 + 0.125, 6),
                "duration": 0.25,
                "velocity": 96,
                "mute": 0,
            }
        )

    mutated.sort(key=lambda n: (n["start_time"], n["pitch"], n["velocity"]))
    return mutated



def _stable_json_size(value: object) -> int:
    text = json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    return len(text)



def _simulate_add_new_notes(notes: Sequence[dict], chunk_size: int) -> tuple[int, int]:
    chunks = kick._chunk_notes(notes, chunk_size=chunk_size)
    payload_bytes = 0
    for chunk in chunks:
        payload_bytes += _stable_json_size({"notes": chunk})
    return len(chunks), payload_bytes



def _adaptive_chunk_size(note_count: int, ack_timeout: float) -> int:
    # Conservative adaptive policy: large note sets use larger chunks, but stay
    # bounded under slower acknowledgement environments.
    if note_count >= 2000:
        return 120 if ack_timeout <= 2.5 else 80
    if note_count >= 1000:
        return 80 if ack_timeout <= 2.5 else 60
    return 40



def _simulate_delta(existing_notes: Sequence[dict], target_notes: Sequence[dict], chunk_size: int) -> tuple[int, int]:
    existing_with_ids = []
    for idx, note in enumerate(existing_notes, start=1):
        enriched = dict(note)
        enriched["note_id"] = idx
        existing_with_ids.append(enriched)

    delta = arrangement._compute_note_delta(existing_with_ids, target_notes)
    if delta.requires_full_replace:
        add_calls, payload_bytes = _simulate_add_new_notes(target_notes, chunk_size)
        return add_calls + 1, payload_bytes

    command_count = 1  # get_notes_extended
    payload_bytes = 0

    if delta.remove_note_ids:
        remove_chunks = kick._chunk_notes(delta.remove_note_ids, chunk_size=128)
        command_count += len(remove_chunks)
        for chunk in remove_chunks:
            payload_bytes += _stable_json_size({"note_ids": list(chunk)})

    if delta.modification_notes:
        mod_chunks = kick._chunk_notes(delta.modification_notes, chunk_size=chunk_size)
        command_count += len(mod_chunks)
        for chunk in mod_chunks:
            payload_bytes += _stable_json_size({"notes": chunk})

    if delta.add_notes:
        add_chunks = kick._chunk_notes(delta.add_notes, chunk_size=chunk_size)
        command_count += len(add_chunks)
        for chunk in add_chunks:
            payload_bytes += _stable_json_size({"notes": chunk})

    return command_count, payload_bytes



def _measure_case(iterations: int, runner: Callable[[], tuple[int, int]], ack_timeout: float) -> tuple[list[IterationSample], int]:
    samples: list[IterationSample] = []
    failures = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            command_count, payload_bytes = runner()
            # Synthetic transport/ack estimate to expose command-count impact.
            # This keeps benchmark deterministic without requiring Live online.
            synthetic_ms = (command_count * 0.35) + (payload_bytes / 180000.0)
            elapsed_ms = ((time.perf_counter() - t0) * 1000.0) + synthetic_ms
            if ack_timeout > 0:
                elapsed_ms += min(ack_timeout * 4.0, 8.0)
            samples.append(IterationSample(elapsed_ms=float(elapsed_ms), command_count=int(command_count)))
        except Exception:
            failures += 1
    return samples, failures



def _summarize(name: str, note_count: int, samples: Sequence[IterationSample], failures: int) -> CaseResult:
    elapsed = [sample.elapsed_ms for sample in samples]
    commands = [sample.command_count for sample in samples]
    return CaseResult(
        name=name,
        note_count=note_count,
        command_count=int(round(mean(commands))) if commands else 0,
        p50_ms=round(_pct(elapsed, 50.0), 3),
        p95_ms=round(_pct(elapsed, 95.0), 3),
        max_ms=round(max(elapsed) if elapsed else 0.0, 3),
        failures=int(failures),
    )



def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Iterations per case")
    parser.add_argument("--ack-timeout", type=float, default=DEFAULT_ACK_TIMEOUT, help="Ack timeout hint")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUT_PATH),
        help=f"Benchmark JSON artifact path (default: {DEFAULT_OUT_PATH})",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run all strategy comparisons including baseline, adaptive, and delta",
    )
    ns = parser.parse_args(list(argv))
    if ns.iterations <= 0:
        parser.error("--iterations must be > 0")
    if ns.ack_timeout < 0:
        parser.error("--ack-timeout must be >= 0")
    return ns



def main(argv: Sequence[str]) -> int:
    ns = parse_args(argv)

    notes_200 = _build_notes(200)
    notes_2000 = _build_notes(2000)
    target_for_delta = _mutate_notes(notes_2000)

    cases: list[tuple[str, int, Callable[[], tuple[int, int]]]] = [
        (
            "case_a_200_one_chunk",
            len(notes_200),
            lambda: _simulate_add_new_notes(notes_200, chunk_size=200),
        ),
        (
            "case_b_2000_chunk40",
            len(notes_2000),
            lambda: _simulate_add_new_notes(notes_2000, chunk_size=40),
        ),
        (
            "case_c_2000_adaptive",
            len(notes_2000),
            lambda: _simulate_add_new_notes(
                notes_2000,
                chunk_size=_adaptive_chunk_size(len(notes_2000), float(ns.ack_timeout)),
            ),
        ),
        (
            "case_d_delta_update",
            len(target_for_delta),
            lambda: _simulate_delta(notes_2000, target_for_delta, chunk_size=40),
        ),
    ]

    if not ns.compare_all:
        # Keep default output focused but still include all required A-D cases.
        pass

    results: list[CaseResult] = []
    for case_name, note_count, runner in cases:
        samples, failures = _measure_case(int(ns.iterations), runner, float(ns.ack_timeout))
        result = _summarize(case_name, note_count, samples, failures)
        results.append(result)
        print(
            "info: benchmark case={case} notes={notes} commands~{commands} "
            "p50_ms={p50} p95_ms={p95} max_ms={maxv} failures={failures}".format(
                case=result.name,
                notes=result.note_count,
                commands=result.command_count,
                p50=f"{result.p50_ms:.3f}",
                p95=f"{result.p95_ms:.3f}",
                maxv=f"{result.max_ms:.3f}",
                failures=result.failures,
            )
        )

    artifact = {
        "version": 1,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "iterations": int(ns.iterations),
        "ack_timeout": float(ns.ack_timeout),
        "cases": [
            {
                "name": result.name,
                "note_count": result.note_count,
                "command_count": result.command_count,
                "p50_ms": result.p50_ms,
                "p95_ms": result.p95_ms,
                "max_ms": result.max_ms,
                "failures": result.failures,
            }
            for result in results
        ],
    }

    output_path = Path(str(ns.output))
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"info: benchmark artifact saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
