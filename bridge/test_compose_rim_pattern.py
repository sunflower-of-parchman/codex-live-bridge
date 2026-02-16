#!/usr/bin/env python3
"""Unit tests for the rim pattern generator."""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import compose_kick_pattern as kick
import compose_rim_pattern as rim


class RimPatternTests(unittest.TestCase):
    def test_anchor_hits_on_beats_two_or_four(self) -> None:
        bars = 8
        beats_per_bar = 4.0
        beat_step = 1.0
        notes, clip_length = rim.build_rim_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            pitch=37,
            velocity=100,
        )
        self.assertEqual(clip_length, float(bars) * beats_per_bar)

        beats_in_bar = rim._beats_in_bar(beats_per_bar, beat_step)
        anchor_candidates = rim._anchor_candidates(beats_in_bar)

        per_bar_offsets: dict[int, set[float]] = {}
        for note in notes:
            start = float(note["start_time"])
            bar_index = int(start // beats_per_bar)
            per_bar_offsets.setdefault(bar_index, set()).add(start - bar_index * beats_per_bar)

        for bar_index in range(bars):
            bar_offsets = per_bar_offsets.get(bar_index, set())
            anchor_beat_index = anchor_candidates[bar_index % len(anchor_candidates)]
            anchor_offset = anchor_beat_index * beat_step
            self.assertIn(anchor_offset, bar_offsets, msg=f"missing anchor at bar {bar_index + 1}")

            # The rim should not mirror the kick on every beat.
            disallowed = {0.0, 2.0}
            self.assertTrue(disallowed.isdisjoint(bar_offsets))

    def test_phrase_syncopation_present_and_rotates(self) -> None:
        bars = 12
        beats_per_bar = 4.0
        beat_step = 1.0
        notes, _ = rim.build_rim_notes(bars=bars, beats_per_bar=beats_per_bar, beat_step=beat_step)

        beat_grid = {0.0, 1.0, 2.0, 3.0}
        per_bar_sync: dict[int, set[float]] = {}

        for note in notes:
            start = float(note["start_time"])
            bar_index = int(start // beats_per_bar)
            offset = start - bar_index * beats_per_bar
            if offset not in beat_grid:
                per_bar_sync.setdefault(bar_index, set()).add(offset)

        sync_bar_4 = per_bar_sync.get(3, set())
        sync_bar_8 = per_bar_sync.get(7, set())
        sync_bar_3 = per_bar_sync.get(2, set())

        self.assertTrue(sync_bar_4, msg="expected syncopation on bar 4")
        self.assertTrue(sync_bar_8, msg="expected syncopation on bar 8")
        self.assertFalse(sync_bar_3, msg="did not expect syncopation on bar 3")
        self.assertNotEqual(sync_bar_4, sync_bar_8, msg="syncopation should rotate by phrase")

    def test_velocity_ranges_and_variation(self) -> None:
        bars = 12
        beats_per_bar = 4.0
        beat_step = 1.0
        velocity = 100
        notes, _ = rim.build_rim_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity=velocity,
        )

        beats_in_bar = rim._beats_in_bar(beats_per_bar, beat_step)
        anchor_candidates = rim._anchor_candidates(beats_in_bar)

        base_velocity = kick._clamp_velocity(velocity)
        velocity_offset = base_velocity - 100
        anchor_min, anchor_max = kick._clamp_velocity_range(98 + velocity_offset, 112 + velocity_offset)
        beat_min, beat_max = kick._clamp_velocity_range(92 + velocity_offset, 106 + velocity_offset)
        sync_min, sync_max = kick._clamp_velocity_range(88 + velocity_offset, 100 + velocity_offset)

        anchor_vels: list[int] = []
        beat_vels: list[int] = []
        sync_vels: list[int] = []

        for note in notes:
            start = float(note["start_time"])
            vel = int(note["velocity"])
            bar_index = int(start // beats_per_bar)
            bar_start = float(bar_index) * beats_per_bar
            on_beat, beat_index = kick._classify_time(start, bar_start, beat_step)
            anchor_index = anchor_candidates[bar_index % len(anchor_candidates)]

            if on_beat and beat_index == anchor_index:
                anchor_vels.append(vel)
            elif on_beat:
                beat_vels.append(vel)
            else:
                sync_vels.append(vel)

        self.assertTrue(anchor_vels, msg="expected anchor velocities")
        self.assertTrue(sync_vels, msg="expected sync velocities")

        self.assertTrue(all(anchor_min <= v <= anchor_max for v in anchor_vels))
        self.assertTrue(all(beat_min <= v <= beat_max for v in beat_vels))
        self.assertTrue(all(sync_min <= v <= sync_max for v in sync_vels))

        self.assertGreater(len(set(anchor_vels)), 1)
        self.assertGreater(len(set(sync_vels)), 1)


if __name__ == "__main__":
    unittest.main()
