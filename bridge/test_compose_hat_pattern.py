#!/usr/bin/env python3
"""Unit tests for the hat pattern generator."""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import compose_kick_pattern as kick
import compose_hat_pattern as hat


class HatPatternTests(unittest.TestCase):
    def test_eighth_note_grid(self) -> None:
        bars = 8
        beats_per_bar = 4.0
        beat_step = 1.0
        notes, clip_length = hat.build_hat_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity=88,
        )
        self.assertEqual(clip_length, float(bars) * beats_per_bar)

        eighth_step = hat._eighth_step(beat_step)

        for note in notes:
            start = float(note["start_time"])
            bar_index = int(start // beats_per_bar)
            bar_start = float(bar_index) * beats_per_bar
            step_pos = (start - bar_start) / eighth_step
            self.assertAlmostEqual(step_pos, round(step_pos), places=6)

    def test_has_gaps_and_not_every_eighth(self) -> None:
        bars = 12
        beats_per_bar = 4.0
        beat_step = 1.0
        notes, _ = hat.build_hat_notes(bars=bars, beats_per_bar=beats_per_bar, beat_step=beat_step)

        eighth_step = hat._eighth_step(beat_step)
        steps_per_bar = hat._steps_per_bar(beats_per_bar, eighth_step)

        per_bar_counts: dict[int, int] = {}
        for note in notes:
            start = float(note["start_time"])
            bar_index = int(start // beats_per_bar)
            per_bar_counts[bar_index] = per_bar_counts.get(bar_index, 0) + 1

        self.assertTrue(per_bar_counts, msg="expected per-bar hat counts")
        self.assertTrue(
            any(count < steps_per_bar for count in per_bar_counts.values()),
            msg="expected at least one bar with missing eighth notes",
        )

    def test_velocity_ranges_and_variation(self) -> None:
        bars = 12
        beats_per_bar = 4.0
        beat_step = 1.0
        velocity = 88
        notes, _ = hat.build_hat_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity=velocity,
        )

        base_velocity = kick._clamp_velocity(velocity)
        velocity_offset = base_velocity - 88

        # Global bounds across phrase scalars, including beat-2/4 accents.
        on_global_min, on_global_max = kick._clamp_velocity_range(
            (80 + velocity_offset) * 0.97,
            (110 + velocity_offset) * 1.05,
        )
        off_global_min, off_global_max = kick._clamp_velocity_range(
            (68 + velocity_offset) * 0.97,
            (92 + velocity_offset) * 1.05,
        )

        on_vels: list[int] = []
        off_vels: list[int] = []

        for note in notes:
            start = float(note["start_time"])
            vel = int(note["velocity"])
            bar_index = int(start // beats_per_bar)
            bar_start = float(bar_index) * beats_per_bar
            on_beat, _beat_index = kick._classify_time(start, bar_start, beat_step)
            if on_beat:
                on_vels.append(vel)
            else:
                off_vels.append(vel)

        self.assertTrue(on_vels, msg="expected on-beat hat velocities")
        self.assertTrue(off_vels, msg="expected off-beat hat velocities")

        self.assertTrue(all(on_global_min <= v <= on_global_max for v in on_vels))
        self.assertTrue(all(off_global_min <= v <= off_global_max for v in off_vels))

        self.assertGreater(len(set(on_vels)), 1)
        self.assertGreater(len(set(off_vels)), 1)

    def test_beats_two_and_four_are_accented(self) -> None:
        bars = 8
        beats_per_bar = 4.0
        beat_step = 1.0
        velocity = 88
        notes, _ = hat.build_hat_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity=velocity,
        )

        accent_beats = set(hat._accent_beat_indices(hat._beats_in_bar(beats_per_bar, beat_step)))
        self.assertTrue(accent_beats, msg="expected accent beats for 4/4")

        base_velocity = kick._clamp_velocity(velocity)
        velocity_offset = base_velocity - 88
        accent_min, accent_max = kick._clamp_velocity_range(94 + velocity_offset, 110 + velocity_offset)

        accent_vels: list[int] = []
        other_on_vels: list[int] = []

        for note in notes:
            start = float(note["start_time"])
            vel = int(note["velocity"])
            bar_index = int(start // beats_per_bar)
            bar_start = float(bar_index) * beats_per_bar
            scalar = hat._phrase_scalar(bar_index)
            scaled_accent_min, scaled_accent_max = kick._clamp_velocity_range(
                accent_min * scalar, accent_max * scalar
            )

            on_beat, beat_index = kick._classify_time(start, bar_start, beat_step)
            if not on_beat:
                continue
            if beat_index in accent_beats:
                accent_vels.append(vel)
                self.assertTrue(scaled_accent_min <= vel <= scaled_accent_max)
            else:
                other_on_vels.append(vel)

        self.assertTrue(accent_vels, msg="expected accented beat-2/4 hats")
        self.assertTrue(other_on_vels, msg="expected other on-beat hats")
        self.assertGreater(sum(accent_vels) / len(accent_vels), sum(other_on_vels) / len(other_on_vels))


if __name__ == "__main__":
    unittest.main()
