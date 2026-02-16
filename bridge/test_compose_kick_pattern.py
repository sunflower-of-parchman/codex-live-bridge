#!/usr/bin/env python3
"""Unit tests for the kick pattern generator."""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import compose_kick_pattern as kick


class KickPatternTests(unittest.TestCase):
    def test_beat_one_present_each_bar(self) -> None:
        bars = 12
        beats_per_bar = 3.0
        notes, clip_length = kick.build_kick_notes(
            bars=bars, beats_per_bar=beats_per_bar, beat_step=1.0
        )
        self.assertEqual(clip_length, float(bars) * beats_per_bar)

        start_times = {float(note["start_time"]) for note in notes}
        for bar_index in range(bars):
            bar_start = float(bar_index) * beats_per_bar
            self.assertIn(bar_start, start_times, msg=f"missing beat 1 at bar {bar_index + 1}")

    def test_four_four_is_sparse_but_keeps_downbeat(self) -> None:
        bars = 8
        beats_per_bar = 4.0
        beat_step = 1.0
        notes, _ = kick.build_kick_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity=110,
        )

        beats_in_bar = int(round(beats_per_bar / beat_step))
        per_bar_hits: dict[int, set[int]] = {}
        for note in notes:
            start = float(note["start_time"])
            bar_index = int(start // beats_per_bar)
            bar_start = float(bar_index) * beats_per_bar
            on_beat, beat_index = kick._classify_time(start, bar_start, beat_step)
            if not on_beat:
                continue
            per_bar_hits.setdefault(bar_index, set()).add(int(beat_index))

        self.assertEqual(len(per_bar_hits), bars)
        patterns: set[tuple[int, ...]] = set()
        for bar_index in range(bars):
            hits = per_bar_hits.get(bar_index, set())
            self.assertIn(0, hits, msg=f"missing beat 1 in bar {bar_index + 1}")
            self.assertLess(len(hits), beats_in_bar, msg="expected sparse kick placement in 4/4")
            patterns.add(tuple(sorted(hits)))

        self.assertGreater(len(patterns), 1, msg="expected per-bar kick variation in 4/4")

    def test_six_four_is_sparse_but_keeps_downbeat(self) -> None:
        bars = 8
        beats_per_bar = 6.0
        beat_step = 1.0
        notes, _ = kick.build_kick_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity=110,
        )

        beats_in_bar = int(round(beats_per_bar / beat_step))
        per_bar_hits: dict[int, set[int]] = {}
        for note in notes:
            start = float(note["start_time"])
            bar_index = int(start // beats_per_bar)
            bar_start = float(bar_index) * beats_per_bar
            on_beat, beat_index = kick._classify_time(start, bar_start, beat_step)
            if not on_beat:
                continue
            per_bar_hits.setdefault(bar_index, set()).add(int(beat_index))

        self.assertEqual(len(per_bar_hits), bars)
        for bar_index in range(bars):
            hits = per_bar_hits.get(bar_index, set())
            self.assertIn(0, hits, msg=f"missing beat 1 in bar {bar_index + 1}")
            self.assertLess(len(hits), beats_in_bar, msg="expected sparse kick placement in 6/4")

    def test_five_four_grouping_leaves_space_but_keeps_beat_one(self) -> None:
        bars = 8
        beats_per_bar = 5.0
        beat_step = 1.0
        notes, _ = kick.build_kick_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity=110,
        )

        beats_in_bar = int(round(beats_per_bar / beat_step))
        per_bar_hits: dict[int, set[int]] = {}

        for note in notes:
            start = float(note["start_time"])
            bar_index = int(start // beats_per_bar)
            bar_start = float(bar_index) * beats_per_bar
            on_beat, beat_index = kick._classify_time(start, bar_start, beat_step)
            if not on_beat:
                continue
            per_bar_hits.setdefault(bar_index, set()).add(int(beat_index))

        self.assertEqual(len(per_bar_hits), bars)
        for bar_index in range(bars):
            hits = per_bar_hits.get(bar_index, set())
            self.assertIn(0, hits, msg=f"missing beat 1 in bar {bar_index + 1}")
            self.assertLess(len(hits), beats_in_bar, msg="5/4 should leave some space")

    def test_five_four_groupings_vary_and_hit_the_hinge(self) -> None:
        bars = 12
        beats_per_bar = 5.0
        beat_step = 1.0
        notes, _ = kick.build_kick_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
        )

        per_bar_patterns: list[tuple[int, ...]] = []
        hinge_hits = 0

        for bar_index in range(bars):
            bar_start = float(bar_index) * beats_per_bar
            hits: set[int] = set()
            for note in notes:
                start = float(note["start_time"])
                if not (bar_start <= start < bar_start + beats_per_bar):
                    continue
                on_beat, beat_index = kick._classify_time(start, bar_start, beat_step)
                if not on_beat:
                    continue
                hits.add(int(beat_index))
            if 3 in hits:
                hinge_hits += 1
            per_bar_patterns.append(tuple(sorted(hits)))

        self.assertGreater(hinge_hits, 0, msg="expected the 3-2 / 2-3 hinge to appear")
        self.assertGreater(len(set(per_bar_patterns)), 1, msg="expected grouping variation")

    def test_velocity_accent_on_beat_one(self) -> None:
        beats_per_bar = 4.0
        beat_step = 1.0
        notes, _ = kick.build_kick_notes(
            bars=4,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity=110,
        )
        beat_one_velocities: list[int] = []
        other_velocities: list[int] = []

        for note in notes:
            start = float(note["start_time"])
            vel = int(note["velocity"])
            bar_index = int(start // beats_per_bar)
            bar_start = float(bar_index) * beats_per_bar
            on_beat, beat_index = kick._classify_time(start, bar_start, beat_step)
            if not on_beat:
                continue
            if beat_index == 0:
                beat_one_velocities.append(vel)
            else:
                other_velocities.append(vel)

        self.assertTrue(beat_one_velocities, msg="expected beat-one velocities")
        self.assertTrue(other_velocities, msg="expected non-downbeat velocities")
        self.assertGreater(min(beat_one_velocities), max(other_velocities))

    def test_velocity_ranges_and_variation(self) -> None:
        bars = 12
        beats_per_bar = 4.0
        beat_step = 1.0
        notes, _ = kick.build_kick_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity=110,
        )

        beat_one_vels: list[int] = []
        other_beat_vels: list[int] = []
        grace_vels: list[int] = []

        for note in notes:
            start = float(note["start_time"])
            vel = int(note["velocity"])
            bar_index = int(start // beats_per_bar)
            bar_start = float(bar_index) * beats_per_bar
            raw_index = (start - bar_start) / beat_step
            beat_index = int(round(raw_index))
            on_beat = abs(raw_index - beat_index) <= 1e-6

            if on_beat and beat_index == 0:
                beat_one_vels.append(vel)
            elif on_beat:
                other_beat_vels.append(vel)
            else:
                grace_vels.append(vel)

        self.assertTrue(beat_one_vels, msg="expected beat-one velocities")
        self.assertTrue(other_beat_vels, msg="expected other beat velocities")
        self.assertTrue(grace_vels, msg="expected grace-note velocities")

        self.assertTrue(all(120 <= v <= 125 for v in beat_one_vels))
        self.assertTrue(all(105 <= v <= 115 for v in other_beat_vels))
        self.assertTrue(all(95 <= v <= 100 for v in grace_vels))

        # Confirm we are not flat-lining at a single value.
        self.assertGreater(len(set(beat_one_vels)), 1)
        self.assertGreater(len(set(other_beat_vels)), 1)
        self.assertGreater(len(set(grace_vels)), 1)

    def test_phrase_boundary_fills_rotate(self) -> None:
        bars = 12
        beats_per_bar = 4.0
        beat_step = 1.0
        notes, _ = kick.build_kick_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
        )

        per_bar: dict[int, set[float]] = {}
        for note in notes:
            start = float(note["start_time"])
            bar_index = int(start // beats_per_bar)
            per_bar.setdefault(bar_index, set()).add(start - bar_index * beats_per_bar)

        beat_offsets = {0.0, 1.0, 2.0, 3.0}
        fill_bar_4 = per_bar.get(3, set()) - beat_offsets
        fill_bar_8 = per_bar.get(7, set()) - beat_offsets
        fill_bar_3 = per_bar.get(2, set()) - beat_offsets

        self.assertTrue(fill_bar_4, msg="expected a fill on bar 4")
        self.assertTrue(fill_bar_8, msg="expected a fill on bar 8")
        self.assertFalse(fill_bar_3, msg="did not expect a fill on bar 3")
        self.assertNotEqual(fill_bar_4, fill_bar_8, msg="fills should rotate")

    def test_invalid_bars_raises(self) -> None:
        with self.assertRaises(ValueError):
            kick.build_kick_notes(bars=0)

    def test_signature_conversion(self) -> None:
        self.assertEqual(kick._beats_per_bar_from_signature(4, 4), 4.0)
        self.assertEqual(kick._beats_per_bar_from_signature(6, 8), 3.0)

    def test_find_groove_id_by_name(self) -> None:
        grooves_children = [{"index": 0}, {"index": 1}]

        def describe_side_effect(_sock, _ack_sock, path: str, _request_id: str, _timeout: float):
            if path.endswith("grooves 0"):
                return {"id": 20, "name": "Swing 16ths 66"}
            if path.endswith("grooves 1"):
                return {"id": 21, "name": "Hip Hop Loosely Flow 16ths 80 bpm"}
            return None

        with mock.patch.object(kick, "_get_children", return_value=grooves_children):
            with mock.patch.object(kick, "_api_describe", side_effect=describe_side_effect):
                groove_id = kick._find_groove_id_by_name(
                    sock=None,  # type: ignore[arg-type]
                    ack_sock=None,  # type: ignore[arg-type]
                    groove_name="Hip Hop Loosely Flow 16ths 80 bpm",
                    timeout_s=1.0,
                )

        self.assertEqual(groove_id, 21)


if __name__ == "__main__":
    unittest.main()
