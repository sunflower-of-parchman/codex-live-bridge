#!/usr/bin/env python3
"""Unit tests for the piano pattern generator."""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import compose_kick_pattern as kick
import compose_piano_pattern as piano


class PianoPatternTests(unittest.TestCase):
    def test_phrase_level_segments_cover_full_clip(self) -> None:
        bars = 32
        beats_per_bar = 4.0
        beat_step = 1.0
        segment_bars = 2

        notes, clip_length = piano.build_piano_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            segment_bars=segment_bars,
        )
        self.assertEqual(clip_length, float(bars) * beats_per_bar)
        self.assertTrue(notes, msg="expected piano notes")

        segment_count = int(math.ceil(bars / segment_bars))
        segment_beats = float(segment_bars) * beats_per_bar
        strum_window = beat_step * 0.06 * 8

        per_segment_min_start: dict[int, float] = {}
        for note in notes:
            start = float(note["start_time"])
            segment_index = int(start // segment_beats)
            per_segment_min_start[segment_index] = min(
                start,
                per_segment_min_start.get(segment_index, start),
            )

        self.assertEqual(len(per_segment_min_start), segment_count)

        for segment_index in range(segment_count):
            segment_start = segment_index * segment_beats
            earliest = per_segment_min_start[segment_index]
            self.assertGreaterEqual(earliest, segment_start)
            self.assertLessEqual(
                earliest,
                segment_start + strum_window,
                msg=f"segment {segment_index} start too late",
            )

    def test_phrase_patterns_rotate_across_8_bar_phrases(self) -> None:
        segment_bars = 2
        segments_per_phrase = piano.PHRASE_BARS // segment_bars

        seq_phrase_1 = [
            piano._chord_for_segment(i, segments_per_phrase)
            for i in range(segments_per_phrase)
        ]
        seq_phrase_2 = [
            piano._chord_for_segment(i + segments_per_phrase, segments_per_phrase)
            for i in range(segments_per_phrase)
        ]

        self.assertEqual(seq_phrase_1, list(piano.PHRASE_PATTERNS[0]))
        self.assertEqual(seq_phrase_2, list(piano.PHRASE_PATTERNS[1]))
        self.assertNotEqual(seq_phrase_1, seq_phrase_2)

    def test_velocity_ranges_and_variation(self) -> None:
        bars = 16
        beats_per_bar = 4.0
        beat_step = 1.0
        velocity_center = 92

        notes, _ = piano.build_piano_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            velocity_center=velocity_center,
        )

        (low_min, low_max), (mid_min, mid_max), (top_min, top_max) = piano._velocity_ranges(
            velocity_center
        )
        scalar_min = min(piano._phrase_scalar(i) for i in range(bars))
        scalar_max = max(piano._phrase_scalar(i) for i in range(bars))
        motion_scalar_min = 0.97

        min_candidates = [
            low_min * scalar_min,
            mid_min * scalar_min,
            top_min * scalar_min,
            mid_min * scalar_min * motion_scalar_min,
            top_min * scalar_min * motion_scalar_min,
        ]
        max_candidates = [
            low_max * scalar_max,
            mid_max * scalar_max,
            top_max * scalar_max,
        ]
        global_min = min(kick._clamp_velocity_range(v, v)[0] for v in min_candidates)
        global_max = max(kick._clamp_velocity_range(v, v)[1] for v in max_candidates)

        velocities = [int(note["velocity"]) for note in notes]
        self.assertTrue(velocities, msg="expected velocities")
        self.assertTrue(all(global_min <= v <= global_max for v in velocities))
        self.assertGreater(len(set(velocities)), 1, msg="expected velocity variation")

    def test_intra_segment_motion_exists_and_stays_within_segment(self) -> None:
        bars = 24
        beats_per_bar = 5.0
        beat_step = 1.0
        segment_bars = 2

        notes, clip_length = piano.build_piano_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            segment_bars=segment_bars,
        )
        self.assertEqual(clip_length, float(bars) * beats_per_bar)

        segment_beats = float(segment_bars) * beats_per_bar
        strum_window = beat_step * 0.06 * 8
        motion_found = False

        for note in notes:
            start = float(note["start_time"])
            segment_index = int(start // segment_beats)
            segment_start = segment_index * segment_beats
            segment_end = min(segment_start + segment_beats, clip_length)

            self.assertLess(start, segment_end, msg="note should not cross segment boundary")
            if start - segment_start > strum_window + beat_step * 0.25:
                motion_found = True

        self.assertTrue(motion_found, msg="expected intra-segment motion beyond the chord strum")

    def test_chord_tones_sustain_each_segment(self) -> None:
        bars = 24
        beats_per_bar = 5.0
        beat_step = 1.0
        segment_bars = 2

        notes, clip_length = piano.build_piano_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            segment_bars=segment_bars,
        )
        self.assertEqual(clip_length, float(bars) * beats_per_bar)
        self.assertTrue(notes, msg="expected piano notes")

        segment_beats = float(segment_bars) * beats_per_bar
        strum_window = beat_step * 0.45
        segment_count = int(math.ceil(bars / segment_bars))

        per_segment: dict[int, list[dict]] = {}
        for note in notes:
            start = float(note["start_time"])
            segment_index = int(start // segment_beats)
            per_segment.setdefault(segment_index, []).append(note)

        self.assertEqual(len(per_segment), segment_count)

        for segment_index in range(segment_count):
            seg_notes = per_segment.get(segment_index, [])
            self.assertTrue(seg_notes, msg=f"missing notes for segment {segment_index}")
            segment_start = segment_index * segment_beats
            segment_end = min(segment_start + segment_beats, clip_length)
            effective_segment_beats = segment_end - segment_start

            early_chord_tones = [
                n for n in seg_notes if float(n["start_time"]) <= segment_start + strum_window
            ]
            self.assertGreaterEqual(
                len(early_chord_tones),
                3,
                msg="expected multiple chord tones near the segment start",
            )

            early_pitches = {int(n["pitch"]) for n in early_chord_tones}
            self.assertGreaterEqual(len(early_pitches), 3, msg="expected distinct chord tones")

            long_early = [
                n for n in early_chord_tones if float(n["duration"]) >= effective_segment_beats * 0.7
            ]
            self.assertTrue(long_early, msg="expected sustained chord tones without pedal")

    def test_no_reattack_of_sustained_pitches_within_segment(self) -> None:
        bars = 24
        beats_per_bar = 5.0
        beat_step = 1.0
        segment_bars = 2

        notes, clip_length = piano.build_piano_notes(
            bars=bars,
            beats_per_bar=beats_per_bar,
            beat_step=beat_step,
            segment_bars=segment_bars,
        )
        self.assertEqual(clip_length, float(bars) * beats_per_bar)
        self.assertTrue(notes, msg="expected piano notes")

        segment_beats = float(segment_bars) * beats_per_bar
        strum_window = beat_step * 0.45
        segment_count = int(math.ceil(bars / segment_bars))

        per_segment: dict[int, list[dict]] = {}
        for note in notes:
            start = float(note["start_time"])
            segment_index = int(start // segment_beats)
            per_segment.setdefault(segment_index, []).append(note)

        self.assertEqual(len(per_segment), segment_count)

        for segment_index in range(segment_count):
            seg_notes = per_segment.get(segment_index, [])
            self.assertTrue(seg_notes, msg=f"missing notes for segment {segment_index}")

            segment_start = segment_index * segment_beats
            segment_end = min(segment_start + segment_beats, clip_length)

            sustained_pitches = {
                int(n["pitch"])
                for n in seg_notes
                if float(n["start_time"]) <= segment_start + strum_window
            }
            self.assertTrue(sustained_pitches, msg="expected sustained pitches near segment start")

            for pitch in sustained_pitches:
                later_hits = [
                    n
                    for n in seg_notes
                    if int(n["pitch"]) == pitch
                    and float(n["start_time"]) > segment_start + strum_window
                    and float(n["start_time"]) < segment_end
                ]
                self.assertFalse(
                    later_hits,
                    msg=f"unexpected reattack for pitch {pitch} in segment {segment_index}",
                )


if __name__ == "__main__":
    unittest.main()
