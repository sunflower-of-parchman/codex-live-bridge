#!/usr/bin/env python3
"""Unit tests for the Live set analysis CLI."""

from __future__ import annotations

import unittest

import analyze_live_set as analyze


def _event(
    *,
    start: float,
    end: float,
    pitch: int,
    velocity: int,
) -> dict:
    return {
        "start_time_global": float(start),
        "end_time_global": float(end),
        "duration": float(end - start),
        "pitch": int(pitch),
        "velocity": int(velocity),
    }


class AnalyzeLiveSetTests(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        cfg = analyze.parse_args([])
        self.assertEqual(cfg.track_query, "piano")
        self.assertEqual(cfg.clip_scope, "both")
        self.assertFalse(cfg.include_note_events)
        self.assertFalse(cfg.dry_run)

    def test_decode_jsonish_handles_double_encoded_json(self) -> None:
        raw = "\"{\\\"notes\\\":[{\\\"pitch\\\":60}]}\""
        decoded = analyze._decode_jsonish(raw)
        self.assertIsInstance(decoded, dict)
        self.assertEqual(decoded["notes"][0]["pitch"], 60)

    def test_extract_notes_payload_from_mapping(self) -> None:
        notes = analyze._extract_notes_payload({"notes": [{"pitch": 64}, {"pitch": 67}]})
        self.assertEqual(len(notes), 2)
        self.assertEqual(notes[1]["pitch"], 67)

    def test_build_harmony_metrics_reports_voicings(self) -> None:
        events = [
            _event(start=0.0, end=1.0, pitch=48, velocity=78),
            _event(start=0.0, end=1.0, pitch=52, velocity=80),
            _event(start=0.0, end=1.0, pitch=55, velocity=82),
            _event(start=1.0, end=2.0, pitch=50, velocity=78),
            _event(start=1.0, end=2.0, pitch=53, velocity=80),
            _event(start=1.0, end=2.0, pitch=57, velocity=82),
        ]
        metrics = analyze._build_harmony_metrics(events)
        self.assertTrue(metrics["enabled"])
        self.assertGreater(metrics["chord_onset_ratio"], 0.5)
        shapes = [item["shape"] for item in metrics["top_voicing_shapes"]]
        self.assertIn("0-4-7", shapes)

    def test_build_velocity_metrics_has_register_summaries(self) -> None:
        events = [
            _event(start=0.0, end=0.5, pitch=40, velocity=62),
            _event(start=0.5, end=1.0, pitch=60, velocity=84),
            _event(start=1.0, end=1.5, pitch=76, velocity=108),
            _event(start=1.5, end=2.0, pitch=79, velocity=96),
        ]
        metrics = analyze._build_velocity_metrics(events)
        self.assertTrue(metrics["enabled"])
        self.assertIn("overall", metrics)
        self.assertGreater(metrics["overall"]["mean"], 60.0)
        self.assertIn("high_73_127", metrics["register_buckets"])

    def test_build_piano_choreography_metrics_detects_two_hand_behavior(self) -> None:
        events = [
            _event(start=0.0, end=2.0, pitch=48, velocity=74),
            _event(start=1.0, end=3.0, pitch=50, velocity=76),
            _event(start=0.25, end=0.5, pitch=72, velocity=92),
            _event(start=1.25, end=1.5, pitch=76, velocity=95),
            _event(start=2.25, end=3.75, pitch=79, velocity=90),
        ]
        metrics = analyze._build_piano_choreography_metrics(events, split_pitch=60)
        self.assertEqual(metrics["status"], "ok")
        self.assertGreater(metrics["left_note_ratio"], 0.2)
        self.assertGreater(metrics["right_note_ratio"], 0.2)
        self.assertGreater(metrics["call_response_ratio"], 0.0)
        self.assertIn("choreography_score", metrics)

    def test_compose_artifact_omits_note_events_by_default(self) -> None:
        cfg = analyze.parse_args([])
        clip = {
            "track_name": "Piano",
            "track_index": 3,
            "clip_source": "arrangement",
            "clip_path": "live_set tracks 3 arrangement_clips 0",
            "clip_name": "Take 1",
            "clip_length": 16.0,
            "clip_start_time": 0.0,
            "note_count": 1,
            "notes": [
                {
                    "track_name": "Piano",
                    "track_index": 3,
                    "clip_source": "arrangement",
                    "clip_path": "x",
                    "clip_name": "Take 1",
                    "pitch": 60,
                    "velocity": 90,
                    "duration": 1.0,
                    "start_time": 0.0,
                    "start_time_global": 0.0,
                    "end_time_global": 1.0,
                    "mute": False,
                }
            ],
        }
        track = analyze.TrackRef(index=3, path="live_set tracks 3", name="Piano", type="Track")
        artifact = analyze._compose_artifact(
            cfg=cfg,
            run_id="rid",
            created_at=analyze._utc_now(),
            live_context={},
            topology={},
            selected_tracks=[track],
            clips_by_track={"Piano": [clip]},
        )
        persisted_clip = artifact["extraction"]["tracks"]["Piano"]["clips"][0]
        self.assertIn("clip_name", persisted_clip)
        self.assertNotIn("notes", persisted_clip)


if __name__ == "__main__":
    unittest.main()
