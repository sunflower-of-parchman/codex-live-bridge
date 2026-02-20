"""Unit tests for composition feedback-loop artifact logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence
import pathlib
import sys
import unittest
from unittest import mock

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import composition_feedback_loop as feedback


@dataclass(frozen=True)
class _Section:
    index: int
    label: str
    piano_mode: str
    hat_density: str
    kick_keep_ratio: float = 0.7
    rim_keep_ratio: float = 0.4
    hat_keep_ratio: float = 0.8


def _notes(count: int) -> list[dict[str, float | int]]:
    return [
        {
            "pitch": 36,
            "start_time": float(i),
            "duration": 0.25,
            "velocity": 100,
            "mute": 0,
        }
        for i in range(count)
    ]


def _arranged_payload(
    sections: Sequence[_Section], hat_mode: str = "quarter"
) -> dict[str, list[tuple[_Section, list[dict[str, float | int]]]]]:
    kick = [(sections[0], _notes(2)), (sections[1], _notes(4)), (sections[2], _notes(6))]
    rim = [(sections[0], _notes(0)), (sections[1], _notes(1)), (sections[2], _notes(2))]
    hat = [
        (sections[0], _notes(3)),
        (sections[1], _notes(6)),
        (sections[2], _notes(12 if hat_mode == "sixteenth" else 8)),
    ]
    piano = [(sections[0], _notes(5)), (sections[1], _notes(5)), (sections[2], _notes(5))]
    return {
        "Kick Drum": kick,
        "RIM": rim,
        "HAT": hat,
        "Piano": piano,
    }


def _piano_variation_payload(
    sections: Sequence[_Section],
) -> dict[str, list[tuple[_Section, list[dict[str, float | int]]]]]:
    return {
        "Piano": [
            (
                sections[0],
                [
                    {"pitch": 48, "start_time": 0.0, "duration": 4.5, "velocity": 72, "mute": 0},
                    {"pitch": 64, "start_time": 0.25, "duration": 0.25, "velocity": 84, "mute": 0},
                    {"pitch": 67, "start_time": 3.0, "duration": 4.0, "velocity": 80, "mute": 0},
                ],
            ),
            (
                sections[1],
                [
                    {"pitch": 50, "start_time": 0.0, "duration": 4.0, "velocity": 74, "mute": 0},
                    {"pitch": 64, "start_time": 0.5, "duration": 0.5, "velocity": 86, "mute": 0},
                    {"pitch": 69, "start_time": 1.25, "duration": 0.25, "velocity": 82, "mute": 0},
                    {"pitch": 71, "start_time": 2.5, "duration": 3.5, "velocity": 78, "mute": 0},
                ],
            ),
            (
                sections[2],
                [
                    {"pitch": 52, "start_time": 0.0, "duration": 4.0, "velocity": 70, "mute": 0},
                    {"pitch": 64, "start_time": 0.75, "duration": 0.25, "velocity": 83, "mute": 0},
                    {"pitch": 67, "start_time": 3.5, "duration": 4.25, "velocity": 76, "mute": 0},
                ],
            ),
        ]
    }


def _marimba_pair_payload(
    sections: Sequence[_Section],
) -> dict[str, list[tuple[_Section, list[dict[str, float | int]]]]]:
    marimba = [
        (sections[0], _notes(4)),
        (sections[1], _notes(5)),
        (sections[2], _notes(6)),
    ]
    vib = [
        (sections[0], _notes(3)),
        (sections[1], _notes(3)),
        (sections[2], _notes(4)),
    ]
    return {
        "Marimba": marimba,
        "Vibraphone": vib,
    }


def _marimba_style_payload(
    sections: Sequence[_Section],
    *,
    stacked: bool,
) -> dict[str, list[tuple[_Section, list[dict[str, float | int]]]]]:
    timeline: list[list[dict[str, float | int]]] = []
    for _section in sections:
        if stacked:
            notes = [
                {"pitch": 62, "start_time": 0.0, "duration": 1.0, "velocity": 96, "mute": 0},
                {"pitch": 65, "start_time": 0.0, "duration": 0.75, "velocity": 88, "mute": 0},
                {"pitch": 69, "start_time": 1.0, "duration": 1.0, "velocity": 98, "mute": 0},
                {"pitch": 72, "start_time": 1.0, "duration": 0.75, "velocity": 86, "mute": 0},
            ]
        else:
            notes = [
                {"pitch": 62, "start_time": 0.0, "duration": 1.0, "velocity": 96, "mute": 0},
                {"pitch": 65, "start_time": 0.5, "duration": 0.75, "velocity": 88, "mute": 0},
                {"pitch": 69, "start_time": 1.0, "duration": 1.0, "velocity": 98, "mute": 0},
                {"pitch": 72, "start_time": 1.5, "duration": 0.75, "velocity": 86, "mute": 0},
            ]
        timeline.append(notes)
    return {
        "Marimba": [
            (section, [dict(note) for note in notes])
            for section, notes in zip(sections, timeline)
        ]
    }


class CompositionFeedbackLoopTests(unittest.TestCase):
    def test_log_composition_run_persists_artifact_and_index(self) -> None:
        sections = [
            _Section(index=0, label="intro", piano_mode="chords", hat_density="quarter"),
            _Section(index=1, label="build", piano_mode="chords", hat_density="eighth"),
            _Section(index=2, label="release", piano_mode="chords", hat_density="quarter"),
        ]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact, artifact_path = feedback.log_composition_run(
                mood="Ambient",
                key_name="D minor",
                bpm=120.0,
                sig_num=4,
                sig_den=4,
                minutes=2.0,
                bars=24,
                section_bars=8,
                sections=sections,
                arranged_by_track=_arranged_payload(sections),
                created_clips_by_track={"Kick Drum": 3, "RIM": 3, "HAT": 3, "Piano": 3},
                status="success",
                run_metadata={"save_policy": "ephemeral", "selected_minutes": 3.5},
                repo_root=root,
            )

            self.assertTrue(artifact_path.exists())
            self.assertEqual(artifact["status"], "success")
            self.assertEqual(artifact["composition"]["mood"], "Ambient")
            self.assertEqual(artifact["fingerprints"]["meter_bpm"], "4/4@120")
            self.assertEqual(artifact["run"]["save_policy"], "ephemeral")
            self.assertIn("-4_4-120bpm-ambient-", artifact["run_id"])
            self.assertEqual(artifact["human_feedback"]["status"], "not_provided")
            self.assertTrue(str(artifact["goal"]["job_to_be_done"]).strip())
            self.assertIn("merit_rubric", artifact["reflection"])
            self.assertIn("piano_variation", artifact["reflection"])
            self.assertIn("instrument_diversity_proxy", artifact["reflection"]["merit_rubric"])
            self.assertIn("similarity_weights", artifact["reflection"])

            index_path = root / feedback.DEFAULT_RELATIVE_INDEX_PATH
            self.assertTrue(index_path.exists())
            entries = feedback._load_json(index_path, {}).get("entries", [])
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["run_id"], artifact["run_id"])

    def test_log_composition_run_persists_human_feedback_section(self) -> None:
        sections = [
            _Section(index=0, label="intro", piano_mode="chords", hat_density="quarter"),
            _Section(index=1, label="build", piano_mode="chords", hat_density="eighth"),
            _Section(index=2, label="release", piano_mode="chords", hat_density="quarter"),
        ]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact, _artifact_path = feedback.log_composition_run(
                mood="Ambient",
                key_name="D minor",
                bpm=120.0,
                sig_num=4,
                sig_den=4,
                minutes=2.0,
                bars=24,
                section_bars=8,
                sections=sections,
                arranged_by_track=_arranged_payload(sections),
                created_clips_by_track={"Kick Drum": 3, "RIM": 3, "HAT": 3, "Piano": 3},
                status="success",
                human_feedback={
                    "mode": "verbal",
                    "text": "Piano and marimba were in different keys; this run was a miss.",
                },
                repo_root=root,
            )

            self.assertEqual(artifact["human_feedback"]["status"], "provided")
            self.assertEqual(artifact["human_feedback"]["mode"], "verbal")
            self.assertEqual(
                artifact["human_feedback"]["text"],
                "Piano and marimba were in different keys; this run was a miss.",
            )

    def test_log_composition_run_stores_goal_and_piano_variation_metrics(self) -> None:
        sections = [
            _Section(index=0, label="intro", piano_mode="chords", hat_density="quarter"),
            _Section(index=1, label="build", piano_mode="motion", hat_density="eighth"),
            _Section(index=2, label="release", piano_mode="out", hat_density="quarter"),
        ]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact, _artifact_path = feedback.log_composition_run(
                mood="Beautiful",
                key_name="E major",
                bpm=142.0,
                sig_num=4,
                sig_den=4,
                minutes=5.0,
                bars=24,
                section_bars=8,
                sections=sections,
                arranged_by_track=_piano_variation_payload(sections),
                created_clips_by_track={"Piano": 3},
                status="success",
                run_metadata={
                    "composition_goal": "Keep left hand stable and vary right hand rhythms with silence.",
                },
                repo_root=root,
            )

            self.assertEqual(
                artifact["goal"]["job_to_be_done"],
                "Keep left hand stable and vary right hand rhythms with silence.",
            )
            piano_variation = artifact["reflection"]["piano_variation"]
            self.assertTrue(piano_variation["enabled"])
            self.assertEqual(piano_variation["track_name"], "Piano")
            self.assertGreater(float(piano_variation["upper_silence_ratio"]), 0.05)
            self.assertGreater(float(piano_variation["upper_short_duration_ratio"]), 0.0)
            self.assertGreater(float(piano_variation["upper_long_duration_ratio"]), 0.0)

    def test_same_second_runs_get_unique_run_ids_and_index_entries(self) -> None:
        sections = [
            _Section(index=0, label="intro", piano_mode="chords", hat_density="quarter"),
            _Section(index=1, label="build", piano_mode="chords", hat_density="eighth"),
            _Section(index=2, label="release", piano_mode="chords", hat_density="quarter"),
        ]
        fixed_now = datetime(2026, 2, 18, 23, 10, 0, 123456, tzinfo=timezone.utc)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                mock.patch.object(feedback, "_utc_now", return_value=fixed_now),
                mock.patch.object(feedback.secrets, "token_hex", side_effect=["a1b2", "c3d4"]),
            ):
                first, first_path = feedback.log_composition_run(
                    mood="Ambient",
                    key_name="D minor",
                    bpm=120.0,
                    sig_num=4,
                    sig_den=4,
                    minutes=2.0,
                    bars=24,
                    section_bars=8,
                    sections=sections,
                    arranged_by_track=_arranged_payload(sections),
                    created_clips_by_track={"Kick Drum": 3, "RIM": 3, "HAT": 3, "Piano": 3},
                    status="success",
                    repo_root=root,
                )
                second, second_path = feedback.log_composition_run(
                    mood="Ambient",
                    key_name="D minor",
                    bpm=120.0,
                    sig_num=4,
                    sig_den=4,
                    minutes=2.0,
                    bars=24,
                    section_bars=8,
                    sections=sections,
                    arranged_by_track=_arranged_payload(sections),
                    created_clips_by_track={"Kick Drum": 3, "RIM": 3, "HAT": 3, "Piano": 3},
                    status="success",
                    repo_root=root,
                )

            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertNotEqual(first_path, second_path)
            entries = feedback._load_json(root / feedback.DEFAULT_RELATIVE_INDEX_PATH, {}).get("entries", [])
            self.assertEqual(len(entries), 2)

    def test_reference_selection_prefers_matching_ensemble_signature(self) -> None:
        current = {
            "fingerprints": {
                "meter_bpm": "4/4@120",
                "note_count_paths": {"Marimba": [4, 6, 8]},
                "ensemble_signature": "marimba",
            },
            "status": "success",
            "run": {"marimba_runtime": {"composition_family": "left_hand_ostinato_right_hand_melody"}},
        }
        candidate_meter_only = {
            "fingerprints": {
                "meter_bpm": "4/4@120",
                "note_count_paths": {"Piano": [4, 6, 8]},
                "ensemble_signature": "piano",
            },
            "status": "success",
            "run": {},
        }
        candidate_ensemble_match = {
            "fingerprints": {
                "meter_bpm": "4/4@120",
                "note_count_paths": {"Marimba": [5, 7, 9]},
                "ensemble_signature": "marimba",
            },
            "status": "success",
            "run": {},
        }

        picked, reason = feedback._pick_reference_artifact(
            current_meter_bpm="4/4@120",
            current_ensemble_signature="marimba",
            current_status="success",
            current_run_metadata=current["run"],
            recent_artifacts=[candidate_meter_only, candidate_ensemble_match],
        )
        self.assertIsNotNone(picked)
        self.assertIs(picked, candidate_ensemble_match)
        self.assertIsNotNone(reason)

    def test_second_similar_run_reports_repetition_flags(self) -> None:
        sections = [
            _Section(index=0, label="intro", piano_mode="chords", hat_density="quarter"),
            _Section(index=1, label="build", piano_mode="chords", hat_density="eighth"),
            _Section(index=2, label="climax", piano_mode="chords", hat_density="sixteenth"),
        ]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            feedback.log_composition_run(
                mood="Energetic",
                key_name="D minor",
                bpm=137.0,
                sig_num=6,
                sig_den=4,
                minutes=2.0,
                bars=46,
                section_bars=8,
                sections=sections,
                arranged_by_track=_arranged_payload(sections, hat_mode="sixteenth"),
                created_clips_by_track={"Kick Drum": 3, "RIM": 3, "HAT": 3, "Piano": 3},
                status="success",
                repo_root=root,
            )

            second, _ = feedback.log_composition_run(
                mood="Mysterious",
                key_name="D minor",
                bpm=137.0,
                sig_num=6,
                sig_den=4,
                minutes=2.0,
                bars=46,
                section_bars=8,
                sections=sections,
                arranged_by_track=_arranged_payload(sections, hat_mode="sixteenth"),
                created_clips_by_track={"Kick Drum": 3, "RIM": 3, "HAT": 3, "Piano": 3},
                status="success",
                repo_root=root,
            )

            reflection = second["reflection"]
            self.assertIsNotNone(reflection["novelty_score"])
            self.assertLessEqual(reflection["novelty_score"], 0.2)
            self.assertIn(
                "overall_structure_highly_similar_to_recent_run",
                reflection["repetition_flags"],
            )
            self.assertIn("repetition_risk", reflection["merit_rubric"])
            self.assertTrue(any("trajectory" in prompt.lower() for prompt in reflection["prompts"]))

    def test_artifact_includes_marimba_identity_metrics(self) -> None:
        sections = [
            _Section(index=0, label="intro", piano_mode="motion", hat_density="quarter"),
            _Section(index=1, label="build", piano_mode="motion", hat_density="eighth"),
            _Section(index=2, label="climax", piano_mode="motion", hat_density="sixteenth"),
        ]
        contract = {
            "track_name": "Marimba",
            "pair_track_name": "Vibraphone",
            "constraints": {
                "midi_min_pitch": 30,
                "midi_max_pitch": 90,
                "max_leap_semitones": 12,
            },
            "pair_rules": {
                "attack_answer": {
                    "min_start_separation_beats": 0.25,
                    "marimba_max_duration_beats": 0.55,
                    "vibraphone_min_duration_beats": 0.7,
                }
            },
            "eval": {
                "target_pair_overlap_max_ratio": 0.34,
                "target_range_adherence_min_ratio": 0.5,
                "target_leap_adherence_min_ratio": 0.5,
            },
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact, _path = feedback.log_composition_run(
                mood="Energetic",
                key_name="D minor",
                bpm=166.0,
                sig_num=6,
                sig_den=8,
                minutes=1.0,
                bars=12,
                section_bars=4,
                sections=sections,
                arranged_by_track=_marimba_pair_payload(sections),
                created_clips_by_track={"Marimba": 3, "Vibraphone": 3},
                status="dry_run",
                instrument_identity_contract=contract,
                repo_root=root,
            )

            identity = artifact["reflection"]["instrument_identity"]
            self.assertTrue(identity["enabled"])
            self.assertEqual(identity["status"], "ok")
            self.assertEqual(identity["marimba_track"], "Marimba")
            self.assertEqual(identity["pair_track"], "Vibraphone")
            self.assertIn("range_adherence_ratio", identity["marimba"])
            self.assertIn("overlap_ratio", identity["pair"])

    def test_style_fingerprint_detects_polyphonic_shift_even_with_same_note_counts(self) -> None:
        sections = [
            _Section(index=0, label="intro", piano_mode="chords", hat_density="quarter"),
            _Section(index=1, label="build", piano_mode="chords", hat_density="eighth"),
            _Section(index=2, label="release", piano_mode="chords", hat_density="quarter"),
        ]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            feedback.log_composition_run(
                mood="Energetic",
                key_name="D minor",
                bpm=120.0,
                sig_num=4,
                sig_den=4,
                minutes=2.0,
                bars=24,
                section_bars=8,
                sections=sections,
                arranged_by_track=_marimba_style_payload(sections, stacked=False),
                created_clips_by_track={"Marimba": 3},
                status="success",
                repo_root=root,
            )

            second, _ = feedback.log_composition_run(
                mood="Energetic",
                key_name="D minor",
                bpm=120.0,
                sig_num=4,
                sig_den=4,
                minutes=2.0,
                bars=24,
                section_bars=8,
                sections=sections,
                arranged_by_track=_marimba_style_payload(sections, stacked=True),
                created_clips_by_track={"Marimba": 3},
                status="success",
                repo_root=root,
            )

            reflection = second["reflection"]
            self.assertIn("style:Marimba", reflection["similarity_breakdown"])
            self.assertLess(reflection["similarity_breakdown"]["style:Marimba"], 0.95)
            self.assertLess(
                reflection["similarity_breakdown"]["style:Marimba"],
                reflection["similarity_breakdown"]["note_shape:Marimba"],
            )
            self.assertGreater(reflection["novelty_score"], 0.0)
            self.assertNotIn("hat_density_trajectory_repeated", reflection["repetition_flags"])
            self.assertNotIn("piano_mode_trajectory_repeated", reflection["repetition_flags"])


if __name__ == "__main__":
    unittest.main()
