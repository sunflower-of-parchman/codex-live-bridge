"""Unit tests for composition feedback-loop artifact logging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence
import pathlib
import sys
import unittest

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
            self.assertIn("merit_rubric", artifact["reflection"])
            self.assertIn("instrument_diversity_proxy", artifact["reflection"]["merit_rubric"])

            index_path = root / feedback.DEFAULT_RELATIVE_INDEX_PATH
            self.assertTrue(index_path.exists())
            entries = feedback._load_json(index_path, {}).get("entries", [])
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["run_id"], artifact["run_id"])

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
