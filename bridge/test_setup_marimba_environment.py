#!/usr/bin/env python3
"""Unit tests for marimba environment setup helpers."""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import setup_marimba_environment as setup


class SetupMarimbaEnvironmentTests(unittest.TestCase):
    def test_parse_meter_accepts_num_den(self) -> None:
        self.assertEqual(setup.parse_meter("5/4"), (5, 4))
        self.assertEqual(setup.parse_meter("7/8"), (7, 8))

    def test_parse_meter_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            setup.parse_meter("five-four")
        with self.assertRaises(ValueError):
            setup.parse_meter("0/4")

    def test_parse_key_name_accepts_sharp_minor(self) -> None:
        root_note, scale_name = setup.parse_key_name("G# minor")
        self.assertEqual(root_note, 8)
        self.assertEqual(scale_name, "Minor")

    def test_parse_key_name_accepts_flat_major(self) -> None:
        root_note, scale_name = setup.parse_key_name("Bb major")
        self.assertEqual(root_note, 10)
        self.assertEqual(scale_name, "Major")

    def test_parse_key_name_rejects_invalid_quality(self) -> None:
        with self.assertRaises(ValueError):
            setup.parse_key_name("C dorian")

    def test_count_pong_acks_counts_only_ack_pong_events(self) -> None:
        acks = [
            ("/ack", ["pong"]),
            ("/ack", ["pong", 1]),
            ("/ack", ["status", 2, 2]),
            ("/other", ["pong"]),
        ]
        self.assertEqual(setup._count_pong_acks(acks), 2)

    def test_build_meter_bpm_filename_uses_meter_underscore_bpm(self) -> None:
        self.assertEqual(setup.build_meter_bpm_filename(5, 4, 120.0), "5_120.als")
        self.assertEqual(setup.build_meter_bpm_filename(7, 8, 97.5), "7_97_5.als")

    def test_build_clip_plan_rounds_up_to_whole_bars(self) -> None:
        plan = setup.build_clip_plan(
            bpm=97.0,
            sig_num=5,
            sig_den=4,
            minutes=5.0,
            start_beats=0.0,
        )
        self.assertEqual(plan.target_beats, 485.0)
        self.assertEqual(plan.bars, 97)
        self.assertEqual(plan.clip_length_beats, 485.0)
        self.assertEqual(plan.clip_end_beats, 485.0)

    def test_build_clip_plan_handles_non_integer_beats_per_bar(self) -> None:
        plan = setup.build_clip_plan(
            bpm=120.0,
            sig_num=7,
            sig_den=8,
            minutes=5.0,
            start_beats=4.0,
        )
        self.assertEqual(plan.beats_per_bar, 3.5)
        self.assertEqual(plan.target_beats, 600.0)
        self.assertEqual(plan.bars, 172)
        self.assertEqual(plan.clip_length_beats, 602.0)
        self.assertEqual(plan.clip_end_beats, 606.0)

    def test_parse_args_defaults_to_five_minute_marimba_blank(self) -> None:
        cfg = setup.parse_args(["--bpm", "132"])
        self.assertEqual(cfg.bpm, 132.0)
        self.assertEqual(cfg.minutes, 5.0)
        self.assertEqual(cfg.sig_num, 4)
        self.assertEqual(cfg.sig_den, 4)
        self.assertEqual(cfg.track_name, "Marimba")
        self.assertEqual(cfg.clip_name, "Marimba Blank")
        self.assertTrue(cfg.launch_ableton)
        self.assertIsNone(cfg.key_name)
        self.assertEqual(cfg.save_policy, "ephemeral")
        self.assertIsNone(cfg.archive_dir)

    def test_parse_args_allows_disabling_launch(self) -> None:
        cfg = setup.parse_args(["--bpm", "132", "--no-launch-ableton"])
        self.assertFalse(cfg.launch_ableton)

    def test_parse_args_meter_overrides_sig_num_sig_den(self) -> None:
        cfg = setup.parse_args(["--bpm", "110", "--sig-num", "3", "--sig-den", "4", "--meter", "5/4"])
        self.assertEqual(cfg.sig_num, 5)
        self.assertEqual(cfg.sig_den, 4)

    def test_parse_args_accepts_archive_save_policy(self) -> None:
        cfg = setup.parse_args(
            [
                "--bpm",
                "120",
                "--meter",
                "4/4",
                "--save-policy",
                "archive",
                "--archive-dir",
                "output/live_sets",
            ]
        )
        self.assertEqual(cfg.save_policy, "archive")
        self.assertEqual(cfg.archive_dir, "output/live_sets")

    def test_parse_args_accepts_current_save_policy(self) -> None:
        cfg = setup.parse_args(["--bpm", "90", "--meter", "4/4", "--save-policy", "current"])
        self.assertEqual(cfg.save_policy, "current")

    def test_parse_args_accepts_key_name(self) -> None:
        cfg = setup.parse_args(["--bpm", "90", "--meter", "4/4", "--key-name", "G# minor"])
        self.assertEqual(cfg.key_name, "G# minor")

    def test_create_blank_clip_sets_clip_signature(self) -> None:
        with (
            mock.patch.object(setup.kick, "_get_children", return_value=[]),
            mock.patch.object(setup.kick, "_api_call", side_effect=[["ok"], ["ok"]]),
            mock.patch.object(setup.kick, "_extract_id_from_call_result", return_value=1234),
            mock.patch.object(setup.kick, "_api_get", return_value=[1234]),
            mock.patch.object(setup.kick, "_api_set") as mock_api_set,
        ):
            clip_path = setup._create_blank_clip(
                sock=mock.Mock(),
                ack_sock=mock.Mock(),
                track_path="live_set tracks 0",
                clip_start=0.0,
                clip_length=300.0,
                clip_name="Marimba Blank",
                sig_num=5,
                sig_den=4,
                timeout_s=1.75,
            )

        self.assertEqual(clip_path, "id 1234")
        mock_api_set.assert_any_call(
            mock.ANY,
            mock.ANY,
            "id 1234",
            "signature_numerator",
            5,
            "setup-clip-sig-num",
            1.75,
        )
        mock_api_set.assert_any_call(
            mock.ANY,
            mock.ANY,
            "id 1234",
            "signature_denominator",
            4,
            "setup-clip-sig-den",
            1.75,
        )


if __name__ == "__main__":
    unittest.main()
