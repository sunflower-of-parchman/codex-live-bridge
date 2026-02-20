#!/usr/bin/env python3
"""Unit tests for the next-track orchestration flow."""

from __future__ import annotations

import argparse
import unittest
from unittest import mock
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import run_next_track as runner


class RunNextTrackTests(unittest.TestCase):
    def test_parse_args_require_bpm_and_meter(self) -> None:
        ns = runner.parse_args(["--bpm", "120", "--meter", "5/4"])
        self.assertEqual(ns.bpm, 120.0)
        self.assertEqual(ns.meter, "5/4")

    def test_run_next_track_composes_directly_without_setup_stage(self) -> None:
        ns = runner.parse_args(
            [
                "--bpm",
                "77",
                "--meter",
                "5/4",
                "--key-name",
                "G# minor",
                "--mood",
                "Beautiful",
                "--archive-dir",
                "output/live_sets",
            ]
        )
        captured_compose_argv: list[str] | None = None

        def fake_arrangement_parse(argv: list[str]) -> argparse.Namespace:
            nonlocal captured_compose_argv
            captured_compose_argv = list(argv)
            return argparse.Namespace()

        with (
            mock.patch.object(runner.arrangement, "parse_args", side_effect=fake_arrangement_parse),
            mock.patch.object(runner.arrangement, "run", return_value=0),
            mock.patch.object(runner.setup, "_launch_ableton_live"),
        ):
            status = runner.run_next_track(ns)

        self.assertEqual(status, 0)
        self.assertIn("--sig-num", captured_compose_argv or [])
        self.assertIn("5", captured_compose_argv or [])
        self.assertIn("--sig-den", captured_compose_argv or [])
        self.assertIn("4", captured_compose_argv or [])
        self.assertIn("--key-name", captured_compose_argv or [])
        self.assertIn("G# minor", captured_compose_argv or [])
        self.assertIn("--instrument-registry-path", captured_compose_argv or [])
        self.assertIn("--clip-write-mode", captured_compose_argv or [])
        self.assertIn("single_clip", captured_compose_argv or [])
        self.assertIn("--no-write-cache", captured_compose_argv or [])
        self.assertTrue(
            any(
                item.endswith("bridge/config/instrument_registry.marimba_piano.v1.json")
                for item in (captured_compose_argv or [])
            )
        )
        self.assertIn("--archive-dir", captured_compose_argv or [])
        self.assertIn("output/live_sets", captured_compose_argv or [])

    def test_run_next_track_returns_arrangement_status(self) -> None:
        ns = runner.parse_args(["--bpm", "88", "--meter", "4/4", "--mood", "Beautiful", "--dry-run"])
        arrangement_status = -7

        with (
            mock.patch.object(runner.arrangement, "parse_args", return_value=argparse.Namespace()),
            mock.patch.object(runner.arrangement, "run", return_value=arrangement_status) as mock_arrangement_run,
            mock.patch.object(runner.setup, "_launch_ableton_live"),
        ):
            status = runner.run_next_track(ns)

        self.assertEqual(status, arrangement_status)
        mock_arrangement_run.assert_called_once()

    def test_compose_args_uses_explicit_duration_seed(self) -> None:
        ns = runner.parse_args(["--bpm", "110", "--meter", "5/4", "--duration-seed", "99"])
        compose_argv = runner._compose_args(ns)
        seed_idx = compose_argv.index("--duration-seed")
        self.assertEqual(compose_argv[seed_idx + 1], "99")

    def test_compose_args_generates_seed_when_omitted(self) -> None:
        ns = runner.parse_args(["--bpm", "110", "--meter", "5/4"])
        with mock.patch.object(runner, "_resolve_duration_seed", return_value=123456):
            compose_argv = runner._compose_args(ns)
        seed_idx = compose_argv.index("--duration-seed")
        self.assertEqual(compose_argv[seed_idx + 1], "123456")

    def test_compose_args_honors_custom_registry_path(self) -> None:
        ns = runner.parse_args(
            [
                "--bpm",
                "96",
                "--meter",
                "4/4",
                "--instrument-registry-path",
                "bridge/config/instrument_registry.marimba_piano.v1.json",
            ]
        )
        compose_argv = runner._compose_args(ns)
        registry_idx = compose_argv.index("--instrument-registry-path")
        self.assertEqual(
            compose_argv[registry_idx + 1],
            "bridge/config/instrument_registry.marimba_piano.v1.json",
        )

    def test_compose_args_passes_composition_goal(self) -> None:
        ns = runner.parse_args(
            [
                "--bpm",
                "96",
                "--meter",
                "4/4",
                "--composition-goal",
                "Maintain left-hand anchors and right-hand rhythmic contrast with silence.",
            ]
        )
        compose_argv = runner._compose_args(ns)
        goal_idx = compose_argv.index("--composition-goal")
        self.assertEqual(
            compose_argv[goal_idx + 1],
            "Maintain left-hand anchors and right-hand rhythmic contrast with silence.",
        )


if __name__ == "__main__":
    unittest.main()
