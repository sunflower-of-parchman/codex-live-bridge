#!/usr/bin/env python3
"""Safety tests for the mutating Ableton Live full-surface smoke script."""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import full_surface_smoke_test as smoke


class FullSurfaceSmokeSafetyTests(unittest.TestCase):
    def test_requires_explicit_mutating_flag(self) -> None:
        self.assertTrue(hasattr(smoke, "parse_args"))
        with self.assertRaises(SystemExit) as raised:
            smoke.parse_args([])
        self.assertNotEqual(raised.exception.code, 0)

    def test_main_does_not_run_without_mutating_flag(self) -> None:
        self.assertTrue(hasattr(smoke, "main"))
        with mock.patch.object(smoke, "run", return_value=0) as run:
            exit_code = smoke.main([])
        self.assertNotEqual(exit_code, 0)
        run.assert_not_called()

    def test_main_runs_with_explicit_mutating_flag(self) -> None:
        with mock.patch.object(smoke, "run", return_value=0) as run:
            exit_code = smoke.main([smoke.MUTATING_FLAG])
        self.assertEqual(exit_code, 0)
        run.assert_called_once_with()

    def test_new_track_index_requires_exactly_one_created_track(self) -> None:
        tracks_before = [{"index": 0}]

        self.assertIsNone(smoke._new_track_index(tracks_before, tracks_before))
        self.assertIsNone(
            smoke._new_track_index(tracks_before, tracks_before + [{"index": 1}, {"index": 2}])
        )
        self.assertEqual(smoke._new_track_index(tracks_before, tracks_before + [{"index": 1}]), 1)


if __name__ == "__main__":
    unittest.main()
