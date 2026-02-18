#!/usr/bin/env python3
"""Runtime-entrypoint tests for compose_arrangement."""

from __future__ import annotations

import io
import pathlib
import sys
import unittest
from contextlib import redirect_stdout

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import compose_arrangement as arrangement


class ComposeArrangementRuntimeTests(unittest.TestCase):
    def test_dry_run_with_marimba_piano_registry_succeeds(self) -> None:
        cfg = arrangement.parse_args(
            [
                "--minutes",
                "2",
                "--bpm",
                "124",
                "--sig-num",
                "5",
                "--sig-den",
                "4",
                "--mood",
                "Beautiful",
                "--key-name",
                "G# minor",
                "--instrument-registry-path",
                "bridge/config/instrument_registry.marimba_piano.v1.json",
                "--clip-write-mode",
                "single_clip",
                "--write-strategy",
                "delta_update",
                "--no-composition-print",
                "--no-eval-log",
                "--no-multi-pass",
                "--dry-run",
            ]
        )

        output_buf = io.StringIO()
        with redirect_stdout(output_buf):
            status = arrangement.run(cfg)

        output = output_buf.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("- instruments: 2", output)
        self.assertIn("- clip mode:   single_clip", output)
        self.assertIn("- marimba strategy: ", output)


if __name__ == "__main__":
    unittest.main()
