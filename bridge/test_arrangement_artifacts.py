#!/usr/bin/env python3
"""Focused tests for arrangement save artifact helpers."""

from __future__ import annotations

import pathlib
import socket
import sys
import tempfile
import unittest
from unittest import mock

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from arrangement import artifacts


class ArrangementArtifactsTests(unittest.TestCase):
    def test_archive_live_set_uses_ui_fallback_when_api_save_does_not_write_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = pathlib.Path(tmp_dir) / "3_60.als"
            with (
                mock.patch.object(artifacts.kick, "_api_call", return_value=0),
                mock.patch.object(artifacts.kick, "_api_get", return_value=0),
                mock.patch.object(artifacts.kick, "_scalar", return_value=None),
                mock.patch.object(
                    artifacts,
                    "_try_ui_save_live_set",
                    side_effect=lambda _path: (archive_path.write_text("ok", encoding="utf-8") or True, "saved via ui fallback"),
                ),
            ):
                ok, message = artifacts._archive_live_set(
                    sock=mock.Mock(spec=socket.socket),
                    ack_sock=mock.Mock(spec=socket.socket),
                    timeout_s=1.0,
                    archive_path=archive_path,
                )

        self.assertTrue(ok)
        self.assertIn("saved via ui fallback", message)

    def test_archive_live_set_reports_ui_fallback_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = pathlib.Path(tmp_dir) / "3_60.als"
            with (
                mock.patch.object(
                    artifacts.kick,
                    "_api_call",
                    side_effect=RuntimeError("save failed"),
                ),
                mock.patch.object(
                    artifacts,
                    "_try_ui_save_live_set",
                    return_value=(False, "ui save unavailable"),
                ),
            ):
                ok, message = artifacts._archive_live_set(
                    sock=mock.Mock(spec=socket.socket),
                    ack_sock=mock.Mock(spec=socket.socket),
                    timeout_s=1.0,
                    archive_path=archive_path,
                )

        self.assertFalse(ok)
        self.assertIn("ui save unavailable", message)


if __name__ == "__main__":
    unittest.main()

