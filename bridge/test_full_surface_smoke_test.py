#!/usr/bin/env python3
"""Safety tests for the mutating Ableton Live full-surface smoke script."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import full_surface_smoke_test as smoke


TEST_AUTH_TOKEN = "test-auth-token-0123456789"


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
            exit_code = smoke.main(
                [
                    smoke.MUTATING_FLAG,
                    "--auth-token",
                    TEST_AUTH_TOKEN,
                ]
            )
        self.assertEqual(exit_code, 0)
        run.assert_called_once_with(TEST_AUTH_TOKEN)

    def test_new_track_index_requires_exactly_one_created_track(self) -> None:
        tracks_before = [{"index": 0}]

        self.assertIsNone(smoke._new_track_index(tracks_before, tracks_before))
        self.assertIsNone(
            smoke._new_track_index(tracks_before, tracks_before + [{"index": 1}, {"index": 2}])
        )
        self.assertEqual(smoke._new_track_index(tracks_before, tracks_before + [{"index": 1}]), 1)

    def test_api_children_extractor_ignores_forged_uncorrelated_ack(self) -> None:
        forged_tracks = [{"index": 0}, {"index": 1}, {"index": 2}]
        legitimate_tracks = [{"index": 0}]
        acks = [
            (
                "/ack",
                [
                    "api_children",
                    "live_set",
                    "tracks",
                    json.dumps(forged_tracks),
                    "attacker-request",
                ],
            ),
            (
                "/ack",
                [
                    "api_children",
                    "live_set",
                    "tracks",
                    json.dumps(legitimate_tracks),
                    "smoke-request",
                ],
            ),
        ]

        self.assertEqual(
            smoke._extract_api_children(acks, "smoke-request"),
            legitimate_tracks,
        )
        self.assertEqual(smoke._extract_api_children(acks, "missing-request"), [])

    def test_api_children_extractor_requires_exact_tracks_context(self) -> None:
        expected_tracks = [{"index": 0}]
        acks = [
            (
                "/ack",
                [
                    "api_children",
                    "live_set tracks 0",
                    "devices",
                    json.dumps([{"index": 99}]),
                    "smoke-request",
                ],
            ),
            (
                "/ack",
                [
                    "api_children",
                    "live_set",
                    "tracks",
                    json.dumps(expected_tracks),
                    "smoke-request",
                ],
            ),
        ]

        self.assertEqual(
            smoke._extract_api_children(acks, "smoke-request"),
            expected_tracks,
        )

    def test_send_and_collect_requires_loopback_ack_sender(self) -> None:
        send_sock = mock.Mock()
        ack_sock = mock.Mock()
        command = smoke.bridge.OscCommand("/status")

        with mock.patch.object(
            smoke.bridge,
            "wait_for_acks",
            return_value=[],
        ) as wait_for_acks:
            smoke._send_and_collect_acks(send_sock, ack_sock, command)

        send_sock.sendto.assert_called_once()
        wait_for_acks.assert_called_once_with(
            ack_sock,
            smoke.ACK_TIMEOUT_S,
            expected_sender_host=smoke.HOST,
            expected_request_id=None,
        )

    def test_send_and_collect_passes_expected_request_id(self) -> None:
        send_sock = mock.Mock()
        ack_sock = mock.Mock()
        command = smoke.bridge.OscCommand(
            "/api/children",
            ("live_set", "tracks", "smoke-request"),
        )

        with mock.patch.object(
            smoke.bridge,
            "wait_for_acks",
            return_value=[],
        ) as wait_for_acks:
            smoke._send_and_collect_acks(
                send_sock,
                ack_sock,
                command,
                expected_request_id="smoke-request",
            )

        wait_for_acks.assert_called_once_with(
            ack_sock,
            smoke.ACK_TIMEOUT_S,
            expected_sender_host=smoke.HOST,
            expected_request_id="smoke-request",
        )

    def test_smoke_request_ids_are_unique_and_unpredictable_length(self) -> None:
        first = smoke._smoke_request_id("tracks")
        second = smoke._smoke_request_id("tracks")

        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^smoke-tracks-[0-9a-f]{32}$")

    def _run_with_final_acks(
        self,
        inspection_acks: list[tuple[str, list[object]]],
        status_acks: list[tuple[str, list[object]]],
    ) -> tuple[int, mock.Mock]:
        ack_socket = mock.Mock()

        def collect_acks(
            _send_socket: object,
            _ack_socket: object,
            command: smoke.bridge.OscCommand,
            **_kwargs: object,
        ) -> list[tuple[str, list[object]]]:
            if command.address == "/api/children" and len(command.args) == 3:
                request_id = str(command.args[2])
                tracks = [{"index": 0}]
                if "tracks-after" in request_id:
                    tracks.append({"index": 1})
                return [
                    (
                        "/ack",
                        ["api_children", "live_set", "tracks", json.dumps(tracks), request_id],
                    )
                ]
            if command.address == "/inspect_session_clip_notes":
                return inspection_acks
            if command.address == "/status":
                return status_acks
            return [("/ack", ["ok"])]

        with (
            mock.patch.object(smoke.bridge, "open_ack_socket", return_value=ack_socket),
            mock.patch.object(smoke.socket, "socket"),
            mock.patch.object(smoke, "_send_and_collect_acks", side_effect=collect_acks),
            mock.patch("sys.stdout"),
            mock.patch("sys.stderr"),
        ):
            return smoke.run(TEST_AUTH_TOKEN), ack_socket

    def test_smoke_fails_when_final_clip_inspection_is_missing(self) -> None:
        exit_code, ack_socket = self._run_with_final_acks([], [])

        self.assertEqual(exit_code, 4)
        ack_socket.close.assert_called_once()

    def test_smoke_fails_when_final_clip_note_count_is_incorrect(self) -> None:
        exit_code, ack_socket = self._run_with_final_acks(
            [("/ack", ["inspect_session_clip_notes", 1, 0, 1])],
            [("/ack", ["status", 2, 2, 0, 0])],
        )

        self.assertEqual(exit_code, 4)
        ack_socket.close.assert_called_once()

    def test_smoke_fails_when_final_bridge_status_is_missing(self) -> None:
        notes, _length = smoke._build_notes()
        exit_code, ack_socket = self._run_with_final_acks(
            [("/ack", ["inspect_session_clip_notes", 1, 0, len(notes)])],
            [],
        )

        self.assertEqual(exit_code, 5)
        ack_socket.close.assert_called_once()

    def test_smoke_succeeds_only_with_matching_clip_and_final_status(self) -> None:
        notes, _length = smoke._build_notes()
        exit_code, ack_socket = self._run_with_final_acks(
            [("/ack", ["inspect_session_clip_notes", 1, 0, len(notes)])],
            [("/ack", ["status", 2, 2, 0, 0])],
        )

        self.assertEqual(exit_code, 0)
        ack_socket.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
