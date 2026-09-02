#!/usr/bin/env python3
"""Safety tests for the mutating Ableton Live full-surface smoke script."""

from __future__ import annotations

import copy
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
        tracks_before = [{"index": 0, "id": 100, "path": "live_set tracks 0"}]
        new_track = {"index": 1, "id": 101, "path": "live_set tracks 1"}

        self.assertIsNone(smoke._new_track_index(tracks_before, tracks_before))
        self.assertIsNone(
            smoke._new_track_index(tracks_before, tracks_before + [new_track, {"index": 2}])
        )
        self.assertEqual(smoke._new_track_index(tracks_before, tracks_before + [new_track]), 1)

    def test_new_track_index_rejects_changed_or_missing_track_identity(self) -> None:
        before = [{"index": 0, "id": 100, "path": "live_set tracks 0"}]
        new_track = {"index": 1, "id": 101, "path": "live_set tracks 1"}
        for after in (
            [{"index": 0, "id": 999, "path": "live_set tracks 0"}, new_track],
            before + [{"index": 1}],
            before + [{**new_track, "id": 100}],
        ):
            with self.subTest(after=after):
                self.assertIsNone(smoke._new_track_index(before, after))

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
        acks[-1][1][1] = '"live_set"'
        self.assertEqual(smoke._extract_api_children(acks, "smoke-request"), expected_tracks)

    def test_send_and_collect_requires_loopback_ack_sender(self) -> None:
        send_sock = mock.Mock()
        ack_sock = mock.Mock()
        command = smoke.bridge.OscCommand("/status")

        with (
            mock.patch.object(smoke.bridge, "_drain_acks_nonblocking", return_value=[]),
            mock.patch.object(smoke.bridge, "wait_for_acks", return_value=[]) as wait_for_acks,
        ):
            smoke._send_and_collect_acks(send_sock, ack_sock, command)

        send_sock.sendto.assert_called_once()
        wait_for_acks.assert_called_once_with(
            ack_sock,
            smoke.ACK_TIMEOUT_S,
            expected_sender_host=smoke.HOST,
            expected_request_id=None,
            expected_event="status",
            expected_commands=(command,),
        )

    def test_send_and_collect_passes_expected_request_id(self) -> None:
        send_sock = mock.Mock()
        ack_sock = mock.Mock()
        command = smoke.bridge.OscCommand(
            "/api/children",
            ("live_set", "tracks", "smoke-request"),
        )

        with (
            mock.patch.object(smoke.bridge, "_drain_acks_nonblocking", return_value=[]),
            mock.patch.object(smoke.bridge, "wait_for_acks", return_value=[]) as wait_for_acks,
        ):
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
            expected_event="api_children",
            expected_commands=(command,),
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
        *,
        fail_address: str | None = None,
        fail_property: str | None = None,
        wrong_readback: str | None = None,
        initial_tracks: list[dict] | None = None,
    ) -> tuple[int, mock.Mock]:
        ack_socket = mock.Mock()
        ack_socket.sent_commands = []
        created = False
        baseline = (
            [{"index": 0, "id": 100, "path": "live_set tracks 0"}]
            if initial_tracks is None else initial_tracks
        )

        def collect_acks(
            _send_socket: object,
            _ack_socket: object,
            command: smoke.bridge.OscCommand,
            **_kwargs: object,
        ) -> list[tuple[str, list[object]]]:
            nonlocal created
            ack_socket.sent_commands.append(command)
            _event, request_id = smoke.bridge._command_ack_expectation(command)
            request = [] if request_id is None else [request_id]
            if command.address == fail_address or (
                command.address == "/api/set" and command.args[2] == fail_property
            ):
                return [("/ack", [
                    "error", "fixture_rejected", "request_correlation", f"req:{request_id or ''}"
                ])]
            if command.address in {"/ping", "/api/ping"}:
                return [("/ack", ["pong", *request])]
            if command.address == "/api/describe":
                return [("/ack", ["api_describe", "live_set", '{"path":"live_set","id":1}', *request])]
            if command.address == "/api/children":
                tracks = list(baseline)
                if created:
                    tracks.append({"index": 1, "id": 101, "path": "live_set tracks 1"})
                return [("/ack", ["api_children", "live_set", "tracks", json.dumps(tracks), *request])]
            if command.address == "/api/call":
                created = True
                return [("/ack", ["api_call", command.args[1], command.args[2], "[]", *request])]
            if command.address == "/api/set":
                return [("/ack", ["api_set", command.args[1], command.args[2], '{"ok":true}', *request])]
            if command.address == "/api/get":
                values = {
                    "tempo": 142.0,
                    "signature_numerator": 5,
                    "signature_denominator": 4,
                    "name": "Full Surface Smoke",
                    "length": 8.0,
                }
                prop = str(command.args[1])
                value = "incorrect readback" if prop == wrong_readback else values[prop]
                return [("/ack", ["api_get", command.args[0], prop, json.dumps([value]), *request])]
            if command.address == "/set_session_clip_notes":
                return [("/ack", ["set_session_clip_notes", 1, 0, 8.0, 16, 16, "Full Surface Smoke"])]
            if command.address == "/inspect_session_clip_notes":
                return inspection_acks
            if command.address == "/status":
                return status_acks if created else [("/ack", ["status", len(baseline), len(baseline), 0, 0])]
            raise AssertionError(f"unexpected command: {command.address}")

        with (
            mock.patch.object(smoke.bridge, "open_ack_socket", return_value=ack_socket),
            mock.patch.object(smoke.socket, "socket"),
            mock.patch.object(smoke, "_send_and_collect_acks", side_effect=collect_acks),
            mock.patch("sys.stdout"),
            mock.patch("sys.stderr"),
        ):
            return smoke.run(TEST_AUTH_TOKEN), ack_socket

    @staticmethod
    def _inspection_acks(
        *,
        track: int = 1,
        slot: int = 0,
        notes: list[dict[str, object]] | None = None,
    ) -> list[tuple[str, list[object]]]:
        pattern, length = smoke._build_notes()
        values = pattern if notes is None else notes
        return [("/ack", [
            "inspect_session_clip_notes", track, slot, len(values),
            min(note["pitch"] for note in values), max(note["pitch"] for note in values),
            length, json.dumps({"notes": values}),
        ])]

    def test_smoke_fails_when_final_clip_inspection_is_missing(self) -> None:
        exit_code, ack_socket = self._run_with_final_acks([], [])

        self.assertEqual(exit_code, 4)
        ack_socket.close.assert_called_once()

    def test_smoke_fails_when_final_clip_note_count_is_incorrect(self) -> None:
        exit_code, ack_socket = self._run_with_final_acks(
            self._inspection_acks(notes=smoke._build_notes()[0][:1]),
            [("/ack", ["status", 2, 2, 0, 0])],
        )

        self.assertEqual(exit_code, 4)
        ack_socket.close.assert_called_once()

    def test_smoke_fails_when_final_bridge_status_is_missing(self) -> None:
        exit_code, ack_socket = self._run_with_final_acks(
            self._inspection_acks(),
            [],
        )

        self.assertEqual(exit_code, 5)
        ack_socket.close.assert_called_once()

    def test_smoke_succeeds_only_with_matching_clip_and_final_status(self) -> None:
        exit_code, ack_socket = self._run_with_final_acks(
            self._inspection_acks(),
            [("/ack", ["status", 2, 2, 0, 0])],
        )

        self.assertEqual(exit_code, 0)
        ack_socket.close.assert_called_once()

    def test_smoke_stops_immediately_when_a_mutation_is_rejected(self) -> None:
        for address in ("/api/call", "/api/set", "/set_session_clip_notes"):
            with self.subTest(address=address):
                code, ack = self._run_with_final_acks(
                    self._inspection_acks(), [("/ack", ["status", 2, 2, 0, 0])],
                    fail_address=address,
                )
                self.assertNotEqual(code, 0)
                self.assertEqual(ack.sent_commands[-1].address, address)
                ack.close.assert_called_once()

    def test_smoke_validates_baseline_track_identities_before_any_mutation(self) -> None:
        track = {"index": 0, "id": 100, "path": "live_set tracks 0"}
        for baseline in (
            [{**track, "id": 0, "error": "resolve_failed"}],
            [{**track, "index": 1}],
            [{**track, "path": "live_set tracks 1"}],
            [{"index": 0, "path": "live_set tracks 0"}],
            [track, {**track, "index": 1, "path": "live_set tracks 1"}],
        ):
            with self.subTest(baseline=baseline):
                code, ack = self._run_with_final_acks([], [], initial_tracks=baseline)
                self.assertEqual(
                    [command.address for command in ack.sent_commands
                     if command.address in smoke.bridge.PROTECTED_OSC_ADDRESSES],
                    [],
                )
                self.assertEqual(code, 2)
                ack.close.assert_called_once()

    def test_smoke_rejects_an_inspection_of_another_clip(self) -> None:
        code, ack = self._run_with_final_acks(
            self._inspection_acks(track=99, slot=77), [("/ack", ["status", 2, 2, 0, 0])]
        )
        self.assertNotEqual(code, 0)
        ack.close.assert_called_once()

    def test_smoke_checks_note_values_not_only_note_count(self) -> None:
        notes = copy.deepcopy(smoke._build_notes()[0])
        notes[0]["pitch"] += 1
        code, _ack = self._run_with_final_acks(
            self._inspection_acks(notes=notes), [("/ack", ["status", 2, 2, 0, 0])]
        )
        self.assertNotEqual(code, 0)

    def test_smoke_requires_property_readbacks(self) -> None:
        for prop in ("tempo", "signature_numerator", "signature_denominator", "name", "length"):
            with self.subTest(property=prop):
                code, _ack = self._run_with_final_acks(
                    self._inspection_acks(), [("/ack", ["status", 2, 2, 0, 0])],
                    wrong_readback=prop,
                )
                self.assertNotEqual(code, 0)

    def test_smoke_requires_expected_final_track_counts(self) -> None:
        code, _ack = self._run_with_final_acks(
            self._inspection_acks(), [("/ack", ["status", 999, 0, 999, 0])]
        )
        self.assertNotEqual(code, 0)

    def test_smoke_uses_unique_request_ids_for_generic_commands(self) -> None:
        code, ack = self._run_with_final_acks(
            self._inspection_acks(), [("/ack", ["status", 2, 2, 0, 0])]
        )
        self.assertEqual(code, 0)
        requests = [
            smoke.bridge._command_ack_expectation(command)[1]
            for command in ack.sent_commands if command.address.startswith("/api/")
        ]
        self.assertTrue(all(requests))
        self.assertEqual(len(requests), len(set(requests)))

    def test_smoke_closes_ack_socket_on_transport_error(self) -> None:
        ack = mock.Mock()
        with (
            mock.patch.object(smoke.bridge, "open_ack_socket", return_value=ack),
            mock.patch.object(smoke.socket, "socket"),
            mock.patch.object(smoke, "_send_and_collect_acks", side_effect=OSError("fixture transport error")),
            mock.patch("sys.stdout"),
            mock.patch("sys.stderr"),
        ):
            code = smoke.run(TEST_AUTH_TOKEN)
        self.assertNotEqual(code, 0)
        ack.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
