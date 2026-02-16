#!/usr/bin/env python3
"""Unit tests for the Ableton Live UDP bridge CLI helpers."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import ableton_udp_bridge as bridge


def _base_args() -> list[str]:
    # Disable all default mutations so tests focus on the API surface.
    return [
        "--ack",
        "--no-tempo",
        "--no-signature",
        "--create-midi-tracks",
        "0",
        "--create-audio-tracks",
        "0",
        "--add-midi-tracks",
        "0",
        "--add-audio-tracks",
        "0",
    ]


class BridgeCliTests(unittest.TestCase):
    def test_parse_defaults_do_not_create_tracks(self) -> None:
        cfg = bridge.parse_args([])
        self.assertEqual(cfg.create_midi_tracks, 0)

    def test_parse_and_build_api_commands(self) -> None:
        cfg = bridge.parse_args(
            _base_args()
            + [
                "--api-get",
                "live_set",
                "tempo",
                "req-1",
                "--api-call",
                "live_set",
                "create_midi_track",
                "[-1]",
            ]
        )

        self.assertEqual(cfg.api_gets, (("live_set", "tempo", "req-1"),))
        self.assertEqual(cfg.api_calls, (("live_set", "create_midi_track", "[-1]", None),))

        commands = bridge.build_commands(cfg)
        addresses = [cmd.address for cmd in commands]

        # /ping should still come first when --ack is enabled.
        self.assertEqual(addresses[0], "/ping")
        # API commands should be present and come before legacy mutations.
        self.assertIn("/api/get", addresses)
        self.assertIn("/api/call", addresses)
        api_indices = [addresses.index("/api/get"), addresses.index("/api/call")]
        legacy_indices = [i for i, addr in enumerate(addresses) if addr.startswith("/tempo")]
        if legacy_indices:
            self.assertLess(max(api_indices), min(legacy_indices))

    def test_rpc_ack_summary_children(self) -> None:
        children = [
            {"index": 0, "id": 1, "path": "live_set tracks 0", "name": "Track 1"},
            {"index": 1, "id": 2, "path": "live_set tracks 1", "name": "Track 2"},
        ]
        args = [
            "api_children",
            "live_set",
            "tracks",
            json.dumps(children),
            "req-2",
        ]
        lines = bridge.summarize_ack("/ack", args)
        self.assertGreaterEqual(len(lines), 2)
        self.assertIn("api_children live_set tracks count=2", lines[1])
        self.assertIn("req=req-2", lines[1])

    def test_parse_and_build_midi_cc_commands(self) -> None:
        cfg = bridge.parse_args(
            _base_args()
            + [
                "--midi-cc",
                "64",
                "127",
                "2",
                "--cc64",
                "0",
            ]
        )

        self.assertEqual(cfg.midi_ccs, ((64, 127, 2),))
        self.assertEqual(cfg.cc64s, ((0, 1),))

        commands = bridge.build_commands(cfg)
        midi_cmds = [cmd for cmd in commands if cmd.address in {"/midi_cc", "/cc64"}]
        self.assertEqual(len(midi_cmds), 2)
        self.assertEqual(midi_cmds[0].address, "/midi_cc")
        self.assertEqual(midi_cmds[0].args, (64, 127, 2))
        self.assertEqual(midi_cmds[1].address, "/cc64")
        self.assertEqual(midi_cmds[1].args, (0, 1))

    def test_ack_summary_midi_cc(self) -> None:
        lines = bridge.summarize_ack("/ack", ["midi_cc", 64, 96, 1, "req-cc"])
        self.assertGreaterEqual(len(lines), 2)
        self.assertIn("midi_cc ctrl=64 value=96 ch=1", lines[1])
        self.assertIn("req=req-cc", lines[1])

    def test_parse_ack_mode_and_metrics_options(self) -> None:
        cfg = bridge.parse_args(
            _base_args()
            + [
                "--ack-mode",
                "flush_interval",
                "--ack-flush-interval",
                "3",
                "--no-metrics",
            ]
        )
        self.assertEqual(cfg.ack_mode, "flush_interval")
        self.assertEqual(cfg.ack_flush_interval, 3)
        self.assertFalse(cfg.report_metrics)

    def test_send_commands_dry_run_returns_metrics(self) -> None:
        cfg = bridge.parse_args(
            _base_args()
            + [
                "--dry-run",
                "--status",
            ]
        )
        commands = bridge.build_commands(cfg)
        metrics = bridge.send_commands(cfg, commands)
        self.assertEqual(metrics.command_count, len(commands))
        self.assertEqual(metrics.elapsed_ms, 0.0)

    def test_wait_for_acks_uses_quiet_window_after_first_packet(self) -> None:
        packet = bridge.encode_osc_message("/ack", ("status",))
        timeouts: list[float] = []

        class _FakeSock:
            def __init__(self) -> None:
                self._packets = [packet]

            def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
                if self._packets:
                    return self._packets.pop(0), ("127.0.0.1", 9001)
                raise BlockingIOError

        fake_sock = _FakeSock()

        def _fake_select(_r: object, _w: object, _e: object, timeout: float) -> tuple[list[object], list[object], list[object]]:
            timeouts.append(float(timeout))
            if len(timeouts) == 1:
                return [fake_sock], [], []
            return [], [], []

        clock = iter([0.0, 0.01, 0.02, 0.03, 0.04])
        with (
            mock.patch("ableton_udp_bridge.time.monotonic", side_effect=lambda: next(clock)),
            mock.patch("ableton_udp_bridge.select.select", side_effect=_fake_select),
        ):
            acks = bridge.wait_for_acks(fake_sock, timeout_s=1.0, quiet_window_s=0.05)

        self.assertEqual(len(acks), 1)
        self.assertGreater(timeouts[0], 0.90)
        self.assertLessEqual(timeouts[1], 0.05 + 1e-6)

    def test_send_commands_drains_stale_acks_before_each_send(self) -> None:
        cfg = bridge.parse_args(_base_args() + ["--status"])
        commands = bridge.build_commands(cfg)
        order: list[str] = []

        class _FakeAckSock:
            def close(self) -> None:
                return None

        class _FakeSendSock:
            def sendto(self, _payload: bytes, _target: tuple[str, int]) -> None:
                return None

            def __enter__(self) -> "_FakeSendSock":
                return self

            def __exit__(self, *_exc: object) -> bool:
                return False

        def _fake_collect(
            _ack_sock: object,
            _timeout: float,
            durations_ms: list[float],
            ack_counts: list[int],
        ) -> None:
            order.append("collect")
            durations_ms.append(0.0)
            ack_counts.append(1)

        with (
            mock.patch("ableton_udp_bridge.open_ack_socket", return_value=_FakeAckSock()),
            mock.patch("ableton_udp_bridge.socket.socket", return_value=_FakeSendSock()),
            mock.patch(
                "ableton_udp_bridge._drain_acks_nonblocking",
                side_effect=lambda _sock: order.append("drain"),
            ),
            mock.patch("ableton_udp_bridge._collect_and_print_acks", side_effect=_fake_collect),
        ):
            bridge.send_commands(cfg, commands)

        self.assertGreater(len(order), 1)
        self.assertEqual(order[0], "drain")
        self.assertEqual(order[1], "collect")


if __name__ == "__main__":
    unittest.main()
