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

    def test_bridge_config_accepts_legacy_kwargs_without_observer_fields(self) -> None:
        cfg = bridge.parse_args(_base_args())
        legacy_kwargs = dict(vars(cfg))
        for field_name in (
            "api_observes",
            "api_unobserves",
            "api_observers",
            "api_clear_observers",
        ):
            legacy_kwargs.pop(field_name, None)

        legacy_cfg = bridge.BridgeConfig(**legacy_kwargs)

        self.assertEqual(legacy_cfg.api_observes, ())
        self.assertEqual(legacy_cfg.api_unobserves, ())
        self.assertEqual(legacy_cfg.api_observers, ())
        self.assertEqual(legacy_cfg.api_clear_observers, ())

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

    def test_parse_and_build_observer_commands(self) -> None:
        options_json = '{"observer_id":"obs-tempo","mode":1,"emit_initial":false}'
        cfg = bridge.parse_args(
            _base_args()
            + [
                "--api-observe",
                "live_set",
                "tempo",
                options_json,
                "req-observe",
                "--api-observers",
                "req-list",
                "--api-unobserve",
                "obs-tempo",
                "req-unobserve",
                "--api-clear-observers",
                "req-clear",
            ]
        )

        self.assertEqual(
            cfg.api_observes,
            (("live_set", "tempo", options_json, "req-observe"),),
        )
        self.assertEqual(cfg.api_observers, ("req-list",))
        self.assertEqual(cfg.api_unobserves, (("obs-tempo", "req-unobserve"),))
        self.assertEqual(cfg.api_clear_observers, ("req-clear",))

        commands = bridge.build_commands(cfg)
        addresses = [cmd.address for cmd in commands]
        self.assertIn("/api_observe", addresses)
        self.assertIn("/api_observers", addresses)
        self.assertIn("/api_unobserve", addresses)
        self.assertIn("/api_clear_observers", addresses)

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

    def test_rpc_ack_summary_observer_event(self) -> None:
        payload = {
            "observer_id": "obs-tempo",
            "requested_path": "live_set",
            "current_path": "live_set",
            "property": "tempo",
            "event_count": 2,
            "timestamp_ms": 123456,
            "value": 121.5,
        }
        lines = bridge.summarize_ack("/ack", ["api_event", "obs-tempo", json.dumps(payload)])
        self.assertGreaterEqual(len(lines), 2)
        self.assertIn("api_event obs-tempo live_set tempo", lines[1])
        self.assertIn("event=2", lines[1])

    def test_parse_ack_event_rpc_payload_and_request_id(self) -> None:
        event = bridge.parse_ack_event(
            "/ack",
            ["api_get", "live_set", "tempo", "142", "req-tempo"],
        )

        self.assertEqual(event.address, "/ack")
        self.assertEqual(event.event, "api_get")
        self.assertEqual(event.request_id, "req-tempo")
        self.assertFalse(event.is_error)
        self.assertEqual(event.payload["path"], "live_set")
        self.assertEqual(event.payload["property"], "tempo")
        self.assertEqual(event.payload["value"], 142)

    def test_parse_ack_event_observer_payload(self) -> None:
        payload = {
            "observer_id": "obs-tempo",
            "requested_path": "live_set",
            "current_path": "live_set",
            "property": "tempo",
            "event_count": 2,
            "dropped_events": 1,
            "timestamp_ms": 123456,
            "value": 121.5,
        }

        event = bridge.parse_ack_event(
            "/ack",
            ["api_event", "obs-tempo", json.dumps(payload)],
        )

        self.assertEqual(event.event, "api_event")
        self.assertIsNone(event.request_id)
        self.assertEqual(event.payload["observer_id"], "obs-tempo")
        self.assertEqual(event.payload["path"], "live_set")
        self.assertEqual(event.payload["property"], "tempo")
        self.assertEqual(event.payload["value"], 121.5)
        self.assertEqual(event.payload["event_count"], 2)
        self.assertEqual(event.payload["dropped_events"], 1)

    def test_js_note_normalizer_supports_live_12_note_fields(self) -> None:
        source = pathlib.Path(__file__).with_name("m4l").joinpath("live_udp_bridge.js").read_text()

        self.assertIn("velocity >= 0 && velocity <= 127", source)
        self.assertIn('"probability"', source)
        self.assertIn('"velocity_deviation"', source)
        self.assertIn('"release_velocity"', source)

    def test_parse_and_build_status_wrapper_commands(self) -> None:
        cfg = bridge.parse_args(
            _base_args()
            + [
                "--api-session-context",
                "req-session",
                "--api-theory-status",
                "req-theory",
                "--api-tuning-status",
                "req-tuning",
            ]
        )

        commands = bridge.build_commands(cfg)
        command_map = {cmd.address: cmd.args for cmd in commands}

        self.assertEqual(command_map["/api/session_context"], ("req-session",))
        self.assertEqual(command_map["/api/theory_status"], ("req-theory",))
        self.assertEqual(command_map["/api/tuning_status"], ("req-tuning",))

    def test_parse_and_build_device_parameter_mixer_wrappers(self) -> None:
        parameter_path = "live_set tracks 0 devices 0 parameters 1"
        cfg = bridge.parse_args(
            _base_args()
            + [
                "--api-device-list",
                "all",
                "req-devices",
                "--api-device-parameters",
                "live_set tracks 0 devices 0",
                "req-params",
                "--api-parameter-set",
                parameter_path,
                "0.5",
                "req-set",
                "--api-mixer-status",
                "0",
                "req-mix",
            ]
        )

        commands = bridge.build_commands(cfg)
        by_address = {cmd.address: cmd.args for cmd in commands}

        self.assertEqual(by_address["/api/device_list"], ("all", "req-devices"))
        self.assertEqual(
            by_address["/api/device_parameters"],
            ("live_set tracks 0 devices 0", "req-params"),
        )
        self.assertEqual(
            by_address["/api/parameter_set"],
            (parameter_path, "0.5", "req-set"),
        )
        self.assertEqual(by_address["/api/mixer_status"], ("0", "req-mix"))

    def test_parse_and_build_insertion_wrappers(self) -> None:
        cfg = bridge.parse_args(
            _base_args()
            + [
                "--api-insert-device",
                "live_set tracks 0",
                "Operator",
                "",
                "req-device",
                "--api-insert-chain",
                "live_set tracks 0 devices 0",
                "",
                "req-chain",
                "--api-drum-chain-in-note",
                "live_set tracks 0 devices 0 chains 0",
                "36",
                "req-note",
            ]
        )

        commands = bridge.build_commands(cfg)
        by_address = {cmd.address: cmd.args for cmd in commands}

        self.assertEqual(
            by_address["/api/insert_device"],
            ("live_set tracks 0", "Operator", "", "req-device"),
        )
        self.assertEqual(
            by_address["/api/insert_chain"],
            ("live_set tracks 0 devices 0", "", "req-chain"),
        )
        self.assertEqual(
            by_address["/api/drum_chain_in_note"],
            ("live_set tracks 0 devices 0 chains 0", 36, "req-note"),
        )

    def test_rpc_ack_summary_api_event_non_dict_payload_does_not_crash(self) -> None:
        lines = bridge.summarize_ack("/ack", ["api_event", "obs-raw", "not-json"])

        self.assertGreaterEqual(len(lines), 2)
        self.assertIn("api_event obs-raw ? ?", lines[1])

    def test_listen_mode_allows_no_send_commands(self) -> None:
        cfg = bridge.parse_args(["--listen", "--listen-timeout", "0.01", "--no-tempo", "--no-signature"])

        self.assertTrue(cfg.listen)
        self.assertTrue(cfg.expect_ack)
        self.assertEqual(bridge.build_commands(cfg), [])

    def test_listen_for_events_stops_on_max_events(self) -> None:
        payload = {"observer_id": "obs-tempo", "property": "tempo", "value": 120}
        packet = bridge.encode_osc_message("/ack", ("api_event", "obs-tempo", json.dumps(payload)))

        class _FakeSock:
            def __init__(self) -> None:
                self._packets = [packet]
                self.closed = False

            def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
                if self._packets:
                    return self._packets.pop(0), ("127.0.0.1", 9001)
                raise BlockingIOError

            def close(self) -> None:
                self.closed = True

        fake_sock = _FakeSock()
        cfg = bridge.parse_args(
            [
                "--listen",
                "--listen-timeout",
                "1",
                "--listen-max-events",
                "1",
                "--no-tempo",
                "--no-signature",
            ]
        )

        with (
            mock.patch("ableton_udp_bridge.open_ack_socket", return_value=fake_sock),
            mock.patch("ableton_udp_bridge.select.select", return_value=([fake_sock], [], [])),
        ):
            count = bridge.listen_for_events(cfg)

        self.assertEqual(count, 1)
        self.assertTrue(fake_sock.closed)

    def test_main_returns_error_when_listen_socket_cannot_bind(self) -> None:
        with mock.patch("ableton_udp_bridge.open_ack_socket", return_value=None):
            exit_code = bridge.main(
                [
                    "--listen",
                    "--listen-timeout",
                    "0.01",
                    "--no-tempo",
                    "--no-signature",
                ]
            )

        self.assertEqual(exit_code, 1)

    def test_js_and_patch_support_api_wrapper_fallback_route(self) -> None:
        m4l_dir = pathlib.Path(__file__).with_name("m4l")
        js_source = m4l_dir.joinpath("live_udp_bridge.js").read_text()
        patch_source = json.loads(m4l_dir.joinpath("LiveUdpBridge.maxpat").read_text())
        boxes = [item["box"] for item in patch_source["patcher"]["boxes"]]
        route_box = next(box for box in boxes if str(box.get("text", "")).startswith("route "))
        js_box = next(box for box in boxes if box.get("text") == "js live_udp_bridge.js")
        route_tokens = str(route_box["text"]).split()[1:]
        fallback_outlet = len(route_tokens)
        patchlines = [item["patchline"] for item in patch_source["patcher"]["lines"]]

        self.assertIn("function api_session_context", js_source)
        self.assertIn("function api_insert_device", js_source)
        self.assertIn("function anything", js_source)
        self.assertNotIn("/api/session_context", route_tokens)
        self.assertNotIn("/api/insert_device", route_tokens)
        self.assertTrue(
            any(
                line.get("source") == [route_box["id"], fallback_outlet]
                and line.get("destination", [None])[0] == js_box["id"]
                for line in patchlines
            )
        )

    def test_js_wrappers_reject_ambiguous_write_values(self) -> None:
        js_source = pathlib.Path(__file__).with_name("m4l").joinpath("live_udp_bridge.js").read_text()

        self.assertIn("function parseOptionalInsertionIndex", js_source)
        self.assertIn("function isNumericScalar", js_source)
        self.assertIn("api_parameter_set_missing_value", js_source)
        self.assertIn("api_parameter_set_invalid_value_type", js_source)
        self.assertIn("api_insert_device_invalid_index", js_source)
        self.assertIn("api_insert_chain_invalid_index", js_source)

    def test_js_fallback_route_uses_explicit_wrapper_allowlist(self) -> None:
        js_source = pathlib.Path(__file__).with_name("m4l").joinpath("live_udp_bridge.js").read_text()

        self.assertIn("var API_FALLBACK_HANDLERS = {", js_source)
        self.assertIn('"api_session_context": api_session_context', js_source)
        self.assertIn('"api_insert_device": api_insert_device', js_source)
        self.assertNotIn("var target = this[targetName];", js_source)

    def test_js_drum_chain_note_rejects_fractional_values(self) -> None:
        js_source = pathlib.Path(__file__).with_name("m4l").joinpath("live_udp_bridge.js").read_text()
        function_source = js_source.split("function api_drum_chain_in_note", 1)[1].split(
            "function api_observe", 1
        )[0]

        self.assertIn("Math.floor(note) !== note", function_source)
        self.assertNotIn("Math.floor(Number(noteValue))", function_source)

    def test_js_cleans_max_dict_payloads(self) -> None:
        js_source = pathlib.Path(__file__).with_name("m4l").joinpath("live_udp_bridge.js").read_text()

        self.assertIn("function clearBuiltPayload", js_source)
        self.assertGreaterEqual(js_source.count("clearBuiltPayload("), 5)

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
