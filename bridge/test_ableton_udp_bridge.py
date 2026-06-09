#!/usr/bin/env python3
"""Unit tests for the Ableton Live UDP bridge CLI helpers."""

from __future__ import annotations

import hashlib
import io
import json
import math
import pathlib
import subprocess
import sys
import unittest
from unittest import mock

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import ableton_udp_bridge as bridge


def _run_bridge_js(body: str) -> object:
    source_path = pathlib.Path(__file__).with_name("m4l").joinpath("live_udp_bridge.js")
    harness = f"""
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync({json.dumps(str(source_path))}, "utf8");
const context = {{
  post: () => {{}},
  outlet: () => {{}},
  ack: () => {{}},
  Dict: function Dict() {{}},
  LiveAPI: function LiveAPI() {{}},
}};
vm.createContext(context);
vm.runInContext(source, context);
const result = (() => {{
{body}
}})();
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


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


def _run_session_clip_inspect_js(
    *,
    notes: list[dict[str, object]],
    devices: list[dict[str, object]],
    request_id: str = "req-inspect",
    clip_ids: list[int] | None = None,
    track_name: str | None = "Inspection Track",
    clip_name: str | None = "Inspection Clip",
    clip_values: dict[str, object] | None = None,
    device_count: int | None = None,
    max_fragments: int | None = None,
) -> list[list[object]]:
    fixture = {
        "notes": notes,
        "devices": devices,
        "clip_ids": clip_ids or [303, 303],
        "track_name": track_name,
        "clip_name": clip_name,
        "clip_values": clip_values
        or {
            "start_marker": 0,
            "end_marker": 8,
            "length": 8,
            "looping": 1,
            "loop_start": 0,
            "loop_end": 8,
        },
        "device_count": device_count,
        "max_fragments": max_fragments,
    }
    return _run_bridge_js(
        f"""
const fixture = {json.dumps(fixture)};
const outputs = [];
let clipOpenCount = 0;
context.outlet = (...args) => outputs.push(args);
context.ensureInitialized = () => true;
if (fixture.max_fragments !== null) {{
  context.SESSION_CLIP_INSPECTION_MAX_FRAGMENTS = fixture.max_fragments;
}}
context.LiveAPI = function LiveAPI(_callback, rawPath) {{
  const path = String(rawPath);
  if (path === "live_set tracks 2") {{
    return {{
      id: 101,
      path,
      get: (property) => {{
        if (property === "has_midi_input") return 1;
        if (property === "name") return fixture.track_name;
        throw new Error("unknown track property " + property);
      }},
      getcount: (child) => {{
        if (child === "clip_slots") return 8;
        if (child === "devices") {{
          return fixture.device_count === null
            ? fixture.devices.length
            : fixture.device_count;
        }}
        throw new Error("unknown track child " + child);
      }},
    }};
  }}
  if (path === "live_set tracks 2 clip_slots 3") {{
    return {{
      id: 202,
      path,
      get: (property) => {{
        if (property === "has_clip") return 1;
        throw new Error("unknown slot property " + property);
      }},
    }};
  }}
  if (path === "live_set tracks 2 clip_slots 3 clip") {{
    const id = fixture.clip_ids[Math.min(clipOpenCount, fixture.clip_ids.length - 1)];
    clipOpenCount += 1;
    return {{
      id,
      path,
      get: (property) => {{
        const values = Object.assign(
          {{ name: fixture.clip_name }},
          fixture.clip_values
        );
        if (Object.prototype.hasOwnProperty.call(values, property)) return values[property];
        throw new Error("unknown clip property " + property);
      }},
      call: (method) => {{
        if (method === "get_all_notes_extended") {{
          return JSON.stringify({{ notes: fixture.notes }});
        }}
        throw new Error("unknown clip method " + method);
      }},
    }};
  }}
  const match = path.match(/^live_set tracks 2 devices (\\d+)$/);
  if (match) {{
    const index = Number(match[1]);
    const item = fixture.devices[index];
    return {{
      id: 400 + index,
      path,
      get: (property) => {{
        if (Object.prototype.hasOwnProperty.call(item, property)) return item[property];
        throw new Error("unavailable device property " + property);
      }},
    }};
  }}
  throw new Error("unknown path " + path);
}};
context.api_session_clip_inspect(2, 3, 1, {json.dumps(request_id)});
return outputs.filter((args) => args[1] === "/ack");
"""
    )


def _inspection_fragment(
    *,
    index: int,
    count: int,
    kind: str,
    data: dict[str, object],
    request_id: str = "req-assembly",
    inspection_id: str = "inspection-1",
) -> dict[str, object]:
    return {
        "schema": "codex-live-bridge.session-midi-clip-inspection",
        "schema_version": 1,
        "producer_version": "3.1.0",
        "inspection_id": inspection_id,
        "correlation": {
            "request_id": request_id,
            "track_index": 2,
            "slot_index": 3,
        },
        "snapshot": {
            "started_ms": 100,
            "completed_ms": 110,
            "atomic": False,
            "consistent": True,
        },
        "transfer": {
            "fragment_index": index,
            "fragment_count": count,
            "fragment_kind": kind,
            "is_last": index == count - 1,
            "packet_budget_bytes": 4096,
        },
        "completeness": {
            "track": "complete",
            "clip": "complete",
            "devices": "complete",
            "notes": "complete",
            "missing_fields": [],
        },
        "data": data,
    }


def _inspection_context_data(
    *,
    note_count: int = 0,
    pitch_min: int | None = None,
    pitch_max: int | None = None,
    track_name: str | None = "Inspection Track",
    clip_name: str | None = "Inspection Clip",
) -> dict[str, object]:
    return {
        "context": "session",
        "track": {
            "index": 2,
            "path": "live_set tracks 2",
            "id": 101,
            "name": track_name,
        },
        "clip": {
            "slot_index": 3,
            "path": "live_set tracks 2 clip_slots 3 clip",
            "id": 303,
            "name": clip_name,
            "start_marker": 0,
            "end_marker": 8,
            "live_length": 8,
            "looping": True,
            "loop_start": 0,
            "loop_end": 8,
        },
        "summary": {
            "note_count": note_count,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
        },
    }


def _inspection_device(
    index: int = 0,
    *,
    device_type: int | None = None,
) -> dict[str, object]:
    return {
        "index": index,
        "path": f"live_set tracks 2 devices {index}",
        "id": 400 + index,
        "name": f"Device {index}",
        "class_name": None,
        "type": device_type,
    }


def _inspection_note(
    *,
    note_id: int = 1,
    pitch: int = 60,
    start_time: float = 0.0,
    duration: float = 0.5,
    velocity: float = 100,
    mute: bool | int | float = False,
    probability: float = 1.0,
    velocity_deviation: float = 0.0,
    release_velocity: float = 64,
) -> dict[str, object]:
    return {
        "note_id": note_id,
        "pitch": pitch,
        "start_time": start_time,
        "duration": duration,
        "velocity": velocity,
        "mute": mute,
        "probability": probability,
        "velocity_deviation": velocity_deviation,
        "release_velocity": release_velocity,
    }


def _complete_inspection_fragment(
    *,
    devices: list[object] | None = None,
    notes: list[object] | None = None,
) -> dict[str, object]:
    device_items = [] if devices is None else devices
    note_items = [] if notes is None else notes
    pitches = [
        int(note["pitch"])
        for note in note_items
        if isinstance(note, dict)
        and isinstance(note.get("pitch"), int)
        and not isinstance(note.get("pitch"), bool)
    ]
    data = _inspection_context_data(
        note_count=len(note_items),
        pitch_min=min(pitches) if pitches else None,
        pitch_max=max(pitches) if pitches else None,
    )
    data.update(
        {
            "device_offset": 0,
            "device_count": len(device_items),
            "device_total": len(device_items),
            "devices": device_items,
            "note_offset": 0,
            "note_count": len(note_items),
            "note_total": len(note_items),
            "notes": note_items,
        }
    )
    return _inspection_fragment(index=0, count=1, kind="complete", data=data)


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

    def test_parse_and_build_session_clip_inspect_commands(self) -> None:
        cfg = bridge.parse_args(
            _base_args()
            + [
                "--api-session-clip-inspect",
                "2",
                "3",
                "req-a",
                "--api-session-clip-inspect",
                "4",
                "5",
                "req-b",
            ]
        )

        self.assertEqual(
            cfg.api_session_clip_inspects,
            ((2, 3, "req-a"), (4, 5, "req-b")),
        )
        commands = [
            command
            for command in bridge.build_commands(cfg)
            if command.address == "/api/session_clip_inspect"
        ]
        self.assertEqual(
            commands,
            [
                bridge.OscCommand("/api/session_clip_inspect", (2, 3, 1, "req-a")),
                bridge.OscCommand("/api/session_clip_inspect", (4, 5, 1, "req-b")),
            ],
        )

    def test_session_clip_inspect_cli_rejects_invalid_inputs(self) -> None:
        invalid_argv = [
            ["--api-session-clip-inspect", "-1", "0", "req"],
            ["--api-session-clip-inspect", "0", "-1", "req"],
            ["--api-session-clip-inspect", "0", "0", ""],
            ["--api-session-clip-inspect", "0", "0", "é" * 65],
        ]

        for argv in invalid_argv:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                bridge.parse_args(_base_args() + argv)

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

    def test_parse_and_summarize_session_clip_inspect_fragment(self) -> None:
        fragment = _inspection_fragment(
            index=0,
            count=1,
            kind="complete",
            request_id="req-fragment",
            data={
                "context": "session",
                "track": {"index": 2},
                "clip": {"slot_index": 3},
                "summary": {"note_count": 0, "pitch_min": None, "pitch_max": None},
                "device_offset": 0,
                "device_count": 0,
                "device_total": 0,
                "devices": [],
                "note_offset": 0,
                "note_count": 0,
                "note_total": 0,
                "notes": [],
            },
        )
        args = ["api_session_clip_inspect", json.dumps(fragment), "req-fragment"]

        event = bridge.parse_ack_event("/ack", args)
        lines = bridge.summarize_ack("/ack", args)

        self.assertEqual(event.request_id, "req-fragment")
        self.assertEqual(event.payload["fragment"], fragment)
        self.assertEqual(len(lines), 1)
        self.assertIn("api_session_clip_inspect complete fragment=1/1", lines[0])
        self.assertIn("notes=0", lines[0])
        self.assertNotIn(json.dumps(fragment), lines[0])

    def test_error_ack_terminal_slot_preserves_details_and_optional_request_id(self) -> None:
        without_request = bridge.parse_ack_event(
            "/ack",
            ["error", "api_example_failed", "path detail", 40, "request_correlation", "req:"],
        )
        with_request = bridge.parse_ack_event(
            "/ack",
            [
                "error",
                "api_example_failed",
                "path detail",
                40,
                "request_correlation",
                "req:req-error",
            ],
        )
        legacy_without_slot = bridge.parse_ack_event(
            "/ack",
            ["error", "api_example_failed", "path detail", 40],
        )
        legacy_req_prefixed_detail = bridge.parse_ack_event(
            "/ack",
            ["error", "api_example_failed", "path detail", "req:Lead"],
        )

        self.assertIsNone(without_request.request_id)
        self.assertEqual(without_request.payload["details"], ["path detail", 40])
        self.assertEqual(with_request.request_id, "req-error")
        self.assertEqual(with_request.payload["details"], ["path detail", 40])
        self.assertIsNone(legacy_without_slot.request_id)
        self.assertEqual(legacy_without_slot.payload["details"], ["path detail", 40])
        self.assertIsNone(legacy_req_prefixed_detail.request_id)
        self.assertEqual(legacy_req_prefixed_detail.payload["details"], ["path detail", "req:Lead"])

        without_summary = bridge.summarize_ack(
            "/ack",
            ["error", "api_example_failed", "path detail", 40, "request_correlation", "req:"],
        )
        with_summary = bridge.summarize_ack(
            "/ack",
            [
                "error",
                "api_example_failed",
                "path detail",
                40,
                "request_correlation",
                "req:req-error",
            ],
        )
        self.assertNotIn("req=", without_summary[1])
        self.assertIn("req=req-error", with_summary[1])

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
        self.assertNotIn("/api/session_clip_inspect", route_tokens)
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
        self.assertIn('"api_session_clip_inspect": api_session_clip_inspect', js_source)
        self.assertNotIn("var target = this[targetName];", js_source)

    def test_js_session_clip_inspect_validation_errors_are_correlated(self) -> None:
        result = _run_bridge_js(
            """
const outputs = [];
context.outlet = (...args) => outputs.push(args);
context.ensureInitialized = () => true;
context.api_session_clip_inspect(0, 0, 1, "");
context.api_session_clip_inspect(-1, 0, 1, "req-track");
context.api_session_clip_inspect(0, 1.5, 1, "req-slot");
context.api_session_clip_inspect(0, 0, 2, "req-schema");
context.api_session_clip_inspect(0, 0, 1, "é".repeat(65));
context.api_session_clip_inspect("0", 0, 1, "req-track-string");
context.api_session_clip_inspect(0, 0, "1", "req-schema-string");
return outputs.filter((args) => args[1] === "/ack");
"""
        )

        self.assertEqual(
            [args[3] for args in result],
            ["api_session_clip_inspect_validation_failed"] * 7,
        )
        self.assertEqual(result[0][-2:], ["request_correlation", "req:"])
        self.assertEqual(result[1][-1], "req:req-track")
        self.assertEqual(result[2][-1], "req:req-slot")
        self.assertEqual(result[3][-1], "req:req-schema")

    def test_js_session_clip_inspect_validation_errors_are_packet_bounded(self) -> None:
        long_ascii = "x" * 5000
        long_multibyte = "𝄞" * 1500
        result = _run_bridge_js(
            f"""
const outputs = [];
context.outlet = (...args) => outputs.push(args);
const longAscii = {json.dumps(long_ascii)};
const longMultibyte = {json.dumps(long_multibyte)};
[
  [0, 0, 1, longAscii],
  [0, 0, 1, longMultibyte],
  [longAscii, 0, 1, "req-track-ascii"],
  [longMultibyte, 0, 1, "req-track-multibyte"],
  [0, longAscii, 1, "req-slot-ascii"],
  [0, longMultibyte, 1, "req-slot-multibyte"],
  [0, 0, longAscii, "req-schema-ascii"],
  [0, 0, longMultibyte, "req-schema-multibyte"],
].forEach((args) => {{
  context.api_session_clip_inspect(...args);
}});
return outputs.filter((args) => args[1] === "/ack");
"""
        )

        self.assertEqual(len(result), 8)
        for args in result:
            packet = bridge.encode_osc_message("/ack", tuple(args[2:]))
            self.assertLessEqual(len(packet), 4096)
            self.assertEqual(args[3], "api_session_clip_inspect_validation_failed")
            self.assertNotIn(long_ascii, args)
            self.assertNotIn(long_multibyte, args)
            self.assertEqual(args[-2], "request_correlation")
            self.assertTrue(str(args[-1]).startswith("req:"))
            self.assertLessEqual(len(str(args[-1])[4:].encode("utf-8")), 128)

    def test_js_session_clip_inspect_empty_clip_is_one_complete_fragment(self) -> None:
        outputs = _run_session_clip_inspect_js(notes=[], devices=[])

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0][2], "api_session_clip_inspect")
        self.assertEqual(outputs[0][4], "req-inspect")
        fragment = json.loads(str(outputs[0][3]))
        self.assertEqual(fragment["transfer"]["fragment_kind"], "complete")
        self.assertEqual(fragment["transfer"]["fragment_count"], 1)
        self.assertTrue(fragment["transfer"]["is_last"])
        self.assertEqual(fragment["data"]["summary"]["note_count"], 0)
        self.assertIsNone(fragment["data"]["summary"]["pitch_min"])
        self.assertIsNone(fragment["data"]["summary"]["pitch_max"])
        self.assertEqual(fragment["data"]["devices"], [])
        self.assertEqual(fragment["data"]["notes"], [])

    def test_js_session_clip_inspect_preserves_metadata_devices_and_all_note_fields(self) -> None:
        note = {
            "note_id": 91,
            "pitch": 64,
            "start_time": 1.25,
            "duration": 0.5,
            "velocity": 102,
            "mute": 0,
            "probability": 0.75,
            "velocity_deviation": -12.5,
            "release_velocity": 88,
            "ignored": "not part of the protocol",
        }
        outputs = _run_session_clip_inspect_js(
            notes=[note],
            devices=[
                {
                    "name": "Operator",
                    "class_name": "Operator",
                    "type": 1,
                },
                {"name": "Optional Fields Device"},
            ],
        )

        fragment = json.loads(str(outputs[0][3]))
        data = fragment["data"]
        self.assertEqual(fragment["schema"], "codex-live-bridge.session-midi-clip-inspection")
        self.assertEqual(fragment["producer_version"], "3.1.0")
        self.assertEqual(data["context"], "session")
        self.assertEqual(
            data["track"],
            {
                "index": 2,
                "path": "live_set tracks 2",
                "id": 101,
                "name": "Inspection Track",
            },
        )
        self.assertEqual(
            data["clip"],
            {
                "slot_index": 3,
                "path": "live_set tracks 2 clip_slots 3 clip",
                "id": 303,
                "name": "Inspection Clip",
                "start_marker": 0,
                "end_marker": 8,
                "live_length": 8,
                "looping": True,
                "loop_start": 0,
                "loop_end": 8,
            },
        )
        self.assertEqual(data["summary"], {"note_count": 1, "pitch_min": 64, "pitch_max": 64})
        self.assertEqual(
            data["devices"][0],
            {
                "index": 0,
                "path": "live_set tracks 2 devices 0",
                "id": 400,
                "name": "Operator",
                "class_name": "Operator",
                "type": 1,
            },
        )
        self.assertEqual(
            data["devices"][1],
            {
                "index": 1,
                "path": "live_set tracks 2 devices 1",
                "id": 401,
                "name": "Optional Fields Device",
                "class_name": None,
                "type": None,
            },
        )
        self.assertEqual(data["notes"], [{key: note[key] for key in note if key != "ignored"}])

    def test_js_session_clip_inspect_preserves_device_type_enum_or_null(self) -> None:
        raw_types: list[object] = [0, 1, 2, 4, None, 3, True, "1", 1.5]
        outputs = _run_session_clip_inspect_js(
            notes=[],
            devices=[
                {} if raw_type is None else {"type": raw_type}
                for raw_type in raw_types
            ],
        )
        fragment = json.loads(str(outputs[0][3]))

        self.assertEqual(
            [device["type"] for device in fragment["data"]["devices"]],
            [0, 1, 2, 4, None, None, None, None, None],
        )

    def test_js_session_clip_inspect_negative_times_roundtrip_through_python(self) -> None:
        note = _inspection_note(start_time=-3.5)
        outputs = _run_session_clip_inspect_js(
            notes=[note],
            devices=[],
            request_id="req-negative-times",
            clip_values={
                "start_marker": -4,
                "end_marker": 4,
                "length": 8,
                "looping": 1,
                "loop_start": -2,
                "loop_end": 2,
            },
        )
        assembler = bridge.SessionClipInspectionAssembler()
        assembled = None
        for output in outputs:
            assembled = assembler.add_fragment(
                json.loads(str(output[3])),
                str(output[4]),
            )

        self.assertIsNotNone(assembled)
        assert assembled is not None
        self.assertEqual(assembled["clip"]["start_marker"], -4)
        self.assertEqual(assembled["clip"]["end_marker"], 4)
        self.assertEqual(assembled["clip"]["loop_start"], -2)
        self.assertEqual(assembled["clip"]["loop_end"], 2)
        self.assertEqual(assembled["notes"], [note])

    def test_js_session_clip_inspect_rejects_invalid_clip_ranges(self) -> None:
        cases = [
            {
                "start_marker": 2,
                "end_marker": 1,
                "length": 1,
                "looping": 1,
                "loop_start": 0,
                "loop_end": 1,
            },
            {
                "start_marker": 0,
                "end_marker": math.inf,
                "length": 1,
                "looping": 1,
                "loop_start": 0,
                "loop_end": 1,
            },
            {
                "start_marker": 0,
                "end_marker": 1,
                "length": -1,
                "looping": 1,
                "loop_start": 2,
                "loop_end": 1,
            },
        ]

        for clip_values in cases:
            with self.subTest(clip_values=clip_values):
                outputs = _run_session_clip_inspect_js(
                    notes=[],
                    devices=[],
                    clip_values=clip_values,
                )
                self.assertEqual(len(outputs), 1)
                self.assertEqual(
                    outputs[0][3],
                    "api_session_clip_inspect_parse_failed",
                )

    def test_js_session_clip_inspect_accepts_v1_inventory_boundaries(self) -> None:
        notes = [
            _inspection_note(
                note_id=index,
                pitch=36 + (index % 48),
                start_time=index * 0.25,
            )
            for index in range(4096)
        ]
        devices = [{} for _ in range(256)]

        outputs = _run_session_clip_inspect_js(
            notes=notes,
            devices=devices,
            request_id="req-max-inventory",
        )
        fragments = [json.loads(str(output[3])) for output in outputs]

        self.assertGreater(len(fragments), 1)
        self.assertLessEqual(len(fragments), 1024)
        self.assertTrue(all(output[2] == "api_session_clip_inspect" for output in outputs))
        self.assertEqual(
            sum(
                fragment["data"].get("device_count", 0)
                for fragment in fragments
            ),
            256,
        )
        self.assertEqual(
            sum(
                fragment["data"].get("note_count", 0)
                for fragment in fragments
            ),
            4096,
        )

    def test_js_session_clip_inspect_rejects_over_limit_inventories(self) -> None:
        cases = [
            (
                "devices",
                _run_session_clip_inspect_js(
                    notes=[],
                    devices=[],
                    device_count=257,
                    request_id="req-too-many-devices",
                ),
            ),
            (
                "notes",
                _run_session_clip_inspect_js(
                    notes=[
                        _inspection_note(note_id=index)
                        for index in range(4097)
                    ],
                    devices=[],
                    request_id="req-too-many-notes",
                ),
            ),
        ]

        for inventory_kind, outputs in cases:
            with self.subTest(inventory_kind=inventory_kind):
                self.assertEqual(len(outputs), 1)
                self.assertEqual(
                    outputs[0][3],
                    "api_session_clip_inspect_limit_exceeded",
                )
                packet = bridge.encode_osc_message(
                    "/ack",
                    tuple(outputs[0][2:]),
                )
                self.assertLessEqual(len(packet), 4096)

    def test_js_session_clip_inspect_enforces_fragment_limit_during_planning(self) -> None:
        outputs = _run_session_clip_inspect_js(
            notes=[],
            devices=[
                {"name": f"Device {index} " + ("X" * 1400)}
                for index in range(3)
            ],
            request_id="req-fragment-limit",
            max_fragments=2,
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(
            outputs[0][3],
            "api_session_clip_inspect_limit_exceeded",
        )
        packet = bridge.encode_osc_message("/ack", tuple(outputs[0][2:]))
        self.assertLessEqual(len(packet), 4096)

    def test_js_session_clip_inspect_rejects_incomplete_or_invalid_extended_notes(self) -> None:
        canonical = _inspection_note()
        cases = [
            {key: value for key, value in canonical.items() if key != missing}
            for missing in canonical
        ] + [
            {**canonical, "note_id": -1},
            {**canonical, "pitch": True},
            {**canonical, "duration": -0.01},
            {**canonical, "velocity": 128},
            {**canonical, "mute": 2},
            {**canonical, "probability": 1.01},
            {**canonical, "velocity_deviation": -128},
            {**canonical, "release_velocity": 128},
        ]

        for note in cases:
            with self.subTest(note=note):
                outputs = _run_session_clip_inspect_js(notes=[note], devices=[])
                self.assertEqual(len(outputs), 1)
                self.assertEqual(outputs[0][2], "error")
                self.assertEqual(
                    outputs[0][3],
                    "api_session_clip_inspect_parse_failed",
                )

    def test_js_session_clip_inspect_complete_fragment_roundtrips_through_python(self) -> None:
        note = _inspection_note(
            velocity=100.5,
            duration=0,
            mute=True,
            release_velocity=64.5,
        )
        outputs = _run_session_clip_inspect_js(
            notes=[note],
            devices=[{}],
            track_name=None,
            clip_name=None,
            request_id="req-roundtrip-complete",
        )
        assembler = bridge.SessionClipInspectionAssembler()
        assembled = None
        for output in outputs:
            fragment = json.loads(str(output[3]))
            assembled = assembler.add_fragment(fragment, str(output[4]))

        self.assertIsNotNone(assembled)
        assert assembled is not None
        self.assertEqual(assembled["track"]["name"], None)
        self.assertEqual(assembled["clip"]["name"], None)
        self.assertEqual(
            assembled["devices"],
            [
                {
                    "index": 0,
                    "path": "live_set tracks 2 devices 0",
                    "id": 400,
                    "name": None,
                    "class_name": None,
                    "type": None,
                }
            ],
        )
        self.assertEqual(assembled["notes"], [note])

    def test_js_session_clip_inspect_paged_fragments_roundtrip_through_python(self) -> None:
        devices = [
            {
                "name": f"Device {index} " + ("D" * 140),
                "class_name": None if index % 2 else "MockDevice",
                "type": [0, 1, 2, 4, None][index % 5],
            }
            for index in range(30)
        ]
        notes = [
            _inspection_note(
                note_id=index,
                pitch=36 + (index % 48),
                start_time=index * 0.25,
                velocity=80.5 + (index % 40),
                mute=bool(index % 2),
                probability=0.5,
                velocity_deviation=-10.5,
                release_velocity=64.5,
            )
            for index in range(180)
        ]
        outputs = _run_session_clip_inspect_js(
            notes=notes,
            devices=devices,
            request_id="req-roundtrip-paged",
        )
        fragments = [json.loads(str(output[3])) for output in outputs]
        kinds = [
            fragment["transfer"]["fragment_kind"] for fragment in fragments
        ]
        assembler = bridge.SessionClipInspectionAssembler()
        assembled = None
        for output in reversed(outputs):
            assembled = assembler.add_fragment(
                json.loads(str(output[3])),
                str(output[4]),
            )

        self.assertGreater(len(fragments), 2)
        self.assertEqual(kinds[0], "context")
        self.assertNotIn("device_page", kinds[kinds.index("note_page") :])
        self.assertIsNotNone(assembled)
        assert assembled is not None
        self.assertEqual(assembled["devices"][1]["class_name"], None)
        self.assertEqual(
            [device["type"] for device in assembled["devices"][:5]],
            [0, 1, 2, 4, None],
        )
        self.assertEqual(assembled["notes"], notes)

    def test_js_session_clip_inspect_pages_large_payloads_with_bounded_packets(self) -> None:
        devices = [
            {
                "name": f"Device {index} " + ("D" * 140),
                "class_name": "MockDevice",
                "type": 1,
            }
            for index in range(30)
        ]
        notes = [
            {
                "note_id": index,
                "pitch": 36 + (index % 48),
                "start_time": index * 0.25,
                "duration": 0.25,
                "velocity": 80 + (index % 40),
                "mute": index % 2,
                "probability": 0.5,
                "velocity_deviation": -10,
                "release_velocity": 64,
            }
            for index in range(180)
        ]
        outputs = _run_session_clip_inspect_js(notes=notes, devices=devices)
        fragments = [json.loads(str(args[3])) for args in outputs]

        self.assertGreater(len(fragments), 2)
        self.assertEqual(
            [fragment["transfer"]["fragment_index"] for fragment in fragments],
            list(range(len(fragments))),
        )
        self.assertTrue(fragments[-1]["transfer"]["is_last"])
        for output in outputs:
            packet = bridge.encode_osc_message(
                "/ack",
                ("api_session_clip_inspect", str(output[3]), str(output[4])),
            )
            self.assertLessEqual(len(packet), 4096)

        device_pages = [
            fragment["data"]
            for fragment in fragments
            if fragment["transfer"]["fragment_kind"] == "device_page"
        ]
        note_pages = [
            fragment["data"]
            for fragment in fragments
            if fragment["transfer"]["fragment_kind"] == "note_page"
        ]
        self.assertEqual(
            [page["device_offset"] for page in device_pages],
            [sum(previous["device_count"] for previous in device_pages[:index]) for index in range(len(device_pages))],
        )
        self.assertEqual(
            [page["note_offset"] for page in note_pages],
            [sum(previous["note_count"] for previous in note_pages[:index]) for index in range(len(note_pages))],
        )
        emitted_devices = [item for page in device_pages for item in page["devices"]]
        emitted_notes = [item for page in note_pages for item in page["notes"]]
        self.assertEqual([item["index"] for item in emitted_devices], list(range(len(devices))))
        self.assertEqual([item["note_id"] for item in emitted_notes], list(range(len(notes))))

    def test_js_session_clip_inspect_packet_size_matches_python_osc_encoding(self) -> None:
        cases = [
            ("{}", "r"),
            ('{"name":"é"}', "req-é"),
            ('{"name":"𝄞","padding":"abc"}', "四拍子"),
        ]
        js_sizes = _run_bridge_js(
            f"""
const cases = {json.dumps(cases)};
return cases.map((item) =>
  context.sessionClipInspectionAckPacketByteLength(item[0], item[1])
);
"""
        )
        python_sizes = [
            len(
                bridge.encode_osc_message(
                    "/ack",
                    ("api_session_clip_inspect", fragment_json, request_id),
                )
            )
            for fragment_json, request_id in cases
        ]

        self.assertEqual(js_sizes, python_sizes)

    def test_js_session_clip_inspection_ids_are_unique(self) -> None:
        inspection_ids = _run_bridge_js(
            """
context.nowMs = () => 123456;
return [
  context.newSessionClipInspectionId(),
  context.newSessionClipInspectionId(),
];
"""
        )

        self.assertEqual(inspection_ids, ["session_clip_123456_1", "session_clip_123456_2"])

    def test_js_session_clip_inspect_snapshot_change_emits_only_correlated_error(self) -> None:
        outputs = _run_session_clip_inspect_js(
            notes=[],
            devices=[],
            clip_ids=[303, 999],
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0][2:5], ["error", "api_session_clip_inspect_snapshot_changed", 303])
        self.assertEqual(outputs[0][-2:], ["request_correlation", "req:req-inspect"])

    def test_js_session_clip_inspect_runtime_errors_use_reserved_codes(self) -> None:
        result = _run_bridge_js(
            """
function runCase(LiveAPI) {
  const outputs = [];
  context.outlet = (...args) => outputs.push(args);
  context.ensureInitialized = () => true;
  context.LiveAPI = LiveAPI;
  context.api_session_clip_inspect(0, 0, 1, "req-runtime");
  return outputs.find((args) => args[1] === "/ack");
}
function trackApi(overrides) {
  return Object.assign({
    id: 10,
    path: "live_set tracks 0",
    get: (property) => {
      if (property === "has_midi_input") return 1;
      if (property === "name") return "Track";
      throw new Error("unknown property");
    },
    getcount: (child) => {
      if (child === "clip_slots") return 1;
      if (child === "devices") return 0;
      throw new Error("unknown child");
    },
  }, overrides || {});
}
function clipApi(callResult) {
  return {
    id: 30,
    path: "live_set tracks 0 clip_slots 0 clip",
    get: (property) => ({
      name: "Clip",
      start_marker: 0,
      end_marker: 4,
      length: 4,
      looping: 1,
      loop_start: 0,
      loop_end: 4,
    })[property],
    call: () => callResult,
  };
}
return [
  runCase(function LiveAPI() { return { id: 0, path: "" }; }),
  runCase(function LiveAPI(_callback, path) {
    if (path === "live_set tracks 0") {
      return trackApi({ get: (property) => property === "has_midi_input" ? 0 : "Track" });
    }
    throw new Error("unexpected path");
  }),
  runCase(function LiveAPI(_callback, path) {
    if (path === "live_set tracks 0") return trackApi();
    if (path === "live_set tracks 0 clip_slots 0") {
      return { id: 20, path, get: () => 0 };
    }
    throw new Error("unexpected path");
  }),
  runCase(function LiveAPI(_callback, path) {
    if (path === "live_set tracks 0") {
      return trackApi({ get: () => { throw new Error("read failed"); } });
    }
    throw new Error("unexpected path");
  }),
  runCase(function LiveAPI(_callback, path) {
    if (path === "live_set tracks 0") return trackApi();
    if (path === "live_set tracks 0 clip_slots 0") {
      return { id: 20, path, get: () => 1 };
    }
    if (path === "live_set tracks 0 clip_slots 0 clip") return clipApi("not-json");
    throw new Error("unexpected path");
  }),
];
"""
        )

        self.assertEqual(
            [args[3] for args in result],
            [
                "api_session_clip_inspect_not_found",
                "api_session_clip_inspect_not_midi",
                "api_session_clip_inspect_no_clip",
                "api_session_clip_inspect_read_failed",
                "api_session_clip_inspect_parse_failed",
            ],
        )
        for args in result:
            self.assertEqual(args[-2:], ["request_correlation", "req:req-runtime"])

    def test_js_session_clip_inspect_reports_serialization_and_item_too_large(self) -> None:
        serialization_error = _run_bridge_js(
            """
const outputs = [];
context.outlet = (...args) => outputs.push(args);
const circular = {};
circular.self = circular;
const built = context.buildSessionClipInspectionFragments(
  {
    inspection_id: "inspection-circular",
    correlation: { request_id: "req-serialization", track_index: 0, slot_index: 0 },
    snapshot: { started_ms: 1, completed_ms: 2, atomic: false, consistent: true },
  },
  {
    context: "session",
    track: circular,
    clip: { slot_index: 0 },
    summary: { note_count: 0, pitch_min: null, pitch_max: null },
  },
  [],
  [],
  "req-serialization"
);
context.sessionClipInspectionError(built.error, built.details, "req-serialization");
return outputs.filter((args) => args[1] === "/ack");
"""
        )
        item_too_large = _run_session_clip_inspect_js(
            notes=[],
            devices=[{"name": "X" * 5000}],
            request_id="req-large-item",
        )

        self.assertEqual(
            serialization_error[0][3],
            "api_session_clip_inspect_serialization_failed",
        )
        self.assertEqual(
            item_too_large[0][3],
            "api_session_clip_inspect_item_too_large",
        )
        self.assertEqual(
            item_too_large[0][-2:],
            ["request_correlation", "req:req-large-item"],
        )

    def test_legacy_session_clip_inspect_source_and_ack_remain_unchanged(self) -> None:
        source = pathlib.Path(__file__).with_name("m4l").joinpath("live_udp_bridge.js").read_text()
        function_source = source.split("function inspect_session_clip_notes", 1)[1].split(
            "\n}\n", 1
        )[0]
        digest = hashlib.sha256(
            ("function inspect_session_clip_notes" + function_source + "\n}\n").encode()
        ).hexdigest()
        self.assertEqual(
            digest,
            "e3cfa4781d667e4c21c3c146cbe06aab398786bed5ef63b3ed0e6f1828388b63",
        )

        raw_notes = '{"notes":[{"pitch":60,"start_time":0,"duration":1,"velocity":100}]}'
        result = _run_bridge_js(
            f"""
const outputs = [];
context.outlet = (...args) => outputs.push(args);
context.ensureInitialized = () => true;
context.song = {{ getcount: (child) => child === "tracks" ? 3 : 0 }};
context.LiveAPI = function LiveAPI(_callback, path) {{
  if (path === "live_set tracks 2") {{
    return {{ get: (property) => property === "has_midi_input" ? 1 : null }};
  }}
  if (path === "live_set tracks 2 clip_slots 3") {{
    return {{ get: (property) => property === "has_clip" ? 1 : null }};
  }}
  if (path === "live_set tracks 2 clip_slots 3 clip") {{
    return {{
      get: (property) => property === "length" ? 4 : null,
      call: () => {json.dumps(raw_notes)},
    }};
  }}
  throw new Error("unknown path");
}};
context.inspect_session_clip_notes(2, 3);
return outputs.filter((args) => args[1] === "/ack");
"""
        )
        self.assertEqual(
            result,
            [[0, "/ack", "inspect_session_clip_notes", 2, 3, 1, 60, 60, 4, raw_notes]],
        )

    def test_session_clip_inspection_assembler_accepts_out_of_order_duplicates(self) -> None:
        fragments = [
            _inspection_fragment(
                index=0,
                count=3,
                kind="context",
                data=_inspection_context_data(
                    note_count=2,
                    pitch_min=60,
                    pitch_max=64,
                ),
            ),
            _inspection_fragment(
                index=1,
                count=3,
                kind="device_page",
                data={
                    "device_offset": 0,
                    "device_count": 1,
                    "device_total": 1,
                    "devices": [_inspection_device()],
                },
            ),
            _inspection_fragment(
                index=2,
                count=3,
                kind="note_page",
                data={
                    "note_offset": 0,
                    "note_count": 2,
                    "note_total": 2,
                    "notes": [
                        _inspection_note(note_id=1, pitch=60),
                        _inspection_note(note_id=2, pitch=64, start_time=0.5),
                    ],
                },
            ),
        ]
        assembler = bridge.SessionClipInspectionAssembler()

        self.assertIsNone(assembler.add_fragment(fragments[2]))
        self.assertIsNone(assembler.add_fragment(fragments[2]))
        self.assertIsNone(assembler.add_fragment(fragments[0]))
        assembled = assembler.add_fragment(fragments[1])

        self.assertIsNotNone(assembled)
        assert assembled is not None
        self.assertEqual(assembled["context"], "session")
        self.assertEqual(assembled["devices"], [_inspection_device()])
        self.assertEqual(
            assembled["notes"],
            [
                _inspection_note(note_id=1, pitch=60),
                _inspection_note(note_id=2, pitch=64, start_time=0.5),
            ],
        )
        self.assertEqual(
            assembled["transport"],
            {
                "complete": True,
                "fragment_count": 3,
                "received_fragment_count": 3,
                "fragment_indexes": [0, 1, 2],
                "packet_budget_bytes": 4096,
            },
        )

    def test_session_clip_inspection_assembler_rejects_conflicts_and_gaps(self) -> None:
        context_fragment = _inspection_fragment(
            index=0,
            count=2,
            kind="context",
            data=_inspection_context_data(),
        )
        assembler = bridge.SessionClipInspectionAssembler()
        assembler.add_fragment(context_fragment)

        conflicting = json.loads(json.dumps(context_fragment))
        conflicting["data"]["track"]["name"] = "Changed Track"
        with self.assertRaisesRegex(bridge.SessionClipInspectionAssemblyError, "conflicting duplicate"):
            assembler.add_fragment(conflicting)
        with self.assertRaisesRegex(bridge.SessionClipInspectionAssemblyError, "missing fragment indexes"):
            assembler.assemble("req-assembly", "inspection-1")

        mixed_assembler = bridge.SessionClipInspectionAssembler()
        mixed_assembler.add_fragment(context_fragment)
        mixed_count = _inspection_fragment(
            index=1,
            count=3,
            kind="note_page",
            data={
                "note_offset": 0,
                "note_count": 0,
                "note_total": 0,
                "notes": [],
            },
        )
        with self.assertRaisesRegex(bridge.SessionClipInspectionAssemblyError, "mixed metadata"):
            mixed_assembler.add_fragment(mixed_count)

    def test_session_clip_inspection_assembler_rejects_malformed_and_noncontiguous_pages(self) -> None:
        assembler = bridge.SessionClipInspectionAssembler()
        malformed = _inspection_fragment(
            index=1,
            count=2,
            kind="note_page",
            data={
                "note_offset": 0,
                "note_count": 2,
                "note_total": 1,
                "notes": [{"note_id": 1}],
            },
        )
        with self.assertRaisesRegex(bridge.SessionClipInspectionAssemblyError, "inconsistent counts"):
            assembler.add_fragment(malformed)

        fragments = [
            _inspection_fragment(
                index=0,
                count=3,
                kind="context",
                data=_inspection_context_data(
                    note_count=2,
                    pitch_min=60,
                    pitch_max=64,
                ),
            ),
            _inspection_fragment(
                index=1,
                count=3,
                kind="note_page",
                data={
                    "note_offset": 0,
                    "note_count": 1,
                    "note_total": 3,
                    "notes": [_inspection_note(note_id=1, pitch=60)],
                },
            ),
            _inspection_fragment(
                index=2,
                count=3,
                kind="note_page",
                data={
                    "note_offset": 2,
                    "note_count": 1,
                    "note_total": 3,
                    "notes": [
                        _inspection_note(note_id=2, pitch=64, start_time=0.5)
                    ],
                },
            ),
        ]
        assembler = bridge.SessionClipInspectionAssembler()
        assembler.add_fragment(fragments[0])
        assembler.add_fragment(fragments[1])
        with self.assertRaisesRegex(bridge.SessionClipInspectionAssemblyError, "noncontiguous note offsets"):
            assembler.add_fragment(fragments[2])

    def test_session_clip_inspection_assembler_rejects_mixed_transfer_metadata(self) -> None:
        fragment = _inspection_fragment(
            index=0,
            count=2,
            kind="context",
            data=_inspection_context_data(),
        )
        fragment["transfer"]["is_last"] = True

        with self.assertRaisesRegex(
            bridge.SessionClipInspectionAssemblyError,
            "mixed transfer metadata",
        ):
            bridge.SessionClipInspectionAssembler().add_fragment(fragment)

    def test_session_clip_inspection_assembler_rejects_unknown_schema_keys(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []

        top_level = _complete_inspection_fragment()
        top_level["invented"] = True
        cases.append(("root", top_level))

        correlation = _complete_inspection_fragment()
        correlation["correlation"]["invented"] = True
        cases.append(("correlation", correlation))

        transfer = _complete_inspection_fragment()
        transfer["transfer"]["invented"] = True
        cases.append(("transfer", transfer))

        data = _complete_inspection_fragment()
        data["data"]["invented"] = True
        cases.append(("data", data))

        for label, fragment in cases:
            with self.subTest(label=label):
                with self.assertRaises(bridge.SessionClipInspectionAssemblyError):
                    bridge.SessionClipInspectionAssembler().add_fragment(fragment)

    def test_session_clip_inspection_assembler_rejects_invalid_context_facts(self) -> None:
        mutations = {
            "empty_track_path": ("track", "path", ""),
            "numeric_track_name": ("track", "name", 42),
            "numeric_clip_name": ("clip", "name", 42),
            "bool_track_id": ("track", "id", True),
            "negative_clip_id": ("clip", "id", -1),
            "nan_marker": ("clip", "start_marker", math.nan),
            "negative_length": ("clip", "live_length", -1),
            "numeric_looping": ("clip", "looping", 1),
            "null_pitch_with_notes": ("summary", "pitch_min", None),
        }
        for label, (section, field, value) in mutations.items():
            fragment = _complete_inspection_fragment(notes=[_inspection_note()])
            fragment["data"][section][field] = value
            with self.subTest(label=label):
                with self.assertRaises(bridge.SessionClipInspectionAssemblyError):
                    bridge.SessionClipInspectionAssembler().add_fragment(fragment)

    def test_session_clip_inspection_assembler_accepts_signed_ordered_clip_times(self) -> None:
        fragment = _complete_inspection_fragment()
        fragment["data"]["clip"].update(
            {
                "start_marker": -8,
                "end_marker": -2,
                "live_length": 6,
                "loop_start": -7.5,
                "loop_end": -2.5,
            }
        )

        assembled = bridge.SessionClipInspectionAssembler().add_fragment(fragment)

        self.assertIsNotNone(assembled)
        assert assembled is not None
        self.assertEqual(assembled["clip"]["start_marker"], -8)
        self.assertEqual(assembled["clip"]["loop_start"], -7.5)

    def test_session_clip_inspection_assembler_rejects_invalid_clip_ranges(self) -> None:
        mutations = [
            ("start_marker", 2, "end_marker", 1),
            ("loop_start", 2, "loop_end", 1),
            ("start_marker", -1e308, "end_marker", 1e308),
            ("loop_start", -1e308, "loop_end", 1e308),
            ("live_length", math.inf, None, None),
            ("live_length", -1, None, None),
        ]

        for first_field, first_value, second_field, second_value in mutations:
            fragment = _complete_inspection_fragment()
            fragment["data"]["clip"][first_field] = first_value
            if second_field is not None:
                fragment["data"]["clip"][second_field] = second_value
            with self.subTest(
                first_field=first_field,
                first_value=first_value,
                second_field=second_field,
            ):
                with self.assertRaises(bridge.SessionClipInspectionAssemblyError):
                    bridge.SessionClipInspectionAssembler().add_fragment(fragment)

    def test_session_clip_inspection_assembler_rejects_resource_counts_before_state(self) -> None:
        cases: list[dict[str, object]] = []

        billion_fragments = _inspection_fragment(
            index=0,
            count=1_000_000_000,
            kind="context",
            data=_inspection_context_data(),
        )
        cases.append(billion_fragments)

        too_many_devices = _inspection_fragment(
            index=1,
            count=2,
            kind="device_page",
            data={
                "device_offset": 0,
                "device_count": 0,
                "device_total": 257,
                "devices": [],
            },
        )
        cases.append(too_many_devices)

        too_many_notes = _inspection_fragment(
            index=1,
            count=2,
            kind="note_page",
            data={
                "note_offset": 0,
                "note_count": 0,
                "note_total": 4097,
                "notes": [],
            },
        )
        cases.append(too_many_notes)

        too_many_summary_notes = _inspection_fragment(
            index=0,
            count=2,
            kind="context",
            data=_inspection_context_data(
                note_count=4097,
                pitch_min=0,
                pitch_max=127,
            ),
        )
        cases.append(too_many_summary_notes)

        for fragment in cases:
            assembler = bridge.SessionClipInspectionAssembler()
            with self.subTest(fragment=fragment):
                with self.assertRaisesRegex(
                    bridge.SessionClipInspectionAssemblyError,
                    "limit",
                ):
                    assembler.add_fragment(fragment)
                self.assertEqual(len(assembler._states), 0)

    def test_session_clip_inspection_assembler_accepts_resource_boundaries(self) -> None:
        device_page = _inspection_fragment(
            index=1,
            count=2,
            kind="device_page",
            inspection_id="boundary-devices",
            data={
                "device_offset": 255,
                "device_count": 1,
                "device_total": 256,
                "devices": [_inspection_device(255)],
            },
        )
        note_page = _inspection_fragment(
            index=1,
            count=2,
            kind="note_page",
            inspection_id="boundary-notes",
            data={
                "note_offset": 4095,
                "note_count": 1,
                "note_total": 4096,
                "notes": [_inspection_note(note_id=4095)],
            },
        )

        self.assertIsNone(
            bridge.SessionClipInspectionAssembler().add_fragment(device_page)
        )
        self.assertIsNone(
            bridge.SessionClipInspectionAssembler().add_fragment(note_page)
        )

    def test_session_clip_inspection_assembler_caps_active_states_and_evicts_errors(self) -> None:
        assembler = bridge.SessionClipInspectionAssembler()
        fragments = [
            _inspection_fragment(
                index=0,
                count=2,
                kind="context",
                request_id=f"req-active-{index}",
                inspection_id=f"inspection-active-{index}",
                data=_inspection_context_data(),
            )
            for index in range(17)
        ]
        for fragment in fragments[:16]:
            self.assertIsNone(assembler.add_fragment(fragment))

        self.assertEqual(len(assembler._states), 16)
        with self.assertRaisesRegex(
            bridge.SessionClipInspectionAssemblyError,
            "active assembly limit",
        ):
            assembler.add_fragment(fragments[16])

        conflicting = json.loads(json.dumps(fragments[0]))
        conflicting["data"]["track"]["name"] = "Conflicting Track"
        with self.assertRaisesRegex(
            bridge.SessionClipInspectionAssemblyError,
            "conflicting duplicate",
        ):
            assembler.add_fragment(conflicting)

        self.assertEqual(len(assembler._states), 15)
        self.assertIsNone(assembler.add_fragment(fragments[16]))
        self.assertEqual(len(assembler._states), 16)

    def test_session_clip_inspection_assembler_mismatched_outer_request_does_not_evict_state(self) -> None:
        fragment = _inspection_fragment(
            index=0,
            count=2,
            kind="context",
            request_id="req-state",
            inspection_id="inspection-state",
            data=_inspection_context_data(),
        )
        assembler = bridge.SessionClipInspectionAssembler()
        assembler.add_fragment(fragment)

        with self.assertRaisesRegex(
            bridge.SessionClipInspectionAssemblyError,
            "mixed metadata",
        ):
            assembler.add_fragment(fragment, request_id="req-other")

        self.assertEqual(len(assembler._states), 1)

    def test_session_clip_inspection_assembler_evicts_completed_state(self) -> None:
        assembler = bridge.SessionClipInspectionAssembler()

        assembled = assembler.add_fragment(_complete_inspection_fragment())

        self.assertIsNotNone(assembled)
        self.assertEqual(len(assembler._states), 0)

    def test_session_clip_inspection_assembler_bounds_missing_index_diagnostic(self) -> None:
        fragment = _inspection_fragment(
            index=0,
            count=1024,
            kind="context",
            data=_inspection_context_data(),
        )
        assembler = bridge.SessionClipInspectionAssembler()
        assembler.add_fragment(fragment)

        with self.assertRaises(bridge.SessionClipInspectionAssemblyError) as raised:
            assembler.assemble("req-assembly", "inspection-1")

        message = str(raised.exception)
        self.assertIn("missing fragment indexes", message)
        self.assertIn("more", message)
        self.assertLess(len(message), 256)
        self.assertEqual(len(assembler._states), 0)

    def test_session_clip_inspection_assembler_rejects_invalid_device_facts(self) -> None:
        invalid_devices: list[object] = [
            "Operator",
            {
                "index": 0,
                "path": "live_set tracks 2 devices 0",
                "id": 400,
                "name": None,
                "class_name": None,
            },
            {**_inspection_device(), "invented": True},
            {**_inspection_device(), "index": True},
            {**_inspection_device(), "name": 42},
            {**_inspection_device(), "class_name": 42},
            {**_inspection_device(), "type": True},
            {**_inspection_device(), "type": "1"},
            {**_inspection_device(), "type": 3},
            {**_inspection_device(), "type": -1},
            {**_inspection_device(), "type": 1.5},
        ]
        for device in invalid_devices:
            with self.subTest(device=device):
                fragment = _complete_inspection_fragment(devices=[device])
                with self.assertRaises(bridge.SessionClipInspectionAssemblyError):
                    bridge.SessionClipInspectionAssembler().add_fragment(fragment)

    def test_session_clip_inspection_assembler_accepts_device_type_enums_and_null(self) -> None:
        devices = [
            _inspection_device(index, device_type=device_type)
            for index, device_type in enumerate((0, 1, 2, 4, None))
        ]
        fragment = _complete_inspection_fragment(devices=devices)

        assembled = bridge.SessionClipInspectionAssembler().add_fragment(fragment)

        self.assertIsNotNone(assembled)
        assert assembled is not None
        self.assertEqual(
            [device["type"] for device in assembled["devices"]],
            [0, 1, 2, 4, None],
        )

    def test_session_clip_inspection_assembler_rejects_invalid_note_facts(self) -> None:
        invalid_notes = [
            {**_inspection_note(), "invented": True},
            {
                key: value
                for key, value in _inspection_note().items()
                if key != "velocity"
            },
            {**_inspection_note(), "pitch": True},
            {**_inspection_note(), "pitch": 128},
            {**_inspection_note(), "start_time": math.inf},
            {**_inspection_note(), "start_time": 10**1000},
            {**_inspection_note(), "duration": -0.01},
            {**_inspection_note(), "start_time": 1e308, "duration": 1e308},
            {**_inspection_note(), "velocity": True},
            {**_inspection_note(), "velocity": 128},
            {**_inspection_note(), "note_id": -1},
            {**_inspection_note(), "mute": 2},
            {**_inspection_note(), "mute": "0"},
            {**_inspection_note(), "probability": True},
            {**_inspection_note(), "probability": 1.01},
            {**_inspection_note(), "velocity_deviation": -128},
            {**_inspection_note(), "release_velocity": 128},
        ]
        for note in invalid_notes:
            with self.subTest(note=note):
                fragment = _complete_inspection_fragment(notes=[note])
                with self.assertRaises(bridge.SessionClipInspectionAssemblyError):
                    bridge.SessionClipInspectionAssembler().add_fragment(fragment)

    def test_session_clip_inspection_assembler_accepts_frozen_v1_scalar_types(self) -> None:
        note = _inspection_note(
            start_time=-0.25,
            duration=0,
            velocity=100.5,
            mute=1.0,
            release_velocity=64.5,
        )
        fragment = _complete_inspection_fragment(
            devices=[
                {
                    **_inspection_device(),
                    "name": None,
                    "class_name": None,
                    "type": None,
                }
            ],
            notes=[note],
        )
        fragment["data"]["track"]["name"] = None
        fragment["data"]["clip"]["name"] = None

        assembled = bridge.SessionClipInspectionAssembler().add_fragment(fragment)

        self.assertIsNotNone(assembled)
        assert assembled is not None
        self.assertEqual(assembled["notes"], [note])
        self.assertIsNone(assembled["track"]["name"])
        self.assertIsNone(assembled["devices"][0]["type"])

    def test_session_clip_inspection_assembler_rejects_invalid_fragment_ordering(self) -> None:
        note_then_device = [
            _inspection_fragment(
                index=0,
                count=3,
                kind="context",
                data=_inspection_context_data(
                    note_count=1,
                    pitch_min=60,
                    pitch_max=60,
                ),
            ),
            _inspection_fragment(
                index=1,
                count=3,
                kind="note_page",
                data={
                    "note_offset": 0,
                    "note_count": 1,
                    "note_total": 1,
                    "notes": [_inspection_note()],
                },
            ),
            _inspection_fragment(
                index=2,
                count=3,
                kind="device_page",
                data={
                    "device_offset": 0,
                    "device_count": 1,
                    "device_total": 1,
                    "devices": [_inspection_device()],
                },
            ),
        ]
        reversed_offsets = [
            _inspection_fragment(
                index=0,
                count=3,
                kind="context",
                data=_inspection_context_data(
                    note_count=2,
                    pitch_min=60,
                    pitch_max=64,
                ),
            ),
            _inspection_fragment(
                index=1,
                count=3,
                kind="note_page",
                data={
                    "note_offset": 1,
                    "note_count": 1,
                    "note_total": 2,
                    "notes": [
                        _inspection_note(
                            note_id=2,
                            pitch=64,
                            start_time=0.5,
                        )
                    ],
                },
            ),
            _inspection_fragment(
                index=2,
                count=3,
                kind="note_page",
                data={
                    "note_offset": 0,
                    "note_count": 1,
                    "note_total": 2,
                    "notes": [_inspection_note()],
                },
            ),
        ]
        zero_total_page = [
            _inspection_fragment(
                index=0,
                count=2,
                kind="context",
                data=_inspection_context_data(),
            ),
            _inspection_fragment(
                index=1,
                count=2,
                kind="device_page",
                data={
                    "device_offset": 0,
                    "device_count": 0,
                    "device_total": 0,
                    "devices": [],
                },
            ),
        ]

        for fragments in (note_then_device, reversed_offsets, zero_total_page):
            with self.subTest(
                kinds=[
                    fragment["transfer"]["fragment_kind"]
                    for fragment in fragments
                ]
            ):
                assembler = bridge.SessionClipInspectionAssembler()
                assembler.add_fragment(fragments[0])
                for fragment in fragments[1:-1]:
                    assembler.add_fragment(fragment)
                with self.assertRaises(
                    bridge.SessionClipInspectionAssemblyError
                ):
                    assembler.add_fragment(fragments[-1])

    def test_session_clip_inspection_assembler_rejects_partial_complete_totals(self) -> None:
        fragments = []

        missing_device = _complete_inspection_fragment()
        missing_device["data"]["device_total"] = 1
        fragments.append(missing_device)

        missing_note = _complete_inspection_fragment(notes=[_inspection_note()])
        missing_note["data"]["note_total"] = 2
        fragments.append(missing_note)

        for fragment in fragments:
            with self.subTest(data=fragment["data"]):
                with self.assertRaises(
                    bridge.SessionClipInspectionAssemblyError
                ):
                    bridge.SessionClipInspectionAssembler().add_fragment(fragment)

    def test_session_clip_inspection_assembler_rejects_invalid_envelope_metadata(self) -> None:
        cases: list[dict[str, object]] = []

        boolean_schema_version = _complete_inspection_fragment()
        boolean_schema_version["schema_version"] = True
        cases.append(boolean_schema_version)

        completed_before_started = _complete_inspection_fragment()
        completed_before_started["snapshot"]["completed_ms"] = 99
        cases.append(completed_before_started)

        mismatched_track = _complete_inspection_fragment()
        mismatched_track["correlation"]["track_index"] = 1
        cases.append(mismatched_track)

        nonzero_context_index = _inspection_fragment(
            index=1,
            count=2,
            kind="context",
            data=_inspection_context_data(),
        )
        cases.append(nonzero_context_index)

        for fragment in cases:
            with self.subTest(fragment=fragment):
                with self.assertRaises(bridge.SessionClipInspectionAssemblyError):
                    bridge.SessionClipInspectionAssembler().add_fragment(fragment)

    def test_js_drum_chain_note_rejects_fractional_values(self) -> None:
        js_source = pathlib.Path(__file__).with_name("m4l").joinpath("live_udp_bridge.js").read_text()
        function_source = js_source.split("function api_drum_chain_in_note", 1)[1].split(
            "function api_observe", 1
        )[0]

        self.assertIn("Math.floor(note) !== note", function_source)
        self.assertNotIn("Math.floor(Number(noteValue))", function_source)

    def test_js_note_schema_accepts_negative_velocity_deviation(self) -> None:
        results = _run_bridge_js(
            """
const deviations = [-127, -64.5, 127, -128];
return deviations.map((velocityDeviation) => context.normalizeNote(
  {
    pitch: 60,
    start_time: 0,
    duration: 1,
    velocity: 100,
    velocity_deviation: velocityDeviation,
  },
  0,
  "test"
) !== null);
"""
        )
        self.assertEqual(results, [True, True, True, False])

    def test_js_error_ack_always_emits_tagged_terminal_correlation_slot(self) -> None:
        result = _run_bridge_js(
            """
const outputs = [];
context.outlet = (...args) => outputs.push(args);
context.ackWithRequest("error", ["api_example_failed", "detail"], null);
context.ackWithRequest("error", ["api_example_failed", "detail"], "req-error");
context.ack("ack", "error", "legacy_failed", 40);
return outputs;
"""
        )
        self.assertEqual(
            result,
            [
                [
                    0,
                    "/ack",
                    "error",
                    "api_example_failed",
                    "detail",
                    "request_correlation",
                    "req:",
                ],
                [
                    0,
                    "/ack",
                    "error",
                    "api_example_failed",
                    "detail",
                    "request_correlation",
                    "req:req-error",
                ],
                [0, "/ack", "error", "legacy_failed", 40, "request_correlation", "req:"],
            ],
        )

    def test_js_request_aware_initialization_and_api_call_helpers_preserve_request_id(self) -> None:
        result = _run_bridge_js(
            """
const outputs = [];
context.outlet = (...args) => outputs.push(args);
context.initialized = false;
context.song = null;
context.init = () => {};
context.api_ping("req-init");
context.ackWithRequest = (eventName, args, requestId) => {
  outputs.push([eventName, ...args, requestId ?? null]);
};
context.ensureInitialized = () => true;
context.resolveApiOrError = () => ({
  path: "live_set tracks 0 clip_slots 0 clip",
  id: 8,
  call: () => null,
});
context.getApiCapabilities = () => ({ hasFunctionsList: false });
context.buildNotesDict = (_notes, _contextName, requestId) => {
  context.ackWithRequest("error", ["api_call_add_new_notes_invalid_pitch"], requestId);
  return null;
};
context.api_call(
  "live_set tracks 0 clip_slots 0 clip",
  "add_new_notes",
  '[{"notes":[{"pitch":-1}]}]',
  "req-call"
);
return outputs.filter((args) => args[1] === "/ack" || args[0] === "error");
"""
        )
        self.assertEqual(
            result[0],
            [
                0,
                "/ack",
                "error",
                "not_initialized",
                "request_correlation",
                "req:req-init",
            ],
        )
        self.assertEqual(
            result[1],
            ["error", "api_call_add_new_notes_invalid_pitch", "req-call"],
        )

    def test_js_dict_builder_exceptions_emit_correlated_errors(self) -> None:
        result = _run_bridge_js(
            """
const acks = [];
context.ackWithRequest = (eventName, args, requestId) => {
  acks.push([eventName, ...args, requestId ?? null]);
};
context.Dict = function DictWithSetparseFailure() {
  return {
    setparse: () => { throw new Error("setparse failed"); },
    get: () => ({}),
    clear: () => {},
  };
};
context.buildNotesDict(
  [{ pitch: 60, start_time: 0, duration: 1, velocity: 100 }],
  "api_call_add_new_notes",
  "req-notes"
);
context.Dict = function DictWithGetFailure() {
  return {
    setparse: () => {},
    get: () => { throw new Error("get failed"); },
    clear: () => {},
  };
};
context.buildGenericDict(
  { note_ids: [1] },
  "api_call_apply_note_modifications",
  "req-generic"
);
return acks;
"""
        )
        self.assertEqual(
            result,
            [
                ["error", "api_call_add_new_notes_notes_dict_build_failed", "req-notes"],
                [
                    "error",
                    "api_call_apply_note_modifications_dict_build_failed",
                    "req-generic",
                ],
            ],
        )

    def test_js_request_aware_status_helpers_stop_after_correlated_errors(self) -> None:
        result = _run_bridge_js(
            """
function runCase(song, LiveAPI, call) {
  const acks = [];
  context.ensureInitialized = () => true;
  context.song = song;
  context.LiveAPI = LiveAPI;
  context.ackWithRequest = (eventName, args, requestId) => {
    acks.push([eventName, ...args, requestId ?? null]);
  };
  call();
  return acks;
}
return {
  sessionTrackCount: runCase(
    { path: "live_set", id: 1, getcount: () => { throw new Error("count failed"); } },
    function LiveAPI() {},
    () => context.api_session_context("req-session-count")
  ),
  deviceTrackCount: runCase(
    { path: "live_set", id: 1, getcount: () => { throw new Error("count failed"); } },
    function LiveAPI() {},
    () => context.api_device_list("all", "req-device-count")
  ),
  sessionMidiCount: runCase(
    { path: "live_set", id: 1, getcount: () => 1 },
    function LiveAPI() { throw new Error("midi inspect failed"); },
    () => context.api_session_context("req-session-midi")
  ),
  sessionAudioCount: runCase(
    { path: "live_set", id: 1, getcount: () => 1 },
    function LiveAPI() {
      return {
        get: (property) => {
          if (property === "has_midi_input") return 1;
          throw new Error("audio inspect failed");
        },
      };
    },
    () => context.api_session_context("req-session-audio")
  ),
};
"""
        )
        self.assertEqual(
            result["sessionTrackCount"],
            [["error", "track_count_failed", "session_context", "req-session-count"]],
        )
        self.assertEqual(
            result["deviceTrackCount"],
            [["error", "track_count_failed", "device_list", "req-device-count"]],
        )
        self.assertEqual(
            result["sessionMidiCount"],
            [["error", "count_midi_tracks_failed", "session_context", 0, "req-session-midi"]],
        )
        self.assertEqual(
            result["sessionAudioCount"],
            [["error", "count_audio_tracks_failed", "session_context", 0, "req-session-audio"]],
        )

    def test_js_track_mutations_stop_after_count_or_inspection_failures(self) -> None:
        result = _run_bridge_js(
            """
function runWithTrackCounts(counts, call, options = {}) {
  const events = [];
  let index = 0;
  context.ensureInitialized = () => true;
  context.song = { call: options.songCall || (() => {}) };
  context.renameTrack = options.renameTrack || (() => true);
  context.getTotalTracksOrError = () => counts[index++];
  context.ack = (_address, eventName, ...details) => events.push([eventName, ...details]);
  call();
  return events;
}
const inspectionEvents = [];
context.LiveAPI = function LiveAPI() { throw new Error("inspect failed"); };
context.ack = (_address, eventName) => inspectionEvents.push(eventName);
const inspectionResult = context.listTrackIndices(2, () => true, "delete_midi_tracks");
return {
  addMidiFinalCount: runWithTrackCounts([1, 1, 2, 0], () => context.add_midi_tracks(1, "MIDI")),
  addAudioFinalCount: runWithTrackCounts([1, 1, 2, 0], () => context.add_audio_tracks(1, "Audio")),
  deleteMidiFinalCount: (() => {
    context.listTrackIndices = () => [1];
    return runWithTrackCounts([2, 0], () => context.delete_midi_tracks(1));
  })(),
  deleteAudioFinalCount: (() => {
    context.listTrackIndices = () => [1];
    return runWithTrackCounts([2, 0], () => context.delete_audio_tracks(1));
  })(),
  addMidiCreate: runWithTrackCounts(
    [1, 1],
    () => context.add_midi_tracks(1, "MIDI"),
    { songCall: () => { throw new Error("create failed"); } }
  ),
  addAudioCreate: runWithTrackCounts(
    [1, 1],
    () => context.add_audio_tracks(1, "Audio"),
    { songCall: () => { throw new Error("create failed"); } }
  ),
  addMidiRename: runWithTrackCounts(
    [1, 1, 2],
    () => context.add_midi_tracks(1, "MIDI"),
    {
      renameTrack: () => {
        context.ack("ack", "error", "rename_track");
        return false;
      },
    }
  ),
  addAudioRename: runWithTrackCounts(
    [1, 1, 2],
    () => context.add_audio_tracks(1, "Audio"),
    {
      renameTrack: () => {
        context.ack("ack", "error", "rename_track");
        return false;
      },
    }
  ),
  deleteMidiFailure: (() => {
    context.listTrackIndices = () => [1];
    return runWithTrackCounts(
      [2],
      () => context.delete_midi_tracks(1),
      { songCall: () => { throw new Error("delete failed"); } }
    );
  })(),
  deleteAudioFailure: (() => {
    context.listTrackIndices = () => [1];
    return runWithTrackCounts(
      [2],
      () => context.delete_audio_tracks(1),
      { songCall: () => { throw new Error("delete failed"); } }
    );
  })(),
  inspectionResult,
  inspectionEvents,
};
"""
        )
        for key, completion_event in [
            ("addMidiFinalCount", "add_midi_tracks"),
            ("addAudioFinalCount", "add_audio_tracks"),
            ("deleteMidiFinalCount", "delete_midi_tracks"),
            ("deleteAudioFinalCount", "delete_audio_tracks"),
            ("addMidiCreate", "add_midi_tracks"),
            ("addAudioCreate", "add_audio_tracks"),
            ("addMidiRename", "add_midi_tracks"),
            ("addAudioRename", "add_audio_tracks"),
            ("deleteMidiFailure", "delete_midi_tracks"),
            ("deleteAudioFailure", "delete_audio_tracks"),
        ]:
            self.assertNotIn(completion_event, [event[0] for event in result[key]])
        self.assertEqual(result["addMidiCreate"], [["error", "add_midi_tracks_create_failed", 0]])
        self.assertEqual(result["addAudioCreate"], [["error", "add_audio_tracks_create_failed", 0]])
        self.assertEqual(result["addMidiRename"], [["error", "rename_track"]])
        self.assertEqual(result["addAudioRename"], [["error", "rename_track"]])
        self.assertEqual(result["deleteMidiFailure"], [["error", "midi_track_delete_failed", 1]])
        self.assertEqual(result["deleteAudioFailure"], [["error", "delete_audio_track_failed", 1]])
        self.assertIsNone(result["inspectionResult"])
        self.assertEqual(result["inspectionEvents"], ["error"])

    def test_js_drum_chain_note_validates_readback_and_applied_write(self) -> None:
        result = _run_bridge_js(
            """
const cases = [
  { requested: 0, payload: { in_note: null } },
  { requested: 36, payload: { errors: { in_note: "read_failed" } } },
  { requested: -1, payload: { in_note: 40 } },
  { requested: 36, payload: { in_note: 36, out_note: 60, choke_group: 0, name: "Chain" } },
];
return cases.map((testCase) => {
  const acks = [];
  const fakeApi = { path: "live_set tracks 0 devices 0 chains 0", id: 9, set: () => {} };
  context.ensureInitialized = () => true;
  context.resolveApiOrError = () => fakeApi;
  context.readApiPropertyBag = () => testCase.payload;
  context.ackWithRequest = (eventName, args, requestId) => {
    acks.push({ eventName, args, requestId: requestId ?? null });
  };
  context.api_drum_chain_in_note(fakeApi.path, testCase.requested, "req-note");
  return acks[0];
});
"""
        )

        self.assertEqual(result[0]["args"][0], "api_drum_chain_in_note_readback_failed")
        self.assertEqual(result[1]["args"][0], "api_drum_chain_in_note_readback_failed")
        self.assertEqual(result[2]["args"][0], "api_drum_chain_in_note_write_not_applied")
        self.assertEqual(result[2]["args"][-2:], [-1, 40])
        self.assertEqual(result[3]["eventName"], "api_drum_chain_in_note")

    def test_js_normalizes_live_api_paths_before_deriving_child_paths(self) -> None:
        js_source = pathlib.Path(__file__).with_name("m4l").joinpath("live_udp_bridge.js").read_text()

        self.assertIn("function normalizeLiveApiPath", js_source)
        self.assertIn("normalizeLiveApiPath(trackApi.path, trackPath)", js_source)
        self.assertIn("normalizeLiveApiPath(deviceApi.path, requestedPath)", js_source)
        self.assertIn("normalizeLiveApiPath(mixerApi.path, mixerPath)", js_source)
        self.assertIn("normalizeLiveApiPath(api.path, path)", js_source)
        self.assertIn("normalizeLiveApiPath(targetApi.path, pathText)", js_source)
        self.assertIn("normalizeLiveApiPath(rackApi.path, pathText)", js_source)
        self.assertIn("normalizeLiveApiPath(chainApi.path, pathText)", js_source)

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

    def test_timeout_parsers_reject_nonfinite_values(self) -> None:
        for flag in ("--ack-timeout", "--listen-timeout"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(flag=flag, value=value):
                    with (
                        mock.patch("sys.stderr", new=io.StringIO()),
                        self.assertRaises(SystemExit),
                    ):
                        bridge.parse_args(
                            [
                                f"{flag}={value}",
                                "--no-tempo",
                                "--no-signature",
                            ]
                        )

        with (
            mock.patch("sys.stderr", new=io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            bridge.parse_args(
                [
                    "--ack-timeout",
                    "0",
                    "--no-tempo",
                    "--no-signature",
                ]
            )
        self.assertEqual(
            bridge.parse_args(
                [
                    "--listen-timeout",
                    "0",
                    "--no-tempo",
                    "--no-signature",
                ]
            ).listen_timeout_s,
            0,
        )

    def test_wait_functions_reject_nonfinite_timeouts(self) -> None:
        sock = mock.Mock()
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(function="generic", value=value):
                with self.assertRaises(ValueError):
                    bridge.wait_for_acks(sock, timeout_s=value)
            with self.subTest(function="inspection", value=value):
                with self.assertRaises(ValueError):
                    bridge.wait_for_session_clip_inspection_acks(
                        sock,
                        timeout_s=value,
                        request_id="req-timeout",
                    )

        with self.assertRaises(ValueError):
            bridge.wait_for_acks(
                sock,
                timeout_s=1,
                quiet_window_s=math.nan,
            )

    def test_session_clip_inspect_waits_for_assembly_instead_of_quiet_window(self) -> None:
        fragments = [
            _inspection_fragment(
                index=0,
                count=2,
                kind="context",
                request_id="req-wait",
                inspection_id="inspection-wait",
                data=_inspection_context_data(
                    note_count=1,
                    pitch_min=60,
                    pitch_max=60,
                ),
            ),
            _inspection_fragment(
                index=1,
                count=2,
                kind="note_page",
                request_id="req-wait",
                inspection_id="inspection-wait",
                data={
                    "note_offset": 0,
                    "note_count": 1,
                    "note_total": 1,
                    "notes": [_inspection_note(note_id=1, pitch=60)],
                },
            ),
        ]
        packets = [
            bridge.encode_osc_message(
                "/ack",
                ("api_session_clip_inspect", json.dumps(fragment), "req-wait"),
            )
            for fragment in fragments
        ]
        timeouts: list[float] = []

        class _FakeSock:
            def __init__(self) -> None:
                self.ready = False
                self.packet_index = 0

            def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
                if not self.ready:
                    raise BlockingIOError
                self.ready = False
                packet = packets[self.packet_index]
                self.packet_index += 1
                return packet, ("127.0.0.1", 9001)

        fake_sock = _FakeSock()

        def _fake_select(
            _r: object,
            _w: object,
            _e: object,
            timeout: float,
        ) -> tuple[list[object], list[object], list[object]]:
            timeouts.append(float(timeout))
            fake_sock.ready = True
            return [fake_sock], [], []

        with mock.patch("ableton_udp_bridge.select.select", side_effect=_fake_select):
            acks = bridge.wait_for_session_clip_inspection_acks(
                fake_sock,
                timeout_s=1.0,
                request_id="req-wait",
            )

        self.assertEqual(len(acks), 2)
        self.assertEqual(fake_sock.packet_index, 2)
        self.assertGreater(timeouts[1], 0.5)

    def test_session_clip_inspect_ack_collection_bounds_unrelated_packet_flood(self) -> None:
        unrelated = bridge.encode_osc_message("/ack", ("status",))
        complete = _complete_inspection_fragment()
        complete["correlation"]["request_id"] = "req-flood"
        success = bridge.encode_osc_message(
            "/ack",
            (
                "api_session_clip_inspect",
                json.dumps(complete),
                "req-flood",
            ),
        )
        packets = [unrelated] * 2000 + [success]

        class _FakeSock:
            def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
                if packets:
                    return packets.pop(0), ("127.0.0.1", 9001)
                raise BlockingIOError

        fake_sock = _FakeSock()
        with mock.patch(
            "ableton_udp_bridge.select.select",
            return_value=([fake_sock], [], []),
        ):
            acks = bridge.wait_for_session_clip_inspection_acks(
                fake_sock,
                timeout_s=1,
                request_id="req-flood",
            )

        self.assertEqual(packets, [])
        self.assertLessEqual(
            len(acks),
            bridge.SESSION_CLIP_INSPECTION_MAX_UNRELATED_ACKS + 1,
        )
        self.assertEqual(acks[-1][1][0], "api_session_clip_inspect")

    def test_session_clip_inspect_ack_collection_enforces_deadline_during_continuous_recv(self) -> None:
        unrelated = bridge.encode_osc_message("/ack", ("status",))
        clock = {"now": 0.0}

        class _FakeSock:
            def __init__(self) -> None:
                self.recv_calls = 0

            def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
                self.recv_calls += 1
                clock["now"] += 0.01
                if self.recv_calls > 100:
                    raise BlockingIOError
                return unrelated, ("127.0.0.1", 9001)

        fake_sock = _FakeSock()
        with (
            mock.patch(
                "ableton_udp_bridge.time.monotonic",
                side_effect=lambda: clock["now"],
            ),
            mock.patch(
                "ableton_udp_bridge.select.select",
                return_value=([fake_sock], [], []),
            ),
        ):
            acks = bridge.wait_for_session_clip_inspection_acks(
                fake_sock,
                timeout_s=0.05,
                request_id="req-deadline",
            )

        self.assertLessEqual(clock["now"], 0.07)
        self.assertLessEqual(fake_sock.recv_calls, 7)
        self.assertLessEqual(
            len(acks),
            bridge.SESSION_CLIP_INSPECTION_MAX_UNRELATED_ACKS,
        )

    def test_session_clip_inspect_ack_collection_completes_across_receive_batches(self) -> None:
        fragments = [
            _inspection_fragment(
                index=0,
                count=2,
                kind="context",
                request_id="req-batches",
                inspection_id="inspection-batches",
                data=_inspection_context_data(
                    note_count=1,
                    pitch_min=60,
                    pitch_max=60,
                ),
            ),
            _inspection_fragment(
                index=1,
                count=2,
                kind="note_page",
                request_id="req-batches",
                inspection_id="inspection-batches",
                data={
                    "note_offset": 0,
                    "note_count": 1,
                    "note_total": 1,
                    "notes": [_inspection_note()],
                },
            ),
        ]
        unrelated = bridge.encode_osc_message("/ack", ("status",))
        packets = (
            [unrelated] * 20
            + [
                bridge.encode_osc_message(
                    "/ack",
                    (
                        "api_session_clip_inspect",
                        json.dumps(fragments[0]),
                        "req-batches",
                    ),
                )
            ]
            + [unrelated] * 20
            + [
                bridge.encode_osc_message(
                    "/ack",
                    (
                        "api_session_clip_inspect",
                        json.dumps(fragments[1]),
                        "req-batches",
                    ),
                )
            ]
        )
        select_calls = 0

        class _FakeSock:
            def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
                if packets:
                    return packets.pop(0), ("127.0.0.1", 9001)
                raise BlockingIOError

        fake_sock = _FakeSock()

        def _fake_select(
            _r: object,
            _w: object,
            _e: object,
            _timeout: float,
        ) -> tuple[list[object], list[object], list[object]]:
            nonlocal select_calls
            select_calls += 1
            return [fake_sock], [], []

        with mock.patch(
            "ableton_udp_bridge.select.select",
            side_effect=_fake_select,
        ):
            acks = bridge.wait_for_session_clip_inspection_acks(
                fake_sock,
                timeout_s=1,
                request_id="req-batches",
            )

        self.assertEqual(packets, [])
        self.assertGreaterEqual(select_calls, 2)
        self.assertEqual(
            [
                args[0]
                for _address, args in acks
                if args and args[0] == "api_session_clip_inspect"
            ],
            ["api_session_clip_inspect", "api_session_clip_inspect"],
        )

    def test_send_commands_uses_completion_aware_collection_for_session_clip_inspect(self) -> None:
        cfg = bridge.parse_args(
            _base_args()
            + [
                "--api-session-clip-inspect",
                "2",
                "3",
                "req-send",
            ]
        )
        command = next(
            command
            for command in bridge.build_commands(cfg)
            if command.address == "/api/session_clip_inspect"
        )
        calls: list[tuple[object, ...]] = []

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

        with (
            mock.patch("ableton_udp_bridge.open_ack_socket", return_value=_FakeAckSock()),
            mock.patch("ableton_udp_bridge.socket.socket", return_value=_FakeSendSock()),
            mock.patch("ableton_udp_bridge._drain_acks_nonblocking", return_value=[]),
            mock.patch(
                "ableton_udp_bridge._collect_and_print_session_clip_inspection_acks",
                side_effect=lambda *args: calls.append(args),
            ),
        ):
            bridge.send_commands(cfg, [command])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], "req-send")

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
