#!/usr/bin/env python3
"""Project-agnostic, read-only Arrangement inspection regressions."""

from __future__ import annotations

import copy
import io
import json
import unittest
from unittest import mock

import ableton_udp_bridge as bridge
from test_ableton_udp_bridge import _base_args, _run_bridge_js


def _note(index: int = 0) -> dict[str, object]:
    return {
        "note_id": 100 + index,
        "pitch": 48 + index % 36,
        "start_time": index * 0.25,
        "duration": 0.2,
        "velocity": 96,
        "mute": False,
        "probability": 1,
        "velocity_deviation": 0,
        "release_velocity": 64,
    }


def _fixture(
    *,
    notes: list[dict[str, object]] | None = None,
    devices: list[dict[str, object]] | None = None,
    clip_ids: list[int] | None = None,
) -> dict[str, object]:
    return {
        "tempo": 118,
        "signature_numerator": 3,
        "signature_denominator": 4,
        "track_count_override": None,
        "return_count_override": None,
        "clip_count_override": None,
        "device_count_override": None,
        "max_fragments": None,
        "max_payload_bytes": None,
        "clip_ids": clip_ids or [301, 301],
        "tracks": [
            {
                "name": "Reusable MIDI track",
                "midi": True,
                "clips": [
                    {
                        "name": "Arrangement phrase",
                        "start_time": 8,
                        "end_time": 16,
                        "notes": notes if notes is not None else [_note()],
                    }
                ],
                "devices": devices
                if devices is not None
                else [
                    {
                        "name": "Reusable Instrument",
                        "class_name": "InstrumentGroupDevice",
                        "type": 1,
                    }
                ],
            },
            {
                "name": "Reusable audio track",
                "midi": False,
                "clips": [
                    {
                        "name": "Audio phrase",
                        "start_time": -2,
                        "end_time": 4,
                        "notes": [],
                    }
                ],
                "devices": [],
            },
        ],
        "returns": [{"name": "Shared delay", "device_count": 2}],
        "main": {"name": "Main", "device_count": 1},
    }


def _run_arrangement_js(
    *,
    scope: str = "clip",
    track_index: int = 0,
    clip_index: int = 0,
    include_notes: bool = False,
    request_id: str = "req-arrangement",
    fixture: dict[str, object] | None = None,
) -> dict[str, object]:
    values = fixture or _fixture()
    command = {
        "project": f"context.api_arrangement_project_inspect(1, {json.dumps(request_id)});",
        "track": (
            f"context.api_arrangement_track_inspect({track_index}, 1, "
            f"{json.dumps(request_id)});"
        ),
        "clip": (
            f"context.api_arrangement_clip_inspect({track_index}, {clip_index}, "
            f"{int(include_notes)}, 1, {json.dumps(request_id)});"
        ),
    }[scope]
    result = _run_bridge_js(
        f"""
const fixture = {json.dumps(values)};
const outputs = [];
let noteCalls = 0;
let mutationCalls = 0;
let firstClipOpens = 0;
context.outlet = (...args) => outputs.push(args);
context.ensureInitialized = () => true;
if (fixture.max_fragments !== null) {{
  context.ARRANGEMENT_INSPECTION_MAX_FRAGMENTS = fixture.max_fragments;
}}
if (fixture.max_payload_bytes !== null) {{
  context.ARRANGEMENT_INSPECTION_MAX_PAYLOAD_BYTES = fixture.max_payload_bytes;
}}
context.song = {{
  get: (property) => fixture[property],
  getcount: (child) => {{
    if (child === "tracks") {{
      return fixture.track_count_override === null
        ? fixture.tracks.length : fixture.track_count_override;
    }}
    if (child === "return_tracks") {{
      return fixture.return_count_override === null
        ? fixture.returns.length : fixture.return_count_override;
    }}
    throw new Error("unknown root child " + child);
  }},
  set: () => {{ mutationCalls += 1; throw new Error("unexpected mutation"); }},
  call: () => {{ mutationCalls += 1; throw new Error("unexpected mutation"); }},
}};
context.LiveAPI = function LiveAPI(_callback, rawPath) {{
  const path = String(rawPath);
  const trackMatch = path.match(/^live_set tracks (\\d+)$/);
  if (trackMatch) {{
    const index = Number(trackMatch[1]);
    const track = fixture.tracks[index];
    if (!track) return {{ id: 0, path }};
    return {{
      id: 100 + index,
      path,
      get: (property) => {{
        if (property === "name") return track.name;
        if (property === "has_midi_input") return Number(track.midi);
        if (property === "has_audio_input") return Number(!track.midi);
        if (property === "mute" || property === "solo") return 0;
        throw new Error("unknown track property " + property);
      }},
      getcount: (child) => {{
        if (child === "arrangement_clips") {{
          return fixture.clip_count_override === null
            ? track.clips.length : fixture.clip_count_override;
        }}
        if (child === "devices") {{
          return fixture.device_count_override === null
            ? track.devices.length : fixture.device_count_override;
        }}
        throw new Error("unknown track child " + child);
      }},
      set: () => {{ mutationCalls += 1; throw new Error("unexpected mutation"); }},
    }};
  }}
  const clipMatch = path.match(/^live_set tracks (\\d+) arrangement_clips (\\d+)$/);
  if (clipMatch) {{
    const track = fixture.tracks[Number(clipMatch[1])];
    const index = Number(clipMatch[2]);
    const clip = track && track.clips[index];
    if (!clip) return {{ id: 0, path }};
    const first = Number(clipMatch[1]) === 0 && index === 0;
    const id = first
      ? fixture.clip_ids[Math.min(firstClipOpens++, fixture.clip_ids.length - 1)]
      : 301 + Number(clipMatch[1]) * 20 + index;
    return {{
      id,
      path,
      get: (property) => {{
        const defaults = {{
          name: clip.name,
          start_time: clip.start_time,
          end_time: clip.end_time,
          start_marker: 0,
          end_marker: clip.end_time - clip.start_time,
          length: clip.end_time - clip.start_time,
          looping: 1,
          loop_start: 0,
          loop_end: clip.end_time - clip.start_time,
          is_midi_clip: Number(track.midi),
          is_audio_clip: Number(!track.midi),
        }};
        if (Object.prototype.hasOwnProperty.call(defaults, property)) return defaults[property];
        throw new Error("unknown clip property " + property);
      }},
      call: (method, payload) => {{
        noteCalls += 1;
        if (method === "get_all_notes_extended") {{
          return JSON.stringify({{
            notes: clip.notes.map((note) => {{
              const projected = {{}};
              payload.return.forEach((field) => {{ projected[field] = note[field]; }});
              return projected;
            }}),
          }});
        }}
        if (method === "get_notes_by_id") {{
          const requested = new Set(payload.note_ids);
          return JSON.stringify({{ notes: clip.notes.filter((note) => requested.has(note.note_id)) }});
        }}
        mutationCalls += 1;
        throw new Error("unexpected mutation method " + method);
      }},
      set: () => {{ mutationCalls += 1; throw new Error("unexpected mutation"); }},
    }};
  }}
  const deviceMatch = path.match(/^live_set tracks (\\d+) devices (\\d+)$/);
  if (deviceMatch) {{
    const device = fixture.tracks[Number(deviceMatch[1])].devices[Number(deviceMatch[2])];
    if (!device) return {{ id: 0, path }};
    return {{
      id: 400 + Number(deviceMatch[2]),
      path,
      get: (property) => device[property],
    }};
  }}
  const returnMatch = path.match(/^live_set return_tracks (\\d+)$/);
  if (returnMatch || path === "live_set master_track") {{
    const index = returnMatch ? Number(returnMatch[1]) : null;
    const track = returnMatch ? fixture.returns[index] : fixture.main;
    if (!track) return {{ id: 0, path }};
    return {{
      id: index === null ? 900 : 800 + index,
      path,
      get: (property) => property === "name" ? track.name : 0,
      getcount: (child) => {{
        if (child === "devices") return track.device_count;
        throw new Error("unknown auxiliary track child " + child);
      }},
    }};
  }}
  return {{ id: 0, path }};
}};
{command}
return {{
  acks: outputs.filter((args) => args[1] === "/ack"),
  noteCalls,
  mutationCalls,
}};
"""
    )
    assert isinstance(result, dict)
    return result


def _assemble(result: dict[str, object]) -> dict[str, object]:
    assembler = bridge.ArrangementInspectionAssembler()
    completed = None
    for output in result["acks"]:
        assert isinstance(output, list)
        event = bridge.parse_ack_event("/ack", output[2:])
        completed = assembler.add_event(event)
    assert isinstance(completed, dict)
    return completed


class ArrangementInspectionTests(unittest.TestCase):
    def test_cli_builds_tokenless_project_track_and_explicit_note_commands(self) -> None:
        cfg = bridge.parse_args(
            _base_args()
            + [
                "--api-arrangement-project-inspect", "req-project",
                "--api-arrangement-track-inspect", "2", "req-track",
                "--api-arrangement-clip-inspect", "2", "3", "req-private",
                "--api-arrangement-clip-inspect-notes", "4", "5", "req-notes",
            ]
        )
        commands = [
            command for command in bridge.build_commands(cfg)
            if "arrangement_" in command.address
        ]
        self.assertEqual(
            commands,
            [
                bridge.OscCommand("/api/arrangement_project_inspect", (1, "req-project")),
                bridge.OscCommand("/api/arrangement_track_inspect", (2, 1, "req-track")),
                bridge.OscCommand("/api/arrangement_clip_inspect", (2, 3, 0, 1, "req-private")),
                bridge.OscCommand("/api/arrangement_clip_inspect", (4, 5, 1, 1, "req-notes")),
            ],
        )
        self.assertTrue(
            all(command.address not in bridge.PROTECTED_OSC_ADDRESSES for command in commands)
        )

    def test_cli_rejects_negative_indexes_missing_and_oversized_request_ids(self) -> None:
        invalid = [
            ["--api-arrangement-project-inspect", ""],
            ["--api-arrangement-project-inspect", "é" * 65],
            ["--api-arrangement-track-inspect", "-1", "req"],
            ["--api-arrangement-clip-inspect", "0", "-1", "req"],
            ["--api-arrangement-clip-inspect-notes", "not-an-index", "0", "req"],
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                    bridge.parse_args(_base_args() + arguments)

    def test_project_reports_tracks_returns_main_tempo_without_reading_notes(self) -> None:
        result = _run_arrangement_js(scope="project")
        inspection = _assemble(result)
        payload = inspection["data"]
        self.assertEqual(payload["project"], {
            "tempo": 118,
            "signature_numerator": 3,
            "signature_denominator": 4,
            "track_count": 2,
            "return_track_count": 1,
        })
        self.assertEqual([track["name"] for track in payload["tracks"]], [
            "Reusable MIDI track", "Reusable audio track"
        ])
        self.assertEqual(payload["return_tracks"][0]["name"], "Shared delay")
        self.assertEqual(payload["main_track"]["name"], "Main")
        self.assertEqual(result["noteCalls"], 0)
        self.assertEqual(result["mutationCalls"], 0)

    def test_track_reports_arrangement_clips_and_devices_without_note_calls(self) -> None:
        result = _run_arrangement_js(scope="track", track_index=0)
        payload = _assemble(result)["data"]
        self.assertEqual(payload["track"]["name"], "Reusable MIDI track")
        self.assertEqual(payload["clips"][0]["start_time"], 8)
        self.assertEqual(payload["devices"][0]["class_name"], "InstrumentGroupDevice")
        self.assertNotIn("notes", payload)
        self.assertEqual(result["noteCalls"], 0)
        self.assertEqual(result["mutationCalls"], 0)

    def test_clip_metadata_defaults_to_no_note_retrieval_or_raw_note_output(self) -> None:
        result = _run_arrangement_js()
        payload = _assemble(result)["data"]
        self.assertEqual(payload["clip"]["name"], "Arrangement phrase")
        self.assertEqual(payload["privacy"], {
            "notes_requested": False, "notes_included": False
        })
        self.assertNotIn("notes", payload)
        self.assertNotIn("summary", payload)
        self.assertEqual(result["noteCalls"], 0)
        self.assertEqual(result["mutationCalls"], 0)
        self.assertNotIn("pitch", json.dumps(result["acks"]))

    def test_clip_raw_notes_require_explicit_opt_in_and_stay_out_of_cli_summary(self) -> None:
        expected_notes = [_note(0), _note(1)]
        result = _run_arrangement_js(include_notes=True, fixture=_fixture(notes=expected_notes))
        inspection = _assemble(result)
        payload = inspection["data"]
        self.assertEqual(payload["notes"], expected_notes)
        self.assertEqual(payload["summary"], {
            "note_count": 2, "pitch_min": 48, "pitch_max": 49
        })
        self.assertEqual(payload["privacy"], {
            "notes_requested": True, "notes_included": True
        })
        self.assertGreater(result["noteCalls"], 0)
        self.assertEqual(result["mutationCalls"], 0)
        for output in result["acks"]:
            lines = bridge.summarize_ack("/ack", output[2:])
            self.assertNotIn("pitch", " ".join(lines))
            self.assertNotIn("note_id", " ".join(lines))

    def test_audio_clip_rejects_explicit_midi_note_requests_without_reading(self) -> None:
        result = _run_arrangement_js(track_index=1, include_notes=True)
        self.assertEqual(result["acks"][0][2:4], [
            "error", "api_arrangement_clip_inspect_not_midi"
        ])
        self.assertEqual(result["noteCalls"], 0)

    def test_multifragment_unicode_payload_is_packet_bounded_and_reassembles(self) -> None:
        devices = [
            {
                "name": f"Instrument {index} " + "🎹" * 35,
                "class_name": "ReusableInstrument",
                "type": 1,
            }
            for index in range(24)
        ]
        notes = [_note(index) for index in range(120)]
        result = _run_arrangement_js(
            include_notes=True,
            fixture=_fixture(notes=notes, devices=devices),
        )
        outputs = result["acks"]
        self.assertGreater(len(outputs), 4)
        for output in outputs:
            packet = bridge.encode_osc_message(
                "/ack", (str(output[2]), str(output[3]), str(output[4]))
            )
            self.assertLessEqual(len(packet), bridge.ARRANGEMENT_INSPECTION_PACKET_BUDGET_BYTES)

        assembler = bridge.ArrangementInspectionAssembler()
        completed = None
        for output in reversed(outputs):
            event = bridge.parse_ack_event("/ack", output[2:])
            completed = assembler.add_event(event)
        self.assertIsNotNone(completed)
        assert completed is not None
        self.assertEqual(completed["data"]["devices"], [
            {
                "index": index,
                "path": f"live_set tracks 0 devices {index}",
                "id": 400 + index,
                "name": item["name"],
                "class_name": item["class_name"],
                "type": item["type"],
            }
            for index, item in enumerate(devices)
        ])
        self.assertEqual(completed["data"]["notes"], notes)
        self.assertEqual(completed["transport"]["fragment_count"], len(outputs))

    def test_inventory_limits_fail_closed_before_expensive_reads(self) -> None:
        cases = [
            ("project", "track_count_override", 257, "tracks"),
            ("project", "return_count_override", 257, "return_tracks"),
            ("track", "clip_count_override", 257, "arrangement_clips"),
            ("clip", "device_count_override", 257, "devices"),
        ]
        for scope, field, count, child in cases:
            fixture = _fixture()
            fixture[field] = count
            result = _run_arrangement_js(scope=scope, fixture=fixture)
            with self.subTest(scope=scope, child=child):
                self.assertEqual(result["acks"][0][2:5], [
                    "error", f"api_arrangement_{scope}_inspect_limit_exceeded", child
                ])
                self.assertEqual(result["noteCalls"], 0)

    def test_explicit_note_inventory_ceiling_rejects_before_full_note_fetch(self) -> None:
        notes = [_note(index) for index in range(4097)]
        result = _run_arrangement_js(
            include_notes=True,
            fixture=_fixture(notes=notes, devices=[]),
        )
        self.assertEqual(result["acks"][0][2:7], [
            "error", "api_arrangement_clip_inspect_limit_exceeded", "notes", 4097, 4096
        ])
        self.assertEqual(result["noteCalls"], 1)

    def test_fragment_and_payload_limits_emit_only_correlated_bounded_errors(self) -> None:
        devices = [{
            "name": "large device " + "🎹" * 1000,
            "class_name": "Instrument",
            "type": 1,
        }]
        for limit_name, value, resource in (
            ("max_fragments", 1, "fragments"),
            ("max_payload_bytes", 64, "payload_bytes"),
        ):
            fixture = _fixture(devices=devices)
            fixture[limit_name] = value
            result = _run_arrangement_js(fixture=fixture)
            with self.subTest(resource=resource):
                self.assertEqual(len(result["acks"]), 1)
                output = result["acks"][0]
                self.assertEqual(output[2:5], [
                    "error", "api_arrangement_clip_inspect_limit_exceeded", resource
                ])
                packet = bridge.encode_osc_message("/ack", tuple(output[2:]))
                self.assertLessEqual(len(packet), 4096)

    def test_clip_snapshot_changes_fail_without_emitting_partial_fragments(self) -> None:
        result = _run_arrangement_js(fixture=_fixture(clip_ids=[301, 302]))
        self.assertEqual(len(result["acks"]), 1)
        self.assertEqual(result["acks"][0][2:4], [
            "error", "api_arrangement_clip_inspect_snapshot_changed"
        ])

    def test_js_validates_request_indexes_schema_and_explicit_note_flag(self) -> None:
        result = _run_bridge_js(
            """
const outputs = [];
context.outlet = (...args) => outputs.push(args);
context.ensureInitialized = () => true;
context.api_arrangement_project_inspect(1, "");
context.api_arrangement_project_inspect(2, "req-version");
context.api_arrangement_project_inspect(1, "é".repeat(65));
context.api_arrangement_track_inspect(-1, 1, "req-track");
context.api_arrangement_clip_inspect(0, -1, 0, 1, "req-clip");
context.api_arrangement_clip_inspect(0, 0, 2, 1, "req-notes");
context.api_arrangement_clip_inspect(0, 0, "1", 1, "req-string");
return outputs.filter((args) => args[1] === "/ack");
"""
        )
        self.assertEqual(len(result), 7)
        self.assertTrue(
            all(str(output[3]).endswith("_validation_failed") for output in result)
        )
        self.assertTrue(all(output[-2] == "request_correlation" for output in result))
        for output in result:
            packet = bridge.encode_osc_message("/ack", tuple(output[2:]))
            self.assertLessEqual(len(packet), 4096)

    def test_fallback_dispatch_explicitly_allows_only_bounded_read_handlers(self) -> None:
        result = _run_bridge_js(
            """
const calls = [];
context.arrayfromargs = (args) => Array.from(args);
context.API_FALLBACK_HANDLERS.api_arrangement_project_inspect = (...args) => {
  calls.push(["project", ...args]);
};
context.API_FALLBACK_HANDLERS.api_arrangement_track_inspect = (...args) => {
  calls.push(["track", ...args]);
};
context.API_FALLBACK_HANDLERS.api_arrangement_clip_inspect = (...args) => {
  calls.push(["clip", ...args]);
};
context.osc_dispatch("/api/arrangement_project_inspect", 1, "req-project");
context.osc_dispatch("/api/arrangement_track_inspect", 0, 1, "req-track");
context.osc_dispatch("/api/arrangement_clip_inspect", 0, 0, 0, 1, "req-clip");
context.osc_dispatch("/api/arrangement_mutate", "req-invalid");
return calls;
"""
        )
        self.assertEqual(result, [
            ["project", 1, "req-project"],
            ["track", 0, 1, "req-track"],
            ["clip", 0, 0, 0, 1, "req-clip"],
        ])

    def test_assembler_rejects_unknown_keys_wrong_request_and_smuggled_notes(self) -> None:
        output = _run_arrangement_js()["acks"][0]
        original = json.loads(str(output[3]))

        unknown = copy.deepcopy(original)
        unknown["unexpected"] = "private"
        with self.assertRaisesRegex(bridge.ArrangementInspectionAssemblyError, "root"):
            bridge.ArrangementInspectionAssembler().add_fragment(unknown, "req-arrangement")

        with self.assertRaisesRegex(bridge.ArrangementInspectionAssemblyError, "mismatch"):
            bridge.ArrangementInspectionAssembler().add_fragment(original, "req-other")

        smuggled = copy.deepcopy(original)
        payload = json.loads(smuggled["data"]["payload_chunk"])
        payload["notes"] = [_note()]
        smuggled["data"]["payload_chunk"] = json.dumps(payload, separators=(",", ":"))
        with self.assertRaisesRegex(bridge.ArrangementInspectionAssemblyError, "privacy"):
            bridge.ArrangementInspectionAssembler().add_fragment(smuggled, "req-arrangement")

        wrong_scope = copy.deepcopy(original)
        wrong_scope["correlation"]["scope"] = "track"
        with self.assertRaises(bridge.ArrangementInspectionAssemblyError):
            bridge.ArrangementInspectionAssembler().add_fragment(
                wrong_scope, "req-arrangement", "api_arrangement_clip_inspect"
            )

    def test_assembler_rejects_oversized_fragments_and_inconsistent_metadata(self) -> None:
        output = _run_arrangement_js()["acks"][0]
        original = json.loads(str(output[3]))
        oversized = copy.deepcopy(original)
        oversized["data"]["payload_chunk"] = "x" * 4097
        with self.assertRaisesRegex(bridge.ArrangementInspectionAssemblyError, "fragment bytes"):
            bridge.ArrangementInspectionAssembler().add_fragment(oversized, "req-arrangement")

        wrong_budget = copy.deepcopy(original)
        wrong_budget["transfer"]["packet_budget_bytes"] = 8192
        with self.assertRaisesRegex(bridge.ArrangementInspectionAssemblyError, "packet_budget"):
            bridge.ArrangementInspectionAssembler().add_fragment(wrong_budget, "req-arrangement")

        fractional_schema = copy.deepcopy(original)
        fractional_schema["schema_version"] = 1.0
        with self.assertRaisesRegex(bridge.ArrangementInspectionAssemblyError, "schema"):
            bridge.ArrangementInspectionAssembler().add_fragment(
                fractional_schema, "req-arrangement"
            )

    def test_send_commands_waits_for_complete_arrangement_transfer(self) -> None:
        cfg = bridge.parse_args(
            _base_args()
            + ["--api-arrangement-clip-inspect", "0", "0", "req-send"]
        )
        command = next(
            command for command in bridge.build_commands(cfg)
            if command.address == "/api/arrangement_clip_inspect"
        )
        ack_sock = mock.Mock()
        send_sock = mock.MagicMock()
        send_sock.__enter__.return_value = send_sock
        with (
            mock.patch("ableton_udp_bridge.open_ack_socket", return_value=ack_sock),
            mock.patch("ableton_udp_bridge.socket.socket", return_value=send_sock),
            mock.patch("ableton_udp_bridge._drain_acks_nonblocking", return_value=[]),
            mock.patch("sys.stdout", new=io.StringIO()),
            mock.patch(
                "ableton_udp_bridge._collect_and_print_arrangement_inspection_acks"
            ) as collector,
        ):
            bridge.send_commands(cfg, [command])
        collector.assert_called_once()
        self.assertEqual(collector.call_args.args[2:4], (
            "req-send", "api_arrangement_clip_inspect"
        ))


if __name__ == "__main__":
    unittest.main()
