import unittest

from live_bridge.protocol import ProtocolError, parse_envelope, supported_commands


class LiveBridgeProtocolTests(unittest.TestCase):
    def test_supported_commands_include_core_controls(self):
        commands = supported_commands()
        self.assertIn("note_insert", commands)
        self.assertIn("create_midi_clip", commands)
        self.assertIn("fire_clip", commands)
        self.assertIn("stop_track", commands)
        self.assertIn("create_automation", commands)
        self.assertIn("set_track_volume", commands)
        self.assertIn("set_track_mute", commands)
        self.assertIn("set_track_solo", commands)
        self.assertIn("set_tempo", commands)
        self.assertIn("set_global_key", commands)
        self.assertIn("get_track_count", commands)
        self.assertIn("get_tempo", commands)

    def test_valid_note_insert_envelope(self):
        envelope = parse_envelope(
            {
                "id": "abc123",
                "command": "note_insert",
                "payload": {
                    "track_index": 0,
                    "clip_slot_index": 1,
                    "notes": [
                        {
                            "pitch": 60,
                            "start_time": 0.0,
                            "duration": 1.0,
                            "velocity": 100,
                            "mute": False,
                        }
                    ],
                },
            }
        )
        self.assertEqual(envelope.command, "note_insert")
        self.assertEqual(envelope.payload["track_index"], 0)

    def test_automation_requires_clip_slot_index(self):
        with self.assertRaises(ProtocolError):
            parse_envelope(
                {
                    "command": "create_automation",
                    "payload": {
                        "track_index": 0,
                        "device_index": 1,
                        "parameter_index": 2,
                        "points": [{"time": 0.0, "value": 0.5}],
                    },
                }
            )

    def test_eq8_band_range_is_validated(self):
        with self.assertRaises(ProtocolError):
            parse_envelope(
                {
                    "command": "set_eq8_band_gain",
                    "payload": {
                        "track_index": 0,
                        "device_index": 0,
                        "band": 9,
                        "gain": 0.7,
                    },
                }
            )

    def test_global_key_range_is_validated(self):
        with self.assertRaises(ProtocolError):
            parse_envelope(
                {
                    "command": "set_global_key",
                    "payload": {"root_note": 12, "scale_name": "Major"},
                }
            )

    def test_create_midi_clip_requires_positive_length(self):
        with self.assertRaises(ProtocolError):
            parse_envelope(
                {
                    "command": "create_midi_clip",
                    "payload": {
                        "track_index": 0,
                        "clip_slot_index": 1,
                        "length_beats": 0,
                    },
                }
            )

    def test_set_track_mute_requires_boolean(self):
        with self.assertRaises(ProtocolError):
            parse_envelope(
                {
                    "command": "set_track_mute",
                    "payload": {"track_index": 0, "value": 1},
                }
            )

    def test_get_track_count_accepts_empty_payload(self):
        envelope = parse_envelope(
            {
                "command": "get_track_count",
                "payload": {},
            }
        )
        self.assertEqual(envelope.command, "get_track_count")

    def test_get_track_count_rejects_extra_payload_fields(self):
        with self.assertRaises(ProtocolError):
            parse_envelope(
                {
                    "command": "get_track_count",
                    "payload": {"foo": "bar"},
                }
            )

    def test_get_tempo_accepts_empty_payload(self):
        envelope = parse_envelope(
            {
                "command": "get_tempo",
                "payload": {},
            }
        )
        self.assertEqual(envelope.command, "get_tempo")

    def test_get_tempo_rejects_extra_payload_fields(self):
        with self.assertRaises(ProtocolError):
            parse_envelope(
                {
                    "command": "get_tempo",
                    "payload": {"foo": "bar"},
                }
            )


if __name__ == "__main__":
    unittest.main()
