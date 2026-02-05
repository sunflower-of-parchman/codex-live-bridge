import unittest

from live_bridge.protocol import ProtocolError, parse_envelope, supported_commands


class LiveBridgeProtocolTests(unittest.TestCase):
    def test_supported_commands_include_core_controls(self):
        commands = supported_commands()
        self.assertIn("note_insert", commands)
        self.assertIn("create_automation", commands)
        self.assertIn("set_track_volume", commands)
        self.assertIn("set_tempo", commands)
        self.assertIn("set_global_key", commands)

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


if __name__ == "__main__":
    unittest.main()
