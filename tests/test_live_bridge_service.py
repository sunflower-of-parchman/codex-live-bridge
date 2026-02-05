import unittest

from live_bridge.adapters import MockLiveAdapter
from live_bridge.protocol import ProtocolError
from live_bridge.service import BridgeService


class LiveBridgeServiceTests(unittest.TestCase):
    def setUp(self):
        self.adapter = MockLiveAdapter()
        self.service = BridgeService(adapter=self.adapter, bind_host="127.0.0.1", bind_port=9000)

    def test_health_reports_port_9000(self):
        health = self.service.health()
        self.assertTrue(health["ok"])
        self.assertEqual(health["bind"]["port"], 9000)

    def test_capabilities_include_commands(self):
        capabilities = self.service.capabilities()
        self.assertTrue(capabilities["ok"])
        self.assertGreater(len(capabilities["commands"]), 0)

    def test_execute_command_routes_to_adapter(self):
        response = self.service.execute_command(
            {
                "id": "cmd-1",
                "command": "set_tempo",
                "payload": {"bpm": 122.0},
            }
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["id"], "cmd-1")
        self.assertEqual(response["command"], "set_tempo")
        self.assertEqual(response["result"]["backend"], "mock")
        self.assertEqual(len(self.adapter.history), 1)

    def test_unsupported_command_raises_protocol_error(self):
        with self.assertRaises(ProtocolError):
            self.service.execute_command(
                {"command": "not_real", "payload": {}}
            )


if __name__ == "__main__":
    unittest.main()
