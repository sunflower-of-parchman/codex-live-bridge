import json
import socket
import threading
import unittest

from live_bridge.adapters import UdpMaxProxyAdapter


class UdpMaxProxyAdapterTests(unittest.TestCase):
    def test_non_query_command_forwards_udp_envelope(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as recv_sock:
            recv_sock.bind(("127.0.0.1", 0))
            recv_sock.settimeout(1.0)
            command_port = recv_sock.getsockname()[1]

            adapter = UdpMaxProxyAdapter(host="127.0.0.1", port=command_port)
            result = adapter.execute("cmd-1", "set_track_mute", {"track_index": 0, "value": True})

            raw, _addr = recv_sock.recvfrom(65535)
            parsed = json.loads(raw.decode("utf-8"))

        self.assertEqual(parsed["id"], "cmd-1")
        self.assertEqual(parsed["command"], "set_track_mute")
        self.assertEqual(parsed["payload"]["track_index"], 0)
        self.assertTrue(parsed["payload"]["value"])
        self.assertEqual(result["status"], "forwarded")

    def test_set_tempo_forwards_udp_envelope(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as recv_sock:
            recv_sock.bind(("127.0.0.1", 0))
            recv_sock.settimeout(1.0)
            command_port = recv_sock.getsockname()[1]

            adapter = UdpMaxProxyAdapter(host="127.0.0.1", port=command_port)
            result = adapter.execute("cmd-tempo", "set_tempo", {"bpm": 121})

            raw, _addr = recv_sock.recvfrom(65535)
            parsed = json.loads(raw.decode("utf-8"))

        self.assertEqual(parsed["command"], "set_tempo")
        self.assertEqual(parsed["payload"]["bpm"], 121)
        self.assertEqual(result["status"], "forwarded")

    def test_query_command_waits_for_matching_udp_response(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as cmd_sock:
            cmd_sock.bind(("127.0.0.1", 0))
            cmd_sock.settimeout(1.0)
            command_port = cmd_sock.getsockname()[1]

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.bind(("127.0.0.1", 0))
                response_port = probe.getsockname()[1]

            def simulate_max_router() -> None:
                raw, _addr = cmd_sock.recvfrom(65535)
                parsed = json.loads(raw.decode("utf-8"))
                response = {
                    "ok": True,
                    "id": parsed["id"],
                    "result": {"track_count": 7},
                }
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as send_sock:
                    send_sock.sendto(
                        json.dumps(response, ensure_ascii=True).encode("utf-8"),
                        ("127.0.0.1", response_port),
                    )

            worker = threading.Thread(target=simulate_max_router, daemon=True)
            worker.start()
            adapter = UdpMaxProxyAdapter(
                host="127.0.0.1",
                port=command_port,
                response_host="127.0.0.1",
                response_port=response_port,
                response_timeout_s=1.0,
            )
            result = adapter.execute("cmd-q", "get_track_count", {})
            worker.join(timeout=1.0)

        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["response"]["track_count"], 7)


if __name__ == "__main__":
    unittest.main()
