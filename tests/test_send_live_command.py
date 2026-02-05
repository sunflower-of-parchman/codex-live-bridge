import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "send_live_command.py"


def load_module():
    spec = importlib.util.spec_from_file_location("send_live_command", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load send_live_command module")
    spec.loader.exec_module(module)
    return module


send_live_command = load_module()


class _EchoHandler(BaseHTTPRequestHandler):
    last_path = None
    last_body = None

    def do_POST(self):  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        _EchoHandler.last_path = self.path
        _EchoHandler.last_body = body
        response = {"ok": True, "echo": json.loads(body)}
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt, *args):  # noqa: D401
        return


class SendLiveCommandTests(unittest.TestCase):
    def test_parse_payload_requires_object(self):
        with self.assertRaises(ValueError):
            send_live_command.parse_payload("[]")

    def test_command_endpoint_appends_command_path(self):
        self.assertEqual(
            send_live_command.command_endpoint("http://127.0.0.1:9000/"),
            "http://127.0.0.1:9000/command",
        )

    def test_build_envelope_includes_optional_id(self):
        envelope = send_live_command.build_envelope(
            command="set_tempo",
            payload={"bpm": 120},
            command_id="abc",
        )
        self.assertEqual(envelope["id"], "abc")
        self.assertEqual(envelope["command"], "set_tempo")

    def test_send_envelope_posts_json_to_command_endpoint(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            response = send_live_command.send_envelope(
                url=f"http://127.0.0.1:{port}",
                envelope={"command": "set_tempo", "payload": {"bpm": 126}},
                timeout_s=2.0,
            )
            self.assertTrue(response["ok"])
            self.assertEqual(_EchoHandler.last_path, "/command")
            body = json.loads(_EchoHandler.last_body)
            self.assertEqual(body["command"], "set_tempo")
            self.assertEqual(body["payload"]["bpm"], 126)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
