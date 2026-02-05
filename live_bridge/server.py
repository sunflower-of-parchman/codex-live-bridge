"""HTTP server exposing the Live bridge on port 9000 by default."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from .adapters import MockLiveAdapter, UdpMaxProxyAdapter
from .protocol import ProtocolError
from .service import BridgeService


def _json_response(handler: BaseHTTPRequestHandler, code: int, body: Dict[str, Any]) -> None:
    encoded = json.dumps(body, ensure_ascii=True).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length)
    if not raw:
        raise ProtocolError("Request body cannot be empty.")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Request body must be valid JSON.") from exc
    if not isinstance(decoded, dict):
        raise ProtocolError("Request body must be a JSON object.")
    return decoded


def build_handler(service: BridgeService):
    class BridgeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                _json_response(self, 200, service.health())
                return
            if self.path == "/capabilities":
                _json_response(self, 200, service.capabilities())
                return
            _json_response(self, 404, {"ok": False, "error": "Route not found."})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/command":
                _json_response(self, 404, {"ok": False, "error": "Route not found."})
                return
            try:
                request_body = _read_json_body(self)
                response = service.execute_command(request_body)
                _json_response(self, 200, response)
            except ProtocolError as exc:
                _json_response(self, 400, service.protocol_error_payload(exc))
            except Exception as exc:  # pragma: no cover
                _json_response(
                    self,
                    500,
                    {"ok": False, "error_type": "server_error", "error": str(exc)},
                )

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep logs readable in terminal sessions.
            print(f"[live-bridge] {self.address_string()} - {fmt % args}")

    return BridgeHandler


def build_service(bind_host: str, bind_port: int, backend: str, udp_host: str, udp_port: int) -> BridgeService:
    if backend == "mock":
        adapter = MockLiveAdapter()
    elif backend == "udp-max-proxy":
        adapter = UdpMaxProxyAdapter(host=udp_host, port=udp_port)
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Use 'mock' or 'udp-max-proxy'.")
    return BridgeService(adapter=adapter, bind_host=bind_host, bind_port=bind_port)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Codex <-> Ableton Live bridge server.")
    parser.add_argument("--host", default=os.getenv("LIVE_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LIVE_BRIDGE_PORT", "9000")))
    parser.add_argument("--backend", default=os.getenv("LIVE_BRIDGE_BACKEND", "udp-max-proxy"))
    parser.add_argument("--udp-host", default=os.getenv("LIVE_BRIDGE_UDP_HOST", "127.0.0.1"))
    parser.add_argument("--udp-port", type=int, default=int(os.getenv("LIVE_BRIDGE_UDP_PORT", "9000")))
    return parser


def main() -> int:
    args = _parser().parse_args()
    service = build_service(
        bind_host=args.host,
        bind_port=args.port,
        backend=args.backend,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
    )
    handler_cls = build_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(
        f"Starting live bridge on http://{args.host}:{args.port} "
        f"(backend={args.backend}, udp_target={args.udp_host}:{args.udp_port})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down live bridge.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
