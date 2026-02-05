"""Adapter implementations for forwarding bridge commands to Ableton-side handlers."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Tuple


class LiveAdapter(Protocol):
    def execute(self, command_id: str, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class MockLiveAdapter:
    """In-memory adapter for tests and local dry-runs."""

    history: List[Dict[str, Any]] = field(default_factory=list)

    def execute(self, command_id: str, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = {
            "id": command_id,
            "command": command,
            "payload": payload,
        }
        self.history.append(event)
        return {"status": "accepted", "backend": "mock", "echo": event}


@dataclass
class UdpMaxProxyAdapter:
    """Forwards validated command envelopes to a Max for Live UDP listener."""

    host: str = "127.0.0.1"
    port: int = 9000
    timeout_s: float = 0.25
    response_host: str = "127.0.0.1"
    response_port: int = 9002
    response_timeout_s: float = 1.0
    query_commands: Tuple[str, ...] = ("set_tempo", "get_track_count", "get_tempo")

    def _target(self) -> str:
        return f"udp://{self.host}:{self.port}"

    def _no_response_message(self, command: str) -> str:
        return (
            f"No UDP response for '{command}' on {self.response_host}:{self.response_port}. "
            "Confirm Max patch includes udpsend for bridge responses."
        )

    def _query_with_response(self, command_id: str, encoded: bytes) -> Dict[str, Any]:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as recv_sock:
            recv_sock.bind((self.response_host, self.response_port))
            recv_sock.settimeout(self.response_timeout_s)

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as send_sock:
                send_sock.settimeout(self.timeout_s)
                send_sock.sendto(encoded, (self.host, self.port))

            while True:
                raw, _addr = recv_sock.recvfrom(65535)
                parsed = json.loads(raw.decode("utf-8"))
                if parsed.get("id") != command_id:
                    continue
                if not parsed.get("ok", False):
                    raise RuntimeError(str(parsed.get("error", "Unknown Max router error.")))
                result = parsed.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("Query response from Max router must include an object 'result'.")
                return result

    def execute(self, command_id: str, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        message = {"id": command_id, "command": command, "payload": payload}
        encoded = json.dumps(message, ensure_ascii=True).encode("utf-8")

        if command in self.query_commands:
            try:
                result = self._query_with_response(command_id, encoded)
            except TimeoutError as exc:  # pragma: no cover
                if command == "set_tempo":
                    return {
                        "status": "forwarded",
                        "backend": "udp-max-proxy",
                        "target": self._target(),
                        "bytes_sent": len(encoded),
                        "warning": self._no_response_message(command),
                    }
                raise RuntimeError(self._no_response_message(command)) from exc
            except socket.timeout as exc:
                if command == "set_tempo":
                    return {
                        "status": "forwarded",
                        "backend": "udp-max-proxy",
                        "target": self._target(),
                        "bytes_sent": len(encoded),
                        "warning": self._no_response_message(command),
                    }
                raise RuntimeError(self._no_response_message(command)) from exc
            return {
                "status": "executed",
                "backend": "udp-max-proxy",
                "target": self._target(),
                "bytes_sent": len(encoded),
                "response": result,
            }

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(self.timeout_s)
            sock.sendto(encoded, (self.host, self.port))

        return {
            "status": "forwarded",
            "backend": "udp-max-proxy",
            "target": self._target(),
            "bytes_sent": len(encoded),
        }
