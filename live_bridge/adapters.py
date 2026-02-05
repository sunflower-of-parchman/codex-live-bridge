"""Adapter implementations for forwarding bridge commands to Ableton-side handlers."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


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
    port: int = 9001
    timeout_s: float = 0.25

    def execute(self, command_id: str, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        message = {"id": command_id, "command": command, "payload": payload}
        encoded = json.dumps(message, ensure_ascii=True).encode("utf-8")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(self.timeout_s)
            sock.sendto(encoded, (self.host, self.port))

        return {
            "status": "forwarded",
            "backend": "udp-max-proxy",
            "target": f"udp://{self.host}:{self.port}",
            "bytes_sent": len(encoded),
        }

