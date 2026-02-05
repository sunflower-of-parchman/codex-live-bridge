"""Business logic for command execution in the Live bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import datetime as dt

from .adapters import LiveAdapter
from .capabilities import CAPABILITIES
from .protocol import ProtocolError, parse_envelope


@dataclass
class BridgeService:
    adapter: LiveAdapter
    bind_host: str
    bind_port: int

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "service": "codex-live-bridge",
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "bind": {"host": self.bind_host, "port": self.bind_port},
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"ok": True, "commands": CAPABILITIES}

    def execute_command(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        envelope = parse_envelope(raw)
        result = self.adapter.execute(
            command_id=envelope.command_id,
            command=envelope.command,
            payload=envelope.payload,
        )
        return {"ok": True, "id": envelope.command_id, "command": envelope.command, "result": result}

    @staticmethod
    def protocol_error_payload(exc: ProtocolError) -> Dict[str, Any]:
        return {"ok": False, "error_type": "protocol_error", "error": str(exc)}

