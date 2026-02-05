#!/usr/bin/env python3
"""Send one command envelope to the local Live bridge server."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional
from urllib import error, request


def parse_payload(payload_text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Payload JSON must decode to an object.")
    return parsed


def command_endpoint(base_url: str) -> str:
    clean = base_url.rstrip("/")
    return f"{clean}/command"


def build_envelope(command: str, payload: Dict[str, Any], command_id: Optional[str] = None) -> Dict[str, Any]:
    envelope: Dict[str, Any] = {"command": command, "payload": payload}
    if command_id:
        envelope["id"] = command_id
    return envelope


def send_envelope(url: str, envelope: Dict[str, Any], timeout_s: float = 3.0) -> Dict[str, Any]:
    body = json.dumps(envelope, ensure_ascii=True).encode("utf-8")
    req = request.Request(
        command_endpoint(url),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8")
        raise RuntimeError(f"Bridge returned HTTP {exc.code}: {text}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach bridge at {url}: {exc.reason}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Bridge response was not valid JSON: {text}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one command to the Live bridge.")
    parser.add_argument("--url", default="http://127.0.0.1:9000", help="Base bridge URL.")
    parser.add_argument("--command", required=True, help="Command name (e.g., set_tempo).")
    parser.add_argument(
        "--payload",
        default="{}",
        help="JSON object payload as a string.",
    )
    parser.add_argument("--id", help="Optional command id.")
    parser.add_argument("--timeout", type=float, default=3.0, help="HTTP timeout in seconds.")
    args = parser.parse_args()

    try:
        payload = parse_payload(args.payload)
        envelope = build_envelope(command=args.command, payload=payload, command_id=args.id)
        response = send_envelope(url=args.url, envelope=envelope, timeout_s=args.timeout)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        return 1

    print(json.dumps(response, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
