#!/usr/bin/env python3
"""Dump exposed Marimba device parameters from Live via the UDP bridge."""

from __future__ import annotations

import argparse
import json
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import ableton_udp_bridge as bridge


HOST = bridge.DEFAULT_HOST
PORT = bridge.DEFAULT_PORT
ACK_PORT = bridge.DEFAULT_ACK_PORT

OscAck = Tuple[str, List[bridge.OscArg]]


@dataclass(frozen=True)
class DumpConfig:
    track_name: str
    device_index: int
    device_name_hint: str | None
    max_parameters: int
    ack_timeout_s: float
    output_path: Path
    force_fallback: bool


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _default_output_path() -> Path:
    return Path("output/marimba") / f"params_{_utc_slug()}.json"


def parse_args(argv: Sequence[str] | None = None) -> DumpConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-name", default="Marimba", help="Track name to inspect")
    parser.add_argument("--device-index", type=int, default=0, help="Device index fallback")
    parser.add_argument("--device-name-hint", default=None, help="Optional device name hint")
    parser.add_argument("--max-parameters", type=int, default=256, help="Max parameters to query")
    parser.add_argument("--ack-timeout", type=float, default=0.8, help="ACK timeout in seconds")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Skip live bridge probing and write fallback artifact",
    )
    ns = parser.parse_args(argv)
    output = Path(ns.output) if ns.output else _default_output_path()
    return DumpConfig(
        track_name=str(ns.track_name),
        device_index=max(0, int(ns.device_index)),
        device_name_hint=None if ns.device_name_hint in (None, "") else str(ns.device_name_hint),
        max_parameters=max(1, int(ns.max_parameters)),
        ack_timeout_s=max(0.1, float(ns.ack_timeout)),
        output_path=output,
        force_fallback=bool(ns.force_fallback),
    )


def _send_and_collect_acks(
    sock: socket.socket,
    ack_sock: socket.socket,
    command: bridge.OscCommand,
    timeout_s: float,
) -> List[OscAck]:
    payload = bridge.encode_osc_message(command.address, command.args)
    sock.sendto(payload, (HOST, PORT))
    return bridge.wait_for_acks(ack_sock, timeout_s)


def _extract_api_children(acks: Sequence[OscAck], request_id: str) -> list[dict[str, Any]] | None:
    for address, args in acks:
        if address != "/ack" or len(args) < 4:
            continue
        if args[0] != "api_children":
            continue
        if str(args[-1]) != request_id:
            continue
        payload = args[3]
        if not isinstance(payload, str):
            return []
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []
    return None


def _extract_api_get(acks: Sequence[OscAck], request_id: str) -> Any:
    for address, args in acks:
        if address != "/ack" or len(args) < 4:
            continue
        if args[0] != "api_get":
            continue
        if str(args[-1]) != request_id:
            continue
        raw = args[3]
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return None


def _extract_error(acks: Sequence[OscAck]) -> str | None:
    for address, args in acks:
        if address != "/ack" or not args:
            continue
        if args[0] == "error" and len(args) > 1:
            return " ".join(str(x) for x in args[1:])
        if args[0] == "json" and len(args) > 1 and isinstance(args[1], str):
            # JSON router compatibility: /ack json {"ok":false,...}
            try:
                payload = json.loads(args[1])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping) and payload.get("ok") is False:
                return str(payload.get("error", "bridge_json_error"))
    return None


def _api_children(
    sock: socket.socket,
    ack_sock: socket.socket,
    path: str,
    child: str,
    request_id: str,
    timeout_s: float,
) -> list[dict[str, Any]]:
    cmd = bridge.OscCommand("/api/children", (path, child, request_id))
    acks = _send_and_collect_acks(sock, ack_sock, cmd, timeout_s)
    parsed = _extract_api_children(acks, request_id)
    if parsed is not None:
        return parsed
    err = _extract_error(acks)
    raise RuntimeError(err or f"api_children failed for {path} {child}")


def _api_get(
    sock: socket.socket,
    ack_sock: socket.socket,
    path: str,
    prop: str,
    request_id: str,
    timeout_s: float,
) -> Any:
    cmd = bridge.OscCommand("/api/get", (path, prop, request_id))
    acks = _send_and_collect_acks(sock, ack_sock, cmd, timeout_s)
    parsed = _extract_api_get(acks, request_id)
    if parsed is not None:
        return parsed
    err = _extract_error(acks)
    raise RuntimeError(err or f"api_get failed for {path}.{prop}")


def _scalar(value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            return None
        return value[-1]
    return value


def _first_device(
    *,
    devices: Sequence[dict[str, Any]],
    device_names: Mapping[str, str],
    index_fallback: int,
    name_hint: str | None,
) -> dict[str, Any]:
    if not devices:
        raise RuntimeError("no devices found on track")
    if name_hint:
        target = str(name_hint).strip().lower()
        for dev in devices:
            path = str(dev.get("path", ""))
            if str(device_names.get(path, "")).strip().lower().find(target) >= 0:
                return dict(dev)
    idx = max(0, min(len(devices) - 1, int(index_fallback)))
    return dict(devices[idx])


def _dump_via_osc(cfg: DumpConfig) -> dict[str, Any]:
    bridge_cfg = bridge.BridgeConfig(
        host=HOST,
        port=PORT,
        ack_port=ACK_PORT,
        ack_timeout_s=cfg.ack_timeout_s,
        expect_ack=True,
        ping_first=False,
        status=False,
        tempo=None,
        sig_num=None,
        sig_den=None,
        create_midi_tracks=0,
        add_midi_tracks=0,
        midi_name="MIDI",
        create_audio_tracks=0,
        add_audio_tracks=0,
        audio_prefix="Audio",
        delete_audio_tracks=0,
        delete_midi_tracks=0,
        rename_track_index=None,
        rename_track_name=None,
        session_clip_track_index=None,
        session_clip_slot_index=None,
        session_clip_length=None,
        session_clip_notes_json=None,
        session_clip_name=None,
        append_session_clip_track_index=None,
        append_session_clip_slot_index=None,
        append_session_clip_notes_json=None,
        inspect_session_clip_track_index=None,
        inspect_session_clip_slot_index=None,
        ensure_midi_tracks=None,
        midi_ccs=(),
        cc64s=(),
        api_pings=(),
        api_gets=(),
        api_sets=(),
        api_calls=(),
        api_children=(),
        api_describes=(),
        ack_mode="per_command",
        ack_flush_interval=10,
        report_metrics=False,
        delay_ms=0,
        dry_run=False,
    )
    ack_sock = bridge.open_ack_socket(bridge_cfg)
    if ack_sock is None:
        raise RuntimeError("failed to open ACK socket")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        tracks = _api_children(sock, ack_sock, "live_set", "tracks", "marimba-tracks", cfg.ack_timeout_s)
        track = None
        target_track = cfg.track_name.strip().lower()
        for item in tracks:
            if str(item.get("name", "")).strip().lower() == target_track:
                track = dict(item)
                break
        if track is None:
            raise RuntimeError(f"track not found: {cfg.track_name}")
        track_path = str(track.get("path", ""))
        devices = _api_children(sock, ack_sock, track_path, "devices", "marimba-devices", cfg.ack_timeout_s)
        device_names: dict[str, str] = {}
        for idx, dev in enumerate(devices):
            path = str(dev.get("path", ""))
            try:
                raw_name = _api_get(sock, ack_sock, path, "name", f"dev-name-{idx}", cfg.ack_timeout_s)
                device_names[path] = str(_scalar(raw_name) or "")
            except Exception:
                device_names[path] = ""
        device = _first_device(
            devices=devices,
            device_names=device_names,
            index_fallback=cfg.device_index,
            name_hint=cfg.device_name_hint,
        )
        device_path = str(device.get("path", ""))
        device_name = str(device_names.get(device_path, ""))
        params = _api_children(sock, ack_sock, device_path, "parameters", "marimba-params", cfg.ack_timeout_s)
        rows: list[dict[str, Any]] = []
        for idx, param in enumerate(params[: cfg.max_parameters]):
            ppath = str(param.get("path", ""))
            row: dict[str, Any] = {
                "index": int(param.get("index", idx)),
                "path": ppath,
            }
            for prop in ("name", "value", "min", "max", "is_quantized"):
                try:
                    value = _api_get(sock, ack_sock, ppath, prop, f"p-{idx}-{prop}", cfg.ack_timeout_s)
                    row[prop] = _scalar(value)
                except Exception as exc:  # noqa: BLE001
                    row[f"{prop}_error"] = str(exc)
            rows.append(row)
    ack_sock.close()
    return {
        "status": "ok",
        "mode": "osc_api",
        "track_name": cfg.track_name,
        "track_path": track_path,
        "device_name": device_name,
        "device_path": device_path,
        "parameter_count": len(rows),
        "parameters": rows,
    }


def _fallback_payload(cfg: DumpConfig, reason: str) -> dict[str, Any]:
    macro_map_path = Path("bridge/config/marimba_macro_map.v1.json")
    macro_map: Mapping[str, Any] = {}
    if macro_map_path.exists():
        try:
            raw = json.loads(macro_map_path.read_text(encoding="utf-8"))
            if isinstance(raw, Mapping):
                macro_map = raw
        except json.JSONDecodeError:
            macro_map = {}
    hints = macro_map.get("macro_targets", []) if isinstance(macro_map, Mapping) else []
    rows: list[dict[str, Any]] = []
    if isinstance(hints, Sequence):
        for idx, item in enumerate(hints):
            if not isinstance(item, Mapping):
                continue
            rows.append(
                {
                    "index": idx,
                    "name": str(item.get("macro_name", f"macro_{idx+1}")),
                    "path": "",
                    "value": item.get("default"),
                    "min": item.get("range_min"),
                    "max": item.get("range_max"),
                    "source": "macro_map_hint",
                    "parameter_name_hints": list(item.get("parameter_name_hints", [])),
                }
            )
    return {
        "status": "fallback",
        "mode": "macro_hints_only",
        "reason": reason,
        "track_name": cfg.track_name,
        "track_path": None,
        "device_name": cfg.device_name_hint,
        "device_path": None,
        "parameter_count": len(rows),
        "parameters": rows,
    }


def _write_summary_markdown(payload: Mapping[str, Any], json_path: Path) -> Path:
    md_path = json_path.with_suffix(".md")
    lines: list[str] = []
    lines.append("# Marimba Parameter Inventory")
    lines.append("")
    lines.append(f"- timestamp_utc: `{payload.get('timestamp_utc')}`")
    lines.append(f"- status: `{payload.get('status')}`")
    lines.append(f"- mode: `{payload.get('mode')}`")
    lines.append(f"- track_name: `{payload.get('track_name')}`")
    lines.append(f"- device_name: `{payload.get('device_name')}`")
    lines.append(f"- parameter_count: `{payload.get('parameter_count')}`")
    if payload.get("reason"):
        lines.append(f"- reason: `{payload.get('reason')}`")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    parameters = payload.get("parameters", [])
    if not isinstance(parameters, Sequence):
        parameters = []
    for row in parameters:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "- `{name}` (index={index}, value={value}, min={minv}, max={maxv})".format(
                name=row.get("name", "(unnamed)"),
                index=row.get("index", "?"),
                value=row.get("value", "?"),
                minv=row.get("min", "?"),
                maxv=row.get("max", "?"),
            )
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main(argv: Sequence[str] | None = None) -> int:
    cfg = parse_args(argv)
    result: dict[str, Any]
    if cfg.force_fallback:
        result = _fallback_payload(cfg, reason="forced_fallback")
    else:
        try:
            result = _dump_via_osc(cfg)
        except Exception as exc:  # noqa: BLE001
            result = _fallback_payload(cfg, reason=str(exc))
    payload = {
        "version": 1,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        **result,
    }
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    md_path = _write_summary_markdown(payload, cfg.output_path)
    print(f"info: marimba parameter dump written to {cfg.output_path} (count={payload.get('parameter_count', 0)})")
    print(f"info: marimba parameter summary written to {md_path}")
    if payload.get("status") != "ok":
        print(f"warning: used fallback inventory mode ({payload.get('reason', 'unknown reason')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
