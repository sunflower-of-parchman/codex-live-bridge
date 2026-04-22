# Codex Live Bridge Protocol

This document defines the public OSC/UDP protocol for the Max for Live bridge. It is intentionally path-first: clients address Live Object Model objects by path strings such as `live_set`, `live_set tracks 0`, and `live_set tracks 0 clip_slots 0 clip`.

Push control is out of scope for this protocol.

## Transport

- Command target: `127.0.0.1:9000`
- ACK/event target: `127.0.0.1:9001`
- Encoding: OSC messages with simple scalar arguments.
- Structured payloads: JSON strings.
- Max for Live runtime: `udpreceive`/`udpsend` plus Max `js`.

The bridge should be loaded in Ableton Live through the shipped Max patch/device. Clients send commands to the command port and listen for acknowledgements on the ACK port.

## Request IDs

Most `/api/*` and observer commands accept an optional trailing `request_id`. When supplied, successful ACKs echo that value as the final argument.

Use request IDs for automation, batching, and retry logic. Object IDs from LiveAPI are dynamic, so clients should correlate their own requests by request ID and target path rather than caching LiveAPI IDs across sessions.

## Core RPC Commands

```text
/api/ping [request_id]
/api/get <path> <property> [request_id]
/api/set <path> <property> <value_json> [request_id]
/api/call <path> <method> <args_json> [request_id]
/api/children <path> <child_name> [request_id]
/api/describe <path> [request_id]
```

Successful ACKs:

```text
/ack api_get <path> <property> <value_json> [request_id]
/ack api_set <path> <property> <result_json> [request_id]
/ack api_call <path> <method> <result_json> [request_id]
/ack api_children <path> <child_name> <children_json> [request_id]
/ack api_describe <path> <describe_json> [request_id]
```

Error ACKs:

```text
/ack error <api_error_code> ... [request_id]
```

`/api/children` returns a JSON array of child records:

```json
[
  {
    "index": 0,
    "id": 1234,
    "path": "live_set tracks 0",
    "name": "Track 1"
  }
]
```

`/api/describe` returns a JSON object with at least the requested path and LiveAPI ID when available. Name, type, child, property, and function metadata depend on what Live exposes for that path.

## Observer Commands

```text
/api_observe <path> <property> [options_json] [request_id]
/api_unobserve <observer_id> [request_id]
/api_observers [request_id]
/api_clear_observers [request_id]
```

`options_json` may include:

```json
{
  "observer_id": "obs-tempo",
  "emit_initial": true,
  "mode": 1
}
```

Successful ACKs:

```text
/ack api_observe <observer_id> <path> <property> <snapshot_json> [request_id]
/ack api_unobserve <observer_id> <result_json> [request_id]
/ack api_observers <observers_json> [request_id]
/ack api_clear_observers <result_json> [request_id]
```

Asynchronous observer events:

```text
/ack api_event <observer_id> <payload_json>
```

Observer payloads include the observer ID, requested path, current path, property, event count, timestamp, and value when available:

```json
{
  "observer_id": "obs-tempo",
  "requested_path": "live_set",
  "current_path": "live_set",
  "property": "tempo",
  "event_count": 2,
  "timestamp_ms": 123456,
  "value": 121.5
}
```

Clients should unobserve or clear observers before shutdown. Mutations triggered by observer events should be queued back through the normal command path; do not mutate Live directly from an observer callback.

## Note Dictionary Schema

Commands that create or append notes accept either a raw note array or an object shaped like `{"notes":[...]}`. The bridge converts JSON into the Max `Dict` shape required by LiveAPI note methods.

Required fields:

```json
{
  "pitch": 60,
  "start_time": 0,
  "duration": 1,
  "velocity": 100
}
```

Supported optional fields:

```json
{
  "mute": 0,
  "probability": 1.0,
  "velocity_deviation": 0,
  "release_velocity": 64
}
```

Validation:

- `pitch`: integer `0..127`
- `start_time`: number `>= 0`
- `duration`: number `> 0`
- `velocity`: integer `0..127`; omitted or invalid values fall back to `100`
- `mute`: truthy values become `1`, otherwise `0`
- `probability`: number `0..1`
- `velocity_deviation`: number `0..127`
- `release_velocity`: integer `0..127`

## Safety Classes

Classify commands before adding named wrappers:

- Read: inspect Live state without mutation.
- Bounded write: set a specific property or parameter with validation.
- Additive mutation: create tracks, clips, devices, chains, or notes.
- Destructive mutation: delete, clear, overwrite, or globally reset state.

The generic RPC layer can reach broad LiveAPI behavior. Public examples and agent defaults should favor read and bounded-write commands, require explicit user intent for additive mutation, and keep destructive operations clearly named.

## LiveAPI Constraints

- Use LiveAPI after device initialization, normally after `live.thisdevice` and `deferlow`.
- Not every property is observable or writable.
- Read back after writes when correctness matters.
- Object IDs are dynamic and should not be persisted across Live sessions.
- Some Live 12.3 APIs, including native device insertion methods, may not exist in older Live versions.
- Native insertion APIs are for native Live devices, not Max for Live devices or plug-ins.
- Arbitrary automation breakpoint writing is not treated as a stable public feature unless verified against the target Live version.

## Payload Guidance

Keep command payloads small enough for local UDP transport. For large note sets or inventory reads, prefer chunked/batched commands with request IDs and explicit ACK handling.

The current bridge is local-first and does not authenticate OSC packets. Keep the host bound to `127.0.0.1` unless a future release adds an authenticated transport.
