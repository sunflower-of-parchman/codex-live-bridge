# Codex Live Bridge Protocol

This document defines the public OSC/UDP protocol for the Max for Live bridge. It is intentionally path-first: clients address Live Object Model objects by path strings such as `live_set`, `live_set tracks 0`, and `live_set tracks 0 clip_slots 0 clip`.

Push control is out of scope for this protocol.

Protocol status: v3.1.

## Transport

- Command target: `127.0.0.1:9000`
- ACK/event target: `127.0.0.1:9001`
- Encoding: OSC messages with simple scalar arguments.
- Structured payloads: JSON strings.
- Max for Live runtime: a dependency-free Node-for-Max UDP receiver,
  `udpsend`, and Max `js`.

The bridge should be loaded in Ableton Live through the shipped Max patch/device. Clients send commands to the command port and listen for acknowledgements on the ACK port.

The shipped command receiver binds explicitly to `127.0.0.1:9000`. The Python
client accepts loopback destinations only. Keep ACK traffic bound to
`127.0.0.1`; UDP does not encrypt the capability token or Live data.

## Authentication

Read-only commands remain tokenless. Mutating commands and observer lifecycle
changes require a capability token as their first OSC argument.

Configure the token locally by replacing `CHANGE_ME_BEFORE_USE` in the Max
patch's `set_auth_token` message, then save and reload the device. Tokens must
be 16 to 256 UTF-8 bytes. The placeholder is rejected and leaves mutations
disabled.

The Python CLI reads the same value from `--auth-token` or
`CODEX_LIVE_BRIDGE_TOKEN`. Missing configuration returns
`auth_not_configured`; a mismatch returns `unauthorized_command`. Neither error
echoes the supplied token.

The Node-for-Max receiver accepts single OSC messages from the loopback socket.
It rejects bundles, malformed messages, unsupported types, nonfinite floats,
and trailing bytes. Max defers decoded work to the low-priority queue and
admits at most one command every 10 ms. Unknown selectors pass through an
explicit wrapper allowlist before they reach the LiveAPI router.

## Product Architecture Boundary

This protocol owns the external and headless Ableton Live route. The bridge
retains persistent observers, generic LiveAPI RPC, raw MIDI output, bounded
writes, additive mutations, and compatibility with release versions of Live
that support Max for Live.

A separate Ableton Extension can provide native context actions, large
contextual reads, modal UX, and future user-invoked undoable transforms. Both
adapters may feed a shared domain core for canonical schema validation, note
normalization, inspection, report formatting, and parity comparison. That
shared core and the Extensions SDK are not part of this public wire protocol.

The bridge emits raw, correlated Live facts. Clients own transfer assembly and
domain interpretation. The public Python client provides strict V1 assembly;
other clients must implement the same validation requirements before treating
an inspection as complete.

## Request IDs

Most `/api/*` and observer commands accept an optional trailing `request_id`. When supplied, successful ACKs echo that value as the final argument.

Use request IDs for automation, batching, and retry logic. Object IDs from LiveAPI are dynamic, so clients should correlate their own requests by request ID and target path rather than caching LiveAPI IDs across sessions.

`/api/session_clip_inspect` requires a non-empty request ID of at most 128
UTF-8 bytes. Its repeated success fragments echo the request ID as the final
ACK argument. Its errors use the same reserved correlation trailer as other v3
errors.

## Core RPC Commands

```text
/api/ping [request_id]
/api/get <path> <property> [request_id]
/api/set <auth_token> <path> <property> <value_json> [request_id]
/api/call <auth_token> <path> <method> <args_json> [request_id]
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
/ack error <error_code> ... request_correlation <request_correlation>
```

Current v3 error ACKs always end with the reserved two-argument correlation
trailer `request_correlation req:<request_id>`, or
`request_correlation req:` when the command supplied no request ID. The
explicit marker keeps variable-length error details unambiguous and lets
clients preserve older untagged ACK details, including details beginning with
`req:`.

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
/api_observe <auth_token> <path> <property> [options_json] [request_id]
/api_unobserve <auth_token> <observer_id> [request_id]
/api_observers [request_id]
/api_clear_observers <auth_token> [request_id]
```

`options_json` may include:

```json
{
  "observer_id": "obs-tempo",
  "emit_initial": true,
  "mode": 1,
  "min_interval_ms": 100
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

Observer payloads include the observer ID, requested path, current path, property, event count, dropped-event count, timestamp, and value when available:

```json
{
  "observer_id": "obs-tempo",
  "requested_path": "live_set",
  "current_path": "live_set",
  "property": "tempo",
  "event_count": 2,
  "dropped_events": 0,
  "timestamp_ms": 123456,
  "value": 121.5
}
```

Clients should unobserve or clear observers before shutdown. The device enforces an observer quota and supports `min_interval_ms` / `throttle_ms` to reduce high-rate event streams. Mutations triggered by observer events should be queued back through the normal command path; do not mutate Live directly from an observer callback.

Caller-supplied observer IDs must be 1 to 128 UTF-8 bytes, begin with an
alphanumeric character, and contain only alphanumerics, `.`, `_`, `:`, or `-`.
The IDs `__proto__`, `prototype`, and `constructor` are rejected.

The Python CLI can stay open for observer events:

```bash
python3 bridge/ableton_udp_bridge.py --listen --listen-timeout 30 --listen-max-events 20 --no-tempo --no-signature
```

## Named Status Wrappers

These wrappers return JSON ACK payloads and accept an optional trailing request ID.

```text
/api/session_context [request_id]
/api/theory_status [request_id]
/api/tuning_status [request_id]
```

Successful ACKs:

```text
/ack api_session_context <context_json> [request_id]
/ack api_theory_status <status_json> [request_id]
/ack api_tuning_status <status_json> [request_id]
```

Safety class: read.

`/api/session_context` reports transport/session fields, track/scene counts, and selected track/scene/device when Live exposes them. `/api/theory_status` reports `root_note`, `scale_name`, `scale_intervals`, and `scale_mode`. `/api/tuning_status` reports `live_set tuning_system` data when the target Live version exposes that path.

## Packet-Bounded Session MIDI Clip Inspection

Protocol 3.1 adds a read-only, additive inspection endpoint:

```text
/api/session_clip_inspect <track_index> <slot_index> 1 <request_id>
```

The final protocol 3.1 implementation establishes V1 with schema version `1`
and producer version `3.1.0`. The endpoint has no compatibility promise with
unpublished development snapshots, including intermediate feature-branch
commits, and does not accept them as alternate V1 shapes.

Validation requires non-negative integer indexes, schema version exactly `1`,
and a non-empty request ID no longer than 128 UTF-8 bytes.

Successful requests emit one or more correlated ACKs:

```text
/ack api_session_clip_inspect <fragment_json> <request_id>
```

Every encoded OSC ACK packet is at most 4096 bytes. The bridge calculates the
actual OSC byte size for `/ack` with three string arguments, including UTF-8
bytes, NUL terminators, type tags, and OSC four-byte padding. A small
inspection is emitted as one `complete` fragment. Larger inspections emit one
`context` fragment followed by ordered `device_page` and `note_page`
fragments. Page sizes adapt to the encoded packet size. A single item that
cannot fit produces a correlated `api_session_clip_inspect_item_too_large`
error.

Consumers reject larger inspection datagrams before OSC or JSON decoding,
require strict OSC string termination, UTF-8, padding, and message length,
and bound aggregate retained fragment bytes across concurrent assemblies.

V1 resource limits are `MAX_NOTES=4096`, `MAX_DEVICES=256`, and
`MAX_FRAGMENTS=1024`. The producer rejects an inventory above these limits
after an ID-only preflight and before fetching full note records. Full note
records are retrieved in batches of at most 256 IDs. Each LiveAPI note response
is byte-bounded before JSON parsing. A final ID-only read detects additions or
deletions during batching. The producer also enforces the fragment ceiling
during adaptive page planning. Limit failures emit a correlated
`api_session_clip_inspect_limit_exceeded` error whose encoded packet remains
within the 4096-byte budget.

Every fragment contains:

```json
{
  "schema": "codex-live-bridge.session-midi-clip-inspection",
  "schema_version": 1,
  "producer_version": "3.1.0",
  "inspection_id": "session_clip_...",
  "correlation": {
    "request_id": "req-inspect",
    "track_index": 0,
    "slot_index": 0
  },
  "snapshot": {
    "started_ms": 0,
    "completed_ms": 0,
    "atomic": false,
    "consistent": true
  },
  "transfer": {
    "fragment_index": 0,
    "fragment_count": 1,
    "fragment_kind": "complete",
    "is_last": true,
    "packet_budget_bytes": 4096
  },
  "completeness": {
    "track": "complete",
    "clip": "complete",
    "devices": "complete",
    "notes": "complete",
    "missing_fields": []
  },
  "data": {}
}
```

Fragment indexes and page offsets are zero-based. Context data contains
`context:"session"`, track identity (`index`, `path`, `id`, `name`), clip
identity and timing (`slot_index`, `path`, `id`, `name`, `start_marker`,
`end_marker`, `live_length`, `looping`, `loop_start`, `loop_end`), and summary
fields (`note_count`, `pitch_min`, `pitch_max`). Empty clips use `null` pitch
bounds. Track and clip `name` values are strings or explicit `null`.
`start_marker`, `end_marker`, `loop_start`, and `loop_end` are finite signed
beat positions. Each start is less than or equal to its matching end, and the
finite difference is representable. `live_length` is finite and non-negative.

Device pages contain `device_offset`, `device_count`, `device_total`, and
ordered device records. Every device record has exactly `index`, `path`, `id`,
`name`, `class_name`, and `type`. `name` and `class_name` are strings or
explicit `null`. `type` preserves Live's integer enum `0`, `1`, `2`, or `4`,
or is explicit `null` when unavailable or invalid. These fields are never
omitted.

Note pages contain `note_offset`, `note_count`, `note_total`, and ordered note
records. Every note has exactly `note_id`, `pitch`, `start_time`, `duration`,
`velocity`, `mute`, `probability`, `velocity_deviation`, and
`release_velocity`. Missing fields cause a correlated parse error. Values are
preserved without coercion and validated as follows:

- `note_id`: non-negative integer
- `pitch`: integer `0..127`
- `start_time`: finite number
- `duration`: finite number `>= 0` with a finite `start_time + duration`
- `velocity`: finite number `0..127`
- `mute`: boolean or numeric `0`/`1`
- `probability`: finite number `0..1`
- `velocity_deviation`: finite number `-127..127`
- `release_velocity`: finite number `0..127`

Empty MIDI clips are successful.

Fragment sequence is defined by `fragment_index`. A multi-fragment transfer
starts with exactly one `context` fragment, followed by zero or more
contiguous `device_page` fragments, then zero or more contiguous `note_page`
fragments. A `device_page` cannot follow a `note_page`. Page offsets advance
contiguously in fragment order, page totals agree, and zero totals are
represented by no page fragments. A `complete` fragment is valid only as the
sole fragment and its counts equal its totals.

Before emitting any success fragment, the bridge re-reads the clip ID. A
changed ID produces only a correlated
`api_session_clip_inspect_snapshot_changed` error. Other endpoint errors use
the reserved correlation trailer and one of these codes. Every error packet
from this handler is also at most 4096 encoded OSC bytes; raw validation
diagnostics are UTF-8 bounded before emission.

```text
api_session_clip_inspect_validation_failed
api_session_clip_inspect_not_found
api_session_clip_inspect_not_midi
api_session_clip_inspect_no_clip
api_session_clip_inspect_read_failed
api_session_clip_inspect_parse_failed
api_session_clip_inspect_serialization_failed
api_session_clip_inspect_item_too_large
api_session_clip_inspect_limit_exceeded
api_session_clip_inspect_snapshot_changed
```

The Python client exposes `SessionClipInspectionAssembler`. It keys assemblies
by request ID and inspection ID, accepts out-of-order identical duplicates,
and rejects conflicting duplicates, malformed fragments, mixed metadata,
missing fragment indexes, invalid fragment-kind ordering, inconsistent
counts, and noncontiguous device or note offsets. It applies the same fragment,
device, and note limits, permits at most 16 active assemblies, caps missing
index diagnostics, and evicts completed or terminally failed states. Inspection
ACK collection retains at most 1024 correlated fragments plus a small bounded
allowance for unrelated traffic while continuing to wait for completion,
correlated error, or timeout.

The V1 bridge note schema is intentionally richer than the qualified
Extensions SDK `1.0.0` response. The bridge preserves `note_id` and
`release_velocity` when LiveAPI returns them; the qualified native SDK runtime
omitted both fields. A cross-surface consumer must use deterministic ID-free
matching and report release velocity as an SDK-side missing field rather than
silently treating the two snapshots as exact parity.

Small reads through the legacy
`/inspect_session_clip_notes <track_index> <slot_index>` command remain
compatible. An ID-only preflight rejects clips above 4096 notes. The bridge
fetches full notes in batches of 256 IDs and rejects a single ACK above 49,152
UTF-8 bytes. Use `/api/session_clip_inspect` for packet-bounded fragments,
metadata, devices, request correlation, and complete Live 12 note fields.

## Device, Parameter, And Mixer Wrappers

```text
/api/device_list <track_ref|all> [request_id]
/api/device_parameters <device_path> [request_id]
/api/mixer_status <track_ref|master|return:N> [request_id]
/api/parameter_set <auth_token> <parameter_path> <value_json> [request_id]
```

Successful ACKs:

```text
/ack api_device_list <target> <devices_json> [request_id]
/ack api_device_parameters <device_path> <parameters_json> [request_id]
/ack api_mixer_status <track_path> <mixer_json> [request_id]
/ack api_parameter_set <parameter_path> <parameter_json> [request_id]
```

Safety classes:

- `device_list`, `device_parameters`, and `mixer_status`: read.
- `parameter_set`: bounded write.

Tokenless collection wrappers accept at most 256 items from one collection and
512 aggregate items per request. They stop after 1,000 ms and reject JSON
responses above 49,152 UTF-8 bytes with a correlated limit error.

`parameter_set` accepts only numeric JSON numbers or numeric strings, checks `is_enabled`, and rejects values outside the parameter's `min`/`max` range when Live exposes that metadata. JSON `null`, booleans, arrays, and objects are rejected before the wrapper calls LiveAPI.

## Live 12.3 Native Insertion Wrappers

```text
/api/insert_device <auth_token> <track_or_chain_path> <native_device_name> <target_index_or_empty> [request_id]
/api/insert_chain <auth_token> <rack_device_path> <target_index_or_empty> [request_id]
/api/drum_chain_in_note <auth_token> <drum_chain_path> <note|-1> [request_id]
```

Successful ACKs:

```text
/ack api_insert_device <target_path> <native_device_name> <result_json> [request_id]
/ack api_insert_chain <rack_path> <result_json> [request_id]
/ack api_drum_chain_in_note <drum_chain_path> <chain_json> [request_id]
```

Safety classes:

- `insert_device`: additive mutation.
- `insert_chain`: additive mutation.
- `drum_chain_in_note`: bounded write accepting integer notes from `-1..127`.

These APIs are gated by LiveAPI capability checks when metadata is available. Optional insertion indexes must be non-negative integers when provided. Native device insertion requires Ableton Live 12.3+ and supports native Live devices only; Max for Live devices and plug-ins are not supported by these insertion calls.

`drum_chain_in_note` reads the property back after writing. It reports
`api_drum_chain_in_note_readback_failed` when the applied value cannot be
verified and `api_drum_chain_in_note_write_not_applied` when Live returns a
different applied value. Live documents `-1` as the Drum Chain "All Notes"
setting, but runtime support can vary by Live build and rack context.

## Protected Core Commands

```text
/tempo <auth_token> <bpm>
/sig_num <auth_token> <numerator>
/sig_den <auth_token> <denominator>
/create_midi_track <auth_token>
/add_midi_tracks <auth_token> <count> [name]
/create_audio_track <auth_token>
/add_audio_tracks <auth_token> <count> [prefix]
/delete_audio_tracks <auth_token> <count>
/delete_midi_tracks <auth_token> <count>
/rename_track <auth_token> <track_index> <name>
/set_session_clip_notes <auth_token> <track_index> <slot_index> <length_beats> <notes_json> [clip_name]
/append_session_clip_notes <auth_token> <track_index> <slot_index> <notes_json>
/ensure_midi_tracks <auth_token> <target_count>
/midi_cc <auth_token> <controller> <value> [channel]
/cc64 <auth_token> <value> [channel]
```

Track create/delete batches are limited to `32` per command.
`ensure_midi_tracks` accepts targets from `0` to `256` and creates at most
`32` missing tracks per command. MIDI values remain bounded to their documented
7-bit and channel ranges.

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
- `velocity_deviation`: number `-127..127`
- `release_velocity`: integer `0..127`

## Safety Classes

Classify commands before adding named wrappers:

- Read: inspect Live state without mutation.
- Bounded write: set a specific property or parameter with validation.
- Additive mutation: create tracks, clips, devices, chains, or notes.
- Destructive mutation: delete, clear, overwrite, or globally reset state.

The generic RPC layer can reach broad LiveAPI behavior. Public examples and automation defaults should favor read and bounded-write commands, require explicit user approval for additive mutation, and keep destructive operations clearly named.

## LiveAPI Constraints

- Use LiveAPI after device initialization, normally after `live.thisdevice` and `deferlow`.
- Not every property is observable or writable.
- Read back after writes when correctness matters.
- Object IDs are dynamic and should not be persisted across Live sessions.
- Some Live 12.3 APIs, including native device insertion methods, may not exist in older Live versions.
- Native insertion APIs are for native Live devices, not Max for Live devices or plug-ins.
- Arbitrary automation breakpoint writing is not treated as a stable public feature unless verified against the target Live version.

## Payload Guidance

Keep command payloads small enough for local UDP transport. The 3.1 session
clip inspector enforces a 4096-byte encoded ACK limit. Tokenless collection
wrappers allow at most 256 items from one collection, 512 items across one
request, 1,000 milliseconds of traversal, and 49,152 UTF-8 bytes of JSON.
Requests that exceed a traversal limit fail with `api_read_limit_exceeded`.
Oversized JSON responses fail with `api_read_response_too_large`. Large note
sets or inventory reads should use chunked commands with request IDs and
explicit ACK handling.
