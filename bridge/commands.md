# Ableton Live UDP Bridge Command Cheat Sheet

This file is a quick reference for commands sent to the Max for Live bridge.
`PROTOCOL.md` is the canonical protocol contract.

Commands are OSC packets sent to `127.0.0.1:9000`. ACKs and observer events are
OSC packets emitted from `127.0.0.1:9001` at address `/ack`.
Current v3 error ACKs always end with the reserved request-ID correlation
trailer `request_correlation req:<request_id>`, or
`request_correlation req:` when the command had no request ID.
Protocol 3.1 session clip inspection ACKs are packet-bounded to 4096 encoded
OSC bytes.

## Authentication

Read commands remain tokenless. Every write, generic call, observer lifecycle
change, insertion, track or clip mutation, and MIDI output command places
`<auth_token>` first. The Python CLI injects it from `--auth-token` or
`CODEX_LIVE_BRIDGE_TOKEN`.

## Read Commands

```text
/ping
/status
/api/ping [request_id]
/api/get <path> <property> [request_id]
/api/children <path> <child_name> [request_id]
/api/describe <path> [request_id]
/api/session_context [request_id]
/api/theory_status [request_id]
/api/tuning_status [request_id]
/api/device_list <track_ref|all> [request_id]
/api/device_parameters <device_path> [request_id]
/api/mixer_status <track_ref|master|return:N> [request_id]
/api/session_clip_inspect <track_index> <slot_index> 1 <request_id>
```

Examples:

```text
/status
/api/get live_set tempo req-tempo
/api/children live_set tracks req-tracks
/api/device_list all req-devices
/api/session_clip_inspect 0 0 1 req-clip
```

`/api/session_clip_inspect` requires non-negative integer indexes, schema
version `1`, and a non-empty request ID of at most 128 UTF-8 bytes. Success
arrives as repeated:

```text
/ack api_session_clip_inspect <fragment_json> <request_id>
```

Small results use one `complete` fragment. Larger results use a `context`
fragment plus adaptive `device_page` and `note_page` fragments. Fragment
indexes define the sequence: context, contiguous device pages, then contiguous
note pages. Device metadata always includes nullable `name`, `class_name`, and
numeric Live `type` enum (`0`, `1`, `2`, or `4`). Each note contains all nine
extended-note fields. Success and error ACKs from this endpoint are at most
4096 encoded OSC bytes. V1 permits at most 4096 notes, 256 devices, and 1024
fragments. Clip marker and loop positions may be negative finite beat values,
with each start less than or equal to its matching end. The Python CLI waits
for complete assembly, a correlated error, or the full ACK timeout:

```bash
python3 bridge/ableton_udp_bridge.py \
  --ack --no-tempo --no-signature \
  --api-session-clip-inspect 0 0 req-clip
```

## Observer Commands

```text
/api_observe <auth_token> <path> <property> [options_json] [request_id]
/api_unobserve <auth_token> <observer_id> [request_id]
/api_observers [request_id]
/api_clear_observers <auth_token> [request_id]
```

Example:

```text
/api_observe <auth_token> live_set tempo {"observer_id":"obs-tempo","emit_initial":true,"min_interval_ms":100} req-observe
```

Observer events arrive as:

```text
/ack api_event <observer_id> <payload_json>
```

## Bounded Writes

```text
/tempo <auth_token> <bpm>
/sig_num <auth_token> <numerator>
/sig_den <auth_token> <denominator>
/rename_track <auth_token> <track_index> <name>
/midi_cc <auth_token> <controller> <value> [channel]
/cc64 <auth_token> <value> [channel]
/api/set <auth_token> <path> <property> <value_json> [request_id]
/api/parameter_set <auth_token> <parameter_path> <value_json> [request_id]
/api/drum_chain_in_note <auth_token> <drum_chain_path> <note|-1> [request_id]
```

Examples:

```text
/tempo <auth_token> 120
/rename_track <auth_token> 0 Lead
/api/parameter_set <auth_token> "live_set tracks 0 devices 0 parameters 1" 0.5 req-param
```

`/api/parameter_set` accepts numeric JSON numbers or numeric strings, checks
`is_enabled`, and rejects ambiguous types and values outside the parameter
`min` and `max` range when Live exposes those fields.

## Additive Mutations

```text
/create_midi_track <auth_token>
/add_midi_tracks <auth_token> <count> [name]
/create_audio_track <auth_token>
/add_audio_tracks <auth_token> <count> [prefix]
/ensure_midi_tracks <auth_token> <target_count>
/append_session_clip_notes <auth_token> <track_index> <slot_index> <notes_json>
/api/call <auth_token> <path> <method> <args_json> [request_id]
/api/insert_device <auth_token> <track_or_chain_path> <native_device_name> <target_index_or_empty> [request_id]
/api/insert_chain <auth_token> <rack_device_path> <target_index_or_empty> [request_id]
```

Examples:

```text
/add_midi_tracks <auth_token> 2 Bridge
/api/call <auth_token> live_set create_midi_track [-1] req-create
/api/insert_device <auth_token> "live_set tracks 0" Operator "" req-operator
```

Live 12.3 native insertion commands support native Live devices. Plug-ins and
Max for Live devices are outside that insertion API. Optional insertion indexes
must be non-negative integers when provided.

Track create/delete batches are capped at `32` per command.
`/ensure_midi_tracks` targets are capped at `256` and may create at most `32`
tracks in one command.

## Destructive Mutations

```text
/delete_audio_tracks <auth_token> <count>
/delete_midi_tracks <auth_token> <count>
/set_session_clip_notes <auth_token> <track_index> <slot_index> <length_beats> <notes_json> [clip_name]
```

`/delete_midi_tracks` protects track index `0`. `/set_session_clip_notes`
deletes any existing clip in the target slot before writing the new clip.

## Note JSON

Commands that create or append notes accept either a JSON array of notes or an
object shaped like `{"notes":[...]}`.

Required fields:

```json
{"pitch":60,"start_time":0,"duration":1,"velocity":100}
```

Optional fields:

```json
{"mute":0,"probability":1.0,"velocity_deviation":0,"release_velocity":64}
```

Validation:

- `pitch`: integer `0..127`
- `start_time`: number `>= 0`
- `duration`: number `> 0`
- `velocity`: integer `0..127`; omitted or invalid values default to `100`
- `mute`: truthy values become `1`, otherwise `0`
- `probability`: number `0..1`
- `velocity_deviation`: number `-127..127`
- `release_velocity`: integer `0..127`

## ACK Examples

```text
/ack ready live_set 1234
/ack pong
/ack status 3 3 0 2 live_set 1234
/ack api_get live_set tempo 120 req-tempo
/ack api_children live_set tracks [{"index":0,"id":1,"path":"live_set tracks 0"}] req-tracks
/ack api_event obs-tempo {"observer_id":"obs-tempo","current_path":"live_set","property":"tempo","value":121.5}
/ack error api_parameter_value_out_of_range "live_set tracks 0 devices 0 parameters 1" 2 0 1 request_correlation req:req-param
```

For full ACK schemas, request ID behavior, LiveAPI constraints, and safety
classes, read `PROTOCOL.md`.

The legacy `/inspect_session_clip_notes <track_index> <slot_index>` command and
its ACK format remain unchanged.
