# Ableton Live UDP Bridge Commands (OSC)

This file defines the OSC (Open Sound Control) UDP messages the Max for Live
bridge will understand.

All messages are sent to `127.0.0.1:9000` as OSC UDP packets.

## Command Format

Each message is an OSC packet with:

- An address that starts with `/`
- Zero or more typed arguments (int, float, string)

Examples:

    /tempo 120
    /sig_num 4
    /sig_den 4
    /create_midi_track
    /add_midi_tracks 2 All Different States
    /create_audio_track
    /add_audio_tracks 10 Audio
    /delete_audio_tracks 3
    /delete_midi_tracks 2
    /rename_track 1 All Different States
    /set_session_clip_notes 1 0 20 {"notes":[...]} CodexArp
    /append_session_clip_notes 1 0 {"notes":[...]}
    /inspect_session_clip_notes 1 0
    /ensure_midi_tracks 4
    /midi_cc 64 127 1
    /cc64 127 1
    /status
    /ping
    /api/get live_set tempo
    /api/call live_set create_midi_track [-1]
    /api/children live_set tracks req-1

## Supported Commands (v0)

- `/ping`
- `/tempo <bpm>`
- `/sig_num <numerator>`
- `/sig_den <denominator>`
- `/create_midi_track`
- `/add_midi_tracks <count> [name]`
- `/create_audio_track`
- `/add_audio_tracks <count> [prefix]`
- `/delete_audio_tracks <count>`
- `/delete_midi_tracks <count>`
- `/rename_track <track_index> <name>`
- `/set_session_clip_notes <track_index> <slot_index> <length_beats> <notes_json> [clip_name]`
- `/append_session_clip_notes <track_index> <slot_index> <notes_json>`
- `/inspect_session_clip_notes <track_index> <slot_index>`
- `/ensure_midi_tracks <target_count>`
- `/midi_cc <controller> <value> [channel]`
- `/cc64 <value> [channel]`
- `/status`
- `/api/ping [request_id]`
- `/api/get <path> <property> [request_id]`
- `/api/set <path> <property> <value_json> [request_id]`
- `/api/call <path> <method> <args_json> [request_id]`
- `/api/children <path> <child_name> [request_id]`
- `/api/describe <path> [request_id]`

`/ensure_midi_tracks` counts MIDI-capable tracks using the Live API property
`has_midi_input`.

`/add_audio_tracks` creates audio tracks at the end of the set and renames
them using the provided prefix (default `Audio`), for example `Audio 01`.

`/delete_midi_tracks` deletes MIDI-capable tracks from the end of the set and
protects track index `0`.

`/rename_track` renames the given track index.

`/set_session_clip_notes` clears a session clip slot, creates a new MIDI clip,
and inserts notes described by the JSON payload.

`/append_session_clip_notes` adds notes to an existing session clip without
deleting it.

`/inspect_session_clip_notes` reports how many notes Live sees in a session
clip slot, plus the pitch range and clip length.

`/midi_cc` emits a MIDI Control Change message directly from the bridge device.
It is most reliable when the bridge device is on the same track as the target
instrument, or when the track's MIDI output is routed appropriately.

`/cc64` is a convenience wrapper for sustain pedal (CC64).

## LiveAPI RPC Surface (`/api/*`)

The `/api/*` commands expose a generic "remote procedure call" layer over the
Live API. This is the broadest possible surface because it lets you target any
Live API path, property, or method that Max for Live can reach.

Definitions used below:

- A `path` is a Live API path such as `live_set`, `live_set tracks 0`, or
  `live_set tracks 0 devices 1`.
- A `property` is a named value you can read or write on that path, such as
  `tempo` on `live_set` or `name` on a track path.
- A `method` is a Live API verb you can call, such as `create_midi_track` on
  `live_set`.
- JSON payloads are string arguments that encode structured data. This keeps
  OSC arguments simple while allowing complex values.
- A `request_id` is an optional trailing string that, when provided, is echoed
  back in the acknowledgement. This makes automation easier to correlate.

Note on MIDI note insertion:

- LiveAPI's `add_new_notes` normally requires a Max `Dict`. The bridge now
  accepts JSON for this method via `/api/call` and will convert it internally.
  You can send either a raw notes array or a wrapper object shaped like
  `{"notes":[...]}` as the first argument.

RPC commands:

- `/api/ping [request_id]`
- `/api/get <path> <property> [request_id]`
- `/api/set <path> <property> <value_json> [request_id]`
- `/api/call <path> <method> <args_json> [request_id]`
- `/api/children <path> <child_name> [request_id]`
- `/api/describe <path> [request_id]`

Acknowledgement shapes:

- `/api/get` returns:

      /ack api_get <path> <property> <value_json> [request_id]

- `/api/set` returns:

      /ack api_set <path> <property> <result_json> [request_id]

  The current `result_json` is `{"ok": true}` on success.

- `/api/call` returns:

      /ack api_call <path> <method> <result_json> [request_id]

- `/api/children` returns:

      /ack api_children <path> <child_name> <children_json> [request_id]

  `children_json` is a JSON array of objects shaped like:

      {"index": 0, "id": 1234, "path": "live_set tracks 0", "name": "Track 1"}

  The `path` value is constructed as:

      <path> <child_name> <index>

  for each index from `0` to `getcount(child_name) - 1`.

- `/api/describe` returns:

      /ack api_describe <path> <describe_json> [request_id]

  `describe_json` includes at least `path` and `id`, and may include `name`
  and `type` when available.

RPC error acknowledgements use:

      /ack error <api_error_code> ... [request_id]

## Acknowledgements

The bridge emits OSC acknowledgement messages to `127.0.0.1:9001` using the
address `/ack`.

Examples:

    /ack ready live_set
    /ack tempo 120
    /ack midi_track_created 12 All Different States
    /ack add_midi_tracks 2 All Different States 2 13
    /ack audio_track_created 12 Audio 01
    /ack add_audio_tracks 10 Audio 10 25
    /ack audio_track_deleted 7
    /ack delete_audio_tracks 3 3 22
    /ack midi_track_deleted 2
    /ack delete_midi_tracks 2 2 10
    /ack track_renamed 1 All Different States
    /ack set_session_clip_notes 3 0 20 40 CodexArp
    /ack append_session_clip_notes 3 0 40 40
    /ack inspect_session_clip_notes 3 0 40 45 75 20
    /ack ensure_midi_tracks 4 3 1 6
    /ack midi_cc 64 127 1
    /ack cc64 127 1
    /ack status 12 4 8 2 live_set 1234
    /ack api_get live_set tempo 142
    /ack api_children live_set tracks [{"index":0,"id":1,"path":"live_set tracks 0"}] req-1
    /ack api_call live_set create_midi_track null req-2
    /ack api_describe live_set {"path":"live_set","id":1}

For `/ack ensure_midi_tracks`, the arguments are:

    /ack ensure_midi_tracks <target_midi_tracks> <current_midi_tracks> <created> <total_tracks>

For `/ack midi_cc`, the arguments are:

    /ack midi_cc <controller> <value> <channel>

For `/ack cc64`, the arguments are:

    /ack cc64 <value> <channel>

For `/ack audio_track_created`, the arguments are:

    /ack audio_track_created <track_index> <track_name>

For `/ack add_audio_tracks`, the arguments are:

    /ack add_audio_tracks <requested_count> <prefix> <created> <total_tracks>

For `/ack midi_track_created`, the arguments are:

    /ack midi_track_created <track_index> <track_name>

For `/ack add_midi_tracks`, the arguments are:

    /ack add_midi_tracks <requested_count> <name> <created> <total_tracks>

For `/ack audio_track_deleted`, the arguments are:

    /ack audio_track_deleted <track_index>

For `/ack delete_audio_tracks`, the arguments are:

    /ack delete_audio_tracks <requested_count> <deleted> <total_tracks>

For `/ack midi_track_deleted`, the arguments are:

    /ack midi_track_deleted <track_index>

For `/ack delete_midi_tracks`, the arguments are:

    /ack delete_midi_tracks <requested_count> <deleted> <total_tracks>

For `/ack track_renamed`, the arguments are:

    /ack track_renamed <track_index> <track_name>

For `/ack set_session_clip_notes`, the arguments are:

    /ack set_session_clip_notes <track_index> <slot_index> <length_beats> <note_count> <clip_name>

For `/ack append_session_clip_notes`, the arguments are:

    /ack append_session_clip_notes <track_index> <slot_index> <note_count> <note_id_count>

For `/ack inspect_session_clip_notes`, the arguments are:

    /ack inspect_session_clip_notes <track_index> <slot_index> <note_count> <min_pitch> <max_pitch> <clip_length>

For `/ack status`, the arguments are:

    /ack status <total_tracks> <midi_tracks> <audio_tracks> <return_tracks> <path> <liveapi_id>

For `/ack api_get`, the arguments are:

    /ack api_get <path> <property> <value_json> [request_id]

For `/ack api_set`, the arguments are:

    /ack api_set <path> <property> <result_json> [request_id]

For `/ack api_call`, the arguments are:

    /ack api_call <path> <method> <result_json> [request_id]

For `/ack api_children`, the arguments are:

    /ack api_children <path> <child_name> <children_json> [request_id]

For `/ack api_describe`, the arguments are:

    /ack api_describe <path> <describe_json> [request_id]

For `/ack error` produced by the RPC layer, the arguments begin with an
`api_*` error code and may end with `[request_id]`.

## Max for Live Patch Expectations

The first bridge device should use this routing shape:

    [udpreceive 9000]
    [route /tempo /sig_num /sig_den /create_midi_track /add_midi_tracks /create_audio_track /add_audio_tracks /delete_audio_tracks /delete_midi_tracks /rename_track /set_session_clip_notes /append_session_clip_notes /inspect_session_clip_notes /ensure_midi_tracks /status /ping /api/ping /api/get /api/set /api/call /api/children /api/describe]
