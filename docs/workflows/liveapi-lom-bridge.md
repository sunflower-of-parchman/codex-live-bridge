# LiveAPI LOM Bridge (Port 9000)

This bridge exposes a local HTTP command API on port `9000` and forwards commands to a Max for Live LiveAPI router over UDP.
Both command transports intentionally use port `9000`: HTTP uses `TCP/9000` and Max receives commands on `UDP/9000`.
This bridge uses a single UDP port (`9000`) for Live control commands.

## Architecture

1. Codex or other client sends JSON commands to `http://127.0.0.1:9000/command`.
2. Python bridge validates payloads against command schemas.
3. Bridge forwards validated envelopes to UDP target `127.0.0.1:9000`.
4. Max for Live `js` router executes LiveAPI calls against the LOM.
5. Commands are executed in Live; this workflow focuses on write operations (notes, tempo, automation, mix).

## Files

- Server entrypoint: `/Users/michaelwall/codex-live-bridge/scripts/run_live_bridge.py`
- CLI sender: `/Users/michaelwall/codex-live-bridge/scripts/send_live_command.py`
- Bridge server: `/Users/michaelwall/codex-live-bridge/live_bridge/server.py`
- Protocol validators: `/Users/michaelwall/codex-live-bridge/live_bridge/protocol.py`
- Max router script: `/Users/michaelwall/codex-live-bridge/max_for_live/live_api_command_router.js`
- Max patch scaffold: `/Users/michaelwall/codex-live-bridge/max_for_live/live_bridge_receiver.maxpat`

## Load into Max MIDI Effect (copy/paste workflow)

1. In Ableton Live, select your Max MIDI Effect and click `Edit` to open in Max.
2. In Max, unlock the patcher (`Cmd+E` on macOS).
3. Open `/Users/michaelwall/codex-live-bridge/max_for_live/live_bridge_receiver.maxpat`.
4. Select all objects in that patch (`Cmd+A`), copy (`Cmd+C`), then paste (`Cmd+V`) into the Max MIDI Effect patcher.
5. Confirm the `js` object path points to:
   `/Users/michaelwall/codex-live-bridge/max_for_live/live_api_command_router.js`
6. Lock patcher (`Cmd+E`) and save the device (`Cmd+S`).
7. Start the bridge server on port `9000` and test with one command.

## Run bridge on port 9000

```bash
python3 /Users/michaelwall/codex-live-bridge/scripts/run_live_bridge.py --port 9000 --backend udp-max-proxy --udp-port 9000
```

## Health and capability checks

```bash
curl -s http://127.0.0.1:9000/health
curl -s http://127.0.0.1:9000/capabilities
```

## CLI sender usage

```bash
python3 /Users/michaelwall/codex-live-bridge/scripts/send_live_command.py --url http://127.0.0.1:9000 --command set_tempo --payload '{"bpm":123}'
```

## Command examples

### Insert notes

```bash
curl -s http://127.0.0.1:9000/command \
  -H 'Content-Type: application/json' \
  -d '{
    "command": "note_insert",
    "payload": {
      "track_index": 0,
      "clip_slot_index": 0,
      "notes": [
        {"pitch": 60, "start_time": 0.0, "duration": 1.0, "velocity": 100, "mute": false},
        {"pitch": 64, "start_time": 1.0, "duration": 1.0, "velocity": 96, "mute": false}
      ]
    }
  }'
```

### Set tempo / BPM

```bash
curl -s http://127.0.0.1:9000/command \
  -H 'Content-Type: application/json' \
  -d '{"command":"set_tempo","payload":{"bpm":120.0}}'
```

### Set global key

```bash
curl -s http://127.0.0.1:9000/command \
  -H 'Content-Type: application/json' \
  -d '{
    "command":"set_global_key",
    "payload":{"root_note":0,"scale_name":"Major","scale_intervals":[0,2,4,5,7,9,11]}
  }'
```

### Create automation points

```bash
curl -s http://127.0.0.1:9000/command \
  -H 'Content-Type: application/json' \
  -d '{
    "command":"create_automation",
    "payload":{
      "track_index":0,
      "clip_slot_index":0,
      "device_index":1,
      "parameter_index":3,
      "points":[{"time":0.0,"value":0.2},{"time":2.0,"value":0.8}]
    }
  }'
```

### Mix and EQ controls

```bash
curl -s http://127.0.0.1:9000/command -H 'Content-Type: application/json' -d '{"command":"set_track_volume","payload":{"track_index":0,"value":0.72}}'
curl -s http://127.0.0.1:9000/command -H 'Content-Type: application/json' -d '{"command":"set_track_pan","payload":{"track_index":0,"value":-0.1}}'
curl -s http://127.0.0.1:9000/command -H 'Content-Type: application/json' -d '{"command":"set_send_level","payload":{"track_index":0,"send_index":1,"value":0.35}}'
curl -s http://127.0.0.1:9000/command -H 'Content-Type: application/json' -d '{"command":"set_eq8_band_gain","payload":{"track_index":0,"device_index":2,"band":4,"gain":0.55}}'
```

### Clip and transport controls

```bash
curl -s http://127.0.0.1:9000/command -H 'Content-Type: application/json' -d '{"command":"create_midi_clip","payload":{"track_index":0,"clip_slot_index":2,"length_beats":8.0}}'
curl -s http://127.0.0.1:9000/command -H 'Content-Type: application/json' -d '{"command":"fire_clip","payload":{"track_index":0,"clip_slot_index":2}}'
curl -s http://127.0.0.1:9000/command -H 'Content-Type: application/json' -d '{"command":"stop_track","payload":{"track_index":0}}'
curl -s http://127.0.0.1:9000/command -H 'Content-Type: application/json' -d '{"command":"set_track_mute","payload":{"track_index":0,"value":true}}'
curl -s http://127.0.0.1:9000/command -H 'Content-Type: application/json' -d '{"command":"set_track_solo","payload":{"track_index":0,"value":false}}'
```

## Next extensions

- Add arrangement-level automation support
- Add device discovery command for named EQ targeting
- Add explicit scene launch commands
