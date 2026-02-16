# codex-live-bridge

`codex-live-bridge` is an open-source bridge that lets the Codex app control
Ableton Live for daily composition workflows.

This project is designed for agent-assisted music production loops, not only
raw transport/control commands.

## What This Enables

- Codex app orchestration of Ableton Live setup and composition tasks
- meter/BPM coverage workflows driven by catalog metadata
- repeatable "prepare the session and start composing" loops
- automation-first creative routines without exposing private user data

## Included

- `bridge/m4l/LiveUdpBridge.amxd`:
  packaged drop-in Max for Live MIDI device
- `bridge/m4l/LiveUdpBridge.maxpat`:
  editable Max patch source
- `bridge/m4l/live_udp_bridge.js`:
  JavaScript router logic used by the patch/device
- `bridge/*.py`:
  bridge control and composition scripts

## Workflow Pattern

Typical usage loop:

1. Query catalog metadata (for example meter/BPM coverage) outside Live.
2. Select the next target context to compose.
3. Use this bridge to configure Ableton Live for that context.
4. Generate, write, or iterate composition material in-session.

## Requirements

- Ableton Live + Max for Live
- Python 3.10+
- local UDP access on ports `9000` (commands) and `9001` (ack/query responses)

## Quick Start

1. Clone:
```bash
git clone https://github.com/sunflower-of-parchman/codex-live-bridge.git
cd codex-live-bridge
```

2. In Ableton Live, drag `bridge/m4l/LiveUdpBridge.amxd` onto a MIDI track.

3. Verify bridge connectivity:
```bash
python3 bridge/ableton_udp_bridge.py --ack --status --no-tempo --no-signature --create-midi-tracks 0 --create-audio-tracks 0 --add-midi-tracks 0 --add-audio-tracks 0
```

4. Compose a one-minute marimba arrangement:
```bash
python3 bridge/compose_arrangement.py --minutes 1
```

## Source Editing

If you modify `bridge/m4l/live_udp_bridge.js` or
`bridge/m4l/LiveUdpBridge.maxpat`:

1. Copy updated JS into your Ableton User Library device folder.
2. Reload the device in Live (remove it from the track, then drag it back in).
3. Re-save `LiveUdpBridge.amxd` from Live/Max.
4. Copy updated `LiveUdpBridge.amxd` back into `bridge/m4l/`.

## Testing

```bash
python3 -m unittest discover -s bridge -p "test_*.py"
```

## License

MIT. See `LICENSE`.
