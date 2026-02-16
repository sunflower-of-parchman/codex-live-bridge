# codex-live-bridge

`codex-live-bridge` is an open-source bridge that lets the Codex app control
Ableton Live across musical workflows.

This project is designed for Codex-driven Live control, not only raw
transport/control commands.

Started during the OpenAI 2026 Hackathon in San Francisco.

## What This Enables

- Codex app orchestration of Ableton Live setup, writing, and iteration tasks
- workflow-specific routines powered by your own documents and notes
  (for example harmony, melody, rhythm, timbre, and velocity references)
- repeatable "prepare the session and start working" loops
- automation-first creative routines while keeping personal project data in a private source repo

## Included

- `bridge/m4l/LiveUdpBridge.amxd`:
  packaged drop-in Max for Live MIDI device
- `bridge/m4l/LiveUdpBridge.maxpat`:
  editable Max patch source
- `bridge/m4l/live_udp_bridge.js`:
  JavaScript router logic used by the patch/device
- `bridge/*.py`:
  bridge control and composition scripts

## Example Use Cases

- composition setup from catalog metadata (for example meter/BPM targeting)
- harmony and arrangement workflows driven by your own music docs
- sound design and session prep routines that Codex can execute repeatedly
- reusable Ableton setup flows for writing, rehearsal, and production sessions

## Workflow Pattern

Typical usage loop (for any musical use case):

1. Query catalog metadata (for example meter/BPM coverage) outside Live.
2. Select the next musical context to work on.
3. Use this bridge to configure Ableton Live for that context.
4. Generate, write, perform, or iterate material in-session.

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

4. Optional composition example:
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
