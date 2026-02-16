# codex-compose

`codex-compose` is an open-source composition workspace for generating musical
ideas with:

- an Ableton Live bridge (OSC/UDP + Max for Live)
- a registry-driven arrangement engine
- durable compositional memory for iterative creative practice

The goal is not one perfect generator. The goal is better range, better
decision-making, and clearer reflection over repeated composition runs.

## Project Status

This repository is actively evolving. The current runtime path is:

- Python scripts in `bridge/`
- Max for Live bridge device in `bridge/m4l/`
- Ableton Live as the execution surface

## Features

- OSC/UDP bridge control for Ableton Live (`bridge/ableton_udp_bridge.py`)
- arrangement generation with section arcs and instrumentation control
- instrument registry support with per-instrument MIDI register constraints
- compositional reflection artifacts and novelty/repetition scoring
- durable memory workflow under `memory/`

## Repository Layout

- `bridge/`:
  Live bridge, composition scripts, and bridge/arrangement tests
- `memory/`:
  canon, fundamentals, mood context, sessions, and work journal
- `docs/`:
  vision and audit documentation
- `output/`:
  generated run artifacts

## Requirements

- Python 3.10+ (stdlib-only scripts for core bridge workflows)
- Ableton Live + Max for Live (for live bridge execution)
- local UDP access on ports `9000` (commands) and `9001` (acks)

## Quick Start

1. Clone the repo:
```bash
git clone <repo-url>
cd <repo-dir>
```

2. Load the Max for Live bridge device in Ableton Live:
- drag `bridge/m4l/LiveUdpBridge.amxd` onto a MIDI track
- source-edit fallback: use `bridge/m4l/LiveUdpBridge.maxpat`
- ensure Live is listening on UDP `9000`

3. Verify bridge connectivity:
```bash
python3 bridge/ableton_udp_bridge.py --ack --status --no-tempo --no-signature --create-midi-tracks 0 --create-audio-tracks 0 --add-midi-tracks 0 --add-audio-tracks 0
```

4. Compose a one-minute marimba arrangement:
```bash
python3 bridge/compose_arrangement.py --minutes 1
```

## Live Registry Workflow

The default runtime path is marimba-only:

- default registry: `bridge/config/instrument_registry.marimba.v1.json`
- default track naming mode: `registry`

Additional registry examples are included:

- `bridge/config/instrument_registry.v1.json`:
  default template registry
- `bridge/config/instrument_registry.live_set.v1.json`:
  current Live-set-aligned registry with explicit register ranges

Use a different registry at runtime:

```bash
python3 bridge/compose_arrangement.py \
  --minutes 1 \
  --instrument-registry-path bridge/config/instrument_registry.live_set.v1.json
```

## Memory Workflow

Durable context lives in `memory/`:

- `memory/index.toml`:
  structured focus and references
- `memory/canon.md`:
  stable compositional guidance
- `memory/fundamentals/`:
  rhythm, timbre, harmony, velocity notes
- `memory/moods.md`:
  local mood shorthand and context links
- `memory/instrument_ranges.md`:
  range source map and range policy
- `memory/sessions/`:
  per-session learning snapshots

Quick usage:

```bash
python3 -m memory.compositional_memory --list
python3 -m memory.compositional_memory --fundamental rhythm
```

## Testing

Run arrangement and bridge tests:

```bash
python3 -m unittest discover -s bridge -p "test_*.py"
```

Run memory tests:

```bash
python3 -m unittest discover -s memory -p "test_*.py"
```

## Contributing

Contributions are welcome.

- open an issue for bugs, regressions, or composition workflow gaps
- submit focused pull requests with clear rationale
- include tests when behavior changes

## License

This project is licensed under the MIT License. See `LICENSE`.
