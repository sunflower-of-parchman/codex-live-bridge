# Listening for the Ghost in the Machine

Open source bridge connecting the Codex app to digital audio workstations (DAWs) for agentic composition assistance and catalog management.

## Vision

This repository is a teaching and build space for composition systems that combine:

- music fundamentals
- emotional and narrative intent from the human condition
- practical DAW execution workflows

The architecture is DAW-agnostic by design. Ableton Live is the first implementation example.

## Project focus

- Build a bridge that can work with any DAW
- Use Ableton Live + Max for Live + Live Object Model (LOM) as the first reference path
- Generate and test composition workflows driven by mood, meter, BPM, and key
- Keep outputs reusable for catalog development

## Workspaces

1. `codex-live-bridge`
2. `composition researcher-TA`

## Composition Researcher-TA output requirements

- instrument ranges (VST-specific when needed)
- mood definitions (written descriptions)
- narrative stories for each mood focused on the human condition and the world

Narrative guardrail: stories must not reference art making, music, or dance.

## Build constraints

- 1 automation
- 2 skills
- 2 workspaces
- dictation-first workflow

## Initial roadmap

1. Build core memory, logging, and repo docs
2. Seed ensemble and mood documentation from composition research outputs
3. Build the Max for Live bridge and connect to LOM
4. Expand bridge controls: note insert, velocity, automation, mixing, EQ, BPM/tempo, global key
5. Run bridge tests, ensemble tests, and eval loops
6. Produce 3-minute tracks and compare run-over-run behavior

## Composition canon

The canon structure is located at:

- `/Users/michaelwall/codex-live-bridge/docs/composition-canon/README.md`

## Open source status

This repo is public from day one:

- [sunflower-of-parchman/codex-live-bridge](https://github.com/sunflower-of-parchman/codex-live-bridge)

## Memory quick start

```bash
python3 /Users/michaelwall/codex-live-bridge/scripts/short_term_memory.py log --role user --text "Starting build now."
python3 /Users/michaelwall/codex-live-bridge/scripts/short_term_memory.py log --role assistant --text "Perfect, let's go."
python3 /Users/michaelwall/codex-live-bridge/scripts/short_term_memory.py show --limit 20
```

## Testing

```bash
python3 -m unittest discover -s /Users/michaelwall/codex-live-bridge/tests -p "test_*.py"
```

## Self-eval

```bash
python3 /Users/michaelwall/codex-live-bridge/scripts/run_self_eval.py
```

## Next best track suggestion

- Workflow doc: `/Users/michaelwall/codex-live-bridge/docs/workflows/supabase-track-gap-suggester.md`
- SQL query: `/Users/michaelwall/codex-live-bridge/sql/next_best_track_gap.sql`

## Launch Ableton Live

- Workflow doc: `/Users/michaelwall/codex-live-bridge/docs/workflows/launch-ableton-live.md`

## LiveAPI LOM bridge (port 9000)

- Run server: `python3 /Users/michaelwall/codex-live-bridge/scripts/run_live_bridge.py --port 9000 --backend udp-max-proxy --udp-port 9001`
- Send command: `python3 /Users/michaelwall/codex-live-bridge/scripts/send_live_command.py --url http://127.0.0.1:9000 --command set_tempo --payload '{"bpm":123}'`
- Bridge workflow: `/Users/michaelwall/codex-live-bridge/docs/workflows/liveapi-lom-bridge.md`
- LOM reference notes: `/Users/michaelwall/codex-live-bridge/docs/research/lom-current-references-2026-02-05.md`
- Max router script: `/Users/michaelwall/codex-live-bridge/max_for_live/live_api_command_router.js`
- Max copy/paste patch: `/Users/michaelwall/codex-live-bridge/max_for_live/live_bridge_receiver.maxpat`
