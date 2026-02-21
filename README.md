# codex-live-bridge

`codex-live-bridge` is an open-source, local-first Codex-to-Ableton Live
control bridge.

More precisely, this repo ships a Max for Live device, a JavaScript command
router, a Python OSC client/CLI, and higher-level workflow scripts that drive
Ableton Live through LiveAPI (the Live Object Model) over OSC/UDP.

Started during the OpenAI 2026 Hackathon in San Francisco, built in tandem with GPT-5.3-Codex.

## Included

- `bridge/m4l/LiveUdpBridge.amxd`: packaged drop-in Max for Live MIDI device
- `bridge/m4l/LiveUdpBridge.maxpat`: editable Max patch source
- `bridge/m4l/live_udp_bridge.js`: JavaScript router logic used by the patch
- `bridge/*.py`: OSC client and workflow scripts

## Live Object Model Control (LiveAPI over OSC/UDP)

The Max for Live device uses LiveAPI (Ableton Live Object Model) and exposes a
generic RPC surface over OSC/UDP. This lets Codex (or any OSC client) query,
set, call, inspect, and enumerate Live Object Model paths and properties.

Current `/api/*` endpoints:

- `/api/ping [request_id]`
- `/api/get <path> <property> [request_id]`
- `/api/set <path> <property> <value_json> [request_id]`
- `/api/call <path> <method> <args_json> [request_id]`
- `/api/children <path> <child_name> [request_id]`
- `/api/describe <path> [request_id]`

This provides broad Live control via LiveAPI/LOM, with an expanding surface
area as workflows are added.

Live Object Model reference:
[Cycling '74 Live Object Model docs](https://docs.cycling74.com/max8/vignettes/live_object_model)

## Data & Training

- This repo ships no trained model weights.
- This repo does not implement model training or fine-tuning pipelines.
- This repo does not include or ingest anyone else's music.
- Any workflow "learning" in this repo refers to optional local logging of your
  own run artifacts (when enabled), not ML training.
- If you are using Codex, that model is external to this repo; this repo is
  the local control and workflow layer around Ableton Live.

## Compositional Studio Assistant Workflow

A supported usage pattern in this repo is:

1. Keep track 1 in Ableton Live as `codex-bridge` (control/bridge, no instrument).
2. Put your first instrument on track 2.
3. Treat the track-2 instrument as the first instrument in your ensemble (first entry in your instrument registry).
4. Choose meter, BPM, and optional mood/key, then compose with workflow scripts.
5. Review eval artifacts, adjust constraints/guidance, and compose again.
6. Grow the ensemble one instrument at a time by updating registry/config and repeating the same compose+eval loop.

This pattern is implemented by shipped scripts such as
`bridge/setup_marimba_environment.py`, `bridge/compose_arrangement.py`,
`bridge/arrangement/marimba.py`, and `bridge/composition_feedback_loop.py`.
Script names currently reflect this repo's marimba-first starter profile, while
the arrangement runtime is registry-driven.

Current starter registry in this repo is marimba-only:
`bridge/config/instrument_registry.marimba.v1.json`.

Starter two-instrument example registry (marimba+piano):
`bridge/config/instrument_registry.marimba_piano.v1.json`.

## Current Composition Architecture

The current runtime uses layered composition decisions:

1. Macro form family (`legacy_arc`, `lift_release`, `wave_train`) is selected per run.
2. Section builder emits section labels plus per-section behavior paths (`piano_mode`, `hat_density`, keep-ratios).
3. Instrument registry maps source materials (`piano_chords`, `piano_motion`, drums) to named tracks.
4. Marimba identity applies family-level shaping plus micro strategy routing (`ostinato_pulse`, `broken_resonance`, `chord_bloom`, `lyrical_roll`).
5. Eval logging records run metadata, fingerprints, and reflection for next-run adaptation.

## Current Eval Coverage

Evals in `bridge/composition_feedback_loop.py` currently score symbolic
composition structure, not rendered audio quality.

Current artifact fields include:

- run metadata (`mood`, `key`, `tempo`, `meter`, minutes, bars, section size, status)
- per-section strategy paths (`form_labels`, `hat_density_path`, `piano_mode_path`)
- per-track note-count paths and created-clip counts
- structural fingerprints and a fingerprint hash

Similarity/novelty behavior:

- compares current fingerprint to a recent reference run using meter+BPM, ensemble signature, run family, and run status
- computes weighted `similarity_to_reference` and `novelty_score = 1 - similarity`
- records `reference_run_id`, `reference_match`, and `similarity_weights` in reflection
- uses high-resolution run IDs and collision-safe artifact paths
- emits repetition flags when trajectories repeat:
  - `overall_structure_highly_similar_to_recent_run`
  - `hat_density_trajectory_repeated`
  - `piano_mode_trajectory_repeated`

Merit rubric proxies (current):

- `pulse_clarity_proxy`
- `form_contrast_proxy`
- `instrument_diversity_proxy`
- `repetition_risk`

Instrument identity checks (when contract is present, e.g. marimba):

- marimba range adherence
- marimba leap-discipline adherence
- marimba attack-duration profile
- marimba/vibraphone overlap and answer ratios
- identity flags + reflection prompts for next run adjustments

Artifacts are persisted to:

- `memory/evals/compositions/<date>/<run_id>.json`
- `memory/evals/composition_index.json`

## Eval-Driven Memory Governance

The repo includes a bounded eval-to-memory loop:

- `python3 -m memory.eval_governance summarize --lookback 30`
- `python3 -m memory.eval_governance summarize --meter 5/4 --bpm 128 --lookback 20`
- `python3 -m memory.eval_governance apply --date YYYY-MM-DD --dry-run`
- `python3 -m memory.eval_governance apply --date YYYY-MM-DD`

What it does:

- Captures repeated eval signals into the current session note first.
- Promotes repeated signals into durable memory docs with `[gov:*]` markers.
- Demotes stale promoted rules into `memory/archive/demoted_guidance.md`.
- Generates active promoted guidance at `memory/governance/active.md`.

`python3 -m memory.retrieval brief ...` automatically includes this active
governance guidance so new compose runs consume it.

## Capabilities

Exact bridge command surface available now:

1. `/ping`
2. `/tempo <bpm>`
3. `/sig_num <numerator>`
4. `/sig_den <denominator>`
5. `/create_midi_track`
6. `/add_midi_tracks <count> [name]`
7. `/create_audio_track`
8. `/add_audio_tracks <count> [prefix]`
9. `/delete_audio_tracks <count>`
10. `/delete_midi_tracks <count>` (track 0 protected)
11. `/rename_track <track_index> <name>`
12. `/set_session_clip_notes <track_index> <slot_index> <length_beats> <notes_json> [clip_name]`
13. `/append_session_clip_notes <track_index> <slot_index> <notes_json>`
14. `/inspect_session_clip_notes <track_index> <slot_index>`
15. `/ensure_midi_tracks <target_count>`
16. `/midi_cc <controller> <value> [channel]`
17. `/cc64 <value> [channel]`
18. `/status`
19. `/api/ping [request_id]`
20. `/api/get <path> <property> [request_id]`
21. `/api/set <path> <property> <value_json> [request_id]`
22. `/api/call <path> <method> <args_json> [request_id]`
23. `/api/children <path> <child_name> [request_id]`
24. `/api/describe <path> [request_id]`

ACK behavior:

- The bridge emits OSC acknowledgements using `/ack`.
- For `/api/*`, an optional trailing `request_id` is echoed in ACK responses
  when provided.
- The Python client can listen on the ACK port and print summarized ACK output.

## Topology (Ports and Transport)

- Default host: `127.0.0.1`
- Command channel: UDP `9000`
- ACK/query response channel: UDP `9001`
- The Python client encodes OSC packets using the Python standard library.
- The Max for Live device routes commands to LiveAPI inside
  `bridge/m4l/live_udp_bridge.js`.

## Shipped Workflows

- `bridge/ableton_udp_bridge.py`: general OSC command client/CLI with ACK
  listening and command batching modes
- `bridge/setup_marimba_environment.py`: set tempo/signature/key, resolve track,
  create/clean arrangement clip, optional save policy
- `bridge/compose_arrangement.py`: registry-driven arrangement writing with
  sections, groove handling, write strategies, save policy, optional logging
- `bridge/arrangement/multi_pass.py`: multi-pass pipeline
  (`seed_layout`, `form_density`, `repetition_development`, `dynamic_arc`,
  `cadence`, then `polish_*`)
- `bridge/arrangement/marimba.py` +
  `bridge/config/marimba_identity.v1.json`: marimba identity and strategy layer
  (including pair-mode shaping)
- `bridge/composition_feedback_loop.py`: optional eval artifacts (novelty,
  similarity, repetition flags, reflection prompts)
- `bridge/compose_kick_pattern.py`: kick arrangement clip writer
- `bridge/compose_rim_pattern.py`: rim arrangement clip writer
- `bridge/compose_hat_pattern.py`: hat arrangement clip writer
- `bridge/compose_piano_pattern.py`: piano arrangement clip writer
- `bridge/compose_codex_arpeggio.py`: demo session-clip arpeggio writer
- `bridge/run_next_track.py`: high-level next-track orchestration wrapper (supports `--instrument-registry-path`)
- `bridge/dump_marimba_params.py`: dump exposed device parameter values via
  `/api/*`
- `bridge/full_surface_smoke_test.py`: full-surface bridge smoke script
- `bridge/benchmark_midi_write.py`: deterministic MIDI write benchmark harness

## Requirements

To run the bridge and workflow scripts:

- Ableton Live with Max for Live support:
  [Ableton Live](https://www.ableton.com/en/live/) and
  [Max for Live](https://www.ableton.com/en/live/max-for-live/)
- Python 3.10+:
  [python.org downloads](https://www.python.org/downloads/)
- local UDP access on ports `9000` (commands) and `9001` (ack/query responses)

To edit bridge/device internals:

- For `bridge/m4l/LiveUdpBridge.maxpat`, use the Max for Live editor in Live or
  [Cycling '74 Max](https://cycling74.com/products/max).
- For `bridge/m4l/live_udp_bridge.js`, edit JavaScript source and reload the
  device in Live (this repo does not require a Node.js runtime for this file).

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

5. Optional two-instrument composition example:
```bash
python3 bridge/compose_arrangement.py --minutes 2 --instrument-registry-path bridge/config/instrument_registry.marimba_piano.v1.json
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
