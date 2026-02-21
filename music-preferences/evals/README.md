# Eval Artifacts

This directory stores local eval artifacts used by retrieval and governance.

## Paths

- `memory/evals/composition_index.json`: index of runs and artifact paths.
- `memory/evals/compositions/<date>/<run_id>.json`: per-run artifact payloads.
- `memory/evals/retrieval_index.sqlite`: generated retrieval index (created by `memory.retrieval`).

## Minimal Artifact Fields

Each artifact should include at least:

- `run_id`
- `timestamp_utc`
- `status`
- `composition` object (for example mood/key/tempo/signature)
- `fingerprints` object (for example meter_bpm)
- `reflection` object (for example novelty_score/repetition_flags/merit_rubric)

The governance loop reads `composition_index.json` first, then falls back to scanning `compositions/**.json`.
