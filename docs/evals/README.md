# Evals

This folder tracks creative, proactive, and reflective self-evaluation.

## Purpose

Use evals to improve composition quality and agent behavior across repeated runs.

## Components

1. `simple-self-eval-rubric.md`:
human-readable rubric for quick review
2. `reflection-log.md`:
rolling log of scores, reflections, and next actions
3. `/Users/michaelwall/codex-live-bridge/evals/self_eval_rubric.json`:
machine-readable rubric with weighted criteria
4. `/Users/michaelwall/codex-live-bridge/scripts/run_self_eval.py`:
report generator from rubric + scores

## Cadence

1. Run eval at the end of each meaningful build phase.
2. Log one reflection and one concrete next action.
3. Re-run after implementation to compare progress.
