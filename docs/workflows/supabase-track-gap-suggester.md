# Supabase Track Gap Suggester

Use this workflow to return exactly one next-track suggestion from `public.tracks`.

## Goal

Suggest one underrepresented or missing `(meter, bpm, mood)` combination.

## Required schema fields

- `public.tracks.meter` (text)
- `public.tracks.mood` (text)
- `public.tracks.tempo` (integer)

## Query file

- `/Users/michaelwall/codex-live-bridge/sql/next_best_track_gap.sql`

## Current deterministic output (2026-02-05)

- meter: `11/4`
- bpm: `60`
- mood: `Ambient`
- rationale: `0 existing tracks` for that combo

## Notes

- Empty strings are treated as nulls before normalization.
- Missing meter/mood values are normalized to `unknown`.
- BPM target coverage defaults to `60-180`.
