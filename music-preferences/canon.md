# Canon

Stable guidance for how we compose in this library. Canon is principles and
operating rules, not a storage area for fixed bar maps or instrument-specific
patterns.

## Normative Language

- MUST = required.
- SHOULD = default behavior; may be overridden if the override is written in the session note.
- MAY = optional.

## Scope

- Canon defines durable principles, operating discipline, and governance triggers.
- `memory/fundamentals/*.md` holds detailed methods and topic strategy.
- `memory/instruments/*.md` holds instrument behavior and track-specific rules.
- Promote guidance into Canon only after it repeats across independent sessions.

## Document Precedence

When guidance conflicts, apply this order:

1. User-authored principles (below)
2. Canon (this document)
3. `memory/fundamentals/*.md`
4. `memory/instruments/*.md`
5. Current run plan + session note

Overrides MUST be written in the session note.

## Workflow Rule: Arrangement Timeline Only

- All work MUST happen on the Arrangement timeline.
- All musical material MUST be placed and developed on the timeline.
- Do not use launch/trigger workflows to build structure.
- If "Back to Arrangement" is lit, restore Arrangement control before continuing and before ending the run.

## Originality and Non-Imitation

- The system MUST generate original music.
- Do not copy, mimic, or emulate specific artists, catalogs, or recognizable templates.
- Do not aim to reproduce a known genre form.
- If a request asks for "in the style of" a specific artist or a recognizable template, translate it into abstract constraints (time behavior, density arc, register focus, timbral palette, contrast plan, spatial approach) and proceed without reference to the source.
- If a familiar resemblance appears during work, change parameters (see Governance).

## Core Terms

- Intent: one sentence describing what the piece should do to the space.
- Anchor: one stable reference to preserve (time behavior, recurring gesture, pitch center or spectral center, timbral signature, register focus, density floor, spatial approach).
- Risk: one deliberate departure from defaults (time elasticity, texture rupture, pitch-field shift, register inversion, density reframe, abrupt contrast, extended restraint).
- Legibility: how quickly the organizing logic becomes clear.
- Macro divergence: a change that creates a new section identity (temporal framework, density regime, pitch field, instrumentation, register, spatial world, form logic).
- Session note: record of what happened and why in this run.

## User-Authored Principles

- The job is fit to context and use.
- The track should change the space when it starts.
- Descriptive language is valid musical input. Translate it into musical constraints.
- Time is treated as structure. Pulse, tempo, meter, and accents are parameters.
- Open time is a valid structure. In open time, prioritize layers and dynamics.
- Timbre is a structural choice. Choose timbres with a purpose. Maintain a growing instrument library and use it with restraint.
- Repetition is allowed. Keep repetition active through internal motion (orchestration, texture, pitch gravity, density).
- Harmony is treated as gravity. Add harmonic change when it improves clarity, motion, or fit.
- Form should be readable. Use sections, returns, transitions, and designed endings.
- Recorded production is a primary medium. The studio is part of composition.
- Finishing is part of the system. Unfinished work does not enter the catalog.
- Define success before starting. Treat the definition as an artifact.
- Work in loops: compose -> render -> evaluate -> store memory -> adjust.

## Temporal Framework

The system MUST choose a temporal framework each run.

- Metered: explicit meter and bars.
- Metric: clear periodicity without strong bar emphasis.
- Open: no persistent periodicity; time shaped by event timing, gesture, texture, and dynamics.
- Hybrid: intentional shifts or overlays of the above.

If temporal framework is not specified, consider at least two options (one open/hybrid and one metric/metered), choose one, and write the reason in the session note.

## Preflight (Required)

Before any composition action:

1. Write intent (one sentence).
2. Choose temporal framework (metered / metric / open / hybrid).
3. Choose one anchor to preserve.
4. Choose one risk to attempt.
5. Set the structural frame.

Metered/metric:

- meter + BPM (metered) OR periodicity character (metric)
- pitch center / pitch field (or intentional ambiguity)
- density floor + register focus
- section contrast plan

Open:

- time-shape plan
- event timing plan (spacing and clustering)
- spectral center / pitch field (or intentional ambiguity)
- density floor + register focus
- section contrast plan

Hybrid:

- transition points
- what remains continuous across the shift

6. Decide what "done for this pass" means.
7. Choose any supporting documents to consult:

- `memory/fundamentals/*.md`
- `memory/instruments/*.md`

## Instrument Registry

- Tracks MUST have stable names.
- Track names are used for automation and must remain consistent.
- Instrument-specific rules belong in `memory/instruments/`.

## Run Logging and Coverage

Each run MUST log:

- temporal framework used
- a coverage target (time behavior, density regime, register focus, texture class, form type, spatial approach)

Coverage selection order:

1. temporal framework
2. structural parameters
3. mood/affect

## Evaluation Rubric

After each pass, evaluate with a simple score (0-5 is enough) for:

- space change at start
- legibility of the organizing logic
- anchor clarity
- risk realized
- temporal coherence
- novelty
- repeat value
- timbre usefulness
- dynamic arc clarity
- mix usability in a room
- deliverable fitness (length, edits, loop points if needed)

Write in the session note:

- what changed vs repeated
- what to keep / cut / exaggerate next
- next-pass directives written plainly
- any candidate guidance for promotion
- any imitation risk detected

## Promotion Rule

- New guidance starts in the session note.
- Promote to Canon only after it repeats across independent sessions and does not conflict with existing Canon.

## Governance Rules

- [gov:low_novelty] If novelty is repeatedly low, preserve one anchor and force one macro divergence next run.
- [gov:temporal_bias] If one temporal framework dominates recent work, force a different framework next run while preserving one anchor.
- [gov:imitation_risk] If the piece resembles a recognizable external template, change at least two of:
- temporal framework or time behavior
- pitch field / spectral center
- instrumentation / timbre world
- form logic
- density regime / register focus

Log the change in the session note.

## Canon Documents (Stable Set)

These documents are expected to exist and evolve:

- Ensemble (instrument families, ranges, layering norms, density mapping)
- Time (temporal frameworks, meter habits, accent habits, when to remove periodicity)
- Harmony (centers, pitch fields, change rates, clarity rules)
- Texture (timbre rules, sound design bounds, environment building)
- Form (section lengths, returns, transitions, endings)
- Mix (clarity rules for rooms, low end, transients, space)
- Economics (constraints planning)
- Release (saving, organizing, finishing)
- Job-well-done (rubric templates)
