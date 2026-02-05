# AGENTS

Operational guide for agents and collaborators working in this repository.

## Project identity

- Project: `Listening for the Ghost in the Machine`
- Asset: open source bridge connecting Codex to DAWs for agentic composition assistance and catalog management

## Hard constraints

1. Keep exactly 1 automation active for the workflow.
2. Use exactly 2 skills for the active build cycle.
3. Operate across exactly 2 workspaces:
   - `codex-live-bridge`
   - `composition researcher-TA`
4. Build through dictation-first interaction whenever practical.

## Workspace responsibilities

### codex-live-bridge

- bridge implementation
- DAW integration
- test and eval harnesses
- documentation assembly

### composition researcher-TA

- instrument ranges (VST-specific when required)
- mood definitions (clear prose)
- narrative stories for each mood

Narrative rule: stories must focus on the human condition and the world, with no references to art making, music, or dance.

## Memory and logging protocol

1. Maintain short-term conversation memory in:
   - `/Users/michaelwall/codex-live-bridge/memory/conversation.jsonl`
2. Keep logging turned on during active sessions.
3. Log after every single commit.

## Commit protocol

1. Commit every turn when meaningful work is completed.
2. Use concise single-line messages that describe scope and change.
3. Keep the `main` branch in a runnable, documented state.

## Documentation deliverables

Create and maintain the following teaching files in markdown:

1. compositional canon (music fundamentals)
2. harmonic process
3. rhythmic process
4. melodic process
5. composition and arrangement process
6. composing for moods process
7. mood documentation
8. ensemble documentation

## Bridge implementation goals

1. Build a Max for Live device connected to the Live Object Model (LOM).
2. Practice and verify automatic Ableton Live open/start workflow.
3. Expand bridge control surface to include:
   - note insertion
   - velocity
   - automation lanes
   - mixing and EQ
   - tempo/BPM
   - global key

## Data and matrix goals

1. Build a Supabase skill that reads database state and creates a matrix view.
2. Use existing Supabase scale conventions.
3. Ensure mood signals remain visible in documentation and matrix outputs.

## Evals and test loop

1. Test the bridge.
2. Run evals.
3. Test ensemble practice.
4. Run evals again.
5. Produce 3-minute tracks from meter/BPM workflows.
6. Run self-comparative reflective evals across repeated runs.

## Capture protocol

Video-record each major stage of build, test, and eval execution.
