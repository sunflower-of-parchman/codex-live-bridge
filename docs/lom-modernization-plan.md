# LOM Modernization Plan

This plan upgrades the public Live Object Model bridge while keeping Ableton Push out of scope. The current target is a clone-safe, Max for Live based OSC/UDP bridge that exposes LiveAPI clearly enough for external agents and scripts to inspect, mutate, observe, and test Ableton Live sets.

## Scope

- Keep the bridge centered on Max for Live, LiveAPI, OSC/UDP, and the Python CLI.
- Keep Push integration out of this plan.
- Prefer path-first LiveAPI access because Live object IDs are dynamic across set loads.
- Treat Live runtime validation as a separate gate from static/unit validation.
- Preserve current local-only defaults: command port `9000`, ACK/event port `9001`, host `127.0.0.1`.

## Milestones

### 1. Public Protocol Baseline

Document the shipped protocol surface before expanding it:

- Core RPC: `/api/ping`, `/api/get`, `/api/set`, `/api/call`, `/api/children`, `/api/describe`.
- Observer lifecycle: `/api_observe`, `/api_unobserve`, `/api_observers`, `/api_clear_observers`, and async `/ack api_event` payloads.
- ACK correlation with optional trailing `request_id`.
- Error shape, payload limits, safety classes, and LiveAPI constraints.
- Modern note dictionary fields for Live 12 era note operations.

Acceptance:

- `PROTOCOL.md` exists and is linked from `README.md`.
- `bridge/commands.md` reflects observer commands and note schema.
- CI checks Python tests and Max JS syntax.

### 2. Structured Client Events

Make the Python client parse ACK/event packets into structured data while preserving the current human-readable summaries:

- Add an `AckEvent` model.
- Extract event name, request ID, parsed JSON payloads, and error status.
- Keep existing CLI output stable for current users.

Acceptance:

- Unit tests cover request-correlated RPC ACKs and observer events.
- Existing summary output tests still pass.

### 3. Modern Note Support

Extend note dictionary normalization without changing the public command shape:

- Allow velocity `0`.
- Preserve optional note fields: `probability`, `velocity_deviation`, and `release_velocity`.
- Keep `pitch`, `start_time`, `duration`, `velocity`, and `mute` validation explicit.

Acceptance:

- Static tests verify the Max JS router includes the modern note fields.
- `node --check bridge/m4l/live_udp_bridge.js` passes.

### 4. Read-Only Status Wrappers

Add named wrappers for high-value read/status operations that are easier for agents to call than generic RPC:

- Session context: tempo, time signature, track counts, selected scene/track if available.
- Theory status: root note, scale name, scale intervals/readability where Live exposes them.
- Tuning status: tuning system and related read-only metadata where Live exposes it.

Acceptance:

- Wrappers degrade gracefully when a Live property is absent.
- Tests assert command construction and ACK parsing.

### 5. Device And Mixer Wrappers

Add bounded wrappers around common device, parameter, mixer, and EQ workflows:

- Enumerate tracks, devices, parameters, names, values, ranges, and automation state.
- Set parameter values by path and parameter index/name when writable.
- Avoid arbitrary automation breakpoint writing unless the LOM exposes a verified method.

Acceptance:

- Docs classify read, bounded write, and destructive operations.
- Runtime smoke scripts can inspect and change a safe parameter in a test Live set.

### 6. Live 12.3 Native Insertion

Gate new native Live device insertion APIs behind runtime feature detection:

- `Track.insert_device`.
- `Chain.insert_device`.
- `RackDevice.insert_chain`.
- `DrumChain.in_note` where applicable.

Acceptance:

- Docs state native Live devices only.
- The bridge reports unsupported methods cleanly on older Live versions or non-native targets.

### 7. Observer Runtime Loop

Make observation useful for long-running agent sessions:

- Add a persistent listen mode in the Python client.
- Add observer quotas/throttling.
- Avoid mutation directly inside observer callbacks.
- Provide examples for tempo, track name, selected track, and clip state monitoring.

Acceptance:

- Unit tests cover listen-mode parsing boundaries.
- Runtime smoke test shows event delivery and cleanup.

### 8. Public Release Gate

Before publishing a release, run the public audit and runtime checklist:

- Python unit tests.
- Max JS syntax check.
- Public hygiene audit.
- Protocol docs link check.
- Ableton Live smoke matrix when Live is available.

Acceptance:

- The branch can be merged without private paths, secrets, generated local logs, or stale mirror artifacts.
- Release notes distinguish statically validated features from Live-runtime validated features.
