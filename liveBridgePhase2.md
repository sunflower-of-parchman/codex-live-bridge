Title: Live Bridge Phase 2

Reference: `/Users/michaelwall/codex-live-bridge/PLANS.md`

Goal

Strengthen the Ableton Live bridge by adding higher-level transport/clip controls, a practical command-line sender, and explicit end-to-end validation so composition operations can be executed reliably from Codex workflows.

Scope

1. Extend bridge protocol and capabilities with additional commands:
   - `create_midi_clip`
   - `fire_clip`
   - `stop_track`
   - `set_track_mute`
   - `set_track_solo`
2. Extend the Max for Live LiveAPI router to execute those commands.
3. Add a CLI sender to post commands to bridge port `9000`.
4. Add/update tests for protocol, service behavior, and CLI command construction.
5. Update workflow docs and README references.

Out of Scope

1. Building a `.amxd` binary package.
2. Advanced arrangement editing or scene graph generation.
3. Cloud deployment or remote networking.

Constraints

1. Keep bridge default host/port local (`127.0.0.1:9000`).
2. Preserve existing command compatibility.
3. Keep implementation dependency-light (standard library only).
4. Maintain commit/push/log protocol for this turn.

Definitions

1. LOM: Live Object Model, the object/property graph exposed by Ableton Live to Max/LiveAPI.
2. Clip slot: Position in a track that may contain one clip.
3. Transport controls: Commands that trigger/stop playback behavior.

Implementation Plan

1. Update `/Users/michaelwall/codex-live-bridge/live_bridge/protocol.py` validators and supported command map.
2. Update `/Users/michaelwall/codex-live-bridge/live_bridge/capabilities.py` with new command metadata.
3. Update `/Users/michaelwall/codex-live-bridge/max_for_live/live_api_command_router.js` with new handlers.
4. Add `/Users/michaelwall/codex-live-bridge/scripts/send_live_command.py` for bridge command posting.
5. Add/update tests:
   - `/Users/michaelwall/codex-live-bridge/tests/test_live_bridge_protocol.py`
   - `/Users/michaelwall/codex-live-bridge/tests/test_send_live_command.py`
6. Update docs:
   - `/Users/michaelwall/codex-live-bridge/docs/workflows/liveapi-lom-bridge.md`
   - `/Users/michaelwall/codex-live-bridge/README.md`

Validation Plan

1. Run:
     python3 -m unittest discover -s /Users/michaelwall/codex-live-bridge/tests -p "test_*.py"
   Expected output: all tests pass.
2. Run a local smoke command in mock backend mode:
     python3 /Users/michaelwall/codex-live-bridge/scripts/run_live_bridge.py --backend mock --port 9010
   and then:
     python3 /Users/michaelwall/codex-live-bridge/scripts/send_live_command.py --url http://127.0.0.1:9010 --command set_tempo --payload '{"bpm":123}'
   Expected output: JSON response with `"ok": true`.

Acceptance Criteria

1. New transport/clip commands are validated by protocol and represented in capabilities.
2. Max router contains executable handler stubs for the same commands.
3. CLI sender can post arbitrary command payloads to bridge endpoint.
4. Docs reflect the new commands and usage path.
5. Full test suite passes.

Risks & Mitigations

1. Risk: LiveAPI method/path differences across Live versions.
   Mitigation: Keep handlers small, path-based, and document compatibility assumptions.
2. Risk: Command mismatch between Python protocol and Max router.
   Mitigation: Add explicit tests for supported command set and keep names consistent.

Progress

- [x] Create `PLANS.md` template for repo-level ExecPlans.
- [x] Extend protocol and capabilities with additional clip/transport commands.
- [x] Extend Max router handlers for new commands.
- [x] Add CLI sender and tests.
- [x] Update docs and run full validation.
- [x] Commit, push, and log this turn.

Surprises & Discoveries

- The repository did not yet contain `PLANS.md`; a template was added first to satisfy ExecPlan workflow requirements.
- Port `9000` can already be occupied during local smoke tests, so validation smoke runs use `9010` when needed while keeping `9000` as default.

Decision Log

- 2026-02-05: Added repo-local `PLANS.md` before writing task plan -> required by skill workflow and improves repeatability.
- 2026-02-05: Chose `liveBridgePhase2.md` as plan filename -> concise lowerCamelCase and aligned with current build milestone.
- 2026-02-05: Added `scripts/send_live_command.py` CLI -> simplifies deterministic testing and operator usage from terminal sessions.

Outcomes & Retrospective

Phase-2 bridge goals were completed in code and tests:

1. Added clip/transport commands (`create_midi_clip`, `fire_clip`, `stop_track`, `set_track_mute`, `set_track_solo`) across protocol, capabilities, and Max router.
2. Added CLI sender (`/Users/michaelwall/codex-live-bridge/scripts/send_live_command.py`) for direct command posting.
3. Added tests for command validation and CLI request behavior.
4. Full test suite passed and smoke command succeeded in mock backend mode.
