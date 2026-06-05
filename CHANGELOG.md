# Changelog

All notable changes to this project are documented in this file.

This project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Corrected the note schema to accept documented negative
  `velocity_deviation` values.
- Made drum-chain note writes distinguish unreadable readback from a verified
  unapplied write.
- Made every error ACK end with an explicit reserved request-ID correlation
  trailer so clients can preserve all error details and parse older untagged
  ACKs safely.
- Made helper failures terminal so request-aware status wrappers and batch
  track mutations do not emit completion ACKs after an error.

## [3.0.0] - 2026-06-03

### Added

- v3 Live Object Model protocol documentation in `PROTOCOL.md`.
- Observer lifecycle commands with CLI listen mode:
  `/api_observe`, `/api_unobserve`, `/api_observers`, `/api_clear_observers`,
  and asynchronous `/ack api_event` summaries.
- Named LiveAPI wrappers for session context, theory status, tuning status,
  device inventory, device parameters, mixer status, bounded parameter writes,
  Live 12.3 native device insertion, rack chain insertion, and drum-chain note
  assignment.
- Modern note dictionary support for `probability`, `velocity_deviation`, and
  `release_velocity`.
- Repository docs tests and Max patch JSON validation in CI.

### Changed

- Recentered the public repo on the bridge: Max patch source, JavaScript
  router, Python OSC client, protocol docs, and validation tests.
- Rewrote `README.md` and `bridge/commands.md` around the current v3 bridge
  surface.
- Restored explicit public language that this repo ships no trained model
  weights, training pipeline, audio corpus, or generative music model.
- Hardened mutating smoke validation behind an explicit confirmation flag.
- Hardened write wrappers so missing/null parameter values, ambiguous
  parameter value types, and invalid insertion indexes fail before reaching
  LiveAPI.
- Restricted unmatched OSC fallback dispatch to the documented named wrappers.
- Made the mutating smoke test abort before writes unless track creation adds
  exactly one appended MIDI track.
- Normalized LiveAPI paths before deriving child paths or returning wrapper
  ACK payloads.
- Documented the Live-hosted Max MIDI Effect packaging path from tracked
  `.maxpat` source to loadable `.amxd` release artifact.
- Made CLI listen-mode socket bind failures return a nonzero process status.

### Removed

- Removed the old local memory package, preference/eval templates, composition
  benchmark, and implementation-plan notes from the public v3 surface. Those
  materials belonged to an earlier higher-level direction and made this repo
  less clear as a standalone Ableton Live bridge.
