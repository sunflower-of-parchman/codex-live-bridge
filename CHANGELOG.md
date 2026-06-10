# Changelog

All notable changes to this project are documented in this file.

This project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.1.0] - 2026-06-10

### Added

- Added the protocol 3.1 `/api/session_clip_inspect` read endpoint with
  required request correlation, stable raw-facts metadata, adaptive device and
  note pages, empty-clip success, snapshot-change detection, and a 4096-byte
  encoded OSC packet budget.
- Added Python CLI construction, concise fragment parsing, completion-aware
  ACK collection, and a public strict fragment assembler.
- Documented the bridge's role in the hybrid Live architecture: external and
  headless automation, observers, generic RPC, raw MIDI, and release-Live
  compatibility remain public bridge responsibilities.

### Changed

- Added a fail-closed local capability token for writes, generic LiveAPI calls,
  observer lifecycle changes, insertion, track and clip mutations, and MIDI
  output. Read-only inspection remains tokenless.
- Moved UDP handling to Max's deferred queue, limited receiver admission to one
  datagram per 10 ms, capped track batches at 32, and capped ensured MIDI track
  targets at 256.
- Documented packet policy, fragment transfer semantics, correlated inspection
  errors, and byte-for-byte compatibility for legacy
  `/inspect_session_clip_notes` ACKs.
- Established the final protocol 3.1 V1 contract at schema version `1` and
  producer version `3.1.0`; unpublished development snapshots have no
  wire-compatibility promise and are not parsed as alternate shapes.
- Froze session clip inspection V1 around complete nine-field extended notes,
  explicit nullable text metadata, index-ordered device/note page phases, and
  bounded diagnostics for every endpoint error ACK.
- Bounded V1 inspection resources to 4096 notes, 256 devices, 1024 fragments,
  and 16 active Python assemblies, with bounded ACK retention and terminal
  state eviction.
- Documented the adapter boundary for shared schema validation, note
  normalization, inspection, reporting, and parity without adding an
  Extensions SDK dependency to this repository.
- Documented the qualified SDK note-ID and release-velocity gap for
  cross-surface consumers. This records an SDK-side limitation and does not
  weaken the bridge's nine-field V1 note contract.

### Fixed

- Preserved Live device `type` as its numeric `0`/`1`/`2`/`4` enum and used
  explicit `null` for unavailable or invalid values.
- Preserved finite signed clip marker and loop positions while rejecting
  reversed, nonfinite, or non-representable ranges.
- Aligned the JavaScript producer and Python assembler on note scalar types,
  nullable device/track/clip metadata, complete-fragment totals, and strict
  fragment ordering.
- Corrected the note schema to accept documented negative
  `velocity_deviation` values.
- Made drum-chain note writes distinguish unreadable readback from a verified
  unapplied write.
- Made every error ACK end with an explicit reserved request-ID correlation
  trailer so clients can preserve all error details and parse older untagged
  ACKs safely.
- Made helper failures terminal so request-aware status wrappers and batch
  track mutations do not emit completion ACKs after an error.
- Made unmatched wrapper exceptions emit correlated internal-error ACKs
  instead of leaving clients to time out.
- Made fallback error correlation follow each wrapper's actual optional
  request-ID position.
- Hardened inspection clients with strict OSC decoding, a 4096-byte inbound
  datagram limit, and aggregate fragment-retention bounds.
- Rejected prototype-sensitive observer IDs and moved observer storage to a
  prototype-free registry.

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
