# Contributing

Thanks for your interest in improving `codex-live-bridge`.

## Before You Start

- Read `README.md` for project scope and runtime requirements.
- Read `INSTALL.md` before exporting a packaged `.amxd` release artifact.
- Read `PROTOCOL.md` before changing OSC addresses, ACK shapes, LiveAPI
  wrapper behavior, or safety classes.
- Keep changes small and focused.
- Do not include secrets, API keys, local absolute paths, rendered media, or
  machine-specific artifacts in commits.

## Local Validation

Run from repo root:

```bash
python3 -m unittest discover -s bridge -p "test_*.py"
python3 -m unittest discover -s tests -p "test_*.py"
node --check bridge/m4l/live_udp_bridge.js
python3 -m json.tool bridge/m4l/LiveUdpBridge.maxpat >/dev/null
bash .github/scripts/audit_public_hygiene.sh
```

If a change touches only one area, include the most relevant command in your
pull request description. Before merging, the full validation set should pass.

## Documentation Sync

Protocol, CLI, Max patch, and JavaScript router changes should update the
public docs in the same pull request:

- `PROTOCOL.md` for exact command contracts, ACK shapes, and safety classes.
- `bridge/commands.md` for short examples.
- `README.md` for capability summaries and user-facing setup.
- Tests under `bridge/` or `tests/` for behavior and docs guardrails.

## Pull Request Expectations

- Explain what changed and why.
- Link any related issue.
- Add or update tests when behavior changes.
- Keep pull requests reviewable; split very large changes when possible.

## Scope and Maintainer Capacity

This project is maintained by one person on a best-effort basis. Not every
feature request can be accepted, and response time may vary.
