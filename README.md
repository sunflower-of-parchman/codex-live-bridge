# codex-live-bridge

`codex-live-bridge` is an open-source, local-first bridge for controlling
Ableton Live from Codex or another local automation client.

The repo currently ships:

- a Python OSC/UDP client and CLI
- an editable Max for Live patch source
- a JavaScript LiveAPI router for the patch
- local memory and eval template tooling
- unit tests for the public Python bridge surface

This project is independent and is not affiliated with or endorsed by OpenAI,
Ableton, or Cycling '74. All trademarks belong to their respective owners.

## Current Status

| Area | Status |
| --- | --- |
| Python OSC client | Implemented in `bridge/ableton_udp_bridge.py` |
| Max for Live patch source | Implemented in `bridge/m4l/LiveUdpBridge.maxpat` |
| Max JavaScript router | Implemented in `bridge/m4l/live_udp_bridge.js` |
| Packaged `.amxd` device | Release artifact workflow; not tracked on current `main` |
| Automated bootstrap/doctor | Planned |
| Composition arrangement generator | Planned / not tracked on current `main` |
| CI | GitHub Actions unit-test workflow |

The implemented target today is Ableton Live with Max for Live. Other DAWs are
architecture targets, not current user-facing implementations.

## Requirements

- Ableton Live with Max for Live support for runtime use
- Python 3.10+ for CLI tools and tests
- Node.js only if you want to syntax-check the Max JavaScript router locally

The Python code uses the standard library.

## Quick Start

Clone and run the local test surface:

```bash
git clone https://github.com/sunflower-of-parchman/codex-live-bridge.git
cd codex-live-bridge
python3 -m unittest discover -s bridge -p "test_*.py"
```

Run a dry-run command build without contacting Ableton:

```bash
python3 bridge/ableton_udp_bridge.py --dry-run --status --no-tempo --no-signature
```

To use the bridge with Ableton Live:

1. Open Ableton Live with Max for Live available.
2. Load or recreate the Max patch from `bridge/m4l/LiveUdpBridge.maxpat`.
3. Keep `bridge/m4l/live_udp_bridge.js` next to the patch so `[js live_udp_bridge.js]` resolves.
4. Run a status check:

```bash
python3 bridge/ableton_udp_bridge.py --ack --status --no-tempo --no-signature
```

Expected ACK shape:

```text
ack:  /ack pong
ack:  /ack status <total_tracks> <midi_tracks> <audio_tracks> <return_tracks> live_set <id>
```

## Bridge Commands

Commands are sent to `127.0.0.1:9000` as OSC/UDP packets. ACK and query
responses are emitted on `127.0.0.1:9001`.

Primary command docs:

- `bridge/commands.md`
- `bridge/ableton_udp_bridge.py --help`

Current command families include:

- transport and status checks: `/ping`, `/status`
- Live set basics: `/tempo`, `/sig_num`, `/sig_den`
- track management: create, add, delete, rename, ensure MIDI tracks
- clip-note workflows: create/replace/append/inspect session clip notes
- MIDI CC workflows: `/midi_cc`, `/cc64`
- generic LiveAPI RPC: `/api/get`, `/api/set`, `/api/call`, `/api/children`, `/api/describe`

## Privacy and Data

This repository should not contain private user data, credentials, local
conversation logs, rendered audio, or machine-specific maintainer paths.

Tracked files are intended to be source code, public documentation, tests, and
blank starter templates. Runtime memory, eval artifacts, logs, local
environment files, and generated media are ignored by default.

This repo does not train on, ingest, copy, or emulate other artists' music.
Any workflow "learning" in this repo means optional local logging of a user's
own run artifacts when enabled.

## Local Security Model

The bridge is a local control surface for a running Ableton Live set. Keep it
bound to loopback addresses such as `127.0.0.1`. Do not expose the command or
ACK ports to a network you do not control.

Default ports:

- command channel: UDP `9000`
- ACK/query response channel: UDP `9001`

## Memory and Eval Templates

`music-preferences/` is the public starter template for user-owned local memory.
It is safe to copy into a private runtime `memory/` tree:

```bash
mkdir -p memory
rsync -a music-preferences/ memory/
```

Runtime memory and eval outputs under `memory/` are local artifacts and should
not be committed.

Useful commands:

```bash
python3 -m memory.retrieval index
python3 -m memory.retrieval status
python3 -m memory.retrieval brief --focus rhythm
python3 -m memory.eval_governance summarize --lookback 30
python3 -m memory.eval_governance apply --date YYYY-MM-DD --dry-run
```

## Testing

Run the public unit suite:

```bash
python3 -m unittest discover -s bridge -p "test_*.py"
```

Optional JavaScript syntax check:

```bash
node --check bridge/m4l/live_udp_bridge.js
```

Run the direct public hygiene scan:

```bash
bash .github/scripts/audit_public_hygiene.sh
```

## Maintainer Files

- `CONTRIBUTING.md`: contribution workflow and pull request expectations
- `SUPPORT.md`: support scope and issue-reporting checklist
- `SECURITY.md`: vulnerability reporting guidance
- `CHANGELOG.md`: human-readable release and change history

## License

MIT. See `LICENSE`.
