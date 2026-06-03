# Support

## Scope

Support is best-effort and provided by a solo maintainer. This repository is an
open-source local control bridge for Ableton Live, not a managed service with
guaranteed response times.

## Where To Ask for Help

- Use GitHub Issues for bug reports and reproducible bridge problems.
- Use GitHub Issues for feature requests; describe the user outcome first.
- For security concerns, follow `SECURITY.md`.

## What to Include in a Bug Report

- Operating system and version.
- Ableton Live version and whether Max for Live is available.
- Max version if known.
- Python version from `python3 --version`.
- Exact command run.
- Full terminal output, including `/ack` lines.
- Whether UDP ports `9000` and `9001` are free and local.
- Whether an exported `LiveUdpBridge.amxd` is loaded in Live.
- Whether `bridge/m4l/live_udp_bridge.js` is next to the exported device and the device
  was reloaded after edits.
- Whether the issue uses a Live 12.3+ wrapper such as `/api/insert_device`.
- Whether the full-surface smoke test was run, and whether it was run in a
  disposable Live set.

Reports missing this information may be closed until repro details are added.
