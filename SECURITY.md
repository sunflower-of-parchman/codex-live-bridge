# Security Policy

## Supported Versions

Security fixes are applied on a best-effort basis to the latest code on `main`.

## Local Runtime Boundary

The bridge protects mutating and persistent-control commands with an
authenticated capability token configured locally in the Max device. Read-only
commands remain tokenless. The shipped command receiver binds explicitly to
`127.0.0.1:9000`, and the Python client rejects non-loopback targets. Keep ACK
traffic on loopback as well. UDP does not encrypt the token or Live data, so do
not expose port `9001` to an untrusted network.

The generic `/api/set` and `/api/call` surface can reach broad LiveAPI behavior.
Treat it as powerful local control of the active Live set. Destructive commands
can delete tracks or overwrite clips, and additive commands can create tracks,
devices, chains, clips, and notes.

Use a unique token of 16 to 256 UTF-8 bytes. Keep the real token in the local
packaged device and `CODEX_LIVE_BRIDGE_TOKEN`; do not commit it. Do not commit
logs, environment files, credentials, rendered audio/video, packaged private
devices, or machine-specific paths.

## Reporting a Vulnerability

Please do not open a public issue with exploitable details.

Preferred path:

- Use GitHub private vulnerability reporting:
  `https://github.com/sunflower-of-parchman/codex-live-bridge/security/advisories/new`

If private reporting is unavailable in your interface, open an issue with a
minimal description and request a private follow-up channel.

## What to Include

- Affected file(s) and component(s)
- Reproduction steps
- Impact assessment
- Any suggested mitigation

I will acknowledge good-faith reports as quickly as possible, triage severity,
and publish remediation notes when a fix is available.
