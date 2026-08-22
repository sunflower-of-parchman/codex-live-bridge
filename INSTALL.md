# Installation

`codex-live-bridge` tracks the editable Max patch and two JavaScript runtime
files. Ableton Live loads an exported Max for Live device alongside both files:

```text
LiveUdpBridge.amxd
live_udp_bridge.js
osc_loopback_receiver.js
```

Keep `bridge/m4l/LiveUdpBridge.maxpat` as the source of truth. Generate packaged
`.amxd` files from a Live-hosted Max MIDI Effect during an explicit release
pass.

## Use a Release Device

The current release contains source code only. Start with
[Package From Source](#package-from-source) below. If a future tagged release
includes `LiveUdpBridge.zip`:

1. Download and extract `LiveUdpBridge.zip`.
2. Keep `LiveUdpBridge.amxd`, `live_udp_bridge.js`, and
   `osc_loopback_receiver.js` next to each other.
3. Open Ableton Live and drag `LiveUdpBridge.amxd` onto a MIDI track.
4. Verify the read-only connection from the repository root:

```bash
python3 bridge/ableton_udp_bridge.py --ack --status --no-tempo --no-signature
```

Read-only inspection does not require a token. Configure one only when you
need writes, observer lifecycle changes, or another protected command.

## Package From Source

Ableton Live's bundled Max editor does not enable standalone export from an
opened `.maxpat` file. To create a loadable device from the tracked source:

1. Open Ableton Live and drag the blank **Max MIDI Effect** device onto a MIDI
   track.
2. Use the device's **Edit in Max** action to open its Live-hosted patch.
3. Update that patch from the reviewed
   `bridge/m4l/LiveUdpBridge.maxpat` source.
4. Use **File > Save As...** in the Live-hosted Max editor and save the device
   as `LiveUdpBridge.amxd`.
5. Place the packaged `LiveUdpBridge.amxd` next to
   `bridge/m4l/live_udp_bridge.js` and
   `bridge/m4l/osc_loopback_receiver.js`.
6. Reload `LiveUdpBridge.amxd` on a MIDI track in Ableton Live.
7. Run the read-only status command shown above.

## Configure Authenticated Writes

Skip this section if you only need read-only inspection. Writes, observer
registration, and other protected commands require the same local token in
both the device and the Python client.

1. Open the device with **Edit in Max**.
2. Replace `CHANGE_ME_BEFORE_USE` in the `set_auth_token` message with a
   unique local token of 16 to 256 UTF-8 bytes.
3. Save and reload the device.
4. Paste the same token into hidden terminal input and export it for the
   Python client:

```bash
read -rs CODEX_LIVE_BRIDGE_TOKEN
export CODEX_LIVE_BRIDGE_TOKEN
```

Leave `CHANGE_ME_BEFORE_USE` unchanged when writes should remain disabled.

## Source Editing

After editing `bridge/m4l/LiveUdpBridge.maxpat` or
either JavaScript runtime file:

1. Package a fresh `LiveUdpBridge.amxd` from a Live-hosted Max MIDI Effect.
2. Keep `live_udp_bridge.js` and `osc_loopback_receiver.js` next to the
   packaged device.
3. Reload the device in Ableton Live.
4. Run the static checks from `README.md`.
5. Run the read-focused status command before any mutating validation.
6. If writes are enabled, confirm the local token still matches
   `CODEX_LIVE_BRIDGE_TOKEN`.

Keep real tokens in local packaged devices and environment state. Do not commit
them to the tracked `.maxpat` source.
