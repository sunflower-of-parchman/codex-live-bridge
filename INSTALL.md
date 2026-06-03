# Installation

`codex-live-bridge` tracks the editable Max patch source and its JavaScript
router. Ableton Live loads an exported Max for Live device:

```text
LiveUdpBridge.amxd
```

Keep `bridge/m4l/LiveUdpBridge.maxpat` as the source of truth. Generate packaged
`.amxd` files from a Live-hosted Max MIDI Effect during an explicit release
pass.

## Use a Release Device

When a tagged release includes `LiveUdpBridge.zip`:

1. Download and extract `LiveUdpBridge.zip`.
2. Keep `LiveUdpBridge.amxd` and `live_udp_bridge.js` next to each other.
3. Open Ableton Live and drag `LiveUdpBridge.amxd` onto a MIDI track.
4. Verify the bridge from the repository root:

```bash
python3 bridge/ableton_udp_bridge.py --ack --status --no-tempo --no-signature
```

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
   `bridge/m4l/live_udp_bridge.js`.
6. Reload `LiveUdpBridge.amxd` on a MIDI track in Ableton Live.
7. Run the status command shown above.

## Source Editing

After editing `bridge/m4l/LiveUdpBridge.maxpat` or
`bridge/m4l/live_udp_bridge.js`:

1. Package a fresh `LiveUdpBridge.amxd` from a Live-hosted Max MIDI Effect.
2. Keep `live_udp_bridge.js` next to the packaged device.
3. Reload the device in Ableton Live.
4. Run the static checks from `README.md`.
5. Run the read-focused status command before any mutating validation.
