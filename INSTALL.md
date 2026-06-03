# Installation

`codex-live-bridge` tracks the editable Max patch source and its JavaScript
router. Ableton Live loads an exported Max for Live device:

```text
LiveUdpBridge.amxd
```

Keep `bridge/m4l/LiveUdpBridge.maxpat` as the source of truth. Generate packaged
`.amxd` files through Max during an explicit release pass.

## Use a Release Device

When a tagged release includes `LiveUdpBridge.amxd`:

1. Download `LiveUdpBridge.amxd`.
2. Put `bridge/m4l/live_udp_bridge.js` next to the downloaded device.
3. Open Ableton Live and drag `LiveUdpBridge.amxd` onto a MIDI track.
4. Verify the bridge from the repository root:

```bash
python3 bridge/ableton_udp_bridge.py --ack --status --no-tempo --no-signature
```

## Export From Source

To create a loadable device from the tracked source:

1. Open `bridge/m4l/LiveUdpBridge.maxpat` in Max.
2. Use **File > Export Max for Live Device...**.
3. Export the project as a Max MIDI Effect named `LiveUdpBridge.amxd`.
4. Place the exported `LiveUdpBridge.amxd` next to
   `bridge/m4l/live_udp_bridge.js`.
5. Drag `LiveUdpBridge.amxd` onto a MIDI track in Ableton Live.
6. Run the status command shown above.

When editing an already exported Max for Live device from Ableton Live, use
**File > Save As...** to preserve the previous package before rebuilding a
release artifact.

## Source Editing

After editing `bridge/m4l/LiveUdpBridge.maxpat` or
`bridge/m4l/live_udp_bridge.js`:

1. Export a fresh `LiveUdpBridge.amxd` through Max.
2. Keep `live_udp_bridge.js` next to the exported device.
3. Reload the device in Ableton Live.
4. Run the static checks from `README.md`.
5. Run the read-focused status command before any mutating validation.
