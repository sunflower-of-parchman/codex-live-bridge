# Installation

The 3.2.0 release includes LiveUdpBridge.zip with the Max for Live device and
its two JavaScript runtime files:

```text
LiveUdpBridge.amxd
live_udp_bridge.js
osc_loopback_receiver.js
```

The repository also includes `bridge/m4l/LiveUdpBridge.maxpat`, the editable
source for the device. Follow [Package From Source](#package-from-source) if
you need to build your own copy.

## Use a Release Device

1. Download and extract
   [LiveUdpBridge.zip](https://github.com/sunflower-of-parchman/codex-live-bridge/releases/download/codex-live-bridge-v3.2.0/LiveUdpBridge.zip).
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

## Update an Installed Device

After the bridge has been installed once, stage an updated copy without
changing the installation:

```bash
node scripts/ableton-device.js
```

On macOS, the default baseline is the existing device at:

```text
~/Music/Ableton/User Library/Presets/MIDI Effects/Max MIDI Effect/LiveUdpBridge.amxd
```

The command rebuilds the device from the tracked Max patch in a private
temporary directory. It preserves the installed device metadata and any
configured local token. An existing baseline does not need to be opened in Max
or rediscovered through the Ableton interface.

To replace the installed files, keep Ableton Live open with the bridge loaded
and run:

```bash
node scripts/ableton-device.js --install --verify-live
```

No installed file changes without `--install`. The `--verify-live` option
requires `--install`. Before installation, the command saves the existing
device and JavaScript files under:

```text
~/Library/Application Support/codex-live-bridge/backups
```

It then checks the running bridge with a token-free localhost status request.
If verification fails, all three installed files are restored from the backup.
Override paths with `--device PATH`, `--output-dir DIR`, `--backup-dir DIR`,
or `--python PATH`. Output and backup directories inside this repository are
rejected.

Staged devices and backups can contain the existing local token. Keep them
private. Do not commit or upload them. Public release packages must use a
placeholder-only device.

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
either JavaScript runtime file, update an existing installation with:

```bash
node scripts/ableton-device.js --install --verify-live
```

Run the static checks from `README.md`. Reload the device when the current Live
set needs to pick up a changed Max patch. If writes are enabled, confirm the
local token still matches `CODEX_LIVE_BRIDGE_TOKEN`.

For a first installation, follow [Package From Source](#package-from-source)
to create the initial device.

Keep real tokens in local packaged devices and environment state. Do not commit
them to the tracked `.maxpat` source.
