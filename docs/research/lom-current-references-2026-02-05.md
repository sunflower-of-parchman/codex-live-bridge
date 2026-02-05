# LOM Current References (2026-02-05)

This note captures the latest docs used to map bridge commands to the Live Object Model.

## Primary docs checked

- Live Object Model index (Live 12.3b9): [docs.cycling74.com/apiref/lom](https://docs.cycling74.com/apiref/lom)
- Song object (tempo, root note, scales): [docs.cycling74.com/apiref/lom/song](https://docs.cycling74.com/apiref/lom/song)
- Track object: [docs.cycling74.com/apiref/lom/track](https://docs.cycling74.com/apiref/lom/track)
- Clip object (notes + envelopes): [docs.cycling74.com/apiref/lom/clip](https://docs.cycling74.com/apiref/lom/clip)
- ClipSlot object: [docs.cycling74.com/apiref/lom/clipslot](https://docs.cycling74.com/apiref/lom/clipslot)
- MixerDevice object: [docs.cycling74.com/apiref/lom/mixerdevice](https://docs.cycling74.com/apiref/lom/mixerdevice)
- DeviceParameter object: [docs.cycling74.com/apiref/lom/deviceparameter](https://docs.cycling74.com/apiref/lom/deviceparameter)

## Confirmed mappings used in this bridge

- `set_tempo` -> `Song.tempo`
- `set_global_key` -> `Song.root_note`, `Song.scale_name`, `Song.scale_intervals`
- `set_track_volume` -> `Track.mixer_device.volume`
- `set_track_pan` -> `Track.mixer_device.panning`
- `set_send_level` -> `Track.mixer_device.sends`
- `set_device_parameter` -> `DeviceParameter.value`
- `note_insert` -> `Clip.add_new_notes` (with note specs)
- `set_note_velocity` -> `Clip.apply_note_modifications`
- `create_automation` -> `Clip.clear_envelope`, `Clip.create_automation_envelope`, envelope step insertion

## Compatibility note

Some LiveAPI method signatures can differ by Live/Max versions and by JS vs Python/Node execution contexts. The bridge validates payloads and forwards clean envelopes, while the Max-side router remains the authoritative execution surface against the running Live set.
