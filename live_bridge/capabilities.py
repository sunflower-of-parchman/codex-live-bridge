"""Capability metadata for the Live bridge commands."""

from __future__ import annotations

from typing import Dict, List


CAPABILITIES: List[Dict[str, str]] = [
    {
        "command": "note_insert",
        "lom_path": "Song.tracks -> Track.clip_slots -> Clip.add_new_notes/set_notes",
        "summary": "Insert one or more MIDI notes into a target clip.",
    },
    {
        "command": "create_midi_clip",
        "lom_path": "Song.tracks -> Track.clip_slots -> ClipSlot.create_clip",
        "summary": "Create a MIDI clip in a target clip slot using beat length.",
    },
    {
        "command": "fire_clip",
        "lom_path": "Song.tracks -> Track.clip_slots -> ClipSlot.fire",
        "summary": "Launch the target clip slot.",
    },
    {
        "command": "stop_track",
        "lom_path": "Song.tracks -> Track.stop_all_clips",
        "summary": "Stop all clips on a target track.",
    },
    {
        "command": "set_note_velocity",
        "lom_path": "Song.tracks -> Track.clip_slots -> Clip.apply_note_modifications",
        "summary": "Update velocity for a specific note match.",
    },
    {
        "command": "create_automation",
        "lom_path": "Track.devices -> Device.parameters -> Clip automation envelope APIs",
        "summary": "Write automation breakpoints for a device parameter in a clip context.",
    },
    {
        "command": "set_track_volume",
        "lom_path": "Track.mixer_device.volume",
        "summary": "Set mixer volume for a track.",
    },
    {
        "command": "set_track_mute",
        "lom_path": "Track.mute",
        "summary": "Toggle track mute state.",
    },
    {
        "command": "set_track_solo",
        "lom_path": "Track.solo",
        "summary": "Toggle track solo state.",
    },
    {
        "command": "set_track_pan",
        "lom_path": "Track.mixer_device.panning",
        "summary": "Set mixer panning for a track.",
    },
    {
        "command": "set_send_level",
        "lom_path": "Track.mixer_device.sends",
        "summary": "Set send amount by send index.",
    },
    {
        "command": "set_device_parameter",
        "lom_path": "Track.devices -> Device.parameters -> DeviceParameter.value",
        "summary": "Set arbitrary device parameter value.",
    },
    {
        "command": "set_eq3",
        "lom_path": "Track.devices (EQ Three) -> Device.parameters",
        "summary": "Set EQ Three band gains and band on/off states.",
    },
    {
        "command": "set_eq8_band_gain",
        "lom_path": "Track.devices (EQ Eight) -> Device.parameters",
        "summary": "Set gain for one EQ Eight band (1-8).",
    },
    {
        "command": "set_tempo",
        "lom_path": "Song.tempo",
        "summary": "Set global tempo in BPM.",
    },
    {
        "command": "set_global_key",
        "lom_path": "Song.root_note, Song.scale_name, Song.scale_intervals",
        "summary": "Set global key center and scale.",
    },
]
