"""Unit tests for arrangement helpers."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import compose_arrangement as arrangement


class ArrangementHelpersTests(unittest.TestCase):
    def test_bars_for_two_minutes_in_six_four_at_137_bpm(self) -> None:
        bars = arrangement._bars_for_minutes(bpm=137.0, beats_per_bar=6.0, minutes=2.0)
        # 2 minutes at 137 BPM = 274 quarter-note beats.
        # 6/4 bars hold 6 beats each -> ceil(274 / 6) = 46 bars.
        self.assertEqual(bars, 46)

    def test_section_rotation_and_last_section_length(self) -> None:
        sections = arrangement._build_sections(total_bars=46, section_bars=8)
        bar_counts = [s.bar_count for s in sections]
        labels = [s.label for s in sections]

        self.assertEqual(bar_counts, [8, 8, 8, 8, 8, 6])
        self.assertEqual(
            labels[:6],
            [
                "intro",
                "build",
                "build",
                "pre_climax",
                "climax",
                "release",
            ],
        )

        last_start, last_end, last_length = arrangement._section_bounds(sections[-1], beats_per_bar=6.0)
        self.assertEqual(last_start, 40 * 6.0)
        self.assertEqual(last_length, 6 * 6.0)
        self.assertEqual(last_end, last_start + last_length)

    def test_slice_and_clamp_shifts_to_section_time_and_limits_duration(self) -> None:
        notes = [
            {"pitch": 60, "start_time": 5.5, "duration": 2.0, "velocity": 100, "mute": 0},
            {"pitch": 62, "start_time": 7.0, "duration": 10.0, "velocity": 100, "mute": 0},
        ]

        # Section spans beats [6, 12).
        section_notes = arrangement._slice_and_clamp_notes(notes, start=6.0, end=12.0, beat_step=1.0)

        self.assertEqual(len(section_notes), 1)
        note = section_notes[0]

        # The note that starts at 7.0 should shift to 1.0 within the section.
        self.assertEqual(note["start_time"], 1.0)

        # Duration must be clamped to the end of the section minus a small gap.
        self.assertLessEqual(note["duration"], 4.98 + 1e-6)

    def test_parse_args_accepts_strategy_and_ack_controls(self) -> None:
        cfg = arrangement.parse_args(
            [
                "--minutes",
                "2",
                "--minutes-min",
                "3",
                "--minutes-max",
                "6",
                "--duration-seed",
                "2026",
                "--bpm",
                "120",
                "--sig-num",
                "4",
                "--sig-den",
                "4",
                "--note-chunk-size",
                "64",
                "--ack-mode",
                "flush_interval",
                "--ack-flush-interval",
                "5",
                "--write-strategy",
                "delta_update",
                "--save-policy",
                "archive",
            ]
        )

        self.assertEqual(cfg.minutes, 2.0)
        self.assertEqual(cfg.minutes_min, 3.0)
        self.assertEqual(cfg.minutes_max, 6.0)
        self.assertEqual(cfg.duration_seed, 2026)
        self.assertEqual(cfg.note_chunk_size, 64)
        self.assertEqual(cfg.ack_mode, "flush_interval")
        self.assertEqual(cfg.ack_flush_interval, 5)
        self.assertEqual(cfg.write_strategy, "delta_update")
        self.assertEqual(cfg.save_policy, "archive")

    def test_parse_args_defaults_to_registry_track_naming_mode(self) -> None:
        cfg = arrangement.parse_args([])
        self.assertEqual(cfg.track_naming_mode, "registry")
        self.assertEqual(cfg.clip_write_mode, "section_clips")
        self.assertEqual(
            cfg.instrument_registry_path,
            "bridge/config/instrument_registry.marimba.v1.json",
        )
        self.assertTrue(cfg.composition_print)

    def test_parse_args_accepts_single_clip_write_mode(self) -> None:
        cfg = arrangement.parse_args(["--clip-write-mode", "single_clip"])
        self.assertEqual(cfg.clip_write_mode, "single_clip")

    def test_parse_args_accepts_composition_print_flags(self) -> None:
        cfg = arrangement.parse_args(
            [
                "--no-composition-print",
                "--composition-print-dir",
                "memory/custom_prints",
                "--composition-print-input",
                "memory/composition_prints/sample.json",
            ]
        )
        self.assertFalse(cfg.composition_print)
        self.assertEqual(cfg.composition_print_dir, "memory/custom_prints")
        self.assertEqual(cfg.composition_print_input, "memory/composition_prints/sample.json")

    def test_parse_args_accepts_marimba_identity_controls(self) -> None:
        cfg = arrangement.parse_args(
            [
                "--marimba-identity-path",
                "bridge/config/marimba_identity.v1.json",
                "--marimba-strategy",
                "broken_resonance",
                "--marimba-pair-mode",
                "attack_answer",
                "--focus",
                "Marimba",
                "--pair",
                "Vibraphone",
            ]
        )
        self.assertEqual(cfg.marimba_identity_path, "bridge/config/marimba_identity.v1.json")
        self.assertEqual(cfg.marimba_strategy, "broken_resonance")
        self.assertEqual(cfg.marimba_pair_mode, "attack_answer")
        self.assertEqual(cfg.focus, "Marimba")
        self.assertEqual(cfg.pair, "Vibraphone")

    def test_parse_args_accepts_multi_pass_controls(self) -> None:
        cfg = arrangement.parse_args(
            [
                "--composition-passes",
                "7",
                "--no-multi-pass",
                "--multi-pass-on-replay",
            ]
        )
        self.assertEqual(cfg.composition_passes, 7)
        self.assertFalse(cfg.multi_pass_enabled)
        self.assertTrue(cfg.multi_pass_on_replay)

    def test_parse_args_accepts_memory_brief_controls(self) -> None:
        cfg = arrangement.parse_args(
            [
                "--memory-brief",
                "--memory-brief-results",
                "4",
            ]
        )
        self.assertTrue(cfg.memory_brief)
        self.assertEqual(cfg.memory_brief_results, 4)

    def test_parse_args_rejects_non_positive_memory_brief_results(self) -> None:
        with self.assertRaises(SystemExit):
            arrangement.parse_args(
                [
                    "--memory-brief-results",
                    "0",
                ]
            )

    def test_build_live_track_names_uses_slot_names(self) -> None:
        specs = [
            arrangement.InstrumentSpec(
                name="Kick Drum",
                source="kick",
                role="pulse",
                priority=1,
                required=True,
                active_min=1,
                active_max=1,
                pitch_shift=0,
                velocity_scale=1.0,
                keep_ratio_scale=1.0,
                allow_labels=(),
                apply_groove=True,
            ),
            arrangement.InstrumentSpec(
                name="RIM",
                source="rim",
                role="clock",
                priority=2,
                required=True,
                active_min=1,
                active_max=1,
                pitch_shift=0,
                velocity_scale=1.0,
                keep_ratio_scale=1.0,
                allow_labels=(),
                apply_groove=True,
            ),
        ]
        names = arrangement._build_live_track_names(specs, "slot")
        self.assertEqual(names["Kick Drum"], "Instrument 01")
        self.assertEqual(names["RIM"], "Instrument 02")

    def test_run_label_uses_meter_numerator_and_bpm(self) -> None:
        label = arrangement._run_label(7, 4, 137.5, "Dark Ambient")
        self.assertEqual(label, "7_137_5")

    def test_notes_payload_hash_is_stable_for_reordered_notes(self) -> None:
        left = [
            {"pitch": 60, "start_time": 1.0, "duration": 0.5, "velocity": 100, "mute": 0},
            {"pitch": 62, "start_time": 0.0, "duration": 0.5, "velocity": 96, "mute": 0},
        ]
        right = list(reversed(left))

        self.assertEqual(arrangement._notes_payload_hash(left), arrangement._notes_payload_hash(right))

    def test_compute_note_delta_detects_remove_modify_add(self) -> None:
        existing = [
            {
                "note_id": 1,
                "pitch": 60,
                "start_time": 0.0,
                "duration": 0.5,
                "velocity": 100,
                "mute": 0,
            },
            {
                "note_id": 2,
                "pitch": 62,
                "start_time": 0.5,
                "duration": 0.5,
                "velocity": 96,
                "mute": 0,
            },
        ]
        target = [
            {"pitch": 60, "start_time": 0.0, "duration": 0.5, "velocity": 110, "mute": 0},
            {"pitch": 64, "start_time": 1.0, "duration": 0.5, "velocity": 90, "mute": 0},
        ]

        delta = arrangement._compute_note_delta(existing, target)
        self.assertFalse(delta.requires_full_replace)
        self.assertEqual(delta.remove_note_ids, (2,))
        self.assertEqual(len(delta.modification_notes), 1)
        self.assertEqual(delta.modification_notes[0]["note_id"], 1)
        self.assertEqual(len(delta.add_notes), 1)
        self.assertEqual(delta.add_notes[0]["pitch"], 64)

    def test_compute_note_delta_requires_full_replace_when_note_id_missing(self) -> None:
        existing = [
            {
                "pitch": 60,
                "start_time": 0.0,
                "duration": 0.5,
                "velocity": 100,
                "mute": 0,
            }
        ]
        target: list[dict] = []
        delta = arrangement._compute_note_delta(existing, target)
        self.assertTrue(delta.requires_full_replace)

    def test_find_matching_arrangement_clip_prefers_reference_with_id(self) -> None:
        refs = [
            arrangement.ArrangementClipRef(path="id 1", clip_id=None, start_time=8.0, end_time=16.0),
            arrangement.ArrangementClipRef(path="id 2", clip_id=42, start_time=8.0, end_time=16.0),
        ]

        match = arrangement._find_matching_arrangement_clip(
            refs,
            section_start_beats=8.0,
            section_length_beats=8.0,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.clip_id, 42)

    def test_list_arrangement_clips_uses_describe_id_fallback(self) -> None:
        def fake_api_get(
            _sock: object,
            _ack: object,
            _path: str,
            prop: str,
            _req_id: str,
            _timeout: float,
        ) -> object:
            if prop == "start_time":
                return 0.0
            if prop == "end_time":
                return 8.0
            if prop == "id":
                return 0
            raise AssertionError(f"unexpected prop {prop}")

        with (
            mock.patch.object(
                arrangement.kick,
                "_get_children",
                return_value=[{"path": "live_set tracks 0 arrangement_clips 0", "id": 0}],
            ),
            mock.patch.object(arrangement.kick, "_api_get", side_effect=fake_api_get),
            mock.patch.object(arrangement.kick, "_api_describe", return_value={"id": 77}),
        ):
            refs = arrangement._list_arrangement_clips(
                sock=object(),  # type: ignore[arg-type]
                ack_sock=object(),  # type: ignore[arg-type]
                track_path="live_set tracks 0",
                timeout_s=1.0,
                request_prefix="t",
            )

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].clip_id, 77)

    def test_delete_overlaps_from_refs_uses_cached_snapshot(self) -> None:
        refs = [
            arrangement.ArrangementClipRef(path="id 10", clip_id=10, start_time=0.0, end_time=8.0),
            arrangement.ArrangementClipRef(path="id 11", clip_id=11, start_time=8.0, end_time=16.0),
            arrangement.ArrangementClipRef(path="id x", clip_id=None, start_time=0.0, end_time=8.0),
        ]
        deleted_ids: list[int] = []

        def fake_api_call(
            _sock: object,
            _ack_sock: object,
            _path: str,
            method: str,
            args: list[int],
            _req_id: str,
            _timeout: float,
        ) -> object:
            self.assertEqual(method, "delete_clip")
            deleted_ids.append(int(args[0]))
            return []

        with mock.patch.object(arrangement.kick, "_api_call", side_effect=fake_api_call):
            deleted, retained = arrangement._delete_overlaps_from_refs(
                sock=object(),  # type: ignore[arg-type]
                ack_sock=object(),  # type: ignore[arg-type]
                track_path="live_set tracks 0",
                refs=refs,
                start=0.0,
                end=8.0,
                timeout_s=1.0,
                preserve_clip_id=None,
                request_prefix="delete",
            )

        self.assertEqual(deleted, 1)
        self.assertEqual(deleted_ids, [10])
        self.assertEqual(len(retained), 2)
        self.assertTrue(any(ref.clip_id == 11 for ref in retained))
        self.assertTrue(any(ref.clip_id is None for ref in retained))

    def test_live_set_identity_includes_path_track_signature_and_untitled_marker(self) -> None:
        def fake_api_get(
            _sock: object,
            _ack_sock: object,
            _path: str,
            prop: str,
            _req_id: str,
            _timeout: float,
        ) -> object:
            if prop == "name":
                return ""
            if prop == "path":
                return ""
            raise AssertionError(f"unexpected prop {prop}")

        with (
            mock.patch.object(arrangement.kick, "_api_describe", return_value={"id": 91}),
            mock.patch.object(arrangement.kick, "_api_get", side_effect=fake_api_get),
            mock.patch.object(
                arrangement.kick,
                "_get_children",
                return_value=[{"name": "Kick Drum"}, {"name": "Piano"}],
            ),
        ):
            ident = arrangement._live_set_identity(
                sock=object(),  # type: ignore[arg-type]
                ack_sock=object(),  # type: ignore[arg-type]
                timeout_s=1.0,
            )

        self.assertIn("id:91", ident)
        self.assertIn("tracks:2", ident)
        self.assertIn("track_sig:", ident)
        self.assertIn("untitled:1", ident)

    def test_select_minutes_is_deterministic_with_seed(self) -> None:
        left = arrangement._select_minutes(
            explicit_minutes=None,
            minutes_min=3.0,
            minutes_max=6.0,
            seed=123,
            bpm=120.0,
            sig_num=4,
            sig_den=4,
            mood="Energetic",
            key_name="D minor",
        )
        right = arrangement._select_minutes(
            explicit_minutes=None,
            minutes_min=3.0,
            minutes_max=6.0,
            seed=123,
            bpm=120.0,
            sig_num=4,
            sig_den=4,
            mood="Energetic",
            key_name="D minor",
        )
        self.assertEqual(left, right)
        self.assertGreaterEqual(left, 3.0)
        self.assertLessEqual(left, 6.0)

    def test_activation_mask_limits_optional_track_density(self) -> None:
        sections = arrangement._build_sections(total_bars=32, section_bars=8)
        specs = [
            arrangement.InstrumentSpec(
                name="A",
                source="kick",
                role="anchor",
                priority=1,
                required=True,
                active_min=4,
                active_max=4,
                pitch_shift=0,
                velocity_scale=1.0,
                keep_ratio_scale=1.0,
                allow_labels=(),
                apply_groove=False,
            ),
            arrangement.InstrumentSpec(
                name="B",
                source="rim",
                role="optional",
                priority=2,
                required=False,
                active_min=4,
                active_max=4,
                pitch_shift=0,
                velocity_scale=1.0,
                keep_ratio_scale=1.0,
                allow_labels=(),
                apply_groove=False,
            ),
            arrangement.InstrumentSpec(
                name="C",
                source="hat",
                role="optional",
                priority=3,
                required=False,
                active_min=4,
                active_max=4,
                pitch_shift=0,
                velocity_scale=1.0,
                keep_ratio_scale=1.0,
                allow_labels=(),
                apply_groove=False,
            ),
        ]
        mask = arrangement._build_activation_mask(specs, sections, seed=1)
        self.assertEqual(mask["A"], {0, 1, 2, 3})
        for idx in range(4):
            active = sum(1 for _, indices in mask.items() if idx in indices)
            self.assertGreaterEqual(active, 1)

    def test_load_instrument_registry_reads_specs(self) -> None:
        with TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "registry.json"
            path.write_text(
                '{\"instruments\":[{\"name\":\"Kick Drum\",\"source\":\"kick\",\"role\":\"pulse\",\"priority\":1,\"required\":true,\"active_min\":1,\"active_max\":2,\"pitch_shift\":0,\"velocity_scale\":1.0,\"keep_ratio_scale\":1.0,\"allow_labels\":[],\"apply_groove\":true,\"midi_min_pitch\":24,\"midi_max_pitch\":48}]}',
                encoding="utf-8",
            )
            specs = arrangement._load_instrument_registry(path)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "Kick Drum")
        self.assertEqual(specs[0].source, "kick")
        self.assertEqual(specs[0].midi_min_pitch, 24)
        self.assertEqual(specs[0].midi_max_pitch, 48)

    def test_fit_pitch_to_register_preserves_pitch_class_when_possible(self) -> None:
        # C6 should fold down by octaves into the target range as C4.
        self.assertEqual(arrangement._fit_pitch_to_register(84, 48, 60), 60)

    def test_fit_pitch_to_register_clamps_when_pitch_class_not_available(self) -> None:
        # A one-note range cannot represent every pitch class.
        self.assertEqual(arrangement._fit_pitch_to_register(61, 60, 60), 60)

    def test_transform_instrument_notes_respects_spec_pitch_range(self) -> None:
        spec = arrangement.InstrumentSpec(
            name="Lead",
            source="piano_motion",
            role="lead",
            priority=1,
            required=True,
            active_min=1,
            active_max=1,
            pitch_shift=24,
            velocity_scale=1.0,
            keep_ratio_scale=1.0,
            allow_labels=(),
            apply_groove=False,
            midi_min_pitch=72,
            midi_max_pitch=96,
        )
        section = arrangement.Section(
            index=0,
            start_bar=0,
            bar_count=4,
            label="build",
            kick_on=True,
            rim_on=True,
            hat_on=True,
            piano_mode="chords",
            kick_keep_ratio=1.0,
            rim_keep_ratio=1.0,
            hat_keep_ratio=1.0,
            hat_density="eighth",
        )
        notes = [
            {"pitch": 52, "start_time": 0.0, "duration": 0.5, "velocity": 100, "mute": 0},
            {"pitch": 76, "start_time": 1.0, "duration": 0.5, "velocity": 100, "mute": 0},
        ]
        transformed = arrangement._transform_instrument_notes(
            notes=notes,
            spec=spec,
            section=section,
            beats_per_bar=4.0,
            beat_step=1.0,
        )
        self.assertTrue(transformed)
        for note in transformed:
            self.assertGreaterEqual(int(note["pitch"]), 72)
            self.assertLessEqual(int(note["pitch"]), 96)

    def test_apply_marimba_identity_shapes_marimba_and_pair(self) -> None:
        sections = arrangement._build_sections(total_bars=8, section_bars=4)
        marimba_notes = [
            {"pitch": 72, "start_time": 0.0, "duration": 1.0, "velocity": 100, "mute": 0},
            {"pitch": 79, "start_time": 0.5, "duration": 1.0, "velocity": 100, "mute": 0},
            {"pitch": 84, "start_time": 1.0, "duration": 1.0, "velocity": 100, "mute": 0},
        ]
        vib_notes = [
            {"pitch": 84, "start_time": 0.0, "duration": 0.2, "velocity": 100, "mute": 0},
            {"pitch": 88, "start_time": 1.0, "duration": 0.2, "velocity": 100, "mute": 0},
        ]
        arranged = {
            "Marimba": [(sections[0], marimba_notes), (sections[1], marimba_notes)],
            "Vibraphone": [(sections[0], vib_notes), (sections[1], vib_notes)],
        }
        specs = [
            arrangement.InstrumentSpec(
                name="Marimba",
                source="piano_motion",
                role="motif",
                priority=1,
                required=True,
                active_min=2,
                active_max=2,
                pitch_shift=0,
                velocity_scale=1.0,
                keep_ratio_scale=1.0,
                allow_labels=(),
                apply_groove=False,
                midi_min_pitch=55,
                midi_max_pitch=96,
            ),
            arrangement.InstrumentSpec(
                name="Vibraphone",
                source="piano_motion",
                role="motif",
                priority=2,
                required=True,
                active_min=2,
                active_max=2,
                pitch_shift=0,
                velocity_scale=1.0,
                keep_ratio_scale=1.0,
                allow_labels=(),
                apply_groove=False,
                midi_min_pitch=60,
                midi_max_pitch=104,
            ),
        ]
        identity = arrangement.MarimbaIdentityConfig(
            path=pathlib.Path("bridge/config/marimba_identity.v1.json"),
            payload={
                "strategy_default": "ostinato_pulse",
                "pair_mode_default": "attack_answer",
                "track_name": "Marimba",
                "pair_track_name": "Vibraphone",
                "constraints": {"max_leap_semitones": 12, "max_density_notes_per_bar": 8},
                "strategies": {
                    "ostinato_pulse": {
                        "anchor_steps_6_8": [0, 3],
                        "pulse_duration_beats": 0.4,
                    }
                },
                "pair_rules": {
                    "attack_answer": {
                        "min_start_separation_beats": 0.25,
                        "marimba_max_duration_beats": 0.55,
                        "vibraphone_min_duration_beats": 0.7,
                        "vibraphone_velocity_scale": 0.9,
                    }
                },
            },
            track_name="Marimba",
            pair_track_name="Vibraphone",
            strategy_default="ostinato_pulse",
            pair_mode_default="attack_answer",
        )

        updated, meta = arrangement._apply_marimba_identity(
            arranged_by_track=arranged,
            specs=specs,
            sections=sections[:2],
            beats_per_bar=3.0,  # 6/8 in quarter-note units
            beat_step=0.5,
            bpm=92.0,
            identity=identity,
            requested_strategy="ostinato_pulse",
            key_name="G# minor",
            pair_mode="attack_answer",
            focus_track="Marimba",
            pair_track="Vibraphone",
        )

        self.assertTrue(meta["enabled"])
        self.assertIn("ostinato_pulse", meta["strategy_usage"])
        mar_first = updated["Marimba"][0][1]
        vib_first = updated["Vibraphone"][0][1]
        self.assertTrue(mar_first)
        self.assertTrue(vib_first)
        self.assertTrue(all(float(n["duration"]) <= 0.55 + 1e-6 for n in mar_first))
        self.assertTrue(all(float(n["duration"]) >= 0.7 - 1e-6 for n in vib_first))
        allowed_g_sharp_minor = {8, 10, 11, 1, 3, 4, 6}
        self.assertTrue(all(int(n["pitch"]) % 12 in allowed_g_sharp_minor for n in mar_first))

    def test_apply_marimba_identity_left_hand_right_hand_family_quantized_and_continuous(self) -> None:
        sections = arrangement._build_sections(total_bars=12, section_bars=4)
        arranged = {
            "Marimba": [(section, []) for section in sections],
        }
        specs = [
            arrangement.InstrumentSpec(
                name="Marimba",
                source="piano_chords",
                role="motif",
                priority=1,
                required=True,
                active_min=len(sections),
                active_max=len(sections),
                pitch_shift=0,
                velocity_scale=1.0,
                keep_ratio_scale=1.0,
                allow_labels=(),
                apply_groove=False,
                midi_min_pitch=55,
                midi_max_pitch=96,
            )
        ]
        identity = arrangement.MarimbaIdentityConfig(
            path=pathlib.Path("bridge/config/marimba_identity.v1.json"),
            payload={
                "enabled": True,
                "track_name": "Marimba",
                "pair_track_name": "Vibraphone",
                "composition_family_default": "left_hand_ostinato_right_hand_melody",
                "hand_model_default": "four_mallet",
                "grid_step_beats": 0.25,
                "mutation_window_bars": [2, 3, 4],
                "harmony_motion_profile": "tonic_subdominant_dominant_cycle",
                "strategy_default": "ostinato_pulse",
                "pair_mode_default": "off",
                "constraints": {
                    "max_leap_semitones": 12,
                    "max_density_notes_per_bar": 12,
                },
            },
            track_name="Marimba",
            pair_track_name="Vibraphone",
            strategy_default="ostinato_pulse",
            pair_mode_default="off",
        )

        updated, meta = arrangement._apply_marimba_identity(
            arranged_by_track=arranged,
            specs=specs,
            sections=sections,
            beats_per_bar=4.0,
            beat_step=1.0,
            bpm=112.0,
            identity=identity,
            requested_strategy="auto",
            key_name="D minor",
            pair_mode="off",
            focus_track="Marimba",
            pair_track=None,
        )

        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["composition_family"], "left_hand_ostinato_right_hand_melody")
        self.assertEqual(meta["strategy_usage"]["left_hand_ostinato_right_hand_melody"], len(sections))
        mar_payload = updated["Marimba"]
        self.assertEqual(len(mar_payload), len(sections))
        self.assertTrue(all(len(notes) > 0 for _section, notes in mar_payload))

        absolute_notes: list[dict] = []
        for section, notes in mar_payload:
            section_start = float(section.start_bar) * 4.0
            for note in notes:
                copied = dict(note)
                copied["start_time"] = float(copied.get("start_time", 0.0)) + section_start
                absolute_notes.append(copied)
        absolute_notes.sort(key=lambda n: (float(n["start_time"]), int(n["pitch"])))

        self.assertTrue(
            all(
                abs((float(note["start_time"]) / 0.25) - round(float(note["start_time"]) / 0.25)) <= 1e-4
                for note in absolute_notes
            )
        )
        self.assertTrue(any(int(note["pitch"]) <= 72 for note in absolute_notes))
        self.assertTrue(any(int(note["pitch"]) >= 74 for note in absolute_notes))

        allowed_d_minor = {2, 4, 5, 7, 9, 10, 0}
        self.assertTrue(all(int(note["pitch"]) % 12 in allowed_d_minor for note in absolute_notes))

        events: list[tuple[float, int]] = []
        for note in absolute_notes:
            start = float(note["start_time"])
            end = start + float(note["duration"])
            events.append((start, 1))
            events.append((end, -1))
        events.sort(key=lambda item: (item[0], item[1]))
        active = 0
        max_active = 0
        for _time, delta in events:
            active += int(delta)
            max_active = max(max_active, active)
        self.assertLessEqual(max_active, 4)

        left_hand_notes = [note for note in absolute_notes if int(note["pitch"]) <= 72]

        def _bar_pattern(bar_index: int) -> list[int]:
            bar_start = float(bar_index) * 4.0
            bar_end = bar_start + 4.0
            return sorted(
                int(round((float(note["start_time"]) - bar_start) / 0.25))
                for note in left_hand_notes
                if bar_start <= float(note["start_time"]) < bar_end
            )

        bar3 = _bar_pattern(3)
        bar4 = _bar_pattern(4)
        overlap = len(set(bar3) & set(bar4))
        self.assertGreaterEqual(overlap, max(1, min(len(bar3), len(bar4)) - 1))

    def test_apply_marimba_identity_fast_tempo_biases_to_eighth_grid(self) -> None:
        sections = arrangement._build_sections(total_bars=8, section_bars=4)
        arranged = {
            "Marimba": [(section, []) for section in sections],
        }
        specs = [
            arrangement.InstrumentSpec(
                name="Marimba",
                source="piano_chords",
                role="motif",
                priority=1,
                required=True,
                active_min=len(sections),
                active_max=len(sections),
                pitch_shift=0,
                velocity_scale=1.0,
                keep_ratio_scale=1.0,
                allow_labels=(),
                apply_groove=False,
                midi_min_pitch=55,
                midi_max_pitch=96,
            )
        ]
        identity = arrangement.MarimbaIdentityConfig(
            path=pathlib.Path("bridge/config/marimba_identity.v1.json"),
            payload={
                "enabled": True,
                "track_name": "Marimba",
                "pair_track_name": "Vibraphone",
                "composition_family_default": "left_hand_ostinato_right_hand_melody",
                "hand_model_default": "four_mallet",
                "grid_step_beats": 0.25,
                "fast_tempo_bpm_threshold": 124,
                "fast_tempo_rhythm_step_beats": 0.5,
                "mutation_window_bars": [2, 3, 4],
                "harmony_block_bars": [4],
                "harmony_motion_profile": "tonic_subdominant_dominant_cycle",
                "form_arc": {
                    "enabled": True,
                    "peak_ratio": 0.66,
                    "density_start_scale": 0.6,
                    "density_peak_scale": 1.0,
                    "density_end_scale": 0.65,
                    "velocity_start_scale": 0.85,
                    "velocity_peak_scale": 1.1,
                    "velocity_end_scale": 0.88,
                },
                "strategy_default": "ostinato_pulse",
                "pair_mode_default": "off",
                "constraints": {
                    "max_leap_semitones": 12,
                    "max_density_notes_per_bar": 12,
                },
            },
            track_name="Marimba",
            pair_track_name="Vibraphone",
            strategy_default="ostinato_pulse",
            pair_mode_default="off",
        )

        updated, meta = arrangement._apply_marimba_identity(
            arranged_by_track=arranged,
            specs=specs,
            sections=sections,
            beats_per_bar=5.0,
            beat_step=1.0,
            bpm=128.0,
            identity=identity,
            requested_strategy="auto",
            key_name="G# minor",
            pair_mode="off",
            focus_track="Marimba",
            pair_track=None,
        )

        self.assertEqual(meta["tempo_profile"], "fast")
        self.assertAlmostEqual(float(meta["rhythm_step_beats"]), 0.5, places=6)
        self.assertEqual(meta["harmony_block_bars"], [4])
        self.assertTrue(meta["form_arc_enabled"])

        mar_payload = updated["Marimba"]
        absolute_notes: list[dict] = []
        for section, notes in mar_payload:
            section_start = float(section.start_bar) * 5.0
            for note in notes:
                copied = dict(note)
                copied["start_time"] = float(copied.get("start_time", 0.0)) + section_start
                absolute_notes.append(copied)
        absolute_notes.sort(key=lambda n: (float(n["start_time"]), int(n["pitch"])))
        self.assertTrue(absolute_notes)
        self.assertTrue(
            all(
                abs((float(note["start_time"]) / 0.5) - round(float(note["start_time"]) / 0.5)) <= 1e-4
                for note in absolute_notes
            )
        )
        bar_counts: dict[int, int] = {}
        for note in absolute_notes:
            bar_idx = int(float(note["start_time"]) // 5.0)
            bar_counts[bar_idx] = bar_counts.get(bar_idx, 0) + 1
        self.assertGreater(bar_counts.get(5, 0), bar_counts.get(0, 0))

    def test_overwrite_existing_clip_sets_clip_signature(self) -> None:
        observed_sets: list[tuple[str, object]] = []

        def fake_api_set(
            _sock: object,
            _ack_sock: object,
            _path: str,
            prop: str,
            value: object,
            _request_id: str,
            _timeout_s: float,
        ) -> object:
            observed_sets.append((prop, value))
            return {"ok": True}

        with (
            mock.patch.object(arrangement.kick, "_api_set", side_effect=fake_api_set),
            mock.patch.object(arrangement.kick, "_api_call", return_value=[]),
        ):
            overwritten = arrangement._overwrite_existing_clip_notes(
                sock=object(),  # type: ignore[arg-type]
                ack_sock=object(),  # type: ignore[arg-type]
                clip_path="id 20",
                track_path="live_set tracks 1",
                section_index=0,
                clip_length_beats=64.0,
                clip_name="Test",
                sig_num=5,
                sig_den=4,
                notes=[],
                timeout_s=1.0,
                note_chunk_size=64,
                groove_id=None,
                apply_groove=False,
            )

        self.assertIn(("signature_numerator", 5), observed_sets)
        self.assertIn(("signature_denominator", 4), observed_sets)
        self.assertFalse(overwritten)

    def test_overwrite_existing_clip_removes_note_ids_when_available(self) -> None:
        observed_remove_args: list[dict] = []

        def fake_api_call(
            _sock: object,
            _ack_sock: object,
            _path: str,
            method: str,
            args: object,
            _request_id: str,
            _timeout_s: float,
        ) -> object:
            if method == "get_notes_extended":
                return {
                    "notes": [
                        {"note_id": 10, "pitch": 60, "start_time": 0.0, "duration": 0.5, "velocity": 90, "mute": 0},
                        {"note_id": 11, "pitch": 62, "start_time": 1.0, "duration": 0.5, "velocity": 90, "mute": 0},
                    ]
                }
            if method == "remove_notes_extended" and isinstance(args, dict):
                observed_remove_args.append(dict(args))
            return []

        with (
            mock.patch.object(arrangement.kick, "_api_set", return_value={"ok": True}),
            mock.patch.object(arrangement.kick, "_api_call", side_effect=fake_api_call),
        ):
            overwritten = arrangement._overwrite_existing_clip_notes(
                sock=object(),  # type: ignore[arg-type]
                ack_sock=object(),  # type: ignore[arg-type]
                clip_path="id 20",
                track_path="live_set tracks 1",
                section_index=0,
                clip_length_beats=64.0,
                clip_name="Test",
                sig_num=5,
                sig_den=4,
                notes=[{"pitch": 60, "start_time": 0.0, "duration": 0.5, "velocity": 90, "mute": 0}],
                timeout_s=1.0,
                note_chunk_size=64,
                groove_id=None,
                apply_groove=False,
            )

        self.assertTrue(overwritten)
        self.assertTrue(observed_remove_args)
        self.assertIn("note_ids", observed_remove_args[0])

    def test_overwrite_existing_clip_scans_note_ids_by_windows_when_full_scan_empty(self) -> None:
        observed_remove_args: list[dict] = []
        observed_scan_args: list[dict] = []

        def fake_api_call(
            _sock: object,
            _ack_sock: object,
            _path: str,
            method: str,
            args: object,
            _request_id: str,
            _timeout_s: float,
        ) -> object:
            if method == "get_notes_extended" and isinstance(args, dict):
                payload = dict(args)
                observed_scan_args.append(payload)
                from_time = float(payload.get("from_time", 0.0))
                if abs(from_time - 32.0) < 1e-6:
                    return {
                        "notes": [
                            {
                                "note_id": 101,
                                "pitch": 60,
                                "start_time": 32.0,
                                "duration": 0.5,
                                "velocity": 90,
                                "mute": 0,
                            }
                        ]
                    }
                return {}
            if method == "remove_notes_extended" and isinstance(args, dict):
                observed_remove_args.append(dict(args))
            return []

        with (
            mock.patch.object(arrangement.kick, "_api_set", return_value={"ok": True}),
            mock.patch.object(arrangement.kick, "_api_call", side_effect=fake_api_call),
        ):
            overwritten = arrangement._overwrite_existing_clip_notes(
                sock=object(),  # type: ignore[arg-type]
                ack_sock=object(),  # type: ignore[arg-type]
                clip_path="id 20",
                track_path="live_set tracks 1",
                section_index=0,
                clip_length_beats=48.0,
                clip_name="Test",
                sig_num=4,
                sig_den=4,
                notes=[],
                timeout_s=1.0,
                note_chunk_size=64,
                groove_id=None,
                apply_groove=False,
            )

        self.assertTrue(overwritten)
        self.assertGreaterEqual(len(observed_scan_args), 2)
        self.assertTrue(observed_remove_args)
        self.assertIn("note_ids", observed_remove_args[0])
        self.assertNotIn("from_time", observed_remove_args[0])

    def test_overwrite_existing_clip_returns_false_when_note_ids_unavailable(self) -> None:
        observed_remove_args: list[dict] = []

        def fake_api_call(
            _sock: object,
            _ack_sock: object,
            _path: str,
            method: str,
            args: object,
            _request_id: str,
            _timeout_s: float,
        ) -> object:
            if method == "get_notes_extended":
                return {}
            if method == "remove_notes_extended" and isinstance(args, dict):
                observed_remove_args.append(dict(args))
            return []

        with (
            mock.patch.object(arrangement.kick, "_api_set", return_value={"ok": True}),
            mock.patch.object(arrangement.kick, "_api_call", side_effect=fake_api_call),
        ):
            overwritten = arrangement._overwrite_existing_clip_notes(
                sock=object(),  # type: ignore[arg-type]
                ack_sock=object(),  # type: ignore[arg-type]
                clip_path="id 20",
                track_path="live_set tracks 1",
                section_index=0,
                clip_length_beats=130.0,
                clip_name="Test",
                sig_num=5,
                sig_den=4,
                notes=[],
                timeout_s=1.0,
                note_chunk_size=64,
                groove_id=None,
                apply_groove=False,
            )

        self.assertFalse(overwritten)
        self.assertEqual(observed_remove_args, [])

    def test_composition_print_roundtrip_preserves_sections_and_notes(self) -> None:
        sections = arrangement._build_sections(total_bars=8, section_bars=4)
        arranged_by_track = {
            "Kick Drum": [
                (sections[0], [{"pitch": 36, "start_time": 0.0, "duration": 0.5, "velocity": 110, "mute": 0}]),
                (sections[1], []),
            ],
            "RIM": [
                (sections[0], []),
                (sections[1], [{"pitch": 50, "start_time": 1.0, "duration": 0.25, "velocity": 90, "mute": 0}]),
            ],
        }
        with TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            print_path = arrangement._persist_composition_print(
                run_label="4_120",
                mood="Energetic",
                key_name="D minor",
                bpm=120.0,
                sig_num=4,
                sig_den=4,
                minutes=2.0,
                bars=8,
                section_bars=4,
                start_beats=0.0,
                registry_path=pathlib.Path("bridge/config/instrument_registry.v1.json"),
                track_naming_mode="slot",
                sections=sections,
                arranged_by_track=arranged_by_track,
                output_dir=root,
            )
            self.assertTrue(print_path.exists())

            raw = json.loads(print_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["run_label"], "4_120")
            self.assertEqual(raw["composition"]["bpm"], 120.0)

            loaded = arrangement._load_composition_print(print_path)
            loaded_sections = loaded["sections"]
            loaded_arranged = loaded["arranged_by_track"]
            self.assertEqual(len(loaded_sections), 2)
            self.assertEqual(loaded_sections[0].label, sections[0].label)
            self.assertEqual(len(loaded_arranged["Kick Drum"]), 2)
            self.assertEqual(loaded_arranged["Kick Drum"][0][1][0]["pitch"], 36)
            self.assertEqual(loaded_arranged["RIM"][1][1][0]["pitch"], 50)

    def test_multi_pass_pipeline_adds_cadential_root_for_last_section(self) -> None:
        sections = arrangement._build_sections(total_bars=8, section_bars=4)
        marimba_notes = [
            {"pitch": 61, "start_time": 0.0, "duration": 0.5, "velocity": 96, "mute": 0},
            {"pitch": 64, "start_time": 1.0, "duration": 0.5, "velocity": 94, "mute": 0},
        ]
        arranged = {
            "Marimba": [
                (sections[0], [dict(note) for note in marimba_notes]),
                (sections[1], [dict(note) for note in marimba_notes]),
            ]
        }
        specs = [
            arrangement.InstrumentSpec(
                name="Marimba",
                source="piano_motion",
                role="motif",
                priority=1,
                required=True,
                active_min=2,
                active_max=2,
                pitch_shift=0,
                velocity_scale=1.0,
                keep_ratio_scale=1.0,
                allow_labels=(),
                apply_groove=False,
                midi_min_pitch=48,
                midi_max_pitch=96,
            )
        ]
        updated, reports = arrangement.run_multi_pass_pipeline(
            arranged_by_track=arranged,
            specs=specs,
            sections=sections,
            beats_per_bar=4.0,
            beat_step=1.0,
            key_name="G minor",
            pass_count=5,
            seed=42,
        )
        self.assertEqual(len(reports), 5)
        self.assertEqual(reports[-1]["pass_name"], "cadence")
        self.assertTrue(any(note["pitch"] % 12 == 7 for note in updated["Marimba"][-1][1]))


if __name__ == "__main__":
    unittest.main()
