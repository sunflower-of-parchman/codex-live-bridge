#!/usr/bin/env python3
"""Repository-level documentation guardrails for the public bridge."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
PROTOCOL = REPO_ROOT / "PROTOCOL.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
PYTHON_CLIENT = REPO_ROOT / "bridge" / "ableton_udp_bridge.py"
JS_PRODUCER = REPO_ROOT / "bridge" / "m4l" / "live_udp_bridge.js"


class PublicDocsTests(unittest.TestCase):
    def test_readme_is_centered_on_bridge_surface(self) -> None:
        text = README.read_text(encoding="utf-8")
        self.assertIn("Max for Live OSC/UDP bridge", text)
        self.assertIn("LiveAPI", text)
        self.assertIn("Python OSC client/CLI", text)

    def test_readme_mentions_observer_surface(self) -> None:
        text = README.read_text(encoding="utf-8")
        for command in (
            "/api_observe",
            "/api_unobserve",
            "/api_observers",
            "/api_clear_observers",
        ):
            self.assertIn(command, text)
        self.assertIn("--i-understand-this-mutates-live-set", text)

    def test_readme_declares_data_training_and_generation_boundary(self) -> None:
        text = README.read_text(encoding="utf-8")
        for phrase in (
            "no trained model weights",
            "training pipeline",
            "audio corpus",
            "generative music system",
            "User intent stays user-authored",
        ):
            self.assertIn(phrase, text)

    def test_readme_code_paths_exist(self) -> None:
        text = README.read_text(encoding="utf-8")
        missing: list[str] = []
        for match in re.finditer(r"`([^`]+)`", text):
            candidate = match.group(1)
            if not re.match(r"^[A-Za-z0-9_./-]+$", candidate):
                continue
            if "/" not in candidate and "." not in candidate:
                continue
            if candidate.startswith(("/", "http", "udp://", "127.")):
                continue
            path = REPO_ROOT / candidate
            if not path.exists():
                missing.append(candidate)
        self.assertEqual(missing, [])

    def test_readme_links_source_device_installation(self) -> None:
        text = README.read_text(encoding="utf-8")
        install_text = (REPO_ROOT / "INSTALL.md").read_text(encoding="utf-8")

        self.assertIn("INSTALL.md", text)
        self.assertIn("LiveUdpBridge.zip", text)
        self.assertIn("LiveUdpBridge.amxd", install_text)
        self.assertIn("LiveUdpBridge.zip", install_text)
        self.assertIn("live_udp_bridge.js", install_text)
        self.assertIn("Max MIDI Effect", install_text)
        self.assertIn("Edit in Max", install_text)
        self.assertIn("Save As", install_text)
        self.assertNotIn("Export Max for Live Device", install_text)

    def test_removed_auxiliary_files_stay_removed(self) -> None:
        stale_paths = [
            REPO_ROOT / "bridge" / "benchmark_midi_write.py",
            REPO_ROOT / "docs" / "lom-modernization-plan.md",
        ]

        existing = [str(path.relative_to(REPO_ROOT)) for path in stale_paths if path.exists()]
        self.assertEqual(existing, [])

    def test_ci_runs_repository_docs_tests(self) -> None:
        ci_text = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('python3 -m unittest discover -s tests -p "test_*.py"', ci_text)

    def test_session_clip_inspection_v1_constants_and_docs_stay_aligned(self) -> None:
        protocol = PROTOCOL.read_text(encoding="utf-8")
        changelog = CHANGELOG.read_text(encoding="utf-8")
        python_source = PYTHON_CLIENT.read_text(encoding="utf-8")
        js_source = JS_PRODUCER.read_text(encoding="utf-8")

        for source in (python_source, js_source):
            self.assertRegex(source, r"SCHEMA_VERSION\s*=\s*1")
            self.assertRegex(source, r'PRODUCER_VERSION\s*=\s*"3\.1\.0"')
            self.assertRegex(source, r"MAX_NOTES\s*=\s*4096")
            self.assertRegex(source, r"MAX_DEVICES\s*=\s*256")
            self.assertRegex(source, r"MAX_FRAGMENTS\s*=\s*1024")

        for text in (protocol, changelog):
            self.assertIn("schema version `1`", text)
            self.assertIn("producer version `3.1.0`", text)
            self.assertIn("unpublished development snapshots", text)

        for limit in (
            "`MAX_NOTES=4096`",
            "`MAX_DEVICES=256`",
            "`MAX_FRAGMENTS=1024`",
        ):
            self.assertIn(limit, protocol)


if __name__ == "__main__":
    unittest.main()
