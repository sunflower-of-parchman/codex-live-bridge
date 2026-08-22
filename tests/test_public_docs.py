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

    def test_readme_distinguishes_the_bridge_from_separate_components(self) -> None:
        text = README.read_text(encoding="utf-8")
        asset = REPO_ROOT / "docs" / "assets" / "hybrid-architecture.svg"

        self.assertTrue(asset.exists())
        self.assertIn("docs/assets/hybrid-architecture.svg", text)
        self.assertIn("standalone external bridge", text)
        self.assertRegex(text, r"are separate components and are not\s+included here")
        self.assertIn("LiveUdpBridge (included)", text)
        self.assertIn("Live Extension", text)
        self.assertIn("Shared inspection core", text)
        self.assertIn("full external automation surface", text)
        self.assertNotIn("private Extensions SDK lab", text)

        image = asset.read_text(encoding="utf-8")
        for label in ("codex / scripts", "OSC / UDP", "127.0.0.1", "LiveUdpBridge"):
            self.assertIn(label, image)
        self.assertIn("Ableton Live", image)
        self.assertNotIn("Live Extension", image)
        self.assertNotIn("Shared inspection core", image)

    def test_public_docs_identify_release_3_1_1(self) -> None:
        readme = README.read_text(encoding="utf-8")
        changelog = CHANGELOG.read_text(encoding="utf-8")
        protocol = PROTOCOL.read_text(encoding="utf-8")

        self.assertIn("Current release: [3.1.1]", readme)
        self.assertIn("releases/tag/codex-live-bridge-v3.1.1", readme)
        self.assertIn("## [3.1.1] - 2026-08-22", changelog)
        self.assertIn("## [3.1.0] - 2026-06-10", changelog)
        self.assertIn("Protocol status: v3.1.", protocol)
        self.assertNotIn("Protocol status: v3.1 draft.", protocol)

    def test_readme_explains_live_compatibility(self) -> None:
        text = README.read_text(encoding="utf-8")
        requirements = text.split("## Requirements\n", 1)[1].split(
            "## Quick Start\n", 1
        )[0]
        bridge_requirements, extension_requirements = requirements.split(
            "The separate Live Extension", 1
        )

        self.assertIn("release or Beta Ableton Live with Max for Live", bridge_requirements)
        self.assertNotIn("Live 12 Suite Beta 12.4.5 or later", bridge_requirements)
        self.assertIn("Live 12 Suite Beta 12.4.5 or later", extension_requirements)
        self.assertIn("https://www.ableton.com/en/live/extensions/", text)
        self.assertIn("https://www.ableton.com/en/beta/", text)

    def test_readme_documents_node_requirement_for_security_tests(self) -> None:
        text = README.read_text(encoding="utf-8")

        self.assertIn("Node.js for the Python security tests and JavaScript syntax checks", text)
        self.assertRegex(
            text,
            r"`bridge/m4l/LiveUdpBridge\.maxpat` or either JavaScript runtime\s+file",
        )

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

    def test_public_docs_define_authenticated_mutation_boundary(self) -> None:
        readme = README.read_text(encoding="utf-8")
        protocol = PROTOCOL.read_text(encoding="utf-8")
        security = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")

        self.assertIn("CODEX_LIVE_BRIDGE_TOKEN", readme)
        self.assertIn("capability token", readme)
        self.assertIn("/api/set <auth_token>", protocol)
        self.assertIn("/api/call <auth_token>", protocol)
        self.assertIn("Read-only commands remain tokenless", protocol)
        self.assertIn("authenticated capability token", security)
        self.assertNotIn("through unauthenticated OSC/UDP", security)

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

    def test_readme_links_packaged_device_and_source_installation(self) -> None:
        text = README.read_text(encoding="utf-8")
        install_text = (REPO_ROOT / "INSTALL.md").read_text(encoding="utf-8")
        download_url = (
            "https://github.com/sunflower-of-parchman/codex-live-bridge/"
            "releases/download/codex-live-bridge-v3.1.1/LiveUdpBridge.zip"
        )

        self.assertIn("INSTALL.md", text)
        self.assertIn("3.1.1 release includes the device", text)
        self.assertIn(download_url, text)
        self.assertIn(download_url, install_text)
        self.assertIn("build the device from", text)
        self.assertIn("LiveUdpBridge.zip", text)
        self.assertNotIn("source code only", text)
        self.assertRegex(
            text,
            r"LiveUdpBridge\.amxd\s+live_udp_bridge\.js\s+osc_loopback_receiver\.js",
        )
        self.assertIn("LiveUdpBridge.amxd", install_text)
        self.assertIn("LiveUdpBridge.zip", install_text)
        self.assertIn("live_udp_bridge.js", install_text)
        self.assertIn("osc_loopback_receiver.js", install_text)
        self.assertIn("Max MIDI Effect", install_text)
        self.assertIn("Edit in Max", install_text)
        self.assertIn("Save As", install_text)
        self.assertNotIn("Export Max for Live Device", install_text)

    def test_readme_checks_read_only_status_before_optional_write_access(self) -> None:
        text = README.read_text(encoding="utf-8")
        quick_start = text.split("## Quick Start\n", 1)[1].split(
            "## Included Files\n", 1
        )[0]

        read_only_index = quick_start.index("Verify the bridge with a read-only status command")
        token_index = quick_start.index("Optional: configure a local token")

        self.assertLess(read_only_index, token_index)
        self.assertIn("Read-only commands do not require a token.", quick_start)
        self.assertIn("printf '%s\\n' \"$CODEX_LIVE_BRIDGE_TOKEN\"", quick_start)

    def test_installation_starts_with_read_only_access(self) -> None:
        text = (REPO_ROOT / "INSTALL.md").read_text(encoding="utf-8")

        self.assertIn("3.1.1 release includes LiveUdpBridge.zip", text)
        self.assertLess(
            text.index("## Use a Release Device"),
            text.index("## Package From Source"),
        )
        self.assertNotIn("source code only", text)
        self.assertRegex(
            text,
            r"LiveUdpBridge\.amxd\s+live_udp_bridge\.js\s+osc_loopback_receiver\.js",
        )
        self.assertLess(
            text.index("Verify the read-only connection"),
            text.index("## Configure Authenticated Writes"),
        )
        self.assertIn("read -rs CODEX_LIVE_BRIDGE_TOKEN", text)
        self.assertIn("Leave `CHANGE_ME_BEFORE_USE` unchanged", text)

    def test_readme_documents_fail_closed_acknowledgements(self) -> None:
        text = README.read_text(encoding="utf-8")

        self.assertRegex(text, r"With `--ack`, the client exits with\s+an error")
        self.assertRegex(text, r"a complete matching response does\s+not arrive")
        self.assertRegex(text, r"does not send the command when\s+the listener cannot open")

    def test_readme_registers_authenticated_observers_before_listening(self) -> None:
        text = README.read_text(encoding="utf-8")
        example_match = re.search(
            r"Register and listen for tempo changes after configuring the local token:"
            r"\n\n```bash\n(?P<command>.*?)\n```",
            text,
            flags=re.DOTALL,
        )

        self.assertIsNotNone(example_match)
        assert example_match is not None
        command = example_match.group("command")
        self.assertIn("--ack --listen", command)
        self.assertIn("--api-observe live_set tempo", command)
        self.assertIn('"observer_id":"obs-tempo"', command)
        self.assertIn("--api-unobserve obs-tempo req-unobserve", text)

    def test_readme_distinguishes_command_and_temporary_ack_listeners(self) -> None:
        text = README.read_text(encoding="utf-8")

        self.assertIn("UDP `9000`, active while the Max for Live device is loaded", text)
        self.assertIn("UDP `9001`, active only while a client is listening", text)
        self.assertIn("`127.0.0.1:9001` only while it is listening", text)
        self.assertNotIn("confirm UDP `9000` and `9001` are active", text)

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
