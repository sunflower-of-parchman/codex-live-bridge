#!/usr/bin/env python3
"""Hermetic integration tests for safe Max for Live device maintenance."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ableton-device.js"
M4L_DIR = REPO_ROOT / "bridge" / "m4l"
PATCH_PATH = M4L_DIR / "LiveUdpBridge.maxpat"
DEVICE_NAME = "LiveUdpBridge.amxd"
ROUTER_NAME = "live_udp_bridge.js"
RECEIVER_NAME = "osc_loopback_receiver.js"
PACKAGE_NAMES = (DEVICE_NAME, ROUTER_NAME, RECEIVER_NAME)
PRIVATE_TOKEN = "fixture-private-token-49bc1d2e-secret"
PRIVATE_METADATA = "fixture-private-project-metadata"
AMPF_HEADER = struct.Struct("<4sI4s4sII4sI")


def _encode_device(document: dict[str, object]) -> bytes:
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8") + b"\0"
    header = AMPF_HEADER.pack(
        b"ampf",
        4,
        b"mmmm",
        b"meta",
        4,
        1,
        b"ptch",
        len(payload),
    )
    return header + payload


def _decode_device(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = path.read_bytes()
    if len(raw) < AMPF_HEADER.size:
        raise ValueError("device header is incomplete")

    magic, version, device_type, metadata, metadata_size, _, patch, size = (
        AMPF_HEADER.unpack(raw[: AMPF_HEADER.size])
    )
    if (magic, version, device_type, metadata, metadata_size, patch) != (
        b"ampf",
        4,
        b"mmmm",
        b"meta",
        4,
        b"ptch",
    ):
        raise ValueError("device header is not a valid Max MIDI Effect")

    payload = raw[AMPF_HEADER.size :]
    if len(payload) != size or not payload.endswith(b"\0"):
        raise ValueError("device patch payload is incomplete")

    return raw[: AMPF_HEADER.size], json.loads(payload[:-1].decode("utf-8"))


class AbletonDeviceCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="ableton-device-test-")
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.install_dir = self.root / "fixture User Library"
        self.install_dir.mkdir(mode=0o700)
        self.device_path = self.install_dir / DEVICE_NAME
        self.output_dir = self.root / "staged device"
        self.backup_root = self.root / "persistent backups"

        source_document = json.loads(PATCH_PATH.read_text(encoding="utf-8"))
        self.source_document = source_document
        installed_document = copy.deepcopy(source_document)
        patcher = installed_document["patcher"]
        patcher["appversion"] = {
            "major": 9,
            "minor": 1,
            "revision": 4,
            "architecture": "x64",
            "modernui": 1,
        }
        patcher["project"] = {
            "amxdtype": 1835887981,
            "fixture_metadata": PRIVATE_METADATA,
            "nested": {"preserve": "installed project state"},
        }
        patcher["installed_only_metadata"] = {
            "fixture": "installed private device metadata"
        }

        for entry in patcher["boxes"]:
            box = entry["box"]
            text = str(box.get("text", ""))
            if text.startswith("set_auth_token "):
                box["text"] = f"set_auth_token {PRIVATE_TOKEN}"
            elif text.startswith("node.script osc_loopback_receiver.js"):
                box["text"] = "udpreceive 9000"

        patcher["dependency_cache"] = [
            entry
            for entry in patcher["dependency_cache"]
            if entry.get("name") != RECEIVER_NAME
        ]
        self.installed_document = installed_document
        self.device_path.write_bytes(_encode_device(installed_document))
        (self.install_dir / ROUTER_NAME).write_bytes(b"// old private router fixture\n")
        (self.install_dir / RECEIVER_NAME).write_bytes(
            b"// old private receiver fixture\n"
        )
        for filename in PACKAGE_NAMES:
            (self.install_dir / filename).chmod(0o600)

    def _run_tool(
        self,
        *arguments: str,
        output_dir: Path | None = None,
        backup_dir: Path | None = None,
        device_path: Path | None = None,
        script_path: Path | None = None,
        extra_environment: dict[str, str] | None = None,
        include_fixture_paths: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = ["node", str(script_path or SCRIPT_PATH)]
        if include_fixture_paths:
            command.extend(
                [
                    "--device",
                    str(device_path or self.device_path),
                    "--output-dir",
                    str(output_dir or self.output_dir),
                    "--backup-dir",
                    str(backup_dir or self.backup_root),
                ]
            )
        command.extend(arguments)

        environment = os.environ.copy()
        if extra_environment:
            environment.update(extra_environment)

        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )
        output = completed.stdout + completed.stderr
        self.assertNotIn(PRIVATE_TOKEN, output, "the configured token must never be logged")
        self.assertNotIn(
            PRIVATE_METADATA,
            output,
            "private installed project metadata must never be logged",
        )
        return completed

    def _summary(self, completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        self.assertTrue(output_lines, "the command must finish with a JSON summary")
        summary = json.loads(output_lines[-1])
        self.assertEqual(
            set(summary),
            {
                "stageDir",
                "installed",
                "liveStatusVerified",
                "runtimeIdentityVerified",
                "verifiedLive",
                "backupDir",
                "tokenConfigured",
                "hashes",
            },
        )
        return summary

    def _installed_snapshot(self) -> dict[str, bytes]:
        return {
            filename: (self.install_dir / filename).read_bytes()
            for filename in PACKAGE_NAMES
        }

    def _assert_private_directory(self, path: Path) -> None:
        self.assertTrue(path.is_dir(), path)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700, path)

    def _assert_private_file(self, path: Path) -> None:
        self.assertTrue(path.is_file(), path)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path)

    def _assert_staged_device(self, summary: dict[str, object]) -> Path:
        stage_dir = Path(str(summary["stageDir"]))
        self.assertEqual(stage_dir.resolve(), self.output_dir.resolve())
        self._assert_private_directory(stage_dir)

        hashes = summary["hashes"]
        self.assertIsInstance(hashes, dict)
        self.assertEqual(set(hashes), set(PACKAGE_NAMES))
        for filename in PACKAGE_NAMES:
            path = stage_dir / filename
            self._assert_private_file(path)
            self.assertEqual(hashes[filename], hashlib.sha256(path.read_bytes()).hexdigest())

        for filename in (ROUTER_NAME, RECEIVER_NAME):
            self.assertEqual(
                (stage_dir / filename).read_bytes(),
                (M4L_DIR / filename).read_bytes(),
            )

        original_header, original = _decode_device(self.device_path)
        staged_header, staged = _decode_device(stage_dir / DEVICE_NAME)
        self.assertEqual(staged_header[:28], original_header[:28])
        self.assertEqual(staged["patcher"]["appversion"], original["patcher"]["appversion"])
        self.assertEqual(staged["patcher"]["project"], original["patcher"]["project"])
        self.assertEqual(
            staged["patcher"]["installed_only_metadata"],
            original["patcher"]["installed_only_metadata"],
        )

        texts = [
            str(entry["box"].get("text", ""))
            for entry in staged["patcher"]["boxes"]
        ]
        self.assertIn(f"set_auth_token {PRIVATE_TOKEN}", texts)
        self.assertTrue(
            any(
                text.startswith("node.script osc_loopback_receiver.js")
                and "@restart 1" in text
                for text in texts
            )
        )
        self.assertFalse(any(text.startswith("udpreceive") for text in texts))
        self.assertEqual(
            len(staged["patcher"]["boxes"]),
            len(self.source_document["patcher"]["boxes"]),
        )
        dependencies = {
            entry["name"] for entry in staged["patcher"]["dependency_cache"]
        }
        self.assertTrue({ROUTER_NAME, RECEIVER_NAME}.issubset(dependencies))
        return stage_dir

    def _fake_python(self, *, exit_code: int) -> tuple[Path, Path, dict[str, str]]:
        executable = self.root / "fake-python"
        record = self.root / "verification-arguments.jsonl"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "import os\n"
            "import sys\n"
            "with open(os.environ['ABLETON_DEVICE_TEST_RECORD'], 'a', "
            "encoding='utf-8') as handle:\n"
            "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "secret = os.environ['ABLETON_DEVICE_TEST_SECRET']\n"
            "print('fixture child stdout: ' + secret)\n"
            "print('fixture child stderr: ' + secret, file=sys.stderr)\n"
            "raise SystemExit(int(os.environ['ABLETON_DEVICE_TEST_EXIT']))\n",
            encoding="utf-8",
        )
        executable.chmod(0o700)
        return (
            executable,
            record,
            {
                "ABLETON_DEVICE_TEST_RECORD": str(record),
                "ABLETON_DEVICE_TEST_SECRET": PRIVATE_TOKEN,
                "ABLETON_DEVICE_TEST_EXIT": str(exit_code),
            },
        )

    def test_default_stages_secure_device_and_preserves_installed_state(self) -> None:
        original = self._installed_snapshot()

        completed = self._run_tool()
        summary = self._summary(completed)

        self.assertFalse(summary["installed"])
        self.assertFalse(summary["verifiedLive"])
        self.assertFalse(summary["liveStatusVerified"])
        self.assertFalse(summary["runtimeIdentityVerified"])
        self.assertIsNone(summary["backupDir"])
        self.assertTrue(summary["tokenConfigured"])
        self.assertFalse(self.backup_root.exists())
        self.assertEqual(self._installed_snapshot(), original)
        self._assert_staged_device(summary)

    def test_disabled_authentication_stays_disabled_in_staged_device(self) -> None:
        installed_document = copy.deepcopy(self.installed_document)
        for entry in installed_document["patcher"]["boxes"]:
            box = entry["box"]
            if str(box.get("text", "")).startswith("set_auth_token "):
                box["text"] = "set_auth_token CHANGE_ME_BEFORE_USE"
        self.device_path.write_bytes(_encode_device(installed_document))
        original = self._installed_snapshot()

        completed = self._run_tool()
        summary = self._summary(completed)

        self.assertFalse(summary["tokenConfigured"])
        self.assertEqual(self._installed_snapshot(), original)
        _, staged = _decode_device(Path(str(summary["stageDir"])) / DEVICE_NAME)
        token_messages = [
            entry["box"].get("text")
            for entry in staged["patcher"]["boxes"]
            if str(entry["box"].get("text", "")).startswith("set_auth_token ")
        ]
        self.assertEqual(token_messages, ["set_auth_token CHANGE_ME_BEFORE_USE"])

    def test_explicit_install_preserves_private_persistent_backup(self) -> None:
        original = self._installed_snapshot()

        completed = self._run_tool("--install")
        summary = self._summary(completed)

        self.assertTrue(summary["installed"])
        self.assertFalse(summary["verifiedLive"])
        self.assertTrue(summary["tokenConfigured"])
        stage_dir = self._assert_staged_device(summary)
        backup_dir = Path(str(summary["backupDir"]))
        self.assertTrue(backup_dir.is_relative_to(self.backup_root))
        self.assertNotEqual(backup_dir, self.backup_root)
        self._assert_private_directory(self.backup_root)
        self._assert_private_directory(backup_dir)

        for filename in PACKAGE_NAMES:
            backup = backup_dir / filename
            self._assert_private_file(backup)
            self.assertEqual(backup.read_bytes(), original[filename])
            self.assertEqual(
                (self.install_dir / filename).read_bytes(),
                (stage_dir / filename).read_bytes(),
            )
        self._assert_private_file(self.device_path)

    def test_successful_live_verification_uses_read_only_status_command(self) -> None:
        fake_python, record, environment = self._fake_python(exit_code=0)

        completed = self._run_tool(
            "--install",
            "--verify-live",
            "--python",
            str(fake_python),
            extra_environment=environment,
        )
        summary = self._summary(completed)

        self.assertTrue(summary["installed"])
        self.assertTrue(summary["verifiedLive"])
        self.assertTrue(summary.get("liveStatusVerified"))
        self.assertIs(summary.get("runtimeIdentityVerified"), False)
        calls = [json.loads(line) for line in record.read_text().splitlines()]
        self.assertEqual(
            calls,
            [
                [
                    str(REPO_ROOT / "bridge" / "ableton_udp_bridge.py"),
                    "--ack",
                    "--status",
                    "--no-tempo",
                    "--no-signature",
                    "--no-metrics",
                    "--ack-timeout",
                    "2",
                ]
            ],
        )

    def test_failed_live_verification_rolls_back_all_installed_files(self) -> None:
        original = self._installed_snapshot()
        fake_python, record, environment = self._fake_python(exit_code=9)

        completed = self._run_tool(
            "--install",
            "--verify-live",
            "--python",
            str(fake_python),
            extra_environment=environment,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(self._installed_snapshot(), original)
        calls = record.read_text().splitlines()
        self.assertGreaterEqual(len(calls), 1)
        self.assertLessEqual(len(calls), 2)
        backups = list(self.backup_root.iterdir())
        self.assertEqual(len(backups), 1)
        self._assert_private_directory(backups[0])
        for filename in PACKAGE_NAMES:
            self.assertEqual((backups[0] / filename).read_bytes(), original[filename])

    def test_invalid_midi_containers_fail_before_changing_installed_files(self) -> None:
        valid_device = self.device_path.read_bytes()
        invalid_devices = {
            "bad-magic": b"nope" + valid_device[4:],
            "not-midi": valid_device[:8] + b"aaaa" + valid_device[12:],
            "missing-terminator": valid_device[:-1],
            "trailing-bytes": valid_device + b"unexpected trailing data",
        }

        for name, raw_device in invalid_devices.items():
            with self.subTest(device=name):
                self.device_path.write_bytes(raw_device)
                original = self._installed_snapshot()
                output_dir = self.root / f"invalid-stage-{name}"
                backup_dir = self.root / f"invalid-backup-{name}"

                completed = self._run_tool(
                    "--install",
                    output_dir=output_dir,
                    backup_dir=backup_dir,
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(self._installed_snapshot(), original)
                self.assertFalse(output_dir.exists())
                self.assertFalse(backup_dir.exists())

    def test_live_verification_requires_explicit_install(self) -> None:
        original = self._installed_snapshot()
        fake_python, record, environment = self._fake_python(exit_code=0)

        completed = self._run_tool(
            "--verify-live",
            "--python",
            str(fake_python),
            extra_environment=environment,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--install", completed.stderr)
        self.assertEqual(self._installed_snapshot(), original)
        self.assertFalse(record.exists())
        self.assertFalse(self.output_dir.exists())
        self.assertFalse(self.backup_root.exists())

    def test_rejects_staging_inside_install_destination(self) -> None:
        original = self._installed_snapshot()

        completed = self._run_tool(output_dir=self.install_dir)

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(self._installed_snapshot(), original)
        self.assertFalse(self.backup_root.exists())

    def test_rejects_overlapping_install_stage_and_backup_directories(self) -> None:
        unsafe_paths = {
            "output-under-install": (
                self.install_dir / "unsafe staged package",
                self.root / "safe backups for installed output",
            ),
            "backup-under-install": (
                self.root / "safe staging for installed backup",
                self.install_dir / "unsafe persistent backup",
            ),
            "output-under-backup": (
                self.root / "backup ancestor" / "unsafe nested stage",
                self.root / "backup ancestor",
            ),
            "backup-under-output": (
                self.root / "staging ancestor",
                self.root / "staging ancestor" / "unsafe nested backup",
            ),
        }

        for name, (output_dir, backup_dir) in unsafe_paths.items():
            with self.subTest(layout=name):
                original = self._installed_snapshot()

                completed = self._run_tool(
                    "--install",
                    output_dir=output_dir,
                    backup_dir=backup_dir,
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(self._installed_snapshot(), original)
                self.assertFalse(output_dir.exists(), output_dir)
                self.assertFalse(backup_dir.exists(), backup_dir)

    def test_rejects_symlinked_installed_device_without_following_it(self) -> None:
        real_device = self.root / "real-device.amxd"
        real_device.write_bytes(self.device_path.read_bytes())
        self.device_path.unlink()
        self.device_path.symlink_to(real_device)
        original = real_device.read_bytes()

        completed = self._run_tool("--install")

        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue(self.device_path.is_symlink())
        self.assertEqual(real_device.read_bytes(), original)
        self.assertFalse(self.output_dir.exists())
        self.assertFalse(self.backup_root.exists())

    def test_rejects_install_inside_staging_or_backup_ancestor(self) -> None:
        for ancestor_option in ("staging", "backup"):
            with self.subTest(ancestor=ancestor_option), tempfile.TemporaryDirectory(
                prefix="ableton-device-ancestor-test-"
            ) as fixture_directory:
                fixture = Path(fixture_directory)
                ancestor = fixture / "artifact parent"
                installed = ancestor / "installed bridge"
                installed.mkdir(parents=True)
                ancestor.chmod(0o755)
                for filename in PACKAGE_NAMES:
                    shutil.copy2(self.install_dir / filename, installed / filename)
                original = {
                    name: (installed / name).read_bytes() for name in PACKAGE_NAMES
                }
                output = ancestor if ancestor_option == "staging" else fixture / "stage"
                backup = ancestor if ancestor_option == "backup" else fixture / "backup"

                completed = self._run_tool(
                    "--install",
                    output_dir=output,
                    backup_dir=backup,
                    device_path=installed / DEVICE_NAME,
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(stat.S_IMODE(ancestor.stat().st_mode), 0o755)
                self.assertEqual(list(ancestor.iterdir()), [installed])
                self.assertEqual(
                    {name: (installed / name).read_bytes() for name in PACKAGE_NAMES},
                    original,
                )
                self.assertFalse((fixture / "stage").exists())
                self.assertFalse((fixture / "backup").exists())

    def test_rejects_symlinked_staging_destination(self) -> None:
        real_output = self.root / "real output directory"
        real_output.mkdir(mode=0o700)
        self.output_dir.symlink_to(real_output, target_is_directory=True)
        original = self._installed_snapshot()

        completed = self._run_tool()

        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue(self.output_dir.is_symlink())
        self.assertEqual(list(real_output.iterdir()), [])
        self.assertEqual(self._installed_snapshot(), original)
        self.assertFalse(self.backup_root.exists())

    def test_rejects_staging_and_backups_inside_source_repository(self) -> None:
        isolated_repo = self.root / "isolated repository"
        isolated_scripts = isolated_repo / "scripts"
        isolated_m4l = isolated_repo / "bridge" / "m4l"
        isolated_scripts.mkdir(parents=True)
        isolated_m4l.mkdir(parents=True)
        isolated_script = isolated_scripts / "ableton-device.js"
        shutil.copy2(SCRIPT_PATH, isolated_script)
        for source in (PATCH_PATH, M4L_DIR / ROUTER_NAME, M4L_DIR / RECEIVER_NAME):
            shutil.copy2(source, isolated_m4l / source.name)

        for protected_option in ("output", "backup"):
            with self.subTest(option=protected_option):
                protected_path = isolated_repo / f"protected-{protected_option}"
                output_dir = (
                    protected_path
                    if protected_option == "output"
                    else self.root / "safe isolated output"
                )
                backup_dir = (
                    protected_path
                    if protected_option == "backup"
                    else self.root / "safe isolated backups"
                )
                original = self._installed_snapshot()

                completed = self._run_tool(
                    "--install",
                    output_dir=output_dir,
                    backup_dir=backup_dir,
                    script_path=isolated_script,
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(self._installed_snapshot(), original)
                self.assertFalse(protected_path.exists())

    def test_rejects_noncanonical_installed_device_filename(self) -> None:
        renamed_device = self.install_dir / "DifferentDevice.amxd"
        renamed_device.write_bytes(self.device_path.read_bytes())
        original = self._installed_snapshot()

        completed = self._run_tool(device_path=renamed_device)

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(self._installed_snapshot(), original)
        self.assertFalse(self.output_dir.exists())
        self.assertFalse(self.backup_root.exists())

    def test_help_documents_safe_install_and_verification_flags(self) -> None:
        completed = self._run_tool("--help", include_fixture_paths=False)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in (
            "--device",
            "--output-dir",
            "--backup-dir",
            "--install",
            "--verify-live",
            "--python",
        ):
            self.assertIn(option, completed.stdout)


if __name__ == "__main__":
    unittest.main()
