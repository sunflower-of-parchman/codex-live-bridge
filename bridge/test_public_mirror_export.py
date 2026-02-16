import pathlib
import subprocess
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPORT_SCRIPT = REPO_ROOT / ".github" / "scripts" / "export_public_mirror.sh"
ALLOWLIST_FILE = REPO_ROOT / ".github" / "public-mirror-allowlist.txt"
PUBLIC_README = REPO_ROOT / "README.public.md"


def _allowlist_paths() -> list[str]:
    paths: list[str] = []
    for raw in ALLOWLIST_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            paths.append(line)
    return paths


class PublicMirrorExportTests(unittest.TestCase):
    def test_export_contains_allowlisted_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = pathlib.Path(tmpdir) / "public_payload"
            subprocess.run(
                [str(EXPORT_SCRIPT), str(destination)],
                check=True,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            for rel_path in _allowlist_paths():
                self.assertTrue((destination / rel_path).exists(), rel_path)

    def test_export_strips_private_memory_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = pathlib.Path(tmpdir) / "public_payload"
            subprocess.run(
                [str(EXPORT_SCRIPT), str(destination)],
                check=True,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertFalse((destination / "memory" / "sessions").exists())
            self.assertFalse((destination / "memory" / "work_journal").exists())

    def test_export_uses_public_readme_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = pathlib.Path(tmpdir) / "public_payload"
            subprocess.run(
                [str(EXPORT_SCRIPT), str(destination)],
                check=True,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            if PUBLIC_README.exists():
                self.assertEqual(
                    (destination / "README.md").read_text(encoding="utf-8"),
                    PUBLIC_README.read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
