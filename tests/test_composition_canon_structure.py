import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANON_DIR = REPO_ROOT / "docs" / "composition-canon"

CANON_FILES = [
    "README.md",
    "00-canon-principles.md",
    "10-rhythm.md",
    "20-harmony.md",
    "30-melody.md",
    "40-timbre.md",
    "50-arranging.md",
    "60-arc-of-piece.md",
    "retrieval-map.md",
]

REQUIRED_HEADINGS = [
    "## Principles",
    "## How You Approach It",
    "## Operational Rules",
    "## Teaching Notes",
    "## Self-Eval Signals",
]


class CompositionCanonStructureTests(unittest.TestCase):
    def test_required_files_exist(self):
        for filename in CANON_FILES:
            with self.subTest(filename=filename):
                self.assertTrue((CANON_DIR / filename).exists(), f"Missing {filename}")

    def test_fundamental_files_have_shared_headings(self):
        fundamental_files = [
            "10-rhythm.md",
            "20-harmony.md",
            "30-melody.md",
            "40-timbre.md",
            "50-arranging.md",
            "60-arc-of-piece.md",
        ]
        for filename in fundamental_files:
            body = (CANON_DIR / filename).read_text(encoding="utf-8")
            for heading in REQUIRED_HEADINGS:
                with self.subTest(filename=filename, heading=heading):
                    self.assertIn(heading, body)

    def test_index_references_all_ordered_files(self):
        index_body = (CANON_DIR / "README.md").read_text(encoding="utf-8")
        for filename in [
            "00-canon-principles.md",
            "10-rhythm.md",
            "20-harmony.md",
            "30-melody.md",
            "40-timbre.md",
            "50-arranging.md",
            "60-arc-of-piece.md",
        ]:
            with self.subTest(filename=filename):
                self.assertIn(filename, index_body)


if __name__ == "__main__":
    unittest.main()
