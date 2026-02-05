import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = REPO_ROOT / "sql" / "next_best_track_gap.sql"


class NextBestTrackSqlTests(unittest.TestCase):
    def test_query_file_exists(self):
        self.assertTrue(QUERY_FILE.exists())

    def test_query_normalizes_required_fields(self):
        sql = QUERY_FILE.read_text(encoding="utf-8")
        self.assertIn("coalesce(nullif(trim(meter), ''), 'unknown')", sql)
        self.assertIn("coalesce(nullif(trim(mood), ''), 'unknown')", sql)
        self.assertIn("tempo::int as bpm", sql)

    def test_query_uses_default_bpm_range(self):
        sql = QUERY_FILE.read_text(encoding="utf-8")
        self.assertIn("generate_series(60, 180)", sql)

    def test_query_returns_single_suggestion(self):
        sql = QUERY_FILE.read_text(encoding="utf-8")
        self.assertIn("limit 1;", sql.lower())


if __name__ == "__main__":
    unittest.main()
