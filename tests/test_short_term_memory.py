import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "short_term_memory.py"
    spec = importlib.util.spec_from_file_location("short_term_memory", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load short_term_memory module")
    spec.loader.exec_module(module)
    return module


memory = load_module()


class ShortTermMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = Path(self.tmpdir.name) / "conversation.jsonl"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_sanitize_text_strips_code_and_output_markers(self):
        raw = "\n".join(
            [
                "Hello",
                "```bash",
                "$ ls -la",
                "```",
                "Output:",
                "Exit code: 0",
                "Wall time: 0.1",
                "World",
            ]
        )

        cleaned = memory.sanitize_text(raw, keep_code=False)
        self.assertEqual(cleaned, "Hello World")

    def test_cmd_log_writes_filtered_entry(self):
        args = SimpleNamespace(
            text="Starting now.\n```bash\n$ ls\n```\nOutput:\nDone.",
            keep_code=False,
            session="session-a",
            role="assistant",
            store=self.store,
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = memory.cmd_log(args)

        self.assertEqual(rc, 0)
        row = json.loads(stdout.getvalue().strip())
        self.assertEqual(row["session_id"], "session-a")
        self.assertEqual(row["role"], "assistant")
        self.assertEqual(row["text"], "Starting now. Done.")

        persisted = list(memory.iter_entries(self.store))
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["text"], "Starting now. Done.")

    def test_cmd_show_filters_by_role_and_limit(self):
        memory.append_entry(
            self.store,
            {
                "id": "1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "session_id": "s1",
                "role": "user",
                "text": "alpha",
            },
        )
        memory.append_entry(
            self.store,
            {
                "id": "2",
                "timestamp": "2026-01-01T00:01:00+00:00",
                "session_id": "s1",
                "role": "assistant",
                "text": "beta",
            },
        )
        memory.append_entry(
            self.store,
            {
                "id": "3",
                "timestamp": "2026-01-01T00:02:00+00:00",
                "session_id": "s1",
                "role": "assistant",
                "text": "gamma",
            },
        )

        args = SimpleNamespace(
            store=self.store,
            session=None,
            role="assistant",
            limit=1,
            json=False,
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = memory.cmd_show(args)

        self.assertEqual(rc, 0)
        output = stdout.getvalue().strip().splitlines()
        self.assertEqual(len(output), 1)
        self.assertIn("assistant: gamma", output[0])

    def test_cmd_clear_can_remove_single_session(self):
        memory.write_entries(
            self.store,
            [
                {
                    "id": "1",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "session_id": "drop-me",
                    "role": "user",
                    "text": "remove",
                },
                {
                    "id": "2",
                    "timestamp": "2026-01-01T00:01:00+00:00",
                    "session_id": "keep-me",
                    "role": "assistant",
                    "text": "keep",
                },
            ],
        )

        args = SimpleNamespace(store=self.store, session="drop-me", yes=True)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = memory.cmd_clear(args)

        self.assertEqual(rc, 0)
        self.assertIn("Removed 1 entries", stdout.getvalue())

        persisted = list(memory.iter_entries(self.store))
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["session_id"], "keep-me")


if __name__ == "__main__":
    unittest.main()
