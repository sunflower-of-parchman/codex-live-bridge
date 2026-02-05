import os
import unittest
from pathlib import Path


SKILL_PATH = Path("/Users/michaelwall/.codex/skills/launch-ableton-live/SKILL.md")
SCRIPT_PATH = Path("/Users/michaelwall/.codex/skills/launch-ableton-live/scripts/launch-ableton-live.sh")


class LaunchAbletonSkillTests(unittest.TestCase):
    def test_skill_file_exists(self):
        self.assertTrue(SKILL_PATH.exists())

    def test_launcher_script_exists_and_executable(self):
        self.assertTrue(SCRIPT_PATH.exists())
        self.assertTrue(os.access(SCRIPT_PATH, os.X_OK))


if __name__ == "__main__":
    unittest.main()
