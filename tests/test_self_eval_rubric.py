import importlib.util
import json
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = REPO_ROOT / "evals" / "self_eval_rubric.json"
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_self_eval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_self_eval", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load run_self_eval module")
    spec.loader.exec_module(module)
    return module


run_self_eval = load_module()


class SelfEvalRubricTests(unittest.TestCase):
    def test_weights_sum_to_one_hundred(self):
        rubric = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
        total_weight = sum(item["weight"] for item in rubric["criteria"])
        self.assertEqual(total_weight, 100)

    def test_criteria_have_required_fields(self):
        rubric = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
        required = {"id", "name", "weight", "question", "improvement_prompt"}
        for criterion in rubric["criteria"]:
            self.assertTrue(required.issubset(criterion.keys()))

    def test_report_contains_reflection_and_next_actions(self):
        rubric = run_self_eval.load_rubric(RUBRIC_PATH)
        criteria_ids = [c["id"] for c in rubric["criteria"]]
        scores = run_self_eval.parse_scores([], criteria_ids, rubric["scale"]["min"], rubric["scale"]["max"])
        report = run_self_eval.build_report(rubric, scores)
        self.assertIn("## Reflection", report)
        self.assertIn("## Next Actions", report)

    def test_script_runs_and_prints_weighted_score(self):
        result = subprocess.run(
            ["python3", str(SCRIPT_PATH)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Weighted score:", result.stdout)


if __name__ == "__main__":
    unittest.main()
