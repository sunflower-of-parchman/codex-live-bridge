#!/usr/bin/env python3
"""Generate a reflective self-eval report from weighted rubric scores."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUBRIC = REPO_ROOT / "evals" / "self_eval_rubric.json"


def load_rubric(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_scores(score_args: List[str], criteria_ids: List[str], min_s: int, max_s: int) -> Dict[str, int]:
    scores = {criterion_id: 3 for criterion_id in criteria_ids}
    for item in score_args:
        if "=" not in item:
            raise ValueError(f"Score must look like criterion=number: {item}")
        criterion_id, value = item.split("=", 1)
        criterion_id = criterion_id.strip()
        if criterion_id not in scores:
            raise ValueError(f"Unknown criterion id: {criterion_id}")
        try:
            numeric = int(value)
        except ValueError as exc:
            raise ValueError(f"Score must be an integer for {criterion_id}") from exc
        if numeric < min_s or numeric > max_s:
            raise ValueError(f"Score for {criterion_id} must be between {min_s} and {max_s}")
        scores[criterion_id] = numeric
    return scores


def calculate_weighted_score(criteria: List[Dict], scores: Dict[str, int], max_s: int) -> float:
    weighted = 0.0
    for criterion in criteria:
        cid = criterion["id"]
        weight = criterion["weight"]
        weighted += (scores[cid] / max_s) * weight
    return round((weighted / 100.0) * max_s, 2)


def lowest_two(criteria: List[Dict], scores: Dict[str, int], max_s: int) -> List[Tuple[Dict, int]]:
    ranked = sorted(criteria, key=lambda c: (scores[c["id"]] / max_s, c["id"]))
    return [(ranked[0], scores[ranked[0]["id"]]), (ranked[1], scores[ranked[1]["id"]])]


def build_report(rubric: Dict, scores: Dict[str, int]) -> str:
    criteria = rubric["criteria"]
    max_s = rubric["scale"]["max"]
    weighted_score = calculate_weighted_score(criteria, scores, max_s)
    lows = lowest_two(criteria, scores, max_s)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("# Self-Eval Report")
    lines.append("")
    lines.append(f"- Timestamp: {now}")
    lines.append(f"- Rubric: `{rubric['rubric_name']}`")
    lines.append(f"- Weighted score: **{weighted_score}/{max_s}.00**")
    lines.append("")
    lines.append("## Scores")
    lines.append("")
    for criterion in criteria:
        cid = criterion["id"]
        lines.append(f"- `{cid}`: {scores[cid]}/{max_s}")
    lines.append("")
    lines.append("## Reflection")
    lines.append("")
    lines.append("Lowest scoring areas this run:")
    for criterion, score in lows:
        lines.append(f"- `{criterion['id']}` ({score}/{max_s}): {criterion['question']}")
    lines.append("")
    lines.append("## Next Actions")
    lines.append("")
    for criterion, _ in lows:
        lines.append(f"- {criterion['improvement_prompt']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a weighted self-eval and print a reflection report.")
    parser.add_argument(
        "--rubric",
        type=Path,
        default=DEFAULT_RUBRIC,
        help=f"Path to rubric JSON (default: {DEFAULT_RUBRIC})",
    )
    parser.add_argument(
        "--score",
        action="append",
        default=[],
        help="Override score as criterion_id=value, e.g. --score proactive_initiative=4",
    )
    parser.add_argument(
        "--write",
        type=Path,
        help="Optional output markdown file path.",
    )
    args = parser.parse_args()

    rubric = load_rubric(args.rubric)
    criteria_ids = [criterion["id"] for criterion in rubric["criteria"]]
    scale_min = rubric["scale"]["min"]
    scale_max = rubric["scale"]["max"]
    scores = parse_scores(args.score, criteria_ids, scale_min, scale_max)
    report = build_report(rubric, scores)

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
