# Memory Runtime Setup

This folder is a blank, user-owned memory scaffold.

## What This Provides

- `index.toml`: structured fundamentals and current focus for retrieval.
- `eval_governance_policy.toml`: bounded policy for eval-to-memory updates.
- `evals/`: local eval artifact index and composition artifact location.
- `governance/active.md`: generated active governance guidance.
- `archive/demoted_guidance.md`: generated archive for demoted guidance.

## Standard Commands

```bash
python3 -m memory.retrieval index
python3 -m memory.retrieval status
python3 -m memory.retrieval brief --focus <FUNDAMENTAL>
python3 -m memory.eval_governance summarize --lookback 30
python3 -m memory.eval_governance apply --date YYYY-MM-DD --dry-run
```

Write-mode apply (mutates memory files):

```bash
python3 -m memory.eval_governance apply --date YYYY-MM-DD
```
