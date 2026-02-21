# User Preference Templates

This directory is a blank, user-owned memory scaffold.

It is designed to initialize runtime `memory/` without shipping private
maintainer content.

## First-Time Setup

```bash
mkdir -p memory
rsync -a music-preferences/ memory/
```

Then fill the docs with your own preferences.

## What This Provides

- `index.toml`: structured fundamentals and current focus for retrieval
- `eval_governance_policy.toml`: bounded policy for eval-to-memory updates
- `evals/`: local eval artifact index and composition artifact location
- `governance/active.md`: generated active governance guidance
- `archive/demoted_guidance.md`: generated archive for demoted guidance

## Runtime CLIs

- `python3 -m memory.compositional_memory`
- `python3 -m memory.retrieval`
- `python3 -m memory.eval_governance`

## Baseline Command Flow

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
