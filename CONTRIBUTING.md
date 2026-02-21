# Contributing

Thanks for your interest in improving `codex-live-bridge`.

## Before You Start

- Read `README.public.md` for project scope and runtime requirements.
- Keep changes small and focused.
- Do not include secrets, API keys, or local absolute paths in commits.

## Local Validation

Run from repo root:

```bash
python3 -m unittest discover -s bridge -p "test_*.py"
python3 -m unittest discover -s memory -p "test_*.py"
```

If your change touches only one area, include at least the most relevant test
command in your pull request description.

## Pull Request Expectations

- Explain what changed and why.
- Link any related issue.
- Add or update tests when behavior changes.
- Update docs when user-facing behavior, CLI flags, or workflows change.
- Keep pull requests reviewable; split very large changes when possible.

## Scope and Maintainer Capacity

This project is maintained by one person on a best-effort basis. Not every
feature request can be accepted, and response time may vary.
