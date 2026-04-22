#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

fail=0

report_fail() {
  printf '[FAIL] %s\n' "$1" >&2
  fail=1
}

report_pass() {
  printf '[PASS] %s\n' "$1"
}

untracked="$(git ls-files -o --exclude-standard)"
if [ -n "$untracked" ]; then
  printf '%s\n' "$untracked" >&2
  report_fail "untracked non-ignored files are present"
else
  report_pass "no untracked non-ignored files"
fi

local_path_hits="$(
  git grep -n -E '(/Users/|/private/|C:\\Users\\)' -- . \
    ':(exclude).github/scripts/audit_public_hygiene.sh' || true
)"
if [ -n "$local_path_hits" ]; then
  printf '%s\n' "$local_path_hits" >&2
  report_fail "tracked files contain machine-local absolute paths"
else
  report_pass "no machine-local absolute paths in tracked files"
fi

secret_hits="$(
  git grep -n -E '(sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|postgres(ql)?://|mysql://|mongodb(\\+srv)?://)' -- . \
    ':(exclude).github/scripts/audit_public_hygiene.sh' || true
)"
if [ -n "$secret_hits" ]; then
  printf '%s\n' "$secret_hits" >&2
  report_fail "tracked files contain high-confidence secret-like patterns"
else
  report_pass "no high-confidence secret-like patterns in tracked files"
fi

risky_files="$(
  git ls-files | grep -Ei '(^|/)(\\.env|.*\\.(pem|key|p12|mobileprovision|sqlite|db|wav|aif|aiff|mp3|m4a|mp4|mov|zip|tar|gz|7z|rar))$' || true
)"
if [ -n "$risky_files" ]; then
  printf '%s\n' "$risky_files" >&2
  report_fail "tracked risky local artifact file types are present"
else
  report_pass "no tracked risky local artifact file types"
fi

if [ "$fail" -ne 0 ]; then
  exit 1
fi

report_pass "public hygiene audit passed"
