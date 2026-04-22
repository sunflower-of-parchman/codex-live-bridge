## Summary

Describe what changed and why.

## Validation

Include the commands you ran, for example:

```bash
python3 -m unittest discover -s bridge -p "test_*.py"
bash .github/scripts/audit_public_hygiene.sh
```

## Public Hygiene

- [ ] No secrets, credentials, local logs, or private memory files are included.
- [ ] No machine-specific absolute paths are included.
- [ ] User-facing docs were updated if behavior changed.
