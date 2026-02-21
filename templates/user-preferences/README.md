# User Preference Templates

These files are intentionally blank templates for user-owned composition context.

Use them as starter docs for your own preferences and workflow.

Suggested setup:

```bash
mkdir -p memory
rsync -a templates/user-preferences/seed/ memory/
```

Then fill the docs with your own preferences.

The mirrored memory runtime CLIs are:

- `python3 -m memory.compositional_memory`
- `python3 -m memory.retrieval`
- `python3 -m memory.eval_governance`

After setup, a common baseline flow is:

```bash
python3 -m memory.retrieval index
python3 -m memory.retrieval brief --focus <FUNDAMENTAL>
python3 -m memory.eval_governance summarize --lookback 30
python3 -m memory.eval_governance apply --date YYYY-MM-DD --dry-run
```
