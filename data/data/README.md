# `data/`

Local copies of Modal volume artifacts (`nla-cache` → `/cache/…`).

Pull everything needed for preview + probes:

```bash
uv run python scripts/pull_from_modal.py
```

Add `--full` to also download full-prompt activation parquets. Files here are gitignored (large); only this README is tracked.
