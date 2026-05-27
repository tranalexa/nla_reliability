# Paper bundle manifest

Upload to Claude:

1. **`PAPER_INPUT.md`** — all tables + methods + figure list
2. **`ANALYSIS_FINDINGS.md`** — full interpretation (optional if PAPER_INPUT is enough)
3. **`figures/*.png`** — four plots
4. **`tables/*.csv`** and **`tables/results_table.tex`** — raw numbers

Or zip the folder:

```bash
cd paper_bundle && zip -r ../nla_paper_bundle.zip .
```

Generated from `nla_reliability/` at export time.
