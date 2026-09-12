# Analysis

Code for the cohort comparisons, forward predictions, and item-level
diagnostics reported in the paper. Every command starts from the released
dataset and the archived model outputs; none calls a model API.

## Inputs

- `data/kangaroo.parquet`: the released dataset (3,886 items). Download it
  from the dataset release; the loader checks its SHA-256.
- `artifacts/runs/<run>/results.parquet`: the final predicted letter of every
  item in the four October 2025 full runs and the nine August 2026 comparison
  runs (German control, English translation, blind diagram). Scoring happens
  at analysis time against the dataset keys, so the archived outputs carry no
  answer keys. `artifacts/runs/run_manifest.json` lists exact model
  identifiers, run times, decoding and image settings, and row counts.
- `artifacts/human_results/`: 23 official cohort summaries (2001-2019 and
  2022-2025). They are not redistributed; reviewers receive them on request.
- `artifacts/form_adjustments.csv`: the two official exam slots that cannot be
  evaluated (2001 grades 9-10 task 30, 2003 grades 3-4 task 21) and the fixed
  credit that keeps those forms on their published maxima.
- `artifacts/gold_key_corrections.csv`: the construction-time verification
  record of the released answer keys (informational; the dataset already
  contains the verified keys).

## Commands

```bash
uv run reproduce-diagnostics      # ~30 s, writes reproduced/diagnostics/
uv run reproduce-surrogate        # ~7 min, writes reproduced/surrogate/
uv run prepare-experiments        # blind-diagram input, reproduced/experiments/
uv run build-translation-subset --translations artifacts/translation/english_translations.json
uv run pytest tests/test_inputs.py tests/test_diagnostics_outputs.py tests/test_surrogate_analysis.py
```

`reproduce-diagnostics` verifies `artifacts/checksums.sha256`, scores the
archived outputs, builds the 115 shared form comparisons, and writes the
pooled and per-grade correlations, composition-controlled regressions,
image-subtype and response-status tables, option-position check, and
per-model contest totals. `reproduce-surrogate` reuses those inputs and writes
the outcome regressions, forward-chaining predictions, variance decomposition,
sensitivity checks, paired ablation comparisons, yearly composition, and the
cohort reference means.

Authors rebuild the archived run tables from raw run directories with
`uv run package-runs --source-root <dir>` (repeatable) and the anonymous
release archive with `uv run build-release-package`.

## Definitions

- The analysis unit is one grade-group form in one year; 23 cohort years and
  five grade groups give 115 shared comparisons (140 forms in total).
- Contest scoring: one starting point per item, the item value for a correct
  letter, a quarter of the item value deducted for a wrong letter, zero for an
  explicit decline. A missing or malformed final selection counts as wrong.
  Form scores are divided by the published maximum (%max).
- The equal-weight ensemble is the mean of the four model %max scores per
  form; the pooled ensemble divides the mean total score by the form maximum.
- Within each grade group, difficulty is the negative z-score of %max; the
  temporal out-of-sample check splits each grade series at its median year.
- Intervals are 95% percentile bootstraps clustered by year (2,000 draws);
  paired item comparisons bootstrap over items.
- Visual composition: `multimodal_share` is the share of items with any
  separate visual element; `associated_image_share` and `option_image_share`
  split it by question diagrams and image-based answer options.
- The translation subset is fixed by `artifacts/translation/english_translations.json`;
  the builder verifies that it meets every sampling quota and records the
  author-review replacement in its manifest.
