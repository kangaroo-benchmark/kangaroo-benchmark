# Kangaroo: evaluation code, model outputs, and analysis

Code and archived outputs behind the paper. The dataset (`kangaroo.parquet`, 3,886 items,
CC BY-NC 4.0) and its 200-item English subset (`kangaroo_english.parquet`) are released
separately; place them under `data/`.

## Evaluation

`eval_run.py` sends every item to a model through OpenRouter (`OPENROUTER_API_KEY`) and
writes the predicted letter, the raw response, and the usage of each item under `runs/`.
`models.json` lists the models and `score_utils.py` holds the contest scoring rule.

```bash
uv sync
uv run python eval_run.py --dataset data/kangaroo.parquet --model openai/gpt-5
uv run python eval_run.py --dataset data/kangaroo.parquet --model openai/gpt-5 --vision-only --no-images
```

The first command is the full evaluation. The second evaluates the items with a question
diagram without any image. Pointing `--dataset` at `data/kangaroo_english.parquet` evaluates
the English subset. Items that a provider error left without a final answer were re-queried
under the same settings.

## Archived outputs

`artifacts/runs/<run>/results.parquet` holds the final predicted letter of every item for the
four October 2025 runs (GPT-5, Qwen3-VL 235B Thinking, Grok 4 Fast, Claude Sonnet 4.5) and the
nine August 2026 comparison runs (GPT-5, Qwen3-VL, Claude Sonnet 4.5 in a German-control, an
English-translation, and a blind arm), next to each run's `config.json`. The blind arm covers
the 1,353 items with a separately extracted question diagram and textual answer options,
evaluated without images. The language arms cover the 200 items of `kangaroo_english.parquet`,
once with the German text and once with the English text. `artifacts/form_adjustments.csv`
records the two official exam slots that cannot be evaluated.

## Analysis

```bash
uv run reproduce-diagnostics   # item-level diagnostics and cohort comparisons, about 30 s
uv run reproduce-surrogate     # regressions, forward-chaining predictions, sensitivity checks, paired comparisons, about 7 min
uv run pytest
```

Both commands score the archived outputs against the dataset keys and write every table and
figure of the paper to `reproduced/`, which is included. They also read the 23 official cohort
summaries (`artifacts/human_results/human_baseline_<year>.json`), which are available to
reviewers on request and are not redistributed.

## Terms

The code is released under the MIT License. The dataset follows its own release terms (CC BY-NC 4.0).
