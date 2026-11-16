# Kangaroo: evaluation code, model outputs, and analysis

Code and archived outputs behind the paper. The dataset (`kangaroo.parquet`, 3,886 items,
CC BY-NC 4.0) and its 200-item English subset (`kangaroo_english.parquet`) are at
https://huggingface.co/datasets/kangaroo-dataset-german/kangaroo_dataset; place them under
`data/`.

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
four October 2025 runs (GPT-5, Qwen3-VL 235B Thinking, Grok 4 Fast, Claude Sonnet 4.5), the
nine August 2026 comparison runs (GPT-5, Qwen3-VL, Claude Sonnet 4.5 in a German, an English,
and a no-image arm), and the seven November 2026 runs, next to each run's `config.json`. The no-image arm covers
the 1,353 items with a separately extracted question diagram and textual answer options,
evaluated without images. The language arms cover the 200 items of `kangaroo_english.parquet`,
once with the German text and once with the English text. The November runs are full
evaluations of GPT-6.1 Sol, GPT-6 Luna, Claude Haiku 5.5, Qwen3.8 Flash, and Ministral 3 8B,
a repeat of Claude Sonnet 4.5, and a Claude Sonnet 4.5 run with shuffled answer options.
Their tables also mark the responses that the first-letter fallback parsed or the output
limit cut off. `tools/shuffle_options.py` builds the dataset of the shuffled run, and
`option_shuffle_mapping.csv` in its folder maps every new option position to the original
letter. `artifacts/form_adjustments.csv` records the two official exam slots that cannot be
evaluated.

## Analysis

```bash
uv run reproduce-diagnostics   # item-level diagnostics and cohort comparisons, about 30 s
uv run reproduce-surrogate     # regressions, predictions of later years, sensitivity checks, paired comparisons, about 7 min
uv run reproduce-robustness    # later runs, recent-mean test, ceiling, non-linear, and exposure checks, about 90 s
uv run pytest
```

The commands score the archived outputs against the dataset keys and write every table and
figure of the paper to `reproduced/`, which is included. They also read the 23 official
summaries (`artifacts/human_results/human_baseline_<year>.json`), which are available to
reviewers on request and are not redistributed.

## Terms

The code is released under the MIT License. The dataset follows its own release terms (CC BY-NC 4.0).
