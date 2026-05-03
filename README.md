Känguru Benchmark – LLM Evaluation
=================================

Quickstart (uses uv for environment management)

- Requirements: Python 3.11+, `uv` installed, OpenRouter API key.
- This repository contains the evaluation runner, dashboard, dataset checker, and retry tooling used for the German Kangaroo benchmark.
- Code is released under the MIT License. The benchmark dataset is a separate artifact and uses its own dataset license.

Setup

- Install deps:
  - `uv sync`
- Set env var:
  - `export OPENROUTER_API_KEY=...`
  - Alternatively, place the key in a local `.env` file; the script auto-loads it.

Artifact Contents

- `eval_run.py`: main model-evaluation runner.
- `check_dataset.py`: schema and data sanity checker for the benchmark Parquet file.
- `tools/retry_missing_answers.py`: retry utility for incomplete result files.
- `dashboard.py`, `dashboard/`, `web/`: local dashboard for inspecting runs and comparing model/human results.
- `score_utils.py`: shared Kangaroo scoring logic.
- `tests/`: regression tests for scoring, evaluation, retry, and dashboard data handling.
- `models.json`: editable model registry for provider/model metadata.

Reproducing the Paper Runs

The paper results were produced from the released benchmark Parquet file. Replace
`/abs/path/to/full.parquet` below with the downloaded dataset artifact path.

1. Install the environment:

   ```bash
   uv sync
   ```

2. Validate the dataset file:

   ```bash
   uv run python check_dataset.py --dataset /abs/path/to/full.parquet
   ```

3. Run a full multimodal evaluation for a model:

   ```bash
   uv run python eval_run.py --dataset /abs/path/to/full.parquet --model openai/gpt-5
   ```

4. Run the text-only diagnostic slice:

   ```bash
   uv run python eval_run.py --dataset /abs/path/to/full.parquet --model openai/gpt-5 --text-only
   ```

5. Inspect generated runs in the dashboard:

   ```bash
   uv run python dashboard.py --host 127.0.0.1 --port 8000
   ```

6. Run the automated tests:

   ```bash
   uv run pytest
   ```

The runner writes each evaluation under `runs/{timestamp}_{model}/`, including
per-question results, aggregate metrics, configuration, failures, and raw model
responses. The main paper tables and plots are computed from these run outputs
using the dashboard aggregation code and the included analysis notebooks.

Input Data

- Store your Känguru benchmark `.parquet` file anywhere inside or outside the repo (for example `datasets/kaenguru_2024.parquet`).
- Pass the path via `--dataset`; relative paths are resolved from the current working directory and `~` is supported.
- The script validates that the supplied file exists and has a `.parquet` extension before starting the run.
- Hard-required columns in the dataset (must all be present, values may be `null`/`None`): `id`, `year`, `group`, `points`, `problem_number`, `problem_statement`, `answer`, `multimodal`, `sol_A`, `sol_B`, `sol_C`, `sol_D`, `sol_E`, `question_image`, `sol_A_image_bin`, `sol_B_image_bin`, `sol_C_image_bin`, `sol_D_image_bin`, `sol_E_image_bin`, `associated_images_bin`, `language`.
- For the "Performance by Tag" analysis, you can also include a `tags` column (a list of strings).
- Image columns should contain raw bytes (preferred), but base64 strings are tolerated and will be decoded best-effort; undecodable blobs simply trigger warnings.
- `answer` is expected to be a single letter in `{A, B, C, D, E}`; anything else is treated as missing ground truth, disabling accuracy for that row.
- Extra columns are ignored, and numeric/text types are coerced as needed (for example, `points` is cast to `float`).

Run Examples

- Vision (GPT‑5):
  - `uv run python eval_run.py --dataset /abs/path/to/dataset.parquet --model openai/gpt-5`
  - Responses may include or omit visible reasoning. Every reply must finish with a line of the form `Final answer: A|B|C|D|E|Declined` so the evaluator can reliably parse the result.

Response Format Expectations

- Reasoning is optional. Models may provide intermediate thoughts, but the evaluator only scores the choice communicated in the final line.
- If the correct option cannot be determined confidently, end with `Final answer: Declined`. This is scored neutrally (0 points, no penalty) and recorded separately.
- Any other outputs (omitted final line or malformed responses) are treated as errors and retried once before being scored as incorrect.
- Text-only evaluation (skips multimodal questions):
  - `uv run python eval_run.py --dataset /abs/path/to/dataset.parquet --model openai/gpt-5 --text-only`
- Add `--sequential` if you need to process requests strictly one at a time.

Advanced Runner Arguments

- `--limit N`: Limit the run to a random sample of N questions.
- `--seed N`: Set the random seed for sampling when using `--limit`.
- `--text-only`: Evaluate only text (non-multimodal) questions.
- `--fail-fast`: Stop the run on the first error.
- For live dashboard customization: `--live-dashboard`, `--no-live-dashboard`, `--dashboard-refresh-hz HZ`, `--recent-items N`, `--ui-compact`.

Dataset Filtering

- `--year <YYYY>`: Select specific competition year(s). Can be used multiple times.
- `--group <GROUP>`: Select specific grade group(s).
- `--language <LANG>`: Select specific language(s).
- `--year-range <START-END>`: Select a continuous range of years.
- `--points-range <MIN-MAX>`: Select a range of question point values.
- `--vision-only`: Run only on multimodal (vision) questions.

Scoring

- LLM runs follow the official Känguru convention: grades 3–6 start with 24 points and grades 7–13 start with 30. Correct answers earn the full task value, unanswered questions score 0, and wrong or unparseable final answers subtract one quarter of the task value (for example, -0.75, -1.0, -1.25). Choosing `Declined` earns 0 points with no penalty; it still counts as an answered question for accuracy reporting. Totals and weighted accuracy include the start capital so model scores align with the published human ranges.

Image Controls (to reduce input bloat)

- The evaluator downsizes images before sending to the model. Flags:
  - `--image_max_dim` (default `1024`): long-edge limit in pixels; images never upscale.
  - `--image_jpeg_quality` (default `85`): JPEG quality (50–100). PNG is used automatically when alpha is present; otherwise JPEG.
  - `--image_detail` (default `auto`): passes `auto|low|high` as an image detail hint for providers that support it.

Token Usage Details

- When the provider returns usage in the response body (or an `X-Usage` header), the runner records:
  - `prompt_tokens`, `completion_tokens`, `total_tokens` and `cost`/`total_cost`.
  - If present, additional details are captured: `completion_tokens_details.reasoning_tokens` and `prompt_tokens_details.{cached_tokens,audio_tokens}`.
- These are stored per row in `results.parquet`/`results.{json,jsonl}` as optional fields:
  - `reasoning_tokens`, `cached_prompt_tokens`, `audio_prompt_tokens`.
- Not all models/providers surface these fields; logging is best‑effort. Unknown usage remains counted under the "Unknown usage rows" metric in `metrics.json` and the dashboard.

Artifacts

- Under `runs/{timestamp}_{model}/` the script writes:
  - `results.parquet` – per-row results
  - `results.json` / `results.jsonl` – same results in human-readable form (`results.jsonl` streams each row as soon as it completes)
- `metrics.json` – aggregates and cost totals
  - `config.json` – run configuration and model info
  - `failures.jsonl` – errors and diagnostics (streamed incrementally)
  - `raw_responses.jsonl` – raw API responses per attempt (useful when a model misbehaves)

Dashboard
---------

- Install/update dependencies: `uv sync`
- Launch the dashboard locally: `uv run python dashboard.py --host 127.0.0.1 --port 8000`
- Open `http://127.0.0.1:8000` in a browser. The UI works entirely offline; all JS/CSS dependencies are vendored.
- Overview page: cards summarize every run (accuracy badges, answered/skipped counts, latency/tokens averages). Failures and unknown-usage are surfaced as warning chips. Use the Compare button to pre-fill selectors on the compare page.
- Run detail: filter sidebar (group, year, language, multimodal, correctness, reasoning mode — legacy runs may show historical modes; new runs appear as “self-directed” — value ranges, warnings) drives server-side pagination and charts. Presets are stored in `localStorage`. Charts (group/year breakdowns, confusion matrix, latency/tokens histograms, predicted-letter distribution) update with the active filters. Click a row to open the drawer with dataset content (problem, answer choices, images, rationale text, warnings); a toggle reveals the raw model response when present. The issues panel surfaces top warning types and the failures timeline, with deep links back to affected ids.
- When human baselines are available, the run detail header surfaces the best human percentile/z-score and the dashboard exposes an aggregate comparison page.
- Compare page: pick two runs to view metric deltas, group/year delta bars, and a per-id diff table. The view selector can restrict to changed/improved/regressed rows; optional limit caps the diff table size. Enable the confusion-matrix toggle to preview both runs' confusion matrices side by side.
- **Scientific Analysis page**: A new page for advanced analysis across multiple runs.
  - **Usage**: Navigate to the "Analysis" tab, select one or more runs from the checkbox list, and click "Analyze".
  - **Human vs. LLM Difficulty**: A scatter plot that correlates the average LLM performance on a question with human performance. This requires a `human_performance.parquet` file in the project root (see below).
  - **Normalized Performance**: A bar chart showing raw average human and LLM scores per year, plus a normalized score (`Human Score / LLM Score`) to track relative performance over time.
  - **Performance by Tag**: A table that breaks down performance by question tags (e.g., "geometry", "algebra"), if a `tags` column is present in the dataset.
- **Human Baseline page**: Select a run or cohort of runs to compare against the yearly human distributions (CDF overlay, bin-delta heatmap, cohort percentile boxplot). Requires the JSON baselines described in `extraction_format.md`.
- Export buttons on the run detail page respect current filters: CSV or JSON downloads via `/api/runs/{id}/results?download={csv|json}`. Filters, sorts, and pagination are all encoded into the request.
- Reload when new run folders appear: `curl -X POST http://127.0.0.1:8000/api/reload` (or use any HTTP client). The index rebuild is fast and re-populates filter facets.
- Known limitations: designed for single-user, local usage (no authentication); very large runs may still take a few seconds to stream filters/aggregates; the vendored chart shim supports only the dashboard’s built-in visualisations.

### Advanced Dashboard Arguments

You can customize the dashboard's behavior with these command-line arguments:
- `--runs-dir PATH`: Directory containing run folders (default: `runs`).
- `--models PATH`: Path to `models.json` (default: `models.json`).
- `--templates PATH`: Template directory (default: `web/templates`).
- `--static PATH`: Static assets directory (default: `web/static`).
- `--human-results PATH`: Directory containing extracted human baseline JSONs (default: `human_results`).
- `--reload`: Enable auto-reload for development.

### Human Performance Data (Optional)

To enable human vs. LLM analysis, you can provide a `human_performance.parquet` file in the project's root directory. If this file exists, the dashboard will automatically load it.

- **Path**: `/human_performance.parquet`
- **Format**: A Parquet file with the following columns:
  - `question_id` (string): The ID of the question, matching the main dataset.
  - `p_correct` (float): The proportion of human students who answered the question correctly (e.g., 0.75 for 75%).
  - `sample_size` (integer): The number of students in the sample.

### Human Baseline JSONs (Optional but recommended)

To enable the dedicated "Human Baseline" dashboard page and the human percentile cards on the run detail view, add per-year JSON files extracted from the official Känguru PDF summaries.

- **Location**: Place the files in the directory passed via `--human-results` (defaults to `human_results/`). Each file must follow the naming convention `human_baseline_YYYY.json`.
- **Schema**: See `extraction_format.md` (schema version `2.0`) for the exact structure. It supports years where early grades are grouped (e.g., `3/4`, `5/6`) and years where grades are listed individually, including the grade-specific scoring maxima.
- **Partial coverage**: Missing years are tolerated—only the available years/grades appear in the UI selectors. Data can be added incrementally; use `/api/reload` or restart the server to pick up new files.

Once the JSONs are in place, open the "Human Baseline" page to explore single-run comparisons, cohort aggregates (micro/macro), CDF overlays, bin-delta heatmaps, and percentile boxplots.

Notes

- The runner issues OpenRouter requests concurrently by default (worker count is derived from available CPUs); add `--sequential` if you prefer single-request processing.
- A shared adaptive rate limiter keeps all workers within the provider’s limits. You can optionally seed per-model limits in `models.json` (see below); the limiter still backs off automatically when throttled.
- The model registry is in `models.json`. Add entries to extend.
- If a text-only model is added without vision, image questions are skipped.

Example `models.json` entry (all optional fields shown):

```jsonc
{
  "models": [
    {
      "id": "openai/gpt-5",
      "label": "OpenAI GPT-5",
      "supports_vision": true,
      "supports_json_response_format": true,
      "rate_limit": {
        "requests_per_second": 3,
        "requests_per_minute": 150,
        "min_interval_seconds": 0.5
      }
    }
  ]
}
```

- You can supply any subset of the `rate_limit` fields. The evaluator converts the first valid value into an initial inter-request delay and adapts from there based on live responses.
- `supports_internal_reasoning` / `supports_disabling_internal_reasoning`: legacy fields that are now ignored; the evaluator always runs in self-directed mode.
