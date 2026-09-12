# Archived inputs

Inputs for the analysis commands documented in `analysis/README.md`.

## Model runs

`runs/` holds one directory per archived run: the four October 2025 full runs
and the nine August 2026 comparison runs. Each directory contains
`results.parquet` (item identifier, source item, exam metadata, and the final
predicted letter), an anonymized `config.json` with credentials and
machine-local paths removed, and gzip-compressed raw provider responses,
first-pass item records, and usage events. `run_manifest.json` records model
identifiers, run times, decoding and image settings, row counts, and file
checksums. Missing final answers were re-queried with
`tools/retry_missing_answers.py`; the raw responses keep every attempt. The
public package ships the compact tables and configurations only.

## Human aggregates

`human_results/` contains 23 JSON files for 2001-2019 and 2022-2025,
structured transcriptions of the official aggregate score summaries with no
participant-level records. They are provided to reviewers on request and are
not redistributed.

## Answer keys and forms

`gold_key_corrections.csv` is the verification record from dataset
construction: 80 rows in five early forms whose printed problem numbers were
shifted (62 of them changing the key letter), one statement-number
correction, and one spurious row removed. All affected form-level answer
sequences were checked against the official solution sheet
(<https://www.mathe-kaenguru.de/wettbewerb/loesung/kaenguru_loesungen_alle.pdf>).
The released dataset already contains the verified keys; the file is kept for
transparency.

`form_adjustments.csv` records two non-evaluable official slots. The 2001
grades 9-10 form had only 29 tasks and awarded every participant five points
for task 30. The 2003 grades 3-4 form's task 21 had no valid answer and kept
the 105-point maximum; adding one starting point and five fixed points is our
analysis convention. Both forms stay on their published scales.

## Translation subset

`translation/english_translations.json` fixes the 200-item English set. Build
the paired evaluation inputs with
`uv run build-translation-subset --translations artifacts/translation/english_translations.json`.

## Image preview

`image_preview/` contains three sample question images in ordinary PNG form
for hosts that do not render image bytes embedded in a Parquet column.

## Integrity and terms

`checksums.sha256` covers every file in this directory. The repository's MIT
license covers the code; the dataset and derived translations follow the
dataset release terms (CC BY-NC 4.0), and the official human summaries are
not covered by either.
