"""Robustness checks of the reference-anchor tests on the archived outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.diagnostics import (
    MODEL_LABELS,
    _cluster_bootstrap_interval,
    _fit_ols,
    run_diagnostics,
)
from analysis.inputs import (
    COMPARISON_RUNS,
    LATER_RUNS,
    RUNS,
    aggregate_exam_scores,
    read_compact_run,
    score_predictions,
)
from analysis.surrogate_analysis import (
    ABLATION_MODELS,
    ENSEMBLE,
    MIN_TRAINING_YEARS,
    RECENT_WINDOW_YEARS,
    SERIES_LABELS,
    _forward_errors,
    _forward_metrics,
    _paired_difference,
    _shared_slope_errors,
    build_panel,
    series_column,
)

NONLINEAR_SPECIFICATIONS = {
    "Model score + visual share": ["model", "visual"],
    "Model score x visual share": ["model", "visual", "model_x_visual"],
    "Model score squared + visual share": ["model", "model_squared", "visual"],
}
CONDITIONAL_VISUAL_SHARES = [0.30, 0.45, 0.60]
LATE_PERIOD_START = 2013
# GPT-5's documented knowledge cutoff (30 September 2024) precedes the March 2025 contest.
POST_CUTOFF = ("openai-gpt-5", 2025)
LETTERS = "ABCDE"
SONNET = "anthropic-claude-sonnet-4.5"
SONNET_REPEAT = f"{SONNET}__repeat"
SONNET_OPTION_ORDER = f"{SONNET}__option_order"
SONNET_TWO_RUNS = f"{SONNET}__two_run_mean"
SONNET_THREE_RUNS = f"{SONNET}__three_run_mean"
MINISTRAL = "mistralai-ministral-8b-2512"
MINISTRAL_STATED = f"{MINISTRAL}__stated"
MINISTRAL_UNPARSED = f"{MINISTRAL}__unparsed"
# The November 2026 series, the two parse variants of the Ministral run, and the means of
# the Sonnet runs with the October run.
LATER_SERIES = {
    "openai-gpt-6.1-sol": "GPT-6.1 Sol",
    "anthropic-claude-haiku-5.5": "Claude Haiku 5.5",
    "openai-gpt-6-luna": "GPT-6 Luna",
    "qwen-qwen3.8-flash": "Qwen3.8 Flash",
    SONNET_REPEAT: "Claude Sonnet 4.5, November 2026",
    MINISTRAL: "Ministral 3 8B",
    SONNET_TWO_RUNS: "Claude Sonnet 4.5, mean of October and November",
    MINISTRAL_STATED: "Ministral 3 8B, stated answers on fallback rows",
    MINISTRAL_UNPARSED: "Ministral 3 8B, fallback rows unparsed",
    SONNET_OPTION_ORDER: "Claude Sonnet 4.5, option order",
    SONNET_THREE_RUNS: "Claude Sonnet 4.5, mean of three runs",
}


def _coefficient(
    frame: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    fixed_effects: list[str],
    term: str,
    seed: int,
    repetitions: int,
) -> tuple[float, float, float]:
    estimate = _fit_ols(frame, outcome, predictors, fixed_effects)["coefficients"][term]
    low, high, _ = _cluster_bootstrap_interval(
        frame,
        lambda sample: float(
            _fit_ols(sample, outcome, predictors, fixed_effects)["coefficients"][term]
        ),
        seed=seed,
        repetitions=repetitions,
    )
    return float(estimate), low, high


def _scored_run(run_dir: Path, dataset: pd.DataFrame) -> pd.DataFrame:
    return score_predictions(read_compact_run(run_dir), dataset).set_index("source_id")


def compute_recent_mean_increment(
    panel: pd.DataFrame,
    series_labels: dict[str, str] = SERIES_LABELS,
    seed: int = 6000,
    repetitions: int = 2000,
) -> pd.DataFrame:
    """The recent-mean baseline against the same baseline plus b times the model score's
    change from its own mean over the same three years. b has no intercept, is shared by
    all grades, and is fitted on every earlier year, so b = 0 recovers the baseline."""
    frame = panel.sort_values(["grade_bucket", "year"]).reset_index(drop=True)
    grouped = frame.groupby("grade_bucket", sort=False)

    def change(column: str) -> np.ndarray:
        recent = grouped[column].transform(
            lambda values: values.shift(1).rolling(RECENT_WINDOW_YEARS).mean()
        )
        return (frame[column] - recent).to_numpy(dtype=float)

    years = frame["year"].to_numpy(dtype=int)
    targets = np.flatnonzero(grouped.cumcount().to_numpy() >= MIN_TRAINING_YEARS)
    human_change = change("human_pct")
    records = []
    for index, (series, label) in enumerate(series_labels.items()):
        model_change = change(series_column(series))
        rows = []
        for target in targets:
            training = (years < years[target]) & ~np.isnan(model_change)
            slope = float(
                np.dot(model_change[training], human_change[training])
                / np.dot(model_change[training], model_change[training])
            )
            rows.append(
                {
                    "year": int(years[target]),
                    "baseline_error": human_change[target],
                    "error": human_change[target] - slope * model_change[target],
                    "slope": slope,
                }
            )
        errors = pd.DataFrame(rows)
        low, high, samples = _cluster_bootstrap_interval(
            errors,
            lambda sample: float(
                (sample["error"].abs() - sample["baseline_error"].abs()).mean()
            ),
            seed=seed + index,
            repetitions=repetitions,
        )
        records.append(
            {
                "series": series,
                "series_label": label,
                **_forward_metrics(errors),
                "mae_change_ci_low": low,
                "mae_change_ci_high": high,
                "final_slope_per_10pp_model": 10 * float(errors["slope"].iloc[-1]),
                "bootstrap_samples": samples,
            }
        )
    return pd.DataFrame(records)


def _slice_panel(
    items: pd.DataFrame, panel: pd.DataFrame, multimodal: bool
) -> pd.DataFrame:
    """Contest %max of each model on one modality slice of every shared exam."""
    sliced = items.loc[items["multimodal"].astype(bool) == multimodal]
    scores = (
        sliced.groupby(["model", "year", "group"])
        .agg(
            items=("id", "size"),
            awarded=("awarded_points", "sum"),
            points=("question_points", "sum"),
        )
        .reset_index()
    )
    scores["pct"] = (
        100
        * (scores["items"] + scores["awarded"])
        / (scores["items"] + scores["points"])
    )
    wide = scores.pivot_table(index=["year", "group"], columns="model", values="pct")
    wide.columns = [series_column(model) for model in wide.columns]
    wide[series_column(ENSEMBLE)] = wide[
        [series_column(model) for model in MODEL_LABELS]
    ].mean(axis=1)
    frame = panel[["year", "exam", "grade_bucket", "human_pct"]].merge(
        wide.reset_index().rename(columns={"group": "exam"}),
        on=["year", "exam"],
        validate="one_to_one",
    )
    if len(frame) != len(panel):
        raise ValueError("A modality slice is missing from a shared exam")
    return frame


def compute_ceiling_checks(
    panel: pd.DataFrame, items: pd.DataFrame, repetitions: int = 2000
) -> pd.DataFrame:
    """Spread, ceiling share, and human association of every model series, on all items
    and on each modality slice."""
    slices = {
        "All items": (panel, None),
        "Text-only items": (_slice_panel(items, panel, multimodal=False), False),
        "Multimodal items": (_slice_panel(items, panel, multimodal=True), True),
    }
    shared_items = items.merge(
        panel[["year", "exam"]].rename(columns={"exam": "group"}),
        on=["year", "group"],
    )
    records = []
    for slice_index, (slice_name, (frame, multimodal)) in enumerate(slices.items()):
        slice_items = (
            shared_items
            if multimodal is None
            else shared_items.loc[shared_items["multimodal"].astype(bool) == multimodal]
        )
        accuracy = slice_items.groupby("model")["answered_correctly"].mean()
        for series_index, (series, label) in enumerate(SERIES_LABELS.items()):
            column = series_column(series)
            within = frame[column] - frame.groupby("grade_bucket")[column].transform(
                "mean"
            )
            human_within = frame["human_pct"] - frame.groupby("grade_bucket")[
                "human_pct"
            ].transform("mean")
            effect, low, high = _coefficient(
                frame,
                "human_pct",
                [column],
                ["grade_bucket"],
                column,
                # All items: the seeds of the grade-FE rows of reproduce-surrogate, so the
                # intervals match human_outcome_regressions.csv.
                seed=(5101 if slice_index == 0 else 6100 + 10 * slice_index)
                + 10 * series_index,
                repetitions=repetitions,
            )
            records.append(
                {
                    "slice": slice_name,
                    "series": series,
                    "series_label": label,
                    "item_accuracy": float(accuracy.get(series, np.nan)),
                    "mean_pct": float(frame[column].mean()),
                    "sd_within_grade": float(within.std(ddof=1)),
                    "share_at_least_95": float((frame[column] >= 95).mean()),
                    "within_grade_r": float(np.corrcoef(within, human_within)[0, 1]),
                    "effect_per_10pp_model": 10 * effect,
                    "ci_low_per_10pp": 10 * low,
                    "ci_high_per_10pp": 10 * high,
                }
            )
    return pd.DataFrame(records)


def compute_change_correlations(
    panel: pd.DataFrame, repetitions: int = 2000
) -> pd.DataFrame:
    """Correlation of changes from one calendar year to the next within each grade, which
    removes every cohort difference that persists from one year to the next, raw and after
    removing the change in visual share from both sides."""
    frame = panel.sort_values(["grade_bucket", "year"]).reset_index(drop=True)
    grouped = frame.groupby("grade_bucket", sort=False)

    def correlation(sample: pd.DataFrame) -> float:
        return float(np.corrcoef(sample["human"], sample["model"])[0, 1])

    def partial_correlation(sample: pd.DataFrame) -> float:
        design = np.column_stack([np.ones(len(sample)), sample["visual"]])
        human, model = (
            sample[name] - design @ np.linalg.lstsq(design, sample[name], rcond=None)[0]
            for name in ("human", "model")
        )
        return float(np.corrcoef(human, model)[0, 1])

    records = []
    for index, (series, label) in enumerate(SERIES_LABELS.items()):
        changes = pd.DataFrame(
            {
                "year": frame["year"],
                "gap": grouped["year"].diff(),
                "human": grouped["human_pct"].diff(),
                "model": grouped[series_column(series)].diff(),
                "visual": grouped["multimodal_share"].diff(),
            }
        )
        # The survey has no 2020 and 2021 results, so 2019 -> 2022 is not a yearly change.
        changes = changes.loc[changes["gap"] == 1].reset_index(drop=True)
        low, high, samples = _cluster_bootstrap_interval(
            changes, correlation, seed=6200 + index, repetitions=repetitions
        )
        partial_low, partial_high, _ = _cluster_bootstrap_interval(
            changes, partial_correlation, seed=6250 + index, repetitions=repetitions
        )
        records.append(
            {
                "series": series,
                "series_label": label,
                "changes": int(len(changes)),
                "correlation": correlation(changes),
                "ci_low": low,
                "ci_high": high,
                "partial_correlation_given_visual": partial_correlation(changes),
                "partial_ci_low": partial_low,
                "partial_ci_high": partial_high,
                "bootstrap_samples": samples,
            }
        )
    return pd.DataFrame(records)


def _nonlinear_terms(panel: pd.DataFrame, series: str) -> pd.DataFrame:
    model = panel[series_column(series)] - panel[series_column(series)].mean()
    visual = panel["multimodal_share"] - panel["multimodal_share"].mean()
    return panel.assign(
        model=model,
        visual=visual,
        model_x_visual=model * visual,
        model_squared=model**2,
    )


def _shared_errors(
    frame: pd.DataFrame, group: pd.DataFrame, predictors: list[str]
) -> pd.DataFrame:
    """Expanding-window predictions with coefficients shared by all grades."""
    group = group.sort_values("year").reset_index(drop=True)
    grade_dummy = f"grade_bucket_{group['grade_bucket'].iloc[0]}"
    records = []
    for index in range(MIN_TRAINING_YEARS, len(group)):
        target = group.iloc[index]
        coefficients = _fit_ols(
            frame.loc[frame["year"] < target["year"]],
            "human_pct",
            predictors,
            ["grade_bucket"],
        )["coefficients"]
        prediction = (
            coefficients["intercept"]
            + coefficients.get(grade_dummy, 0.0)
            + sum(coefficients[name] * float(target[name]) for name in predictors)
        )
        records.append(
            {
                "year": int(target["year"]),
                "observed": float(target["human_pct"]),
                "predicted": float(prediction),
            }
        )
    return pd.DataFrame(records)


def compute_nonlinear_fits(
    panel: pd.DataFrame, repetitions: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """An interaction of model score with visual share and a quadratic model term:
    in-sample fit with grade fixed effects, and forward errors per grade and shared."""
    in_sample = []
    forward = []
    for series_index, (series, label) in enumerate(SERIES_LABELS.items()):
        frame = _nonlinear_terms(panel, series)
        additive_r2 = _fit_ols(
            frame, "human_pct", ["model", "visual"], ["grade_bucket"]
        )["r2"]
        # Each form is compared with the visual-share-only forecast of the same form.
        references = {
            "baseline_error": [],
            "visual_error": [],
            "shared_visual_error": [],
        }
        for _, group in frame.groupby("grade_bucket", sort=False):
            for name, errors in [
                ("baseline_error", _forward_errors(group, [])),
                ("visual_error", _forward_errors(group, ["multimodal_share"])),
                (
                    "shared_visual_error",
                    _shared_errors(frame, group, ["multimodal_share"]),
                ),
            ]:
                errors["grade_bucket"] = group["grade_bucket"].iloc[0]
                errors[name] = errors["observed"] - errors["predicted"]
                references[name].append(errors[["grade_bucket", "year", name]])
        reference = pd.concat(references["baseline_error"])
        for name in ["visual_error", "shared_visual_error"]:
            reference = reference.merge(
                pd.concat(references[name]),
                on=["grade_bucket", "year"],
                validate="one_to_one",
            )
        for spec_index, (spec, predictors) in enumerate(
            NONLINEAR_SPECIFICATIONS.items()
        ):
            fit = _fit_ols(frame, "human_pct", predictors, ["grade_bucket"])
            seed = 6300 + 10 * series_index + spec_index
            record = {
                "series": series,
                "series_label": label,
                "specification": spec,
                "r2": float(fit["r2"]),
                "adjusted_r2": 1
                - (1 - float(fit["r2"]))
                * (len(frame) - 1)
                / (len(frame) - fit["parameters"]),
                "r2_increment_over_additive": float(fit["r2"] - additive_r2),
            }
            added = {"model_x_visual", "model_squared"} & set(predictors)
            if added:
                (term,) = added
                estimate, low, high = _coefficient(
                    frame,
                    "human_pct",
                    predictors,
                    ["grade_bucket"],
                    term,
                    seed,
                    repetitions,
                )
                record.update(
                    {
                        "added_term": estimate,
                        "added_term_ci_low": low,
                        "added_term_ci_high": high,
                    }
                )
            if "model_x_visual" in predictors:
                for share in CONDITIONAL_VISUAL_SHARES:
                    offset = share - panel["multimodal_share"].mean()

                    def slope(sample: pd.DataFrame, offset: float = offset) -> float:
                        coefficients = _fit_ols(
                            sample, "human_pct", predictors, ["grade_bucket"]
                        )["coefficients"]
                        return 10 * float(
                            coefficients["model"]
                            + offset * coefficients["model_x_visual"]
                        )

                    low, high, _ = _cluster_bootstrap_interval(
                        frame,
                        slope,
                        seed=seed + 1000 + round(100 * share),
                        repetitions=repetitions,
                    )
                    key = f"slope_per_10pp_at_visual_{round(100 * share)}"
                    record.update(
                        {
                            key: slope(frame),
                            f"{key}_ci_low": low,
                            f"{key}_ci_high": high,
                        }
                    )
            in_sample.append(record)

            for form_index, form in enumerate(["Per grade", "Shared across grades"]):
                if form == "Per grade" and spec == "Model score + visual share":
                    continue  # reported by reproduce-surrogate
                errors = []
                for grade, group in frame.groupby("grade_bucket", sort=False):
                    grade_errors = (
                        _forward_errors(group, predictors)
                        if form == "Per grade"
                        else _shared_errors(frame, group, predictors)
                    )
                    grade_errors["grade_bucket"] = grade
                    errors.append(grade_errors)
                errors = (
                    pd.concat(errors, ignore_index=True)
                    .assign(error=lambda f: f["observed"] - f["predicted"])
                    .merge(
                        reference, on=["grade_bucket", "year"], validate="one_to_one"
                    )
                )
                visual_column = (
                    "visual_error" if form == "Per grade" else "shared_visual_error"
                )
                intervals = {}
                for name, column in [
                    ("", "baseline_error"),
                    ("_vs_visual", visual_column),
                ]:
                    low, high, _ = _cluster_bootstrap_interval(
                        errors,
                        lambda sample, column=column: float(
                            (sample["error"].abs() - sample[column].abs()).mean()
                        ),
                        seed=seed + 2000 + 100 * form_index + (50 if name else 0),
                        repetitions=repetitions,
                    )
                    intervals[f"mae_change{name}_ci_low"] = low
                    intervals[f"mae_change{name}_ci_high"] = high
                forward.append(
                    {
                        "series": series,
                        "series_label": label,
                        "specification": spec,
                        "form": form,
                        **_forward_metrics(errors),
                        "visual_mae": float(errors[visual_column].abs().mean()),
                        "mae_change_vs_visual": float(
                            (errors["error"].abs() - errors[visual_column].abs()).mean()
                        ),
                        **intervals,
                    }
                )
    return pd.DataFrame(in_sample), pd.DataFrame(forward)


def compute_exposure_checks(
    project_root: Path,
    dataset: pd.DataFrame,
    items: pd.DataFrame,
    panel: pd.DataFrame,
    repetitions: int = 2000,
) -> pd.DataFrame:
    """Checks a training-exposure account predicts: higher accuracy on older items at equal
    composition, a smaller image-removal loss on older items, and lower accuracy after a
    documented knowledge cutoff."""
    items = items.assign(
        correct=items["answered_correctly"].astype(float),
        decade=items["year"] / 10,
        subtype=items["has_associated_image"].astype(int).astype(str)
        + items["has_option_image"].astype(int).astype(str),
        points_tier=items["points"].astype(int).astype(str),
    )
    controls = ["subtype", "group", "points_tier"]
    records = []
    for index, (series, label) in enumerate(MODEL_LABELS.items()):
        frame = items.loc[items["model"] == series].reset_index(drop=True)
        estimate, low, high = _coefficient(
            frame, "correct", ["decade"], controls, "decade", 6400 + index, repetitions
        )
        records.append(
            {
                "check": "Accuracy per later decade, at equal composition",
                "series": series,
                "series_label": label,
                "units": int(len(frame)),
                "estimate": 100 * estimate,
                "ci_low": 100 * low,
                "ci_high": 100 * high,
                "unit": "pp",
            }
        )
        if series == POST_CUTOFF[0]:
            recent = (
                frame.loc[frame["year"] >= LATE_PERIOD_START]
                .assign(
                    after_cutoff=lambda f: (f["year"] == POST_CUTOFF[1]).astype(float)
                )
                .reset_index(drop=True)
            )
            # One contest year cannot be resampled as a cluster, so this interval resamples items.
            estimate = _fit_ols(recent, "correct", ["after_cutoff"], controls)[
                "coefficients"
            ]["after_cutoff"]
            rng = np.random.default_rng(6410)
            draws = [
                _fit_ols(
                    recent.iloc[rng.integers(0, len(recent), len(recent))],
                    "correct",
                    ["after_cutoff"],
                    controls,
                )["coefficients"]["after_cutoff"]
                for _ in range(repetitions)
            ]
            low, high = np.percentile(draws, [2.5, 97.5])
            records.append(
                {
                    "check": f"Accuracy on {POST_CUTOFF[1]} minus {LATE_PERIOD_START}-"
                    f"{POST_CUTOFF[1] - 1} items, at equal composition (item bootstrap)",
                    "series": series,
                    "series_label": label,
                    "units": int(len(recent)),
                    "estimate": 100 * estimate,
                    "ci_low": 100 * low,
                    "ci_high": 100 * high,
                    "unit": "pp",
                }
            )

    runs_dir = project_root / "artifacts" / "runs"
    full_runs = {model: run_id for run_id, model in RUNS.items()}
    blind_runs = {
        model: run_id
        for run_id, (model, arm) in COMPARISON_RUNS.items()
        if arm == "blind"
    }
    for index, series in enumerate(ABLATION_MODELS.values()):
        blind = _scored_run(runs_dir / blind_runs[series], dataset)
        october = _scored_run(runs_dir / full_runs[series], dataset).loc[blind.index]
        frame = pd.DataFrame(
            {
                "year": october["year"].to_numpy(),
                "decade": october["year"].to_numpy() / 10,
                "group": october["group"].to_numpy(),
                "points_tier": october["points"].astype(int).astype(str).to_numpy(),
                "removal_effect": blind["answered_correctly"].to_numpy(dtype=float)
                - october["answered_correctly"].to_numpy(dtype=float),
            }
        )
        estimate, low, high = _coefficient(
            frame,
            "removal_effect",
            ["decade"],
            ["group", "points_tier"],
            "decade",
            6420 + index,
            repetitions,
        )
        records.append(
            {
                "check": "Image-removal effect per later decade",
                "series": series,
                "series_label": MODEL_LABELS[series],
                "units": int(len(frame)),
                "estimate": 100 * estimate,
                "ci_low": 100 * low,
                "ci_high": 100 * high,
                "unit": "pp",
            }
        )

    column = series_column(ENSEMBLE)
    for index, (period, frame) in enumerate(
        [
            (
                f"before {LATE_PERIOD_START}",
                panel.loc[panel["year"] < LATE_PERIOD_START],
            ),
            (
                f"{LATE_PERIOD_START} onward",
                panel.loc[panel["year"] >= LATE_PERIOD_START],
            ),
        ]
    ):
        frame = frame.reset_index(drop=True)
        estimate, low, high = _coefficient(
            frame,
            "human_pct",
            [column],
            ["grade_bucket"],
            column,
            6430 + index,
            repetitions,
        )
        records.append(
            {
                "check": f"Grade-FE association, contests {period}",
                "series": ENSEMBLE,
                "series_label": SERIES_LABELS[ENSEMBLE],
                "units": int(len(frame)),
                "estimate": 10 * estimate,
                "ci_low": 10 * low,
                "ci_high": 10 * high,
                "unit": "per 10 pp model %max",
            }
        )
    later = (panel["year"] >= LATE_PERIOD_START).astype(float)
    frame = panel.assign(later=later, model_x_later=panel[column] * later)
    for index, (check, predictors, fixed_effects, term) in enumerate(
        [
            # Year effects absorb any exposure that is shared by all exams of a contest year.
            (
                "Grade + year FE association, all contests",
                [column],
                ["grade_bucket", "year"],
                column,
            ),
            (
                f"Change in grade-FE association from {LATE_PERIOD_START} on (interaction)",
                [column, "later", "model_x_later"],
                ["grade_bucket"],
                "model_x_later",
            ),
        ]
    ):
        estimate, low, high = _coefficient(
            frame,
            "human_pct",
            predictors,
            fixed_effects,
            term,
            6432 + index,
            repetitions,
        )
        records.append(
            {
                "check": check,
                "series": ENSEMBLE,
                "series_label": SERIES_LABELS[ENSEMBLE],
                "units": int(len(frame)),
                "estimate": 10 * estimate,
                "ci_low": 10 * low,
                "ci_high": 10 * high,
                "unit": "per 10 pp model %max",
            }
        )
    return pd.DataFrame(records)


def _agreement(
    first: pd.DataFrame, second: pd.DataFrame, seed: int, repetitions: int
) -> dict[str, object]:
    """Item-level agreement of two scored runs on the same items in the same order."""
    correct_first = first["answered_correctly"].to_numpy()
    correct_second = second["answered_correctly"].to_numpy()
    agreement = float((correct_first == correct_second).mean())
    chance = correct_first.mean() * correct_second.mean() + (
        1 - correct_first.mean()
    ) * (1 - correct_second.mean())
    difference, low, high = _paired_difference(
        correct_first, correct_second, seed, repetitions
    )
    changed = correct_first != correct_second
    multimodal = first["multimodal"].astype(bool).to_numpy()
    return {
        "items": int(len(first)),
        "accuracy_first": float(correct_first.mean()),
        "accuracy_second": float(correct_second.mean()),
        "difference_pp": 100 * difference,
        "ci_low_pp": 100 * low,
        "ci_high_pp": 100 * high,
        "same_outcome": agreement,
        "kappa": (agreement - chance) / (1 - chance),
        "same_letter": float(
            (
                first["predicted_normalized"].to_numpy()
                == second["predicted_normalized"].to_numpy()
            ).mean()
        ),
        # Wrong and declined answers score differently, so points can change
        # where correctness does not.
        "changed_points": int(
            (
                first["awarded_points"].to_numpy()
                != second["awarded_points"].to_numpy()
            ).sum()
        ),
        "changed_text_only": int((changed & ~multimodal).sum()),
        "changed_multimodal": int((changed & multimodal).sum()),
        "multimodal_items": int(multimodal.sum()),
    }


def compute_rerun_agreement(
    project_root: Path, dataset: pd.DataFrame, repetitions: int = 2000
) -> pd.DataFrame:
    """Agreement of the October 2025 runs (first) with the August 2026 German control arm
    (second) on the 200 shared items. The arms differ in date, provider state, and the
    omitted question crop, so disagreement mixes run-to-run noise with these changes."""
    runs_dir = project_root / "artifacts" / "runs"
    full_runs = {model: run_id for run_id, model in RUNS.items()}
    control_runs = {
        model: run_id
        for run_id, (model, arm) in COMPARISON_RUNS.items()
        if arm == "german_control"
    }
    records = []
    for index, series in enumerate(ABLATION_MODELS.values()):
        control = _scored_run(runs_dir / control_runs[series], dataset)
        october = _scored_run(runs_dir / full_runs[series], dataset).loc[control.index]
        records.append(
            {
                "series": series,
                "series_label": MODEL_LABELS[series],
                **_agreement(october, control, 6500 + index, repetitions),
            }
        )
    return pd.DataFrame(records)


def load_later_items(project_root: Path, dataset: pd.DataFrame) -> pd.DataFrame:
    """Scored items of the November 2026 runs, one row per series and item. Answers of the
    option-order arm are mapped back to the original letters. The Ministral run enters a
    second and third time, with its first-letter fallback rows replaced by the answer the
    response states and by no parsed answer."""
    runs_dir = project_root / "artifacts" / "runs"
    frames = {}
    for run_id, (model, arm) in LATER_RUNS.items():
        run = read_compact_run(runs_dir / run_id)
        if arm == "full":
            frames[model] = run
        elif arm == "repeat":
            frames[SONNET_REPEAT] = run
        else:
            mapping = pd.read_csv(
                runs_dir / run_id / "option_shuffle_mapping.csv", keep_default_na=False
            ).set_index("id")
            sources = mapping.loc[
                run["source_id"], [f"new_{letter}_source" for letter in LETTERS]
            ].to_numpy()
            position = run["predicted"].fillna("").str.strip().str.upper()
            chosen = position.map(LETTERS.find).to_numpy()
            frames[SONNET_OPTION_ORDER] = run.assign(
                position=position,
                predicted=np.where(
                    chosen >= 0,
                    sources[np.arange(len(run)), chosen.clip(0)],
                    run["predicted"],
                ),
            )
    ministral = frames[MINISTRAL]
    fallback = ministral["fallback_parse"].to_numpy()
    frames[MINISTRAL_STATED] = ministral.assign(
        predicted=np.where(fallback, ministral["stated_answer"], ministral["predicted"])
    )
    frames[MINISTRAL_UNPARSED] = ministral.assign(
        predicted=np.where(fallback, None, ministral["predicted"])
    )
    return pd.concat(
        [
            score_predictions(frame, dataset).assign(model=series)
            for series, frame in frames.items()
        ],
        ignore_index=True,
    )


def add_later_series(
    panel: pd.DataFrame, later_items: pd.DataFrame, form_adjustments: pd.DataFrame
) -> pd.DataFrame:
    """The shared-exam panel with the %max series of the November 2026 runs."""
    scores = aggregate_exam_scores(later_items, form_adjustments).pivot_table(
        index=["year", "exam"], columns="model", values="llm_pct"
    )
    scores.columns = [series_column(series) for series in scores.columns]
    extended = panel.merge(
        scores.reset_index(), on=["year", "exam"], validate="one_to_one"
    )
    runs = [series_column(s) for s in (SONNET, SONNET_REPEAT, SONNET_OPTION_ORDER)]
    extended[series_column(SONNET_TWO_RUNS)] = extended[runs[:2]].mean(axis=1)
    extended[series_column(SONNET_THREE_RUNS)] = extended[runs].mean(axis=1)
    if len(extended) != len(panel) or extended.isna().any().any():
        raise ValueError("A later run does not cover every shared exam")
    return extended


def _grade_errors(panel: pd.DataFrame, errors_of_grade) -> pd.DataFrame:
    frame = pd.concat(
        [
            errors_of_grade(group).assign(grade_bucket=grade)
            for grade, group in panel.groupby("grade_bucket", sort=False)
        ],
        ignore_index=True,
    )
    return frame.assign(error=frame["observed"] - frame["predicted"])


def _mae_change(
    errors: pd.DataFrame, reference: pd.DataFrame, seed: int, repetitions: int
) -> tuple[float, float, float]:
    frame = errors.merge(
        reference[["grade_bucket", "year", "error"]].rename(
            columns={"error": "baseline_error"}
        ),
        on=["grade_bucket", "year"],
        validate="one_to_one",
    )
    low, high, _ = _cluster_bootstrap_interval(
        frame,
        lambda sample: float(
            (sample["error"].abs() - sample["baseline_error"].abs()).mean()
        ),
        seed=seed,
        repetitions=repetitions,
    )
    return float(_forward_metrics(frame)["mae_change"]), low, high


def compute_later_runs(
    panel: pd.DataFrame, later_items: pd.DataFrame, repetitions: int = 2000
) -> pd.DataFrame:
    """The reference-anchor tests on each November 2026 series: spread and ceiling share,
    the grade-fixed-effect association with the human mean and with the exam's visual share,
    and the change in forward MAE against the grade mean (slope per grade and shared),
    against visual share when the model score is added to it, and against the recent mean."""
    baseline = _grade_errors(panel, lambda group: _forward_errors(group, []))
    visual = _grade_errors(
        panel, lambda group: _forward_errors(group, ["multimodal_share"])
    )
    recent = compute_recent_mean_increment(
        panel, LATER_SERIES, seed=7602, repetitions=repetitions
    ).set_index("series")
    human_within = panel["human_pct"] - panel.groupby("grade_bucket")[
        "human_pct"
    ].transform("mean")
    records = []
    for index, (series, label) in enumerate(LATER_SERIES.items()):
        column = series_column(series)
        within = panel[column] - panel.groupby("grade_bucket")[column].transform("mean")
        effect, low, high = _coefficient(
            panel,
            "human_pct",
            [column],
            ["grade_bucket"],
            column,
            # The Sonnet rows use the seed of the October Sonnet row in ceiling_checks.csv,
            # so their intervals differ through the scores only.
            seed=5131 if series.startswith(SONNET) else 7100,
            repetitions=repetitions,
        )
        record = {"series": series, "series_label": label}
        items = later_items.loc[later_items["model"] == series]
        if len(items):
            multimodal = items["multimodal"].astype(bool)
            record.update(
                {
                    "items": int(len(items)),
                    "item_accuracy": float(items["answered_correctly"].mean()),
                    "text_only_accuracy": float(
                        items.loc[~multimodal, "answered_correctly"].mean()
                    ),
                    "multimodal_accuracy": float(
                        items.loc[multimodal, "answered_correctly"].mean()
                    ),
                    "declined_share": float(items["declined"].mean()),
                }
            )
        record.update(
            {
                "mean_pct": float(panel[column].mean()),
                "sd_within_grade": float(within.std(ddof=1)),
                "share_at_least_95": float((panel[column] >= 95).mean()),
                "share_at_100": float(np.isclose(panel[column], 100).mean()),
                "within_grade_r": float(np.corrcoef(within, human_within)[0, 1]),
                "effect_per_10pp_model": 10 * effect,
                "ci_low_per_10pp": 10 * low,
                "ci_high_per_10pp": 10 * high,
            }
        )
        visual_effect, visual_low, visual_high = _coefficient(
            panel,
            column,
            ["multimodal_share"],
            ["grade_bucket"],
            "multimodal_share",
            seed=7200 + index,
            repetitions=repetitions,
        )
        record.update(
            {
                "pct_per_10pp_visual_share": visual_effect / 10,
                "pct_per_10pp_visual_share_ci_low": visual_low / 10,
                "pct_per_10pp_visual_share_ci_high": visual_high / 10,
            }
        )
        comparisons = {
            "per_grade": (
                lambda group: _forward_errors(group, [column]),
                baseline,
                7302,
            ),
            "shared_slope": (
                lambda group: _shared_slope_errors(panel, group, column),
                baseline,
                7402,
            ),
            "with_visual_vs_visual": (
                lambda group: _forward_errors(group, [column, "multimodal_share"]),
                visual,
                7502,
            ),
        }
        for name, (errors_of_grade, reference, seed) in comparisons.items():
            change, low, high = _mae_change(
                _grade_errors(panel, errors_of_grade),
                reference,
                seed + index,
                repetitions,
            )
            record.update(
                {
                    f"mae_change_{name}": change,
                    f"mae_change_{name}_ci_low": low,
                    f"mae_change_{name}_ci_high": high,
                }
            )
        record.update(
            {
                "mae_change_recent_mean": recent.at[series, "mae_change"],
                "mae_change_recent_mean_ci_low": recent.at[series, "mae_change_ci_low"],
                "mae_change_recent_mean_ci_high": recent.at[
                    series, "mae_change_ci_high"
                ],
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _option_order_mapping(project_root: Path) -> pd.DataFrame:
    (run_id,) = [run for run, (_, arm) in LATER_RUNS.items() if arm == "option_order"]
    return pd.read_csv(
        project_root / "artifacts" / "runs" / run_id / "option_shuffle_mapping.csv",
        keep_default_na=False,
    ).set_index("id")


def compute_sonnet_run_agreement(
    project_root: Path,
    items: pd.DataFrame,
    later_items: pd.DataFrame,
    panel: pd.DataFrame,
    repetitions: int = 2000,
) -> pd.DataFrame:
    """Item-level agreement of the three full Claude Sonnet 4.5 runs: October 2025, its
    November 2026 repeat, and the option-order arm, on all items and on the items whose
    options were or were not reordered. The option-order arm also omits the question crop,
    so its disagreement with the other runs mixes run-to-run noise, the crop, and the
    order; "same_letter" compares the chosen option after mapping back."""
    october = items.loc[items["model"] == SONNET].set_index("source_id")
    runs = {
        SONNET: october,
        SONNET_REPEAT: later_items.loc[later_items["model"] == SONNET_REPEAT]
        .set_index("source_id")
        .loc[october.index],
        SONNET_OPTION_ORDER: later_items.loc[
            later_items["model"] == SONNET_OPTION_ORDER
        ]
        .set_index("source_id")
        .loc[october.index],
    }
    mapping = _option_order_mapping(project_root).loc[october.index]
    reordered = (mapping["kept_order_reason"] == "").to_numpy()
    moved = (mapping["old_answer"] != mapping["new_answer"]).to_numpy()
    multimodal = october["multimodal"].astype(bool).to_numpy()
    subsets = {
        "All items": np.ones(len(october), dtype=bool),
        "Reordered": reordered,
        "Reordered, key moved": reordered & moved,
        "Reordered, text-only": reordered & ~multimodal,
        "Reordered, multimodal": reordered & multimodal,
        "Order kept": ~reordered,
    }
    pairs = [
        (SONNET, SONNET_REPEAT),
        (SONNET_REPEAT, SONNET_OPTION_ORDER),
        (SONNET, SONNET_OPTION_ORDER),
    ]
    labels = {**MODEL_LABELS, **LATER_SERIES, SONNET: "Claude Sonnet 4.5, October 2025"}
    records = []
    for subset, mask in subsets.items():
        for index, (first, second) in enumerate(pairs):
            record = {
                "subset": subset,
                "first": labels[first],
                "second": labels[second],
                **_agreement(
                    runs[first].loc[mask],
                    runs[second].loc[mask],
                    7300 + index,
                    repetitions,
                ),
            }
            if subset == "All items":
                scores = panel[[series_column(first), series_column(second)]]
                within = scores - scores.groupby(panel["grade_bucket"]).transform(
                    "mean"
                )
                record.update(
                    {
                        "exam_score_r": float(scores.corr().iloc[0, 1]),
                        "exam_score_r_within_grade": float(within.corr().iloc[0, 1]),
                        "mean_abs_exam_score_change": float(
                            scores.diff(axis=1).iloc[:, 1].abs().mean()
                        ),
                    }
                )
            records.append(record)
    return pd.DataFrame(records)


def compute_option_positions(
    project_root: Path, later_items: pd.DataFrame
) -> pd.DataFrame:
    """Position effects on the reordered items of the option-order arm, where the position
    of the key is random: accuracy by key position, and how often each position is chosen,
    holds the key, and receives a wrong answer. The November repeat gives the accuracy by
    key position in the printed order on the same items."""
    mapping = _option_order_mapping(project_root)
    reordered = mapping.index[mapping["kept_order_reason"] == ""]
    arm = (
        later_items.loc[later_items["model"] == SONNET_OPTION_ORDER]
        .set_index("source_id")
        .loc[reordered]
    )
    repeat = (
        later_items.loc[later_items["model"] == SONNET_REPEAT]
        .set_index("source_id")
        .loc[reordered]
    )
    key = mapping.loc[reordered, "new_answer"]
    wrong = arm.loc[~arm["answered_correctly"] & arm["position"].isin(list(LETTERS))]
    records = []
    for letter in LETTERS:
        records.append(
            {
                "position": letter,
                "key_items": int((key == letter).sum()),
                "accuracy": float(arm.loc[key == letter, "answered_correctly"].mean()),
                "key_share": float((key == letter).mean()),
                "chosen_share": float((arm["position"] == letter).mean()),
                "wrong_answer_share": float((wrong["position"] == letter).mean()),
                "printed_order_key_items": int((repeat["answer"] == letter).sum()),
                "printed_order_accuracy": float(
                    repeat.loc[repeat["answer"] == letter, "answered_correctly"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def run_robustness(
    project_root: Path, dataset_path: Path, write_outputs: bool = True
) -> dict[str, object]:
    results = run_diagnostics(project_root, dataset_path, write_outputs=False)
    panel = build_panel(results["exam_data"], results["exam_scores"])
    items, dataset = results["items"], results["dataset"]
    later_items = load_later_items(project_root, dataset)
    later_panel = add_later_series(panel, later_items, results["form_adjustments"])
    tables = {
        "recent_mean_increment": compute_recent_mean_increment(panel),
        "ceiling_checks": compute_ceiling_checks(panel, items),
        "change_correlations": compute_change_correlations(panel),
        "exposure_checks": compute_exposure_checks(project_root, dataset, items, panel),
        "rerun_agreement": compute_rerun_agreement(project_root, dataset),
        "later_runs": compute_later_runs(later_panel, later_items),
        "sonnet_run_agreement": compute_sonnet_run_agreement(
            project_root, items, later_items, later_panel
        ),
        "option_positions": compute_option_positions(project_root, later_items),
    }
    tables["nonlinear_fits"], tables["nonlinear_forward"] = compute_nonlinear_fits(
        panel
    )
    within_visual = panel["multimodal_share"] - panel.groupby("grade_bucket")[
        "multimodal_share"
    ].transform("mean")
    within_year = panel["year"] - panel.groupby("grade_bucket")["year"].transform(
        "mean"
    )
    human_within = panel["human_pct"] - panel.groupby("grade_bucket")[
        "human_pct"
    ].transform("mean")
    metrics = {
        "within_grade_visual_share_year_correlation": float(
            np.corrcoef(within_visual, within_year)[0, 1]
        ),
        "human_sd_within_grade": float(human_within.std(ddof=1)),
        "human_r2_grade": float(
            _fit_ols(panel, "human_pct", [], ["grade_bucket"])["r2"]
        ),
        "human_r2_grade_year": float(
            _fit_ols(panel, "human_pct", [], ["grade_bucket", "year"])["r2"]
        ),
    }
    ministral = later_items.loc[later_items["model"] == MINISTRAL]
    fallback = ministral.loc[ministral["fallback_parse"].astype(bool)]
    stated = fallback.loc[fallback["stated_answer"].notna()]
    mapping = _option_order_mapping(project_root)
    reordered = mapping.loc[mapping["kept_order_reason"] == ""]
    metrics.update(
        {
            "ministral_fallback_rows": int(len(fallback)),
            "ministral_capped_rows": int(ministral["output_capped"].sum()),
            "ministral_fallback_or_capped_rows": int(
                (ministral["fallback_parse"] | ministral["output_capped"]).sum()
            ),
            "ministral_fallback_first_letter_accuracy": float(
                fallback["answered_correctly"].mean()
            ),
            "ministral_fallback_rows_with_stated_answer": int(len(stated)),
            "ministral_fallback_stated_answer_accuracy": float(
                (stated["stated_answer"] == stated["answer"]).mean()
            ),
            "option_order_reordered_items": int(len(reordered)),
            "option_order_key_moved_items": int(
                (reordered["old_answer"] != reordered["new_answer"]).sum()
            ),
        }
    )
    if write_outputs:
        output_dir = project_root / "reproduced" / "robustness"
        table_dir = output_dir / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name, table in tables.items():
            table.to_csv(table_dir / f"{name}.csv", index=False)
    return {"metrics": metrics, **tables}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/kangaroo.parquet"),
        help="The released dataset file (default: data/kangaroo.parquet).",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = args.dataset.expanduser()
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    run_robustness(project_root, dataset_path)
    print(f"Outputs: {project_root / 'reproduced' / 'robustness'}")


if __name__ == "__main__":
    main()
