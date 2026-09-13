"""Surrogate-validity analysis with aggregate human performance as the outcome."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figure_style import (
    BRICK,
    FILLS,
    GRADE_COLORS,
    INK,
    MUTED,
    POINT,
    SERIES_COLORS,
    TEAL,
    styled_subplots,
)
from analysis.diagnostics import (
    MODEL_LABELS,
    _cluster_bootstrap_interval,
    _fit_ols,
    run_diagnostics,
)
from analysis.inputs import (
    COMPARISON_RUNS,
    RUNS,
    load_human_scores,
    read_compact_run,
    score_predictions,
)

ENSEMBLE = "ensemble"
SERIES_LABELS = {**MODEL_LABELS, ENSEMBLE: "Equal-weight ensemble"}
COMPOSITION_COLUMNS = [
    "multimodal_share",
    "associated_image_share",
    "option_image_share",
]
OUTCOME_SPECIFICATIONS = [
    ("Raw", [], []),
    ("Grade fixed effects", [], ["grade_bucket"]),
    ("Grade FE + visual share", ["multimodal_share"], ["grade_bucket"]),
    (
        "Grade FE + image-location shares",
        ["associated_image_share", "option_image_share"],
        ["grade_bucket"],
    ),
]
FORWARD_SPECIFICATIONS = {
    "Grade mean": [],
    "Visual share": ["multimodal_share"],
    "Model score": ["<series>"],
    "Model score + visual share": ["<series>", "multimodal_share"],
}
FORWARD_BASELINE = "Grade mean"
VISUAL_SPECIFICATION = "Visual share"
GRADE_ORDER = ["Grade 3-4", "Grade 5-6", "Grade 7-8", "Grade 9-10", "Grade 11-13"]
MIN_TRAINING_YEARS = 8
RECENT_WINDOW_YEARS = 3
SHARED_COMPARISONS = 115


def series_column(series: str) -> str:
    return f"model_pct__{series}"


def build_panel(exam_data: pd.DataFrame, exam_scores: pd.DataFrame) -> pd.DataFrame:
    """Combine the shared-exam panel with every model's own %max series."""
    panel = exam_data[
        [
            "year",
            "exam",
            "grade_bucket",
            "human_pct",
            "human_mean_source",
            "pooled_ensemble_pct",
            "students",
            "item_count",
            *COMPOSITION_COLUMNS,
        ]
    ].copy()
    model_pct = (
        exam_scores.pivot_table(
            index=["year", "exam"], columns="model", values="llm_pct"
        )
        .rename(columns={model: series_column(model) for model in MODEL_LABELS})
        .reset_index()
    )
    panel = panel.merge(
        model_pct, on=["year", "exam"], how="left", validate="one_to_one"
    )
    panel[series_column(ENSEMBLE)] = panel["pooled_ensemble_pct"]
    if len(panel) != SHARED_COMPARISONS or panel.isna().any().any():
        raise ValueError("Panel must cover the 115 shared exams with every model score")
    model_mean = panel[[series_column(model) for model in MODEL_LABELS]].mean(axis=1)
    if not np.allclose(model_mean, panel["pooled_ensemble_pct"]):
        raise ValueError("Pooled ensemble must equal the mean of the model %max series")
    grade_rank = panel["grade_bucket"].map({g: i for i, g in enumerate(GRADE_ORDER)})
    if grade_rank.isna().any():
        raise ValueError("Unexpected grade bucket label in the shared-exam panel")
    return (
        panel.assign(grade_rank=grade_rank)
        .sort_values(["grade_rank", "year"])
        .drop(columns="grade_rank")
        .reset_index(drop=True)
    )


def compute_outcome_regressions(
    panel: pd.DataFrame, repetitions: int = 2000
) -> pd.DataFrame:
    """Regress human %max on each model series under composition controls."""
    records = []
    for series_index, (series, label) in enumerate(SERIES_LABELS.items()):
        column = series_column(series)
        for spec_index, (spec, covariates, fixed_effects) in enumerate(
            OUTCOME_SPECIFICATIONS
        ):
            predictors = [column, *covariates]
            full = _fit_ols(panel, "human_pct", predictors, fixed_effects)
            baseline = _fit_ols(panel, "human_pct", covariates, fixed_effects)
            coefficient = float(full["coefficients"][column])
            interval = _cluster_bootstrap_interval(
                panel,
                lambda sample, predictors=predictors, fixed_effects=fixed_effects: (
                    float(
                        _fit_ols(sample, "human_pct", predictors, fixed_effects)[
                            "coefficients"
                        ][column]
                    )
                ),
                seed=5100 + 10 * series_index + spec_index,
                repetitions=repetitions,
            )
            records.append(
                {
                    "series": series,
                    "series_label": label,
                    "specification": spec,
                    "effect_per_10pp_model": 10 * coefficient,
                    "ci_low_per_10pp": 10 * interval[0],
                    "ci_high_per_10pp": 10 * interval[1],
                    "bootstrap_samples": interval[2],
                    "r2_full": float(full["r2"]),
                    "r2_baseline": float(baseline["r2"]),
                    "r2_increment": float(full["r2"] - baseline["r2"]),
                    "covariates": ", ".join(covariates) or "none",
                    "fixed_effects": ", ".join(fixed_effects) or "none",
                }
            )
    return pd.DataFrame(records)


def _forward_errors(group: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    group = group.sort_values("year").reset_index(drop=True)
    records = []
    for index in range(MIN_TRAINING_YEARS, len(group)):
        training = group.iloc[:index]
        target = group.iloc[index]
        coefficients = _fit_ols(training, "human_pct", predictors, [])["coefficients"]
        prediction = coefficients["intercept"] + sum(
            coefficients[predictor] * float(target[predictor])
            for predictor in predictors
        )
        records.append(
            {
                "year": int(target["year"]),
                "training_years": int(index),
                "observed": float(target["human_pct"]),
                "predicted": float(prediction),
            }
        )
    return pd.DataFrame(records)


def compute_forward_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window prospective prediction of human %max within each grade."""
    frames = []
    for grade, group in panel.groupby("grade_bucket", sort=False):
        for series in SERIES_LABELS:
            for spec, template in FORWARD_SPECIFICATIONS.items():
                predictors = [
                    series_column(series) if item == "<series>" else item
                    for item in template
                ]
                errors = _forward_errors(group, predictors)
                errors["grade_bucket"] = grade
                errors["series"] = series
                errors["series_label"] = SERIES_LABELS[series]
                errors["specification"] = spec
                frames.append(errors)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["error"] = predictions["observed"] - predictions["predicted"]
    baseline = predictions.loc[
        predictions["specification"] == FORWARD_BASELINE,
        ["grade_bucket", "series", "year", "error"],
    ].rename(columns={"error": "baseline_error"})
    return predictions.merge(
        baseline, on=["grade_bucket", "series", "year"], validate="many_to_one"
    )


def _forward_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    squared = float(np.sum(frame["error"] ** 2))
    baseline_squared = float(np.sum(frame["baseline_error"] ** 2))
    return {
        "predictions": int(len(frame)),
        "mae": float(frame["error"].abs().mean()),
        "baseline_mae": float(frame["baseline_error"].abs().mean()),
        "mae_change": float(
            (frame["error"].abs() - frame["baseline_error"].abs()).mean()
        ),
        "forward_r2": 1.0 - squared / baseline_squared,
        "years_beating_baseline": int(
            (frame["error"].abs() < frame["baseline_error"].abs()).sum()
        ),
    }


def summarize_forward_predictions(
    predictions: pd.DataFrame, repetitions: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-grade and pooled forward-prediction metrics against the grade-mean baseline."""
    per_grade = pd.DataFrame(
        [
            {
                "grade_bucket": grade,
                "series": series,
                "series_label": SERIES_LABELS[series],
                "specification": spec,
                **_forward_metrics(frame),
            }
            for (grade, series, spec), frame in predictions.groupby(
                ["grade_bucket", "series", "specification"], sort=False
            )
        ]
    )
    visual_errors = (
        predictions.loc[predictions["specification"] == VISUAL_SPECIFICATION]
        .drop_duplicates(["grade_bucket", "year"])[["grade_bucket", "year", "error"]]
        .rename(columns={"error": "visual_error"})
    )
    pooled_records = []
    for index, ((series, spec), frame) in enumerate(
        predictions.groupby(["series", "specification"], sort=False)
    ):
        frame = frame.merge(
            visual_errors,
            on=["grade_bucket", "year"],
            how="left",
            validate="one_to_one",
        )
        if frame["visual_error"].isna().any():
            raise ValueError(
                "Visual-share predictions do not cover every held-out year"
            )
        interval = _cluster_bootstrap_interval(
            frame,
            lambda sample: float(
                (sample["error"].abs() - sample["baseline_error"].abs()).mean()
            ),
            seed=5200 + index,
            repetitions=repetitions,
        )
        visual_interval = _cluster_bootstrap_interval(
            frame,
            lambda sample: float(
                (sample["error"].abs() - sample["visual_error"].abs()).mean()
            ),
            seed=5300 + index,
            repetitions=repetitions,
        )
        pooled_records.append(
            {
                "series": series,
                "series_label": SERIES_LABELS[series],
                "specification": spec,
                **_forward_metrics(frame),
                "mae_change_ci_low": interval[0],
                "mae_change_ci_high": interval[1],
                "mae_change_vs_visual": float(
                    (frame["error"].abs() - frame["visual_error"].abs()).mean()
                ),
                "mae_change_vs_visual_ci_low": visual_interval[0],
                "mae_change_vs_visual_ci_high": visual_interval[1],
                "bootstrap_samples": interval[2],
                "grades_beating_baseline_mae": int(
                    (
                        per_grade.loc[
                            (per_grade["series"] == series)
                            & (per_grade["specification"] == spec),
                            "mae",
                        ].to_numpy()
                        < per_grade.loc[
                            (per_grade["series"] == series)
                            & (per_grade["specification"] == spec),
                            "baseline_mae",
                        ].to_numpy()
                    ).sum()
                ),
            }
        )
    return per_grade, pd.DataFrame(pooled_records)


def _recent_mean_errors(group: pd.DataFrame, window: int) -> pd.DataFrame:
    group = group.sort_values("year").reset_index(drop=True)
    records = []
    for index in range(MIN_TRAINING_YEARS, len(group)):
        target = group.iloc[index]
        records.append(
            {
                "year": int(target["year"]),
                "training_years": int(index),
                "observed": float(target["human_pct"]),
                "predicted": float(
                    group.iloc[index - window : index]["human_pct"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def _shared_slope_errors(
    panel: pd.DataFrame, group: pd.DataFrame, column: str
) -> pd.DataFrame:
    group = group.sort_values("year").reset_index(drop=True)
    grade_dummy = f"grade_bucket_{group['grade_bucket'].iloc[0]}"
    records = []
    for index in range(MIN_TRAINING_YEARS, len(group)):
        target = group.iloc[index]
        training = panel.loc[panel["year"] < target["year"]]
        coefficients = _fit_ols(training, "human_pct", [column], ["grade_bucket"])[
            "coefficients"
        ]
        prediction = (
            coefficients["intercept"]
            + coefficients.get(grade_dummy, 0.0)
            + coefficients[column] * float(target[column])
        )
        records.append(
            {
                "year": int(target["year"]),
                "training_years": int(index),
                "observed": float(target["human_pct"]),
                "predicted": float(prediction),
            }
        )
    return pd.DataFrame(records)


def compute_forward_sensitivity(
    panel: pd.DataFrame, predictions: pd.DataFrame, repetitions: int = 2000
) -> pd.DataFrame:
    """Forward errors of a recent-history baseline and of shared-slope calibrations."""
    ensemble_rows = predictions.loc[predictions["series"] == ENSEMBLE]
    reference = {
        "baseline_error": FORWARD_BASELINE,
        "visual_error": VISUAL_SPECIFICATION,
    }
    variants = [("Recent mean (last three years)", None)]
    variants += [("Model score, shared slope", series) for series in SERIES_LABELS]
    records = []
    for index, (spec, series) in enumerate(variants):
        frames = []
        for grade, group in panel.groupby("grade_bucket", sort=False):
            errors = (
                _recent_mean_errors(group, RECENT_WINDOW_YEARS)
                if series is None
                else _shared_slope_errors(panel, group, series_column(series))
            )
            errors["grade_bucket"] = grade
            frames.append(errors)
        frame = pd.concat(frames, ignore_index=True)
        frame["error"] = frame["observed"] - frame["predicted"]
        for column, reference_spec in reference.items():
            frame = frame.merge(
                ensemble_rows.loc[
                    ensemble_rows["specification"] == reference_spec,
                    ["grade_bucket", "year", "error"],
                ].rename(columns={"error": column}),
                on=["grade_bucket", "year"],
                validate="one_to_one",
            )
        interval = _cluster_bootstrap_interval(
            frame,
            lambda sample: float(
                (sample["error"].abs() - sample["baseline_error"].abs()).mean()
            ),
            seed=5500 + index,
            repetitions=repetitions,
        )
        visual_interval = _cluster_bootstrap_interval(
            frame,
            lambda sample: float(
                (sample["error"].abs() - sample["visual_error"].abs()).mean()
            ),
            seed=5600 + index,
            repetitions=repetitions,
        )
        records.append(
            {
                "series": series or "none",
                "series_label": SERIES_LABELS[series] if series else "Cohort history",
                "specification": spec,
                **_forward_metrics(frame),
                "mae_change_ci_low": interval[0],
                "mae_change_ci_high": interval[1],
                "mae_change_vs_visual": float(
                    (frame["error"].abs() - frame["visual_error"].abs()).mean()
                ),
                "mae_change_vs_visual_ci_low": visual_interval[0],
                "mae_change_vs_visual_ci_high": visual_interval[1],
                "bootstrap_samples": interval[2],
            }
        )
    return pd.DataFrame(records)


def compute_human_mean_sensitivity(
    panel: pd.DataFrame, histogram_scores: pd.DataFrame, repetitions: int = 2000
) -> pd.DataFrame:
    """Bin-midpoint means against the reported means, and forward errors using them."""
    merged = panel.merge(
        histogram_scores[["year", "exam", "human_pct"]].rename(
            columns={"human_pct": "histogram_pct"}
        ),
        on=["year", "exam"],
        validate="one_to_one",
    )
    reported = merged.loc[merged["human_mean_source"] == "reported"]
    deviation = reported["histogram_pct"] - reported["human_pct"]
    alternative = panel.assign(human_pct=merged["histogram_pct"].to_numpy())
    predictions = compute_forward_predictions(alternative)
    _, pooled = summarize_forward_predictions(
        predictions.loc[predictions["series"] == ENSEMBLE], repetitions=repetitions
    )
    ensemble = pooled.set_index("specification")
    return pd.DataFrame(
        [
            {
                "forms_reported": int(len(reported)),
                "forms_estimated": int(
                    (merged["human_mean_source"] == "histogram").sum()
                ),
                "midpoint_minus_reported_mean": float(deviation.mean()),
                "midpoint_minus_reported_mae": float(deviation.abs().mean()),
                "midpoint_minus_reported_max_abs": float(deviation.abs().max()),
                "forward_baseline_mae": float(ensemble.loc[FORWARD_BASELINE, "mae"]),
                "forward_visual_mae": float(ensemble.loc[VISUAL_SPECIFICATION, "mae"]),
                "forward_ensemble_mae": float(ensemble.loc["Model score", "mae"]),
                "forward_ensemble_mae_change": float(
                    ensemble.loc["Model score", "mae_change"]
                ),
                "forward_ensemble_mae_change_ci_low": float(
                    ensemble.loc["Model score", "mae_change_ci_low"]
                ),
                "forward_ensemble_mae_change_ci_high": float(
                    ensemble.loc["Model score", "mae_change_ci_high"]
                ),
                "bootstrap_samples": int(
                    ensemble.loc["Model score", "bootstrap_samples"]
                ),
            }
        ]
    )


def compute_variance_decomposition(
    panel: pd.DataFrame, repetitions: int = 2000
) -> pd.DataFrame:
    """Variance explained by grade and visual composition; visual-share coefficients."""
    outcomes = {
        "human_pct": "Cohort",
        **{series_column(series): label for series, label in SERIES_LABELS.items()},
    }
    records = []
    for index, (column, label) in enumerate(outcomes.items()):
        grade_only = _fit_ols(panel, column, [], ["grade_bucket"])
        visual = _fit_ols(panel, column, ["multimodal_share"], ["grade_bucket"])
        location = _fit_ols(
            panel,
            column,
            ["associated_image_share", "option_image_share"],
            ["grade_bucket"],
        )
        interval = _cluster_bootstrap_interval(
            panel,
            lambda sample, column=column: float(
                _fit_ols(sample, column, ["multimodal_share"], ["grade_bucket"])["r2"]
                - _fit_ols(sample, column, [], ["grade_bucket"])["r2"]
            ),
            seed=5300 + index,
            repetitions=repetitions,
        )
        calendar = {}
        for name, predictors, fixed_effects, seed in [
            ("", ["multimodal_share"], ["grade_bucket"], 5700),
            ("_year_fe", ["multimodal_share"], ["grade_bucket", "year"], 5800),
            ("_trend", ["multimodal_share", "year"], ["grade_bucket"], 5900),
        ]:
            fit = _fit_ols(panel, column, predictors, fixed_effects)
            calendar_interval = _cluster_bootstrap_interval(
                panel,
                lambda sample, column=column, predictors=predictors, fe=fixed_effects: (
                    float(
                        _fit_ols(sample, column, predictors, fe)["coefficients"][
                            "multimodal_share"
                        ]
                    )
                ),
                seed=seed + index,
                repetitions=repetitions,
            )
            calendar[f"effect_per_10pp_visual_share{name}"] = 0.1 * float(
                fit["coefficients"]["multimodal_share"]
            )
            calendar[f"effect_per_10pp_visual_share{name}_ci_low"] = (
                0.1 * calendar_interval[0]
            )
            calendar[f"effect_per_10pp_visual_share{name}_ci_high"] = (
                0.1 * calendar_interval[1]
            )
        records.append(
            {
                "outcome": column,
                "outcome_label": label,
                "sd_total": float(panel[column].std(ddof=1)),
                "sd_within_grade": float(np.std(grade_only["residuals"], ddof=1)),
                "r2_grade": float(grade_only["r2"]),
                "r2_grade_visual": float(visual["r2"]),
                "r2_grade_location": float(location["r2"]),
                "visual_r2_increment": float(visual["r2"] - grade_only["r2"]),
                "visual_r2_increment_ci_low": interval[0],
                "visual_r2_increment_ci_high": interval[1],
                **calendar,
                "bootstrap_samples": interval[2],
            }
        )
    return pd.DataFrame(records)


def plot_outcome_coefficients(regressions: pd.DataFrame) -> plt.Figure:
    figure, axis = styled_subplots(figsize=(7.2, 3.0))
    specifications = [spec for spec, _, _ in OUTCOME_SPECIFICATIONS]
    short_labels = {
        "Raw": "Raw",
        "Grade fixed effects": "+ grade FE",
        "Grade FE + visual share": "+ visual share",
        "Grade FE + image-location shares": "diagram + option shares",
    }
    colors = SERIES_COLORS
    offsets = np.linspace(-0.3, 0.3, len(SERIES_LABELS))
    axis.axvline(0, color=INK, linewidth=0.7)
    for offset, color, (series, label) in zip(
        offsets, colors, SERIES_LABELS.items(), strict=True
    ):
        rows = regressions.loc[regressions["series"] == series].set_index(
            "specification"
        )
        estimates = rows.loc[specifications, "effect_per_10pp_model"].to_numpy()
        lower = rows.loc[specifications, "ci_low_per_10pp"].to_numpy()
        upper = rows.loc[specifications, "ci_high_per_10pp"].to_numpy()
        positions = np.arange(len(specifications))[::-1] + offset
        axis.errorbar(
            estimates,
            positions,
            xerr=np.vstack([estimates - lower, upper - estimates]),
            fmt="o" if series == ENSEMBLE else "s",
            markersize=4 if series == ENSEMBLE else 3,
            color=color,
            ecolor=color,
            capsize=1.5,
            linewidth=0.8,
            markeredgewidth=0.8,
            label=label,
        )
    axis.set_yticks(
        np.arange(len(specifications))[::-1],
        [short_labels[spec] for spec in specifications],
        fontsize=8,
    )
    axis.set_xlabel("Change in cohort %max per +10 pp model %max", fontsize=8)
    axis.tick_params(axis="x", labelsize=7)
    axis.grid(axis="x")
    axis.legend(fontsize=6.5, loc="upper right", frameon=False)
    figure.tight_layout(pad=0.6)
    return figure


def plot_forward_predictions(predictions: pd.DataFrame) -> plt.Figure:
    present = set(predictions["grade_bucket"])
    grades = [grade for grade in GRADE_ORDER if grade in present]
    figure, axes = styled_subplots(1, len(grades), figsize=(7.2, 2.4), sharey=True)
    for axis, grade in zip(np.atleast_1d(axes), grades, strict=True):
        frame = predictions.loc[
            (predictions["grade_bucket"] == grade) & (predictions["series"] == ENSEMBLE)
        ]
        observed = frame.loc[frame["specification"] == FORWARD_BASELINE]
        axis.plot(
            observed["year"],
            observed["observed"],
            color=INK,
            marker="o",
            markersize=2.2,
            linewidth=1.0,
            label="Observed cohort",
        )
        axis.plot(
            observed["year"],
            observed["predicted"],
            color=MUTED,
            linestyle="--",
            linewidth=0.9,
            label="Grade-mean baseline",
        )
        surrogate = frame.loc[frame["specification"] == "Model score"]
        axis.plot(
            surrogate["year"],
            surrogate["predicted"],
            color=BRICK,
            marker="s",
            markersize=2.2,
            linewidth=1.0,
            label="Ensemble calibration",
        )
        axis.set_title(grade, fontsize=8)
        axis.tick_params(labelsize=6.5)
        axis.grid(axis="y")
    np.atleast_1d(axes)[0].set_ylabel("Cohort %max", fontsize=8)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=6.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(pad=0.5, w_pad=0.6, rect=(0, 0.1, 1, 1))
    return figure


def plot_variance_decomposition(decomposition: pd.DataFrame) -> plt.Figure:
    figure, axis = styled_subplots(figsize=(7.2, 2.6))
    positions = np.arange(len(decomposition))
    width = 0.27
    bar_style = {"edgecolor": INK, "linewidth": 0.5}
    axis.bar(
        positions - width,
        decomposition["r2_grade"],
        width,
        color=FILLS[3],
        label="Grade FE",
        **bar_style,
    )
    axis.bar(
        positions,
        decomposition["r2_grade_visual"],
        width,
        color=FILLS[0],
        label="+ visual share",
        **bar_style,
    )
    axis.bar(
        positions + width,
        decomposition["r2_grade_location"],
        width,
        color=FILLS[1],
        label="diagram + option shares",
        **bar_style,
    )
    axis.set_xticks(
        positions,
        [textwrap.fill(label, 12) for label in decomposition["outcome_label"]],
        fontsize=7.5,
    )
    axis.set_ylabel("$R^2$ of form-level %max", fontsize=8)
    axis.set_ylim(0, 1)
    axis.tick_params(axis="y", labelsize=7)
    axis.grid(axis="y")
    axis.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper left")
    figure.tight_layout(pad=0.6)
    return figure


def compute_yearly_composition(item_scores: pd.DataFrame) -> pd.DataFrame:
    """Share of items with each kind of auxiliary visual content, by contest year."""
    items = item_scores.drop_duplicates("id")
    if len(items) != 3886:
        raise ValueError("Expected 3,886 evaluated items")
    flags = items.assign(
        auxiliary_visual=items["has_associated_image"] | items["has_option_image"]
    )
    yearly = (
        flags.groupby("year")
        .agg(
            items=("id", "size"),
            auxiliary_visual_share=("auxiliary_visual", "mean"),
            question_diagram_share=("has_associated_image", "mean"),
            option_image_share=("has_option_image", "mean"),
        )
        .reset_index()
    )
    return yearly


def plot_yearly_composition(yearly: pd.DataFrame) -> plt.Figure:
    figure, axis = styled_subplots(figsize=(7.2, 2.6))
    axis.plot(
        yearly["year"],
        100 * yearly["auxiliary_visual_share"],
        color=INK,
        marker="o",
        markersize=2.6,
        linewidth=1.4,
        label="Any auxiliary visual content",
    )
    axis.plot(
        yearly["year"],
        100 * yearly["question_diagram_share"],
        color=TEAL,
        marker="s",
        markersize=2.4,
        linewidth=1.1,
        label="Question diagram",
    )
    axis.plot(
        yearly["year"],
        100 * yearly["option_image_share"],
        color=BRICK,
        marker="^",
        markersize=2.4,
        linewidth=1.1,
        label="Image-based answer options",
    )
    axis.set_xlabel("Contest year")
    axis.set_ylabel("Share of items (%)")
    axis.set_ylim(0, 80)
    axis.grid(axis="y")
    axis.legend(frameon=False, loc="upper left", fontsize=8)
    figure.tight_layout()
    return figure


def plot_composition_effects(panel: pd.DataFrame) -> plt.Figure:
    """Grade-demeaned human and ensemble %max against the form's visual share."""
    frame = panel.copy()
    ensemble = series_column(ENSEMBLE)
    for column in ["human_pct", ensemble, "multimodal_share"]:
        frame[column + "_dm"] = frame[column] - frame.groupby("grade_bucket")[
            column
        ].transform("mean")
    figure, axes = styled_subplots(1, 2, figsize=(7.2, 2.7), sharex=True)
    colors = GRADE_COLORS
    for axis, column, label in [
        (axes[0], "human_pct_dm", "Official cohort %max"),
        (axes[1], ensemble + "_dm", "Equal-weight ensemble %max"),
    ]:
        x = 100 * frame["multimodal_share_dm"].to_numpy()
        y = frame[column].to_numpy()
        for grade, color in zip(GRADE_ORDER, colors):
            mask = (frame["grade_bucket"] == grade).to_numpy()
            axis.scatter(
                x[mask],
                y[mask],
                s=14,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.3,
                label=grade,
            )
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 50)
        axis.plot(grid, intercept + slope * grid, color=INK, linewidth=1.2)
        axis.annotate(
            f"{10 * slope:+.2f} per +10 pp",
            xy=(0.03, 0.93),
            xycoords="axes fraction",
            fontsize=9,
            va="top",
        )
        axis.axhline(0, color=MUTED, linewidth=0.5)
        axis.axvline(0, color=MUTED, linewidth=0.5)
        axis.set_title(label, fontsize=9)
        axis.set_xlabel(
            "Share of items with auxiliary visual content\n(pp, within-grade centered)"
        )
        axis.grid(axis="y")
    axes[0].set_ylabel("%max (within-grade centered)")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left", ncol=2, title=None)
    figure.tight_layout()
    return figure


ADJUSTED_FORMS = [(2001, "9-10"), (2003, "3-4")]


def _drop_adjusted_forms(panel: pd.DataFrame) -> pd.DataFrame:
    keep = ~panel.set_index(["year", "exam"]).index.isin(ADJUSTED_FORMS)
    if (~keep).sum() != len(ADJUSTED_FORMS):
        raise ValueError("Expected both fixed-credit forms in the shared panel")
    return panel.loc[keep].reset_index(drop=True)


def compute_sensitivity_checks(
    panel: pd.DataFrame, repetitions: int = 2000
) -> pd.DataFrame:
    """Robustness of the grade-fixed-effects association."""
    records = []
    reduced = _drop_adjusted_forms(panel)
    for series_index, (series, label) in enumerate(SERIES_LABELS.items()):
        column = series_column(series)
        checks = [
            ("Grade FE, excluding fixed-credit forms", reduced, ["grade_bucket"]),
            ("Grade + year FE", panel, ["grade_bucket", "year"]),
        ]
        for check_index, (check, frame, fixed_effects) in enumerate(checks):
            fit = _fit_ols(frame, "human_pct", [column], fixed_effects)
            interval = _cluster_bootstrap_interval(
                frame,
                lambda sample, fixed_effects=fixed_effects: float(
                    _fit_ols(sample, "human_pct", [column], fixed_effects)[
                        "coefficients"
                    ][column]
                ),
                seed=5400 + 10 * series_index + check_index,
                repetitions=repetitions,
            )
            records.append(
                {
                    "series": series,
                    "series_label": label,
                    "check": check,
                    "comparisons": int(len(frame)),
                    "effect_per_10pp_model": 10 * float(fit["coefficients"][column]),
                    "ci_low_per_10pp": 10 * interval[0],
                    "ci_high_per_10pp": 10 * interval[1],
                }
            )
        leave_one_out = [
            10
            * float(
                _fit_ols(
                    panel.loc[panel["year"] != year],
                    "human_pct",
                    [column],
                    ["grade_bucket"],
                )["coefficients"][column]
            )
            for year in sorted(panel["year"].unique())
        ]
        records.append(
            {
                "series": series,
                "series_label": label,
                "check": "Grade FE, leave one year out",
                "comparisons": int(len(panel)) - 5,
                "effect_per_10pp_model": float(np.mean(leave_one_out)),
                "ci_low_per_10pp": float(np.min(leave_one_out)),
                "ci_high_per_10pp": float(np.max(leave_one_out)),
            }
        )
    return pd.DataFrame(records)


def build_human_reference_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Official cohort %max by year and grade group, as used in every analysis."""
    table = panel.pivot(index="year", columns="grade_bucket", values="human_pct")
    return table[GRADE_ORDER].reset_index()


def plot_composition_overview(yearly: pd.DataFrame, panel: pd.DataFrame) -> plt.Figure:
    """Composition drift and its associations with cohort and ensemble %max."""
    figure, axes = styled_subplots(
        1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.35, 1, 1]}
    )
    drift = axes[0]
    drift.plot(
        yearly["year"],
        100 * yearly["auxiliary_visual_share"],
        color=INK,
        marker="o",
        markersize=2.4,
        linewidth=1.3,
        label="Any auxiliary visual content",
    )
    drift.plot(
        yearly["year"],
        100 * yearly["question_diagram_share"],
        color=TEAL,
        marker="s",
        markersize=2.1,
        linewidth=1.0,
        label="Question diagram",
    )
    drift.plot(
        yearly["year"],
        100 * yearly["option_image_share"],
        color=BRICK,
        marker="^",
        markersize=2.1,
        linewidth=1.0,
        label="Image-based options",
    )
    drift.set_ylim(0, 80)
    drift.set_xlabel("Contest year")
    drift.set_ylabel("Share of items (%)")
    drift.set_title("(a) Composition of the archive", fontsize=9)
    drift.legend(frameon=False, fontsize=6.5, loc="upper left")
    drift.grid(axis="y")

    frame = panel.copy()
    ensemble = series_column(ENSEMBLE)
    for column in ["human_pct", ensemble, "multimodal_share"]:
        frame[column + "_dm"] = frame[column] - frame.groupby("grade_bucket")[
            column
        ].transform("mean")
    x = 100 * frame["multimodal_share_dm"].to_numpy()
    grid = np.linspace(x.min(), x.max(), 50)
    for axis, column, title in [
        (axes[1], "human_pct_dm", "(b) Official cohort %max"),
        (axes[2], ensemble + "_dm", "(c) Ensemble %max"),
    ]:
        y = frame[column].to_numpy()
        axis.scatter(
            x, y, s=11, color=POINT, alpha=0.9, edgecolor="white", linewidth=0.3
        )
        slope, intercept = np.polyfit(x, y, 1)
        axis.plot(grid, intercept + slope * grid, color=BRICK, linewidth=1.3)
        axis.annotate(
            f"{10 * slope:+.2f} per +10 pp",
            xy=(0.04, 0.94),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
        )
        axis.axhline(0, color=MUTED, linewidth=0.5)
        axis.axvline(0, color=MUTED, linewidth=0.5)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Visual share (pp)")
        axis.grid(axis="y")
    axes[1].set_ylabel("%max (centered)")
    figure.tight_layout(w_pad=1.0)
    return figure


ABLATION_MODELS = {
    "gpt5": "openai-gpt-5",
    "qwen": "qwen-qwen3-vl-235b-a22b-thinking",
    "claude": "anthropic-claude-sonnet-4.5",
}


def _paired_difference(
    first: np.ndarray, second: np.ndarray, seed: int, repetitions: int = 2000
) -> tuple[float, float, float]:
    """Mean of second minus first over items, with a percentile bootstrap over items."""
    difference = second.astype(float) - first.astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(difference, size=(repetitions, len(difference)), replace=True)
    means = draws.mean(axis=1)
    return (
        float(difference.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _scored_arm(run_dir: Path, dataset: pd.DataFrame) -> pd.DataFrame:
    """Correctness and abstention of an archived run, indexed by source item."""
    scored = score_predictions(read_compact_run(run_dir), dataset)
    return pd.DataFrame(
        {
            "correct": scored["answered_correctly"].to_numpy(),
            "declined": scored["declined"].to_numpy(),
        },
        index=pd.Index(scored["source_id"], name="id"),
    )


def _pair_record(
    series: str,
    comparison: str,
    reference: pd.DataFrame,
    treated: pd.DataFrame,
    seed: int,
    repetitions: int,
) -> dict[str, object]:
    pairs = reference.join(treated, how="inner", lsuffix="_ref", rsuffix="_treat")
    estimate, low, high = _paired_difference(
        pairs["correct_ref"].to_numpy(),
        pairs["correct_treat"].to_numpy(),
        seed=seed,
        repetitions=repetitions,
    )
    return {
        "series": series,
        "series_label": MODEL_LABELS[series],
        "comparison": comparison,
        "items": int(len(pairs)),
        "accuracy_reference": float(pairs["correct_ref"].mean()),
        "accuracy_treated": float(pairs["correct_treat"].mean()),
        "difference_pp": 100 * estimate,
        "ci_low_pp": 100 * low,
        "ci_high_pp": 100 * high,
        "lost": int((pairs["correct_ref"] & ~pairs["correct_treat"]).sum()),
        "gained": int((~pairs["correct_ref"] & pairs["correct_treat"]).sum()),
        "decline_rate_reference": float(pairs["declined_ref"].mean()),
        "decline_rate_treated": float(pairs["declined_treat"].mean()),
    }


def compute_ablation_pairs(
    project_root: Path, dataset: pd.DataFrame, repetitions: int = 2000
) -> pd.DataFrame:
    """Paired accuracy differences: English versus German on the 200 translation pairs, and
    the blind rerun of the question-diagram items versus the October 2025 runs."""
    runs_dir = project_root / "artifacts" / "runs"
    comparison_runs = {
        (model, arm): run_id for run_id, (model, arm) in COMPARISON_RUNS.items()
    }
    full_runs = {model: run_id for run_id, model in RUNS.items()}
    records = []
    for index, series in enumerate(ABLATION_MODELS.values()):
        arms = {
            arm: _scored_arm(runs_dir / comparison_runs[(series, arm)], dataset)
            for arm in ["german_control", "english", "blind"]
        }
        records.append(
            _pair_record(
                series,
                "English minus German",
                arms["german_control"],
                arms["english"],
                5500 + index,
                repetitions,
            )
        )
        october = _scored_arm(runs_dir / full_runs[series], dataset)
        records.append(
            _pair_record(
                series,
                "Images removed minus retained",
                october.loc[arms["blind"].index],
                arms["blind"],
                5700 + index,
                repetitions,
            )
        )
    return pd.DataFrame(records)


def run_surrogate_analysis(
    project_root: Path, dataset_path: Path, write_outputs: bool = True
) -> dict[str, object]:
    results = run_diagnostics(project_root, dataset_path, write_outputs=False)
    yearly_composition = compute_yearly_composition(results["items"])
    panel = build_panel(results["exam_data"], results["exam_scores"])
    sensitivity = compute_sensitivity_checks(panel)
    human_reference = build_human_reference_table(panel)
    ablation_pairs = compute_ablation_pairs(project_root, results["dataset"])
    regressions = compute_outcome_regressions(panel)
    predictions = compute_forward_predictions(panel)
    per_grade, pooled = summarize_forward_predictions(predictions)
    forward_sensitivity = compute_forward_sensitivity(panel, predictions)
    human_mean_sensitivity = compute_human_mean_sensitivity(
        panel,
        load_human_scores(
            project_root / "artifacts" / "human_results", histogram_means=True
        ),
    )
    decomposition = compute_variance_decomposition(panel)

    def series_rows(frame: pd.DataFrame, series: str) -> pd.DataFrame:
        return frame.loc[frame["series"] == series].set_index("specification")

    metrics = {
        "panel": {
            "shared_exams": int(len(panel)),
            "grade_buckets": int(panel["grade_bucket"].nunique()),
            "forward_predictions_per_grade": int(
                predictions.groupby(["grade_bucket", "series", "specification"])
                .size()
                .iloc[0]
            ),
            "minimum_training_years": MIN_TRAINING_YEARS,
        },
        "outcome_regressions": {
            SERIES_LABELS[series]: {
                spec: {
                    "effect_per_10pp_model": float(row["effect_per_10pp_model"]),
                    "ci_95": [
                        float(row["ci_low_per_10pp"]),
                        float(row["ci_high_per_10pp"]),
                    ],
                    "r2_increment": float(row["r2_increment"]),
                }
                for spec, row in series_rows(regressions, series).iterrows()
            }
            for series in SERIES_LABELS
        },
        "forward_prediction": {
            SERIES_LABELS[series]: {
                spec: {
                    "mae": float(row["mae"]),
                    "baseline_mae": float(row["baseline_mae"]),
                    "mae_change_ci_95": [
                        float(row["mae_change_ci_low"]),
                        float(row["mae_change_ci_high"]),
                    ],
                    "forward_r2": float(row["forward_r2"]),
                    "years_beating_baseline": int(row["years_beating_baseline"]),
                    "predictions": int(row["predictions"]),
                    "grades_beating_baseline_mae": int(
                        row["grades_beating_baseline_mae"]
                    ),
                }
                for spec, row in series_rows(pooled, series).iterrows()
            }
            for series in SERIES_LABELS
        },
        "variance_decomposition": {
            row["outcome_label"]: {
                "r2_grade": float(row["r2_grade"]),
                "r2_grade_visual": float(row["r2_grade_visual"]),
                "r2_grade_location": float(row["r2_grade_location"]),
                "visual_r2_increment_ci_95": [
                    float(row["visual_r2_increment_ci_low"]),
                    float(row["visual_r2_increment_ci_high"]),
                ],
                "sd_within_grade": float(row["sd_within_grade"]),
            }
            for _, row in decomposition.iterrows()
        },
    }

    if write_outputs:
        output_dir = project_root / "reproduced" / "surrogate"
        table_dir = output_dir / "tables"
        figure_dir = output_dir / "figures"
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        panel.to_csv(table_dir / "shared_exam_panel.csv", index=False)
        regressions.to_csv(table_dir / "human_outcome_regressions.csv", index=False)
        predictions.to_csv(table_dir / "forward_predictions.csv", index=False)
        per_grade.to_csv(table_dir / "forward_prediction_by_grade.csv", index=False)
        pooled.to_csv(table_dir / "forward_prediction_pooled.csv", index=False)
        forward_sensitivity.to_csv(
            table_dir / "forward_prediction_sensitivity.csv", index=False
        )
        human_mean_sensitivity.to_csv(
            table_dir / "human_mean_sensitivity.csv", index=False
        )
        decomposition.to_csv(table_dir / "variance_decomposition.csv", index=False)
        yearly_composition.to_csv(table_dir / "yearly_composition.csv", index=False)
        sensitivity.to_csv(table_dir / "sensitivity_checks.csv", index=False)
        ablation_pairs.to_csv(table_dir / "ablation_pairs.csv", index=False)
        human_reference.to_csv(table_dir / "human_reference_means.csv", index=False)
        for name, figure in [
            ("human_outcome_coefficients", plot_outcome_coefficients(regressions)),
            ("forward_prediction", plot_forward_predictions(predictions)),
            ("variance_decomposition", plot_variance_decomposition(decomposition)),
            ("dataset_composition", plot_yearly_composition(yearly_composition)),
            ("composition_effects", plot_composition_effects(panel)),
            (
                "composition_overview",
                plot_composition_overview(yearly_composition, panel),
            ),
        ]:
            figure.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
            figure.savefig(figure_dir / f"{name}.png", dpi=220, bbox_inches="tight")
            plt.close(figure)

    return {
        "metrics": metrics,
        "panel": panel,
        "regressions": regressions,
        "predictions": predictions,
        "per_grade": per_grade,
        "pooled": pooled,
        "forward_sensitivity": forward_sensitivity,
        "human_mean_sensitivity": human_mean_sensitivity,
        "decomposition": decomposition,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether model %max is a valid surrogate for human %max."
    )
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
    results = run_surrogate_analysis(project_root, dataset_path)
    ensemble = SERIES_LABELS[ENSEMBLE]
    regressions = results["metrics"]["outcome_regressions"][ensemble]
    forward = results["metrics"]["forward_prediction"][ensemble]["Model score"]
    print("Surrogate-validity analysis completed successfully.")
    for spec, row in regressions.items():
        print(
            f"{ensemble} / {spec}: {row['effect_per_10pp_model']:.3f} "
            f"[{row['ci_95'][0]:.3f}, {row['ci_95'][1]:.3f}] "
            f"(incremental R2 {row['r2_increment']:.3f})"
        )
    print(
        f"Forward prediction ({ensemble}): MAE {forward['mae']:.3f} vs "
        f"grade-mean baseline {forward['baseline_mae']:.3f}; forward R2 "
        f"{forward['forward_r2']:.3f}; beats baseline in "
        f"{forward['years_beating_baseline']}/{forward['predictions']} years"
    )
    print(f"Outputs: {project_root / 'reproduced' / 'surrogate'}")


if __name__ == "__main__":
    main()
