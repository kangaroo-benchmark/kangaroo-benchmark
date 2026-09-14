"""Item-level diagnostics and cohort comparisons built from the archived model runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.figure_style import (
    BRICK,
    FILLS,
    INK,
    SEQUENTIAL,
    styled_subplots,
)
from analysis.inputs import (
    aggregate_exam_scores,
    build_comparison,
    compute_statistics,
    load_dataset,
    load_human_scores,
    load_item_scores,
)

MODEL_LABELS = {
    "openai-gpt-5": "GPT-5",
    "qwen-qwen3-vl-235b-a22b-thinking": "Qwen3-VL",
    "x-ai-grok-4-fast": "Grok 4 Fast",
    "anthropic-claude-sonnet-4.5": "Claude Sonnet 4.5",
}
SUBTYPE_ORDER = [
    "Text-only",
    "Question diagram only",
    "Image answers only",
    "Question diagram and image answers",
]


def _fit_ols(
    frame: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    fixed_effects: list[str],
) -> dict[str, object]:
    design_parts = [np.ones((len(frame), 1), dtype=float)]
    names = ["intercept"]
    for predictor in predictors:
        design_parts.append(frame[[predictor]].to_numpy(dtype=float))
        names.append(predictor)
    for fixed_effect in fixed_effects:
        dummies = pd.get_dummies(
            frame[fixed_effect].astype(str),
            prefix=fixed_effect,
            drop_first=True,
            dtype=float,
        )
        design_parts.append(dummies.to_numpy(dtype=float))
        names.extend(dummies.columns.tolist())

    design = np.column_stack(design_parts)
    values = frame[outcome].to_numpy(dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    total_sum = float(np.sum((values - values.mean()) ** 2))
    residual_sum = float(np.sum((values - fitted) ** 2))
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0
        else float("inf")
    )
    return {
        "coefficients": dict(zip(names, coefficients, strict=True)),
        "r2": 1.0 - residual_sum / total_sum,
        "rank": int(rank),
        "parameters": int(design.shape[1]),
        "condition_number": condition_number,
        "residuals": values - fitted,
    }


def _cluster_bootstrap_interval(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    seed: int,
    repetitions: int = 2000,
) -> tuple[float, float, int]:
    years = np.asarray(sorted(frame["year"].unique()), dtype=int)
    rows_by_year = {
        year: np.flatnonzero(frame["year"].to_numpy(dtype=int) == year)
        for year in years
    }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled_years = rng.choice(years, size=len(years), replace=True)
        sampled_rows = np.concatenate([rows_by_year[year] for year in sampled_years])
        sample = frame.iloc[sampled_rows]
        value = statistic(sample)
        if np.isfinite(value):
            values.append(float(value))
    if len(values) < int(0.95 * repetitions):
        raise ValueError("Too many invalid year-cluster bootstrap samples")
    lower, upper = np.percentile(np.asarray(values), [2.5, 97.5])
    return float(lower), float(upper), len(values)


def _correlation(frame: pd.DataFrame, first: str, second: str, rank: bool) -> float:
    first_values = frame[first].to_numpy(dtype=float)
    second_values = frame[second].to_numpy(dtype=float)
    if np.std(first_values) == 0 or np.std(second_values) == 0:
        return float("nan")
    result = (
        spearmanr(first_values, second_values).statistic
        if rank
        else pearsonr(first_values, second_values).statistic
    )
    return float(result)


def build_exam_diagnostics(
    comparison: pd.DataFrame, item_scores: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_items = item_scores.drop_duplicates("id").copy()
    if len(base_items) != 3886:
        raise ValueError("Expected 3,886 evaluated items")

    composition_records = []
    for (year, exam), group in base_items.groupby(["year", "group"], sort=True):
        composition_records.append(
            {
                "year": int(year),
                "exam": exam,
                "item_count": int(len(group)),
                "multimodal_share": float(group["multimodal"].mean()),
                "associated_image_share": float(group["has_associated_image"].mean()),
                "option_image_share": float(group["has_option_image"].mean()),
                "three_point_share": float((group["question_points"] == 3).mean()),
                "five_point_share": float((group["question_points"] == 5).mean()),
            }
        )
    composition = pd.DataFrame(composition_records)
    exam_data = comparison.merge(
        composition, on=["year", "exam"], how="left", validate="one_to_one"
    )
    if (
        exam_data[
            [
                "multimodal_share",
                "associated_image_share",
                "option_image_share",
                "three_point_share",
                "five_point_share",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Exam composition did not cover all shared human comparisons")
    exam_data["year_per_decade"] = (exam_data["year"] - exam_data["year"].mean()) / 10

    slice_records = []
    slices = {
        "All items": pd.Series(True, index=item_scores.index),
        "Text-only": ~item_scores["multimodal"],
        "Auxiliary visual content": item_scores["multimodal"],
    }
    for slice_name, mask in slices.items():
        sliced = item_scores.loc[mask]
        by_model = (
            sliced.groupby(["model", "year", "group"], sort=True)
            .agg(
                accuracy=("answered_correctly", "mean"),
                decline_rate=("declined", "mean"),
                items=("id", "size"),
            )
            .reset_index()
        )
        ensemble = (
            by_model.groupby(["year", "group"], sort=True)
            .agg(
                ensemble_accuracy=("accuracy", "mean"),
                ensemble_decline_rate=("decline_rate", "mean"),
                items_per_model=("items", "first"),
                model_count=("model", "nunique"),
            )
            .reset_index()
        )
        ensemble["slice"] = slice_name
        slice_records.append(ensemble)
    slice_scores = pd.concat(slice_records, ignore_index=True).rename(
        columns={"group": "exam"}
    )
    return exam_data, slice_scores


def compute_alignment_diagnostics(
    exam_data: pd.DataFrame, slice_scores: pd.DataFrame
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_pearson = _correlation(
        exam_data, "human_pct", "pooled_ensemble_pct", rank=False
    )
    pooled_spearman = _correlation(
        exam_data, "human_pct", "pooled_ensemble_pct", rank=True
    )
    pooled_pearson_ci = _cluster_bootstrap_interval(
        exam_data,
        lambda sample: _correlation(
            sample, "human_pct", "pooled_ensemble_pct", rank=False
        ),
        seed=4201,
    )
    pooled_spearman_ci = _cluster_bootstrap_interval(
        exam_data,
        lambda sample: _correlation(
            sample, "human_pct", "pooled_ensemble_pct", rank=True
        ),
        seed=4202,
    )

    specifications = [
        ("Raw", [], []),
        ("Grade fixed effects", [], ["grade_bucket"]),
        (
            "Grade FE + multimodal share",
            ["multimodal_share"],
            ["grade_bucket"],
        ),
        (
            "Grade FE + image-location shares",
            ["associated_image_share", "option_image_share"],
            ["grade_bucket"],
        ),
    ]
    controlled_records = []
    for index, (label, covariates, fixed_effects) in enumerate(specifications):
        predictors = ["human_pct", *covariates]
        fitted = _fit_ols(
            exam_data,
            outcome="pooled_ensemble_pct",
            predictors=predictors,
            fixed_effects=fixed_effects,
        )
        coefficient = float(fitted["coefficients"]["human_pct"])
        human_adjustment = _fit_ols(
            exam_data,
            outcome="human_pct",
            predictors=covariates,
            fixed_effects=fixed_effects,
        )
        model_adjustment = _fit_ols(
            exam_data,
            outcome="pooled_ensemble_pct",
            predictors=covariates,
            fixed_effects=fixed_effects,
        )
        partial_correlation = float(
            pearsonr(
                human_adjustment["residuals"], model_adjustment["residuals"]
            ).statistic
        )
        interval = _cluster_bootstrap_interval(
            exam_data,
            lambda sample, predictors=predictors, fixed_effects=fixed_effects: float(
                _fit_ols(
                    sample,
                    outcome="pooled_ensemble_pct",
                    predictors=predictors,
                    fixed_effects=fixed_effects,
                )["coefficients"]["human_pct"]
            ),
            seed=4300 + index,
        )
        controlled_records.append(
            {
                "specification": label,
                "human_coefficient_per_1pp": coefficient,
                "effect_per_10pp_human": 10 * coefficient,
                "ci_low_per_10pp": 10 * interval[0],
                "ci_high_per_10pp": 10 * interval[1],
                "bootstrap_samples": interval[2],
                "r2": float(fitted["r2"]),
                "rank": int(fitted["rank"]),
                "parameters": int(fitted["parameters"]),
                "condition_number": float(fitted["condition_number"]),
                "partial_correlation": partial_correlation,
                "covariates": ", ".join(covariates) or "none",
                "fixed_effects": ", ".join(fixed_effects) or "none",
            }
        )
    controlled = pd.DataFrame(controlled_records)

    grade_records = []
    for index, (grade, group) in enumerate(
        exam_data.groupby("grade_bucket", sort=False)
    ):
        correlation = _correlation(
            group, "human_pct", "pooled_ensemble_pct", rank=False
        )
        human_detrended = _fit_ols(
            group, outcome="human_pct", predictors=["year"], fixed_effects=[]
        )["residuals"]
        model_detrended = _fit_ols(
            group,
            outcome="pooled_ensemble_pct",
            predictors=["year"],
            fixed_effects=[],
        )["residuals"]
        detrended_correlation = float(
            pearsonr(human_detrended, model_detrended).statistic
        )
        interval = _cluster_bootstrap_interval(
            group,
            lambda sample: _correlation(
                sample, "human_pct", "pooled_ensemble_pct", rank=False
            ),
            seed=4400 + index,
        )
        grade_records.append(
            {
                "grade_bucket": grade,
                "comparisons": int(len(group)),
                "pearson_r": correlation,
                "year_detrended_pearson_r": detrended_correlation,
                "ci_low": interval[0],
                "ci_high": interval[1],
                "bootstrap_samples": interval[2],
            }
        )
    per_grade = pd.DataFrame(grade_records)

    slice_data = slice_scores.merge(
        exam_data[["year", "exam", "grade_bucket", "human_pct"]],
        on=["year", "exam"],
        how="inner",
        validate="many_to_one",
    )
    slice_records = []
    for index, (slice_name, group) in enumerate(
        slice_data.groupby("slice", sort=False)
    ):
        correlation = _correlation(group, "human_pct", "ensemble_accuracy", rank=False)
        correlation_ci = _cluster_bootstrap_interval(
            group,
            lambda sample: _correlation(
                sample, "human_pct", "ensemble_accuracy", rank=False
            ),
            seed=4500 + index,
        )
        fitted = _fit_ols(
            group,
            outcome="ensemble_accuracy",
            predictors=["human_pct"],
            fixed_effects=["grade_bucket"],
        )
        coefficient_ci = _cluster_bootstrap_interval(
            group,
            lambda sample: float(
                _fit_ols(
                    sample,
                    outcome="ensemble_accuracy",
                    predictors=["human_pct"],
                    fixed_effects=["grade_bucket"],
                )["coefficients"]["human_pct"]
            ),
            seed=4600 + index,
        )
        slice_records.append(
            {
                "slice": slice_name,
                "comparisons": int(len(group)),
                "pearson_r": correlation,
                "pearson_ci_low": correlation_ci[0],
                "pearson_ci_high": correlation_ci[1],
                "grade_fe_accuracy_pp_per_10pp_human": 1000
                * float(fitted["coefficients"]["human_pct"]),
                "grade_fe_ci_low_accuracy_pp": 1000 * coefficient_ci[0],
                "grade_fe_ci_high_accuracy_pp": 1000 * coefficient_ci[1],
            }
        )
    slice_summary = pd.DataFrame(slice_records)

    metrics = {
        "pooled": {
            "comparisons": int(len(exam_data)),
            "pearson_r": pooled_pearson,
            "pearson_year_cluster_ci_95": list(pooled_pearson_ci[:2]),
            "spearman_rho": pooled_spearman,
            "spearman_year_cluster_ci_95": list(pooled_spearman_ci[:2]),
        },
        "composition_control": {
            record["specification"]: {
                "effect_per_10pp_human": record["effect_per_10pp_human"],
                "ci_95": [record["ci_low_per_10pp"], record["ci_high_per_10pp"]],
                "r2": record["r2"],
                "partial_correlation": record["partial_correlation"],
            }
            for record in controlled_records
        },
        "text_only_diagnostic": {
            record["slice"]: {
                "pearson_r": record["pearson_r"],
                "pearson_ci_95": [
                    record["pearson_ci_low"],
                    record["pearson_ci_high"],
                ],
                "grade_fe_accuracy_pp_per_10pp_human": record[
                    "grade_fe_accuracy_pp_per_10pp_human"
                ],
            }
            for record in slice_records
        },
    }
    return metrics, controlled, per_grade, slice_summary


def compute_model_correlations(
    exam_scores: pd.DataFrame, human_scores: pd.DataFrame
) -> pd.DataFrame:
    """Pooled correlations between each model series and the cohort means, with
    year-cluster bootstrap intervals."""
    records = []
    index_columns = ["year", "exam", "grade_bucket"]
    model_scores = exam_scores.pivot_table(
        index=index_columns,
        columns="model",
        values="llm_pct",
    )
    pooled = exam_scores.groupby(index_columns).agg(
        mean_total_score=("total_score", "mean"),
        possible_points=("possible_points", "first"),
    )
    model_scores["equal-weight ensemble"] = (
        100.0 * pooled["mean_total_score"] / pooled["possible_points"]
    )
    comparison = model_scores.reset_index().merge(
        human_scores[index_columns + ["human_pct"]],
        on=index_columns,
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != 115:
        raise ValueError(f"Expected 115 shared comparisons, found {len(comparison)}")

    series = [*MODEL_LABELS, "equal-weight ensemble"]
    for series_index, column in enumerate(series):
        label = (
            "Equal-weight ensemble"
            if column == "equal-weight ensemble"
            else MODEL_LABELS[column]
        )
        pearson = _correlation(comparison, "human_pct", column, rank=False)
        spearman = _correlation(comparison, "human_pct", column, rank=True)
        pearson_interval = _cluster_bootstrap_interval(
            comparison,
            lambda sample, column=column: _correlation(
                sample, "human_pct", column, rank=False
            ),
            seed=4900 + series_index * 2,
        )
        spearman_interval = _cluster_bootstrap_interval(
            comparison,
            lambda sample, column=column: _correlation(
                sample, "human_pct", column, rank=True
            ),
            seed=4901 + series_index * 2,
        )
        records.append(
            {
                "series": label,
                "comparisons": int(len(comparison)),
                "pearson_r": pearson,
                "pearson_ci_low": pearson_interval[0],
                "pearson_ci_high": pearson_interval[1],
                "spearman_rho": spearman,
                "spearman_ci_low": spearman_interval[0],
                "spearman_ci_high": spearman_interval[1],
                "bootstrap_samples": min(pearson_interval[2], spearman_interval[2]),
            }
        )
    return pd.DataFrame(records)


def compute_image_subtypes(
    dataset: pd.DataFrame, item_scores: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    image_columns = [f"sol_{letter}_image_bin" for letter in "ABCDE"]
    image_metadata = dataset[
        ["id", "multimodal", "question_image", "associated_images_bin", *image_columns]
    ].copy()
    image_metadata["id"] = image_metadata["id"].astype(str)
    evaluated_ids = set(item_scores["id"].astype(str))
    image_metadata = image_metadata.loc[image_metadata["id"].isin(evaluated_ids)].copy()
    image_metadata["has_stem_image"] = (
        image_metadata["associated_images_bin"].map(len) > 0
    )
    image_metadata["has_option_images"] = (
        image_metadata[image_columns].notna().any(axis=1)
    )
    image_metadata["image_subtype"] = np.select(
        [
            image_metadata["has_stem_image"] & image_metadata["has_option_images"],
            image_metadata["has_stem_image"],
            image_metadata["has_option_images"],
        ],
        [
            "Question diagram and image answers",
            "Question diagram only",
            "Image answers only",
        ],
        default="Text-only",
    )
    derived_multimodal = image_metadata["image_subtype"] != "Text-only"
    if not derived_multimodal.equals(image_metadata["multimodal"]):
        raise ValueError("Released multimodal flag disagrees with image fields")

    scores = item_scores.copy()
    scores["id"] = scores["id"].astype(str)
    scores = scores.merge(
        image_metadata[["id", "image_subtype"]],
        on="id",
        how="left",
        validate="many_to_one",
    )
    if scores["image_subtype"].isna().any():
        raise ValueError("Released dataset does not cover every archived prediction")

    records = []
    for (model, subtype), group in scores.groupby(
        ["model", "image_subtype"], sort=False
    ):
        correct = int(group["answered_correctly"].sum())
        count = int(len(group))
        proportion = correct / count
        z = 1.959963984540054
        denominator = 1 + z**2 / count
        center = (proportion + z**2 / (2 * count)) / denominator
        half_width = (
            z
            * np.sqrt(proportion * (1 - proportion) / count + z**2 / (4 * count**2))
            / denominator
        )
        attempted = group.loc[group["attempted"]]
        records.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "image_subtype": subtype,
                "items": count,
                "accuracy": proportion,
                "accuracy_ci_low": center - half_width,
                "accuracy_ci_high": center + half_width,
                "decline_rate": float(group["declined"].mean()),
                "attempted_accuracy": float(attempted["answered_correctly"].mean()),
            }
        )
    subtype_results = pd.DataFrame(records)
    subtype_results["_subtype_order"] = subtype_results["image_subtype"].map(
        {label: index for index, label in enumerate(SUBTYPE_ORDER)}
    )
    subtype_results = subtype_results.sort_values(
        ["_subtype_order", "model_label"]
    ).drop(columns="_subtype_order")

    subtype_counts = (
        image_metadata["image_subtype"].value_counts().reindex(SUBTYPE_ORDER).to_dict()
    )
    metadata = {
        "released_rows": int(len(dataset)),
        "evaluated_items": int(len(image_metadata)),
        "question_crop_present": int(image_metadata["question_image"].notna().sum()),
        "subtype_counts": {key: int(value) for key, value in subtype_counts.items()},
    }
    return subtype_results.reset_index(drop=True), scores, metadata


def compute_behavioral_diagnostics(
    scored_subtypes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = scored_subtypes.copy()
    scores["visual_condition"] = np.where(
        scores["image_subtype"] == "Text-only",
        "Text-only",
        "Auxiliary visual content",
    )
    scores["valid_selection"] = scores["predicted_normalized"].isin(list("ABCDE"))
    scores["no_valid_selection"] = ~scores["valid_selection"] & ~scores["declined"]

    response_status = (
        scores.groupby(["model", "visual_condition"], sort=False)
        .agg(
            items=("id", "size"),
            correct=("answered_correctly", "sum"),
            accuracy=("answered_correctly", "mean"),
            valid_selections=("valid_selection", "sum"),
            abstentions=("declined", "sum"),
            abstention_rate=("declined", "mean"),
            no_valid_selections=("no_valid_selection", "sum"),
            no_valid_selection_rate=("no_valid_selection", "mean"),
        )
        .reset_index()
    )
    response_status["model_label"] = response_status["model"].map(MODEL_LABELS)

    point_tier = (
        scores.groupby(["model", "question_points", "visual_condition"], sort=False)
        .agg(
            items=("id", "size"),
            accuracy=("answered_correctly", "mean"),
            abstention_rate=("declined", "mean"),
        )
        .reset_index()
    )
    point_tier["model_label"] = point_tier["model"].map(MODEL_LABELS)

    by_grade = (
        scores.groupby(["model", "group", "visual_condition"], sort=False)
        .agg(
            items=("id", "size"),
            accuracy=("answered_correctly", "mean"),
            abstention_rate=("declined", "mean"),
        )
        .reset_index()
        .rename(columns={"group": "grade_bucket"})
    )
    by_grade["model_label"] = by_grade["model"].map(MODEL_LABELS)

    no_visual_by_year = (
        scores.loc[scores["visual_condition"] == "Text-only"]
        .groupby(["model", "year"], sort=False)
        .agg(
            items=("id", "size"),
            accuracy=("answered_correctly", "mean"),
            abstention_rate=("declined", "mean"),
        )
        .reset_index()
    )
    no_visual_by_year["model_label"] = no_visual_by_year["model"].map(MODEL_LABELS)
    return response_status, point_tier, by_grade, no_visual_by_year


def compute_option_position_diagnostics(item_scores: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model, group in item_scores.groupby("model", sort=False):
        attempted = group.loc[group["predicted_normalized"].isin(list("ABCDE"))]
        gold_counts = group["answer"].astype(str).str.upper().value_counts()
        predicted_counts = attempted["predicted_normalized"].value_counts()
        for letter in "ABCDE":
            gold_subset = group.loc[group["answer"].astype(str).str.upper() == letter]
            records.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "option": letter,
                    "gold_count": int(gold_counts.get(letter, 0)),
                    "gold_share": float(gold_counts.get(letter, 0) / len(group)),
                    "predicted_count": int(predicted_counts.get(letter, 0)),
                    "predicted_share_among_attempts": float(
                        predicted_counts.get(letter, 0) / len(attempted)
                    ),
                    "accuracy_when_gold": float(
                        gold_subset["answered_correctly"].mean()
                    ),
                    "decline_rate_when_gold": float(gold_subset["declined"].mean()),
                }
            )
    return pd.DataFrame(records)


def plot_alignment_summary(
    controlled: pd.DataFrame, exam_data: pd.DataFrame
) -> plt.Figure:
    figure, axes = styled_subplots(
        1,
        2,
        figsize=(7.2, 3.1),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )

    scatter_axis = axes[0]
    scatter = scatter_axis.scatter(
        exam_data["human_pct"],
        exam_data["pooled_ensemble_pct"],
        c=exam_data["multimodal_share"],
        cmap=SEQUENTIAL,
        s=20,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.3,
    )
    coefficients = np.polyfit(
        exam_data["human_pct"], exam_data["pooled_ensemble_pct"], 1
    )
    x_values = np.linspace(exam_data["human_pct"].min(), exam_data["human_pct"].max())
    scatter_axis.plot(
        x_values,
        coefficients[0] * x_values + coefficients[1],
        color=INK,
        linewidth=1.0,
    )
    scatter_axis.set_xlabel("Human exam score (%max)", fontsize=8)
    scatter_axis.set_ylabel("Model ensemble score (%max)", fontsize=8)
    scatter_axis.set_title("(a) Raw association", fontsize=9)
    scatter_axis.tick_params(labelsize=7)
    scatter_axis.grid(axis="y")
    scatter_axis.text(
        0.04,
        0.06,
        r"$r=-0.490$, $n=115$",
        transform=scatter_axis.transAxes,
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )
    colorbar = figure.colorbar(scatter, ax=scatter_axis, pad=0.02)
    colorbar.set_label("Visual-item share", fontsize=7.5)
    colorbar.ax.tick_params(labelsize=7)

    coefficient_plot = controlled.iloc[::-1].reset_index(drop=True)
    short_labels = {
        "Raw": "Raw",
        "Grade fixed effects": "+ grade FE",
        "Grade FE + multimodal share": "+ visual share",
        "Grade FE + image-location shares": "+ diagram/answer shares",
    }
    coefficient_axis = axes[1]
    coefficient_axis.axvline(0, color=INK, linewidth=0.7)
    coefficient_axis.errorbar(
        coefficient_plot["effect_per_10pp_human"],
        np.arange(len(coefficient_plot)),
        xerr=np.vstack(
            [
                coefficient_plot["effect_per_10pp_human"]
                - coefficient_plot["ci_low_per_10pp"],
                coefficient_plot["ci_high_per_10pp"]
                - coefficient_plot["effect_per_10pp_human"],
            ]
        ),
        fmt="s",
        color=BRICK,
        ecolor=BRICK,
        capsize=2,
        linewidth=0.8,
    )
    coefficient_axis.set_yticks(
        np.arange(len(coefficient_plot)),
        coefficient_plot["specification"].map(short_labels),
        fontsize=7.5,
    )
    for index, row in coefficient_plot.iterrows():
        coefficient_axis.annotate(
            f"{row['effect_per_10pp_human']:.2f}",
            (row["effect_per_10pp_human"], index),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    coefficient_axis.set_xlabel("Model-score change per +10 pp human score", fontsize=8)
    coefficient_axis.set_title("(b) Estimate and 95% CI", fontsize=9)
    coefficient_axis.tick_params(axis="x", labelsize=7)
    coefficient_axis.grid(axis="x")

    figure.tight_layout(pad=0.6, w_pad=0.8)
    return figure


def plot_image_subtypes(subtype_results: pd.DataFrame) -> plt.Figure:
    models = list(MODEL_LABELS)
    x = np.arange(len(SUBTYPE_ORDER))
    width = 0.19
    colors = FILLS[:4]
    figure, axis = styled_subplots(figsize=(7.2, 3.4))
    for index, (model, color) in enumerate(zip(models, colors, strict=True)):
        group = (
            subtype_results.loc[subtype_results["model"] == model]
            .set_index("image_subtype")
            .reindex(SUBTYPE_ORDER)
        )
        positions = x + (index - 1.5) * width
        axis.bar(
            positions,
            100 * group["accuracy"],
            width,
            label=MODEL_LABELS[model],
            color=color,
            edgecolor=INK,
            linewidth=0.5,
        )
    axis.set_xticks(x, SUBTYPE_ORDER, rotation=0, ha="center")
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y")
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    return figure


def run_diagnostics(
    project_root: Path,
    dataset_path: Path,
    write_outputs: bool = True,
) -> dict[str, object]:
    dataset = load_dataset(dataset_path)
    human_scores = load_human_scores(project_root / "artifacts" / "human_results")
    items = load_item_scores(project_root / "artifacts" / "runs", dataset)
    form_adjustments = pd.read_csv(project_root / "artifacts" / "form_adjustments.csv")
    numeric_adjustment_columns = [
        "year",
        "evaluated_items",
        "official_items",
        "additional_start_points",
        "automatic_credit",
        "fixed_bonus",
        "official_max",
    ]
    form_adjustments[numeric_adjustment_columns] = form_adjustments[
        numeric_adjustment_columns
    ].apply(pd.to_numeric)
    if not np.allclose(
        form_adjustments["fixed_bonus"],
        form_adjustments["additional_start_points"]
        + form_adjustments["automatic_credit"],
    ):
        raise ValueError("Official form fixed bonuses do not match their components")
    image_columns = [f"sol_{letter}_image_bin" for letter in "ABCDE"]
    image_flags = dataset[["id", "associated_images_bin", *image_columns]].copy()
    image_flags["has_associated_image"] = (
        image_flags["associated_images_bin"].map(len) > 0
    )
    image_flags["has_option_image"] = image_flags[image_columns].notna().any(axis=1)
    items = items.merge(
        image_flags[["id", "has_associated_image", "has_option_image"]],
        on="id",
        how="left",
        validate="many_to_one",
    )
    if items[["has_associated_image", "has_option_image"]].isna().any().any():
        raise ValueError("Image metadata does not cover every scored item")

    exam_scores = aggregate_exam_scores(items, form_adjustments)
    exam_counts = exam_scores.drop_duplicates(["year", "exam"])[
        [
            "year",
            "exam",
            "questions",
            "official_questions",
            "fixed_bonus",
            "possible_points",
        ]
    ]
    adjustment_coverage = form_adjustments.merge(
        exam_counts,
        left_on=["year", "group"],
        right_on=["year", "exam"],
        how="left",
        validate="one_to_one",
    )
    if not (
        adjustment_coverage["evaluated_items"] == adjustment_coverage["questions"]
    ).all():
        raise ValueError("Official form adjustments do not match evaluated row counts")
    shared_coverage = human_scores.merge(
        exam_counts, on=["year", "exam"], how="left", validate="one_to_one"
    )
    shared_coverage["official_items"] = shared_coverage["human_max"] / 5
    mismatched_forms = shared_coverage.loc[
        shared_coverage["official_questions"] != shared_coverage["official_items"]
    ]
    if not mismatched_forms.empty:
        raise ValueError("Model and human official item counts differ")
    comparison, lomo = build_comparison(exam_scores, human_scores)
    if not np.allclose(comparison["human_max"], comparison["possible_points"]):
        raise ValueError("Model and human score maxima differ on complete shared exams")
    pooled_statistics, calibration, out_of_sample = compute_statistics(comparison, lomo)

    exam_data, slice_scores = build_exam_diagnostics(comparison, items)
    metrics, controlled, per_grade, slice_summary = compute_alignment_diagnostics(
        exam_data, slice_scores
    )
    pooled_statistics["pooled"].pop("pearson_bootstrap_ci_95", None)
    pooled_statistics["pooled"].pop("spearman_bootstrap_ci_95", None)
    pooled_statistics["pooled"].pop("year_fe_effect_per_10pp", None)
    pooled_statistics["pooled"].pop("year_fe_human_pct_coefficient", None)
    pooled_statistics["pooled"]["pearson_year_cluster_ci_95"] = metrics["pooled"][
        "pearson_year_cluster_ci_95"
    ]
    pooled_statistics["pooled"]["spearman_year_cluster_ci_95"] = metrics["pooled"][
        "spearman_year_cluster_ci_95"
    ]
    subtype_results, scored_subtypes, image_metadata = compute_image_subtypes(
        dataset, items
    )
    response_status, point_tier, by_grade, no_visual_by_year = (
        compute_behavioral_diagnostics(scored_subtypes)
    )
    option_positions = compute_option_position_diagnostics(items)
    model_correlations = compute_model_correlations(exam_scores, human_scores)
    complete_exam_scores = exam_scores.loc[exam_scores["complete_exam"]]
    winner_counts = complete_exam_scores.loc[
        complete_exam_scores.groupby(["year", "exam"])["llm_pct"].idxmax()
    ]["model"].value_counts()
    model_performance = (
        items.groupby("model", sort=False)
        .agg(
            items=("id", "size"),
            correct=("answered_correctly", "sum"),
            accuracy=("answered_correctly", "mean"),
            abstentions=("declined", "sum"),
            decline_rate=("declined", "mean"),
        )
        .reset_index()
    )
    model_performance["no_valid_selections"] = model_performance["model"].map(
        items.assign(
            no_valid_selection=(
                ~items["predicted_normalized"].isin(list("ABCDE")) & ~items["declined"]
            )
        )
        .groupby("model")["no_valid_selection"]
        .sum()
    )
    model_performance["model_label"] = model_performance["model"].map(MODEL_LABELS)
    model_performance["contest_points"] = model_performance["model"].map(
        complete_exam_scores.groupby("model")["total_score"].sum()
    )
    model_performance["exams_won"] = (
        model_performance["model"].map(winner_counts).fillna(0).astype(int)
    )
    metrics["dataset"] = image_metadata
    metrics["form_adjustments"] = {
        "non_evaluable_official_slots": int(len(form_adjustments)),
        "fixed_credit_form_adjustments": form_adjustments[
            ["year", "group", "fixed_bonus", "official_max"]
        ].to_dict(orient="records"),
        "shared_comparisons": int(len(comparison)),
    }
    metrics["pooled_statistics"] = pooled_statistics
    metrics["contest_totals"] = {
        row.model_label: {
            "forms": int((complete_exam_scores["model"] == row.model).sum()),
            "possible_points": float(
                complete_exam_scores.loc[
                    complete_exam_scores["model"] == row.model, "possible_points"
                ].sum()
            ),
            "contest_points": float(row.contest_points),
        }
        for row in model_performance.itertuples(index=False)
    }
    metrics["response_policy"] = {
        row.model_label: {
            "items": int(row.items),
            "accuracy": float(row.accuracy),
            "abstentions": int(row.abstentions),
            "abstention_rate": float(row.decline_rate),
            "no_valid_selections": int(row.no_valid_selections),
        }
        for row in model_performance.itertuples(index=False)
    }
    metrics["model_correlations"] = model_correlations.to_dict(orient="records")
    metrics["image_subtype"] = {
        f"{record.model_label} / {record.image_subtype}": {
            "items": int(record.items),
            "accuracy": float(record.accuracy),
            "decline_rate": float(record.decline_rate),
        }
        for record in subtype_results.itertuples(index=False)
    }

    if write_outputs:
        output_dir = project_root / "reproduced" / "diagnostics"
        table_dir = output_dir / "tables"
        figure_dir = output_dir / "figures"
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exam_data.to_csv(table_dir / "shared_exam_composition.csv", index=False)
        controlled.to_csv(table_dir / "composition_controlled_models.csv", index=False)
        per_grade.to_csv(table_dir / "per_grade_correlations.csv", index=False)
        slice_summary.to_csv(table_dir / "text_only_diagnostic.csv", index=False)
        subtype_results.to_csv(table_dir / "image_subtype_results.csv", index=False)
        option_positions.to_csv(
            table_dir / "option_position_diagnostic.csv", index=False
        )
        response_status.to_csv(
            table_dir / "response_status_by_visual_condition.csv", index=False
        )
        point_tier.to_csv(table_dir / "visual_accuracy_by_point_tier.csv", index=False)
        by_grade.to_csv(table_dir / "visual_accuracy_by_grade.csv", index=False)
        no_visual_by_year.to_csv(
            table_dir / "no_visual_accuracy_by_year.csv", index=False
        )
        model_correlations.to_csv(table_dir / "model_correlations.csv", index=False)
        form_adjustments.to_csv(table_dir / "form_adjustments.csv", index=False)
        model_performance.to_csv(table_dir / "model_performance.csv", index=False)
        calibration.to_csv(table_dir / "per_grade_calibration.csv", index=False)
        out_of_sample.to_csv(table_dir / "out_of_sample_r2.csv", index=False)
        comparison.to_csv(table_dir / "shared_exam_comparisons.csv", index=False)

        alignment_figure = plot_alignment_summary(controlled, exam_data)
        alignment_figure.savefig(
            figure_dir / "alignment_summary.pdf", bbox_inches="tight"
        )
        alignment_figure.savefig(
            figure_dir / "alignment_summary.png", dpi=220, bbox_inches="tight"
        )
        plt.close(alignment_figure)
        subtype_figure = plot_image_subtypes(subtype_results)
        subtype_figure.savefig(
            figure_dir / "image_subtype_accuracy.pdf", bbox_inches="tight"
        )
        subtype_figure.savefig(
            figure_dir / "image_subtype_accuracy.png", dpi=220, bbox_inches="tight"
        )
        plt.close(subtype_figure)

    return {
        "metrics": metrics,
        "exam_data": exam_data,
        "controlled": controlled,
        "per_grade": per_grade,
        "slice_summary": slice_summary,
        "subtype_results": subtype_results,
        "response_status": response_status,
        "point_tier": point_tier,
        "by_grade": by_grade,
        "no_visual_by_year": no_visual_by_year,
        "option_positions": option_positions,
        "model_correlations": model_correlations,
        "model_performance": model_performance,
        "comparison": comparison,
        "exam_scores": exam_scores,
        "items": items,
        "dataset": dataset,
        "calibration": calibration,
        "out_of_sample": out_of_sample,
        "form_adjustments": form_adjustments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/kangaroo.parquet"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = args.dataset.expanduser()
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    results = run_diagnostics(project_root, dataset_path)
    pooled = results["metrics"]["pooled"]
    print(f"Shared exam comparisons: {pooled['comparisons']}")
    print(f"Pearson r: {pooled['pearson_r']:.6f}")
    print(f"Spearman rho: {pooled['spearman_rho']:.6f}")
    print(f"Outputs: {project_root / 'reproduced' / 'diagnostics'}")


if __name__ == "__main__":
    main()
