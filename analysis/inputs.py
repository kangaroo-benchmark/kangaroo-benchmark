"""Archived inputs: the released dataset, official cohort summaries, and model runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from score_utils import DECLINED_TOKEN, PENALTY_FACTOR

DATASET_SHA256 = "131e4eeb45bae89d7bf3796461b009cc8cba7e514aacde2f8d36849ddbe911d9"
DATASET_ITEMS = 3886

# Full evaluations of the released dataset (October 2025).
RUNS = {
    "20251013_191510_openai-gpt-5": "openai-gpt-5",
    "20251013_191528_x-ai-grok-4-fast": "x-ai-grok-4-fast",
    "20251014_080823_qwen-qwen3-vl-235b-a22b-thinking": (
        "qwen-qwen3-vl-235b-a22b-thinking"
    ),
    "20251020_212534_anthropic-claude-sonnet-4.5": "anthropic-claude-sonnet-4.5",
}
# Paired comparisons (August 2026): (model, arm).
COMPARISON_RUNS = {
    "20260807_163720_anthropic-claude-sonnet-4.5": (
        "anthropic-claude-sonnet-4.5",
        "german_control",
    ),
    "20260807_164146_anthropic-claude-sonnet-4.5": (
        "anthropic-claude-sonnet-4.5",
        "english",
    ),
    "20260807_164637_anthropic-claude-sonnet-4.5": (
        "anthropic-claude-sonnet-4.5",
        "blind",
    ),
    "20260807_164656_openai-gpt-5": ("openai-gpt-5", "german_control"),
    "20260807_171242_openai-gpt-5": ("openai-gpt-5", "english"),
    "20260807_182113_openai-gpt-5": ("openai-gpt-5", "blind"),
    "20260807_164703_qwen-qwen3-vl-235b-a22b-thinking": (
        "qwen-qwen3-vl-235b-a22b-thinking",
        "german_control",
    ),
    "20260807_182113_qwen-qwen3-vl-235b-a22b-thinking": (
        "qwen-qwen3-vl-235b-a22b-thinking",
        "english",
    ),
    "20260807_191427_qwen-qwen3-vl-235b-a22b-thinking": (
        "qwen-qwen3-vl-235b-a22b-thinking",
        "blind",
    ),
}
ARM_INPUTS = {
    "full": "kangaroo.parquet",
    "german_control": "translation/german_control.parquet",
    "english": "translation/english_translation.parquet",
    "blind": "experiments/blind_question_diagram_only.parquet",
}
# Translation-arm rows carry the source item id plus an arm suffix.
ARM_ID_SUFFIX = {
    "full": "",
    "german_control": "_de_control",
    "english": "_en",
    "blind": "",
}
REFERENCE_MODEL = "openai-gpt-5"
GRADE_MEMBERS = {
    "3-4": (3, 4),
    "5-6": (5, 6),
    "7-8": (7, 8),
    "9-10": (9, 10),
    "11-13": (11, 12, 13),
}
GRADE_LABELS = {group: f"Grade {group}" for group in GRADE_MEMBERS}
GRADE_ORDER = {label: index for index, label in enumerate(GRADE_LABELS.values())}
OFFICIAL_FORM_ADJUSTMENTS = {(2001, "9-10"), (2003, "3-4")}
METADATA_COLUMNS = ["year", "group", "problem_number", "points", "multimodal"]
COMPACT_RUN_COLUMNS = ["id", "source_id", *METADATA_COLUMNS, "predicted"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """The released dataset, verified against its pinned checksum."""
    digest = file_sha256(dataset_path)
    if digest != DATASET_SHA256:
        raise ValueError(f"Unexpected dataset hash {digest}; expected {DATASET_SHA256}")
    dataset = pd.read_parquet(dataset_path)
    dataset["id"] = dataset["id"].astype(str)
    if len(dataset) != DATASET_ITEMS or dataset["id"].nunique() != DATASET_ITEMS:
        raise ValueError(f"Expected {DATASET_ITEMS:,} unique items in the dataset")
    return dataset


def _histogram_mean(data: dict[str, object], grade_id: str) -> float:
    counts_by_grade = data["counts_by_grade"]
    bins = data["bins"]
    assert isinstance(counts_by_grade, dict)
    assert isinstance(bins, list)
    counts = np.asarray(counts_by_grade[grade_id], dtype=float)
    midpoints = []
    for item in bins:
        assert isinstance(item, dict)
        ranges_by_grade = item.get("ranges_by_grade", {})
        grade_range = ranges_by_grade.get(grade_id) if ranges_by_grade else None
        selected_range = grade_range or item["range_default"]
        midpoints.append(
            (float(selected_range["min"]) + float(selected_range["max"])) / 2
        )
    return float(np.average(np.asarray(midpoints, dtype=float), weights=counts))


def load_human_scores(human_dir: Path) -> pd.DataFrame:
    """Official cohort means per grade group, weighting listed grades by participants."""
    records: list[dict[str, object]] = []
    files = sorted(human_dir.glob("human_baseline_*.json"))
    if not files:
        raise FileNotFoundError(f"No human baseline files found in {human_dir}")

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["schema_version"] != "2.0":
            raise ValueError(f"Unsupported schema in {path}: {data['schema_version']}")

        year = int(data["year"])
        totals = data["totals_by_grade"]
        average_scores = data["avg_score_by_grade"]
        grade_records = []
        for grade in data["grades"]:
            grade_id = str(grade["id"])
            if grade_id == "overall":
                continue
            average = average_scores.get(grade_id)
            if average is None:
                average = _histogram_mean(data, grade_id)
            grade_records.append(
                {
                    "grade_id": grade_id,
                    "members": tuple(int(member) for member in grade["members"]),
                    "human_score": float(average),
                    "human_max": float(grade["max_points"]),
                    "students": int(totals[grade_id]),
                }
            )

        for exam, members in GRADE_MEMBERS.items():
            exact = [record for record in grade_records if record["grade_id"] == exam]
            selected = exact.copy()
            remaining = set() if exact else set(members)
            if not exact:
                for record in grade_records:
                    record_members = set(record["members"])
                    if record_members and record_members.issubset(remaining):
                        selected.append(record)
                        remaining -= record_members
                if remaining:
                    for record in grade_records:
                        record_members = set(record["members"])
                        if record_members & remaining and record not in selected:
                            selected.append(record)
                            remaining -= record_members
            if remaining:
                raise ValueError(
                    f"Human grades for {year} {exam} do not cover {sorted(remaining)}"
                )

            student_counts = np.asarray(
                [record["students"] for record in selected], dtype=float
            )
            scores = np.asarray(
                [record["human_score"] for record in selected], dtype=float
            )
            human_score = float(np.average(scores, weights=student_counts))
            human_max = float(np.mean([record["human_max"] for record in selected]))
            records.append(
                {
                    "year": year,
                    "exam": exam,
                    "grade_bucket": GRADE_LABELS[exam],
                    "human_score": human_score,
                    "human_max": human_max,
                    "human_pct": 100.0 * human_score / human_max,
                    "students": int(student_counts.sum()),
                    "human_grades_used": ",".join(
                        str(record["grade_id"]) for record in selected
                    ),
                }
            )

    human_scores = pd.DataFrame(records)
    return human_scores.sort_values(["year", "exam"]).reset_index(drop=True)


def read_compact_run(run_dir: Path) -> pd.DataFrame:
    """The compact result table of an archived run."""
    path = run_dir / "results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing model results: {path}")
    frame = pd.read_parquet(path)
    missing = set(COMPACT_RUN_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["id"] = frame["id"].astype(str)
    frame["source_id"] = frame["source_id"].astype(str)
    return frame


def score_predictions(predictions: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """Score predicted letters against the dataset keys with the contest rule."""
    reference = dataset.set_index("id")
    scored = predictions.copy()
    if not scored["source_id"].isin(reference.index).all():
        raise ValueError("Predictions reference items outside the dataset")
    for column in [*METADATA_COLUMNS, "answer"]:
        scored[column] = scored["source_id"].map(reference[column]).to_numpy()
    scored["year"] = scored["year"].astype(int)
    scored["group"] = scored["group"].astype(str)
    scored["answer"] = scored["answer"].astype(str).str.strip().str.upper()
    scored["predicted_normalized"] = (
        scored["predicted"].fillna("").astype(str).str.strip().str.upper()
    )
    scored["declined"] = scored["predicted_normalized"] == DECLINED_TOKEN
    scored["attempted"] = ~scored["declined"]
    scored["question_points"] = scored["points"].astype(float)
    scored["answered_correctly"] = scored["predicted_normalized"] == scored["answer"]
    scored["awarded_points"] = np.where(
        scored["answered_correctly"],
        scored["question_points"],
        np.where(scored["attempted"], -PENALTY_FACTOR * scored["question_points"], 0.0),
    )
    return scored


def load_item_scores(runs_dir: Path, dataset: pd.DataFrame) -> pd.DataFrame:
    """Scored item-level outputs of the full runs, one row per model and item."""
    frames = []
    for run_id, model in RUNS.items():
        frame = read_compact_run(runs_dir / run_id)
        if len(frame) != DATASET_ITEMS or frame["source_id"].nunique() != DATASET_ITEMS:
            raise ValueError(f"{run_id} must cover every dataset item exactly once")
        frame = score_predictions(frame, dataset)
        frame["model"] = model
        frames.append(frame)
    item_scores = pd.concat(frames, ignore_index=True)
    if not set(item_scores["group"].unique()).issubset(GRADE_MEMBERS):
        raise ValueError("Unexpected exam group in the archived runs")
    return item_scores


def aggregate_exam_scores(
    item_scores: pd.DataFrame, form_adjustments: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Contest scores per model and form: one starting point per item, item values for
    correct answers, a quarter-value penalty for wrong answers, fixed credit for the two
    non-evaluable official slots."""
    adjustments = {}
    if form_adjustments is not None:
        adjustments = {
            (int(row.year), str(row.group)): row
            for row in form_adjustments.itertuples(index=False)
        }
        if set(adjustments) != OFFICIAL_FORM_ADJUSTMENTS:
            raise ValueError(
                f"Unexpected official form adjustments: {set(adjustments)}"
            )

    records = []
    for (model, year, exam), group in item_scores.groupby(
        ["model", "year", "group"], sort=True
    ):
        adjustment = adjustments.get((int(year), str(exam)))
        evaluated_questions = int(len(group))
        official_questions = (
            int(adjustment.official_items) if adjustment else evaluated_questions
        )
        fixed_bonus = float(adjustment.fixed_bonus) if adjustment else 0.0
        start_points = float(evaluated_questions)
        question_points = float(group["question_points"].sum())
        possible_points = start_points + question_points + fixed_bonus
        expected_maximum = (
            float(adjustment.official_max) if adjustment else 5.0 * official_questions
        )
        complete_exam = bool(np.isclose(possible_points, expected_maximum))
        total_score = start_points + float(group["awarded_points"].sum()) + fixed_bonus
        records.append(
            {
                "model": model,
                "year": int(year),
                "exam": exam,
                "grade_bucket": f"Grade {exam}",
                "questions": evaluated_questions,
                "official_questions": official_questions,
                "fixed_bonus": fixed_bonus,
                "correct": int(group["answered_correctly"].sum()),
                "attempted": int(group["attempted"].sum()),
                "declined": int(group["declined"].sum()),
                "multimodal_share": float(group["multimodal"].mean()),
                "total_score": total_score,
                "possible_points": possible_points,
                "llm_pct": 100.0 * total_score / possible_points,
                "complete_exam": complete_exam,
            }
        )
    return pd.DataFrame(records).sort_values(["year", "exam", "model"])


def _z_score(series: pd.Series) -> pd.Series:
    standard_deviation = series.std(ddof=0)
    if standard_deviation == 0 or np.isnan(standard_deviation):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / standard_deviation


def build_comparison(
    exam_scores: pd.DataFrame,
    human_scores: pd.DataFrame,
    expected_comparisons: int = 115,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the shared form comparisons and grade-normalized difficulty series."""
    index_columns = ["year", "exam", "grade_bucket"]
    model_scores = exam_scores.pivot_table(
        index=index_columns, columns="model", values="llm_pct"
    )
    if model_scores.isna().any().any():
        raise ValueError("Every shared exam must contain all four model scores")

    model_summary = pd.DataFrame(
        {
            "ensemble_pct": model_scores.mean(axis=1),
            "reference_pct": model_scores[REFERENCE_MODEL],
            "model_count": model_scores.notna().sum(axis=1),
        }
    )
    pooled_ensemble = exam_scores.groupby(index_columns).agg(
        mean_total_score=("total_score", "mean"),
        possible_points=("possible_points", "first"),
    )
    model_summary["pooled_ensemble_pct"] = (
        100.0 * pooled_ensemble["mean_total_score"] / pooled_ensemble["possible_points"]
    )
    model_summary["possible_points"] = pooled_ensemble["possible_points"]
    for model in model_scores.columns:
        model_summary[f"lomo_pct__{model}"] = model_scores.drop(columns=model).mean(
            axis=1
        )

    comparison = model_summary.reset_index().merge(
        human_scores, on=index_columns, how="inner", validate="one_to_one"
    )
    if len(comparison) != expected_comparisons:
        raise ValueError(
            f"Expected {expected_comparisons} shared exams, found {len(comparison)}"
        )

    comparison["human_difficulty"] = -comparison.groupby("grade_bucket")[
        "human_pct"
    ].transform(_z_score)
    comparison["ensemble_difficulty"] = -comparison.groupby("grade_bucket")[
        "ensemble_pct"
    ].transform(_z_score)
    comparison["reference_difficulty"] = -comparison.groupby("grade_bucket")[
        "reference_pct"
    ].transform(_z_score)

    lomo_columns = []
    for model in model_scores.columns:
        lomo_pct_column = f"lomo_pct__{model}"
        lomo_difficulty_column = f"lomo_difficulty__{model}"
        comparison[lomo_difficulty_column] = -comparison.groupby("grade_bucket")[
            lomo_pct_column
        ].transform(_z_score)
        lomo_columns.append(lomo_difficulty_column)

    comparison["opposite_difficulty_sign"] = (
        comparison["human_difficulty"] * comparison["ensemble_difficulty"] < 0
    )
    comparison["_grade_order"] = comparison["grade_bucket"].map(GRADE_ORDER)
    comparison = comparison.sort_values(["_grade_order", "year"]).drop(
        columns="_grade_order"
    )

    lomo_records = []
    for column in lomo_columns:
        model = column.removeprefix("lomo_difficulty__")
        for grade, group in comparison.groupby("grade_bucket", sort=False):
            deviations = np.abs(
                group[column].to_numpy(dtype=float)
                - group["ensemble_difficulty"].to_numpy(dtype=float)
            )
            lomo_records.append(
                {
                    "grade_bucket": grade,
                    "model_omitted": model,
                    "mean_abs_deviation_sd": float(deviations.mean()),
                    "max_abs_deviation_sd": float(deviations.max()),
                    "correlation_with_human": float(
                        np.corrcoef(
                            group["human_difficulty"].to_numpy(dtype=float),
                            group[column].to_numpy(dtype=float),
                        )[0, 1]
                    ),
                }
            )
    lomo = pd.DataFrame(lomo_records)
    lomo["_grade_order"] = lomo["grade_bucket"].map(GRADE_ORDER)
    lomo = lomo.sort_values(["_grade_order", "model_omitted"]).drop(
        columns="_grade_order"
    )
    return comparison.reset_index(drop=True), lomo.reset_index(drop=True)


def _bootstrap_correlation(
    first: np.ndarray, second: np.ndarray, seed: int, rank: bool = False
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    bootstrapped = []
    for _ in range(2000):
        indices = rng.integers(0, len(first), len(first))
        sampled_first = first[indices]
        sampled_second = second[indices]
        if np.std(sampled_first) == 0 or np.std(sampled_second) == 0:
            continue
        correlation = (
            spearmanr(sampled_first, sampled_second).statistic
            if rank
            else pearsonr(sampled_first, sampled_second).statistic
        )
        bootstrapped.append(float(correlation))
    lower, upper = np.percentile(np.asarray(bootstrapped), [2.5, 97.5])
    return float(lower), float(upper)


def _fit_and_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> float:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    coefficients = np.linalg.lstsq(design, train_y, rcond=None)[0]
    predictions = np.column_stack([np.ones(len(test_x)), test_x]) @ coefficients
    residual_sum = float(np.sum((test_y - predictions) ** 2))
    total_sum = float(np.sum((test_y - test_y.mean()) ** 2))
    return 1.0 - residual_sum / total_sum


def compute_statistics(
    comparison: pd.DataFrame, lomo: pd.DataFrame, items_per_model: int = DATASET_ITEMS
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Pooled correlations, per-grade calibrations, and early/late out-of-sample fits."""
    pooled_comparison = comparison.sort_values(["year", "exam"])
    human_pct = pooled_comparison["human_pct"].to_numpy(dtype=float)
    ensemble_pct = pooled_comparison["pooled_ensemble_pct"].to_numpy(dtype=float)
    pearson = pearsonr(human_pct, ensemble_pct)
    spearman = spearmanr(human_pct, ensemble_pct)
    pearson_ci = _bootstrap_correlation(human_pct, ensemble_pct, seed=2026)
    spearman_ci = _bootstrap_correlation(human_pct, ensemble_pct, seed=2026, rank=True)

    year_dummies = pd.get_dummies(
        pooled_comparison["year"].astype(str), drop_first=True, dtype=float
    )
    year_fixed_effect_design = np.column_stack(
        [np.ones(len(comparison)), human_pct, year_dummies.to_numpy(dtype=float)]
    )
    year_fixed_effect_coefficients = np.linalg.lstsq(
        year_fixed_effect_design, ensemble_pct, rcond=None
    )[0]

    calibration_records = []
    out_of_sample_records = []
    series = {
        "gpt5": "reference_difficulty",
        "ensemble_equal": "ensemble_difficulty",
    }
    for grade_index, (grade, group) in enumerate(
        comparison.groupby("grade_bucket", sort=False)
    ):
        group = group.sort_values("year")
        years = group["year"].to_numpy(dtype=float)
        human_difficulty = group["human_difficulty"].to_numpy(dtype=float)
        split_year = float(np.median(years))
        early = years < split_year
        late = ~early

        for series_index, (name, column) in enumerate(series.items()):
            model_difficulty = group[column].to_numpy(dtype=float)
            correlation = float(np.corrcoef(human_difficulty, model_difficulty)[0, 1])
            correlation_ci = _bootstrap_correlation(
                human_difficulty,
                model_difficulty,
                seed=3000 + grade_index * 10 + series_index,
            )
            calibration_design = np.column_stack(
                [np.ones(len(model_difficulty)), model_difficulty]
            )
            calibration_coefficients = np.linalg.lstsq(
                calibration_design, human_difficulty, rcond=None
            )[0]
            fitted = calibration_design @ calibration_coefficients
            calibration_r2 = 1.0 - float(
                np.sum((human_difficulty - fitted) ** 2)
                / np.sum((human_difficulty - human_difficulty.mean()) ** 2)
            )
            calibration_records.append(
                {
                    "grade_bucket": grade,
                    "series": name,
                    "correlation": correlation,
                    "correlation_ci_low": correlation_ci[0],
                    "correlation_ci_high": correlation_ci[1],
                    "same_side_rate": float(
                        np.mean((human_difficulty > 0) == (model_difficulty > 0))
                    ),
                    "intercept": float(calibration_coefficients[0]),
                    "slope": float(calibration_coefficients[1]),
                    "r2": calibration_r2,
                }
            )
            out_of_sample_records.append(
                {
                    "grade_bucket": grade,
                    "series": name,
                    "split_year": int(split_year),
                    "early_to_late_r2": _fit_and_score(
                        model_difficulty[early],
                        human_difficulty[early],
                        model_difficulty[late],
                        human_difficulty[late],
                    ),
                    "late_to_early_r2": _fit_and_score(
                        model_difficulty[late],
                        human_difficulty[late],
                        model_difficulty[early],
                        human_difficulty[early],
                    ),
                }
            )

    calibration = pd.DataFrame(calibration_records)
    out_of_sample = pd.DataFrame(out_of_sample_records)
    opposite_sign_count = int(comparison["opposite_difficulty_sign"].sum())
    reference_correlations = calibration.loc[
        calibration["series"] == "gpt5", "correlation"
    ]
    ensemble_correlations = calibration.loc[
        calibration["series"] == "ensemble_equal", "correlation"
    ]
    metrics = {
        "inputs": {
            "models": len(RUNS),
            "items_per_model": items_per_model,
            "human_years": int(comparison["year"].nunique()),
            "grade_buckets": int(comparison["grade_bucket"].nunique()),
        },
        "pooled": {
            "comparisons": int(len(comparison)),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "pearson_bootstrap_ci_95": list(pearson_ci),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "spearman_bootstrap_ci_95": list(spearman_ci),
            "year_fe_human_pct_coefficient": float(year_fixed_effect_coefficients[1]),
            "year_fe_effect_per_10pp": float(10.0 * year_fixed_effect_coefficients[1]),
        },
        "difficulty": {
            "opposite_sign_count": opposite_sign_count,
            "opposite_sign_rate": opposite_sign_count / len(comparison),
            "mean_per_grade_reference_correlation": float(
                reference_correlations.mean()
            ),
            "mean_per_grade_ensemble_correlation": float(ensemble_correlations.mean()),
            "max_lomo_deviation_sd": float(lomo["max_abs_deviation_sd"].max()),
            "all_oos_r2_negative": bool(
                (out_of_sample[["early_to_late_r2", "late_to_early_r2"]] < 0)
                .all()
                .all()
            ),
        },
    }
    return metrics, calibration, out_of_sample
