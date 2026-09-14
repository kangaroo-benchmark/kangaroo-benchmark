import numpy as np
import pandas as pd

from analysis.surrogate_analysis import (
    ENSEMBLE,
    MODEL_LABELS,
    compute_forward_predictions,
    compute_outcome_regressions,
    compute_variance_decomposition,
    series_column,
    summarize_forward_predictions,
)

YEARS = [*range(2001, 2020), 2022, 2023, 2024, 2025]
GRADES = ["Grade 3-4", "Grade 5-6", "Grade 7-8", "Grade 9-10", "Grade 11-13"]


def _synthetic_panel(slope: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for grade_index, grade in enumerate(GRADES):
        for year in YEARS:
            visual_share = float(rng.uniform(0.2, 0.7))
            model_scores = {
                series_column(model): 55.0
                + 5.0 * grade_index
                + 30.0 * visual_share
                + float(rng.normal(0.0, 2.0))
                for model in MODEL_LABELS
            }
            ensemble = float(np.mean(list(model_scores.values())))
            rows.append(
                {
                    "year": year,
                    "exam": grade.removeprefix("Grade "),
                    "grade_bucket": grade,
                    "human_pct": 30.0 + 4.0 * grade_index + slope * (ensemble - 60.0),
                    "pooled_ensemble_pct": ensemble,
                    series_column(ENSEMBLE): ensemble,
                    "students": 1000,
                    "item_count": 30,
                    "multimodal_share": visual_share,
                    "associated_image_share": visual_share * 0.8,
                    "option_image_share": visual_share * 0.3,
                    **model_scores,
                }
            )
    return pd.DataFrame(rows)


def test_forward_prediction_recovers_exact_linear_surrogate():
    panel = _synthetic_panel(slope=0.5)
    predictions = compute_forward_predictions(panel)
    per_grade, pooled = summarize_forward_predictions(predictions, repetitions=50)

    ensemble = per_grade.loc[
        (per_grade["series"] == ENSEMBLE)
        & (per_grade["specification"] == "Model score")
    ]
    assert len(ensemble) == len(GRADES)
    assert (ensemble["predictions"] == len(YEARS) - 8).all()
    assert (ensemble["mae"] < 1e-9).all()
    assert (ensemble["years_beating_baseline"] == ensemble["predictions"]).all()

    pooled_ensemble = pooled.loc[
        (pooled["series"] == ENSEMBLE) & (pooled["specification"] == "Model score")
    ].iloc[0]
    assert np.isclose(pooled_ensemble["forward_r2"], 1.0)
    assert pooled_ensemble["mae_change_ci_high"] < 0

    baseline = pooled.loc[
        (pooled["series"] == ENSEMBLE) & (pooled["specification"] == "Grade mean")
    ].iloc[0]
    assert baseline["forward_r2"] == 0.0
    assert baseline["mae"] == baseline["baseline_mae"]


def test_outcome_regression_recovers_slope_under_grade_fixed_effects():
    panel = _synthetic_panel(slope=0.5)
    regressions = compute_outcome_regressions(panel, repetitions=50)
    row = regressions.loc[
        (regressions["series"] == ENSEMBLE)
        & (regressions["specification"] == "Grade fixed effects")
    ].iloc[0]
    assert np.isclose(row["effect_per_10pp_model"], 5.0)
    assert row["ci_low_per_10pp"] <= 5.0 <= row["ci_high_per_10pp"]
    assert np.isclose(row["r2_full"], 1.0)
    assert row["r2_increment"] > 0.05


def test_variance_decomposition_explains_constructed_model_scores():
    panel = _synthetic_panel(slope=0.0)
    decomposition = compute_variance_decomposition(panel, repetitions=50).set_index(
        "outcome_label"
    )
    human = decomposition.loc["Human mean"]
    ensemble = decomposition.loc["Equal-weight ensemble"]
    assert np.isclose(human["r2_grade"], 1.0)
    assert np.isclose(human["visual_r2_increment"], 0.0, atol=1e-9)
    assert ensemble["r2_grade_visual"] > 0.95
    assert ensemble["visual_r2_increment"] > 0.2
    assert np.isclose(ensemble["effect_per_10pp_visual_share"], 3.0, atol=0.5)
