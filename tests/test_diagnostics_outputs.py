from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path(__file__).resolve().parents[1] / "reproduced/diagnostics/tables"


def test_model_correlations_match_reported_values():
    table = pd.read_csv(TABLES / "model_correlations.csv").set_index("series")

    assert len(table) == 5
    assert (table["comparisons"] == 115).all()
    ensemble = table.loc["Equal-weight ensemble"]
    assert np.isclose(ensemble["pearson_r"], -0.49037806881338686)
    assert np.isclose(ensemble["spearman_rho"], -0.4567138210082801)


def test_response_status_counts_use_all_rows():
    table = pd.read_csv(TABLES / "response_status_by_visual_condition.csv")
    totals = table.groupby("model_label").agg(
        items=("items", "sum"),
        abstentions=("abstentions", "sum"),
        no_valid_selections=("no_valid_selections", "sum"),
    )

    assert (totals["items"] == 3886).all()
    assert totals["abstentions"].to_dict() == {
        "Claude Sonnet 4.5": 29,
        "GPT-5": 117,
        "Grok 4 Fast": 460,
        "Qwen3-VL": 0,
    }
    assert totals["no_valid_selections"].to_dict() == {
        "Claude Sonnet 4.5": 0,
        "GPT-5": 0,
        "Grok 4 Fast": 0,
        "Qwen3-VL": 4,
    }


def test_visual_gap_persists_within_every_point_tier_and_grade_bucket():
    for filename, index_columns in [
        ("visual_accuracy_by_point_tier.csv", ["model", "question_points"]),
        ("visual_accuracy_by_grade.csv", ["model", "grade_bucket"]),
    ]:
        table = pd.read_csv(TABLES / filename)
        comparison = table.pivot(
            index=index_columns,
            columns="visual_condition",
            values="accuracy",
        )
        assert (
            comparison["Text-only"]
            > comparison["Auxiliary visual content"]
        ).all()


def test_composition_and_temporal_claims_match_generated_tables():
    composition = pd.read_csv(TABLES / "composition_controlled_models.csv").set_index(
        "specification"
    )
    assert np.isclose(
        composition.loc["Grade FE + multimodal share", "effect_per_10pp_human"],
        -2.1069270370791737,
    )
    assert np.isclose(
        composition.loc["Grade FE + image-location shares", "effect_per_10pp_human"],
        -0.6250186396575044,
    )

    out_of_sample = pd.read_csv(TABLES / "out_of_sample_r2.csv")[
        ["early_to_late_r2", "late_to_early_r2"]
    ].to_numpy()
    assert int((out_of_sample < 0).sum()) == 19
    assert np.isclose(out_of_sample[out_of_sample >= 0][0], 0.002916107536899193)


def test_option_and_recent_year_diagnostics_match_reported_values():
    options = pd.read_csv(TABLES / "option_position_diagnostic.csv")
    c_shares = options.loc[options["option"] == "C"].set_index("model_label")[
        "predicted_share_among_attempts"
    ]
    assert np.isclose(c_shares["GPT-5"], 0.21915627487397188)
    assert np.isclose(c_shares["Grok 4 Fast"], 0.22825452422650322)
    assert np.isclose(c_shares["Qwen3-VL"], 0.23235445646573932)
    assert np.isclose(c_shares["Claude Sonnet 4.5"], 0.24163857920663728)

    by_year = pd.read_csv(TABLES / "no_visual_accuracy_by_year.csv")
    by_year["period"] = np.where(by_year["year"] >= 2024, "recent", "older")
    by_year["correct"] = by_year["accuracy"] * by_year["items"]
    periods = by_year.groupby(["model_label", "period"]).agg(
        items=("items", "sum"), correct=("correct", "sum")
    )
    periods["accuracy"] = periods["correct"] / periods["items"]
    assert (periods.xs("recent", level="period")["items"] == 87).all()
    assert (periods.xs("older", level="period")["items"] == 2053).all()
    assert np.isclose(periods.loc[("GPT-5", "recent"), "accuracy"], 1.0)


def test_no_valid_qwen_outputs_receive_the_wrong_answer_penalty():
    performance = pd.read_csv(TABLES / "model_performance.csv").set_index("model_label")

    assert performance.loc["Qwen3-VL", "no_valid_selections"] == 4
    assert performance.loc["Qwen3-VL", "contest_points"] == 16803.75
