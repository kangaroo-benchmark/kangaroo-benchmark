import pandas as pd

from analysis.inputs import aggregate_exam_scores, score_predictions


def test_contest_scoring_uses_dataset_keys_and_quarter_penalty():
    dataset = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "year": [2020, 2020, 2020],
            "group": ["3-4", "3-4", "3-4"],
            "problem_number": [1, 2, 3],
            "points": [3, 4, 5],
            "multimodal": [False, True, False],
            "answer": ["A", "B", "C"],
        }
    )
    predictions = pd.DataFrame(
        {
            "id": ["a_en", "b_en", "c_en"],
            "source_id": ["a", "b", "c"],
            "predicted": ["a", "DECLINED", "D"],
        }
    )

    scored = score_predictions(predictions, dataset).set_index("source_id")

    assert (
        scored.loc["a", "answered_correctly"] and scored.loc["a", "awarded_points"] == 3
    )
    assert scored.loc["b", "declined"] and scored.loc["b", "awarded_points"] == 0
    assert scored.loc["c", "attempted"] and scored.loc["c", "awarded_points"] == -1.25
    assert scored["multimodal"].tolist() == [False, True, False]


def test_exam_start_capital_equals_item_count():
    rows = []
    for index, points in enumerate([3.0, 4.0, 5.0]):
        rows.append(
            {
                "id": str(index),
                "model": "model",
                "year": 2020,
                "group": "3-4",
                "question_points": points,
                "awarded_points": points,
                "answered_correctly": True,
                "attempted": True,
                "declined": False,
                "multimodal": False,
            }
        )

    exam = aggregate_exam_scores(pd.DataFrame(rows)).iloc[0]

    assert exam["possible_points"] == 15.0
    assert exam["total_score"] == 15.0
    assert exam["llm_pct"] == 100.0
    assert bool(exam["complete_exam"])


def test_official_form_adjustment_restores_fixed_credit_and_maximum():
    rows = []
    for index, points in enumerate([3.0] * 10 + [4.0] * 10 + [5.0] * 9):
        rows.append(
            {
                "id": str(index),
                "model": "model",
                "year": 2001,
                "group": "9-10",
                "question_points": points,
                "awarded_points": points,
                "answered_correctly": True,
                "attempted": True,
                "declined": False,
                "multimodal": False,
            }
        )
    adjustments = pd.DataFrame(
        [
            {
                "year": 2001,
                "group": "9-10",
                "official_items": 30,
                "fixed_bonus": 6,
                "official_max": 150,
            },
            {
                "year": 2003,
                "group": "3-4",
                "official_items": 21,
                "fixed_bonus": 6,
                "official_max": 105,
            },
        ]
    )

    exam = aggregate_exam_scores(pd.DataFrame(rows), adjustments).iloc[0]

    assert exam["questions"] == 29
    assert exam["official_questions"] == 30
    assert exam["fixed_bonus"] == 6
    assert exam["possible_points"] == 150
    assert exam["total_score"] == 150
    assert bool(exam["complete_exam"])
