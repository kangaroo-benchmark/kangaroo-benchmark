from pathlib import Path

import pandas as pd

from analysis.translation_subset import (
    add_visual_type,
    allocate_proportional_quotas,
    build_evaluation_frames,
    render_review_html,
)


def test_add_visual_type_distinguishes_diagrams_and_image_answers():
    rows = []
    for diagram, image_answer in [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]:
        row = {"associated_images_bin": [b"diagram"] if diagram else []}
        for letter in "ABCDE":
            row[f"sol_{letter}_image_bin"] = (
                b"answer" if image_answer and letter == "A" else None
            )
        rows.append(row)

    result = add_visual_type(pd.DataFrame(rows))

    assert result["visual_type"].tolist() == [
        "no_separate_visual_element",
        "question_diagram_only",
        "image_answers_only",
        "question_diagram_and_image_answers",
    ]


def test_proportional_quotas_use_largest_remainders():
    dataset = pd.DataFrame(
        {
            "group": ["a"] * 6 + ["b"] * 3 + ["c"],
            "points": [3] * 10,
            "visual_type": ["none"] * 10,
        }
    )

    quotas = allocate_proportional_quotas(
        dataset, ["group", "points", "visual_type"], sample_size=4
    )

    assert dict(zip(quotas["group"], quotas["sample_quota"], strict=True)) == {
        "a": 2,
        "b": 1,
        "c": 1,
    }


def test_evaluation_frames_pair_text_and_omit_german_question_crop():
    translated = pd.DataFrame(
        [
            {
                "id": "item-1",
                "language": "de",
                "problem_statement": "Rohtext",
                "german_problem_statement": "Bereinigter Text",
                "english_problem_statement": "Clean text",
                "question_image": b"German crop",
                **{f"sol_{letter}": str(index) for index, letter in enumerate("ABCDE")},
                **{
                    f"english_sol_{letter}": str(index)
                    for index, letter in enumerate("ABCDE")
                },
            }
        ]
    )

    german, english = build_evaluation_frames(translated)

    assert german.loc[0, "id"] == "item-1_de_control"
    assert english.loc[0, "id"] == "item-1_en"
    assert german.loc[0, "problem_statement"] == "Bereinigter Text"
    assert english.loc[0, "problem_statement"] == "Clean text"
    assert german.loc[0, "question_image"] is None
    assert english.loc[0, "question_image"] is None
    assert "german_problem_statement" not in english.columns


def test_review_html_keeps_newline_regex_on_one_line(tmp_path: Path):
    row = {
        "id": "item-1",
        "year": 2025,
        "group": "7-8",
        "points": 3,
        "problem_number": "1",
        "multimodal": False,
        "visual_type": "no_separate_visual_element",
        "answer": "A",
        "question_image": None,
        "associated_images_bin": [],
    }
    for letter in "ABCDE":
        row[f"sol_{letter}"] = letter
        row[f"english_sol_{letter}"] = letter
        row[f"sol_{letter}_image_bin"] = None
    row["problem_statement"] = "Zeile eins\nZeile zwei"
    row["english_problem_statement"] = "Line one\nLine two"
    output_path = tmp_path / "translation_review.html"

    render_review_html(pd.DataFrame([row]), output_path)

    document = output_path.read_text(encoding="utf-8")
    assert 'replace(/\\n/g,"<br>")' in document
    assert 'replace(/\n/g,"<br>")' not in document
