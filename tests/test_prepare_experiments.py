import pandas as pd

from analysis.prepare_experiments import select_blind_question_diagram_items


def test_blind_question_diagram_selection_excludes_image_answers():
    rows = []
    for item_id, diagram, image_answer in [
        ("diagram", [b"diagram"], False),
        ("image-answer", [b"diagram"], True),
        ("plain", [], False),
    ]:
        row = {"id": item_id, "associated_images_bin": diagram}
        for letter in "ABCDE":
            row[f"sol_{letter}_image_bin"] = (
                b"answer" if image_answer and letter == "A" else None
            )
        rows.append(row)

    selected = select_blind_question_diagram_items(pd.DataFrame(rows))

    assert selected["id"].tolist() == ["diagram"]
