"""Prepare the blind-diagram evaluation input from the released dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.inputs import DATASET_SHA256, load_dataset

LETTERS = list("ABCDE")


def select_blind_question_diagram_items(dataset: pd.DataFrame) -> pd.DataFrame:
    """Items with a separately extracted question diagram and textual options."""
    image_columns = [f"sol_{letter}_image_bin" for letter in LETTERS]
    has_question_diagram = dataset["associated_images_bin"].map(len) > 0
    has_image_answers = dataset[image_columns].notna().any(axis=1)
    return dataset.loc[has_question_diagram & ~has_image_answers].copy()


def prepare_experiments(dataset_path: Path, output_dir: Path) -> None:
    dataset = load_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    blind = select_blind_question_diagram_items(dataset)
    blind.to_parquet(output_dir / "blind_question_diagram_only.parquet", index=False)
    blind[["id", "year", "group", "points", "problem_number"]].to_csv(
        output_dir / "blind_question_diagram_only_ids.csv", index=False
    )
    manifest = {
        "dataset_sha256": DATASET_SHA256,
        "blind_question_diagram_only_items": int(len(blind)),
        "blind_question_diagram_design": (
            "separately extracted question diagram with textual answer options; "
            "evaluated without the question crop and without the diagram"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the blind-diagram evaluation input"
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/kangaroo.parquet"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reproduced/experiments")
    )
    args = parser.parse_args()
    prepare_experiments(args.dataset, args.output_dir)
    print(f"Prepared the blind-diagram input in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
