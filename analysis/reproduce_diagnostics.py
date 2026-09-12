"""Regenerate the item-level diagnostics and cohort comparisons from archived inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.diagnostics import run_diagnostics
from analysis.inputs import file_sha256


def verify_checksums(project_root: Path) -> None:
    manifest = project_root / "artifacts" / "checksums.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", maxsplit=1)
        if file_sha256(project_root / relative) != expected:
            raise ValueError(f"Checksum mismatch for {relative}")


def resolve_dataset(project_root: Path, dataset: Path) -> Path:
    dataset_path = dataset.expanduser()
    return dataset_path if dataset_path.is_absolute() else project_root / dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the item-level diagnostics and cohort comparisons."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/kangaroo.parquet"),
        help="The released dataset file (default: data/kangaroo.parquet).",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    verify_checksums(project_root)
    results = run_diagnostics(project_root, resolve_dataset(project_root, args.dataset))

    pooled = results["metrics"]["pooled"]
    controlled = results["controlled"]
    modality_row = controlled.loc[
        controlled["specification"] == "Grade FE + multimodal share"
    ].iloc[0]
    print("Diagnostics completed successfully.")
    print(f"Shared exam comparisons: {pooled['comparisons']}")
    print(f"Pearson r: {pooled['pearson_r']:.6f}")
    print(f"Spearman rho: {pooled['spearman_rho']:.6f}")
    print(
        "Composition-controlled effect per +10 pp cohort %max: "
        f"{modality_row['effect_per_10pp_human']:.3f} "
        f"[{modality_row['ci_low_per_10pp']:.3f}, "
        f"{modality_row['ci_high_per_10pp']:.3f}]"
    )
    print(f"Outputs: {project_root / 'reproduced' / 'diagnostics'}")


if __name__ == "__main__":
    main()
