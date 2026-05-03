from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import pandas as pd
import shutil

try:  # Optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


DEFAULT_DATASET_PATH = Path("datasets") / "dataset_full.edited.corrected.parquet"
REPO_ROOT = Path(__file__).resolve().parents[1]
QUESTION_ID_CANDIDATES = ("question_id", "id")
VALID_PREDICTIONS_DEFAULT = {"A", "B", "C", "D", "E", "DECLINED"}
RETRY_METADATA_COLUMNS = (
    "retry_run_id",
    "retry_timestamp",
    "retry_attempt",
    "retry_source",
)
REPORTS_DIR = Path("reports")
FILLED_SUFFIX = ".filled"
TMP_DIR_NAME = ".retry_tmp"


class RetryToolError(RuntimeError):
    """Raised when the retry tool encounters an unrecoverable error."""


@dataclass
class DetectionStats:
    total_rows: int
    missing_rows: int
    invalid_counts: Dict[str, int]
    question_ids: List[str]


def normalize_prediction(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper()


def coerce_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes)):
        return value
    if isinstance(value, pd.Series):
        if len(value) == 1:
            return coerce_cell(value.iloc[0])
        return [coerce_cell(item) for item in value.tolist()]
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict)):
        try:
            converted = value.tolist()
            return coerce_cell(converted)
        except Exception:
            return value
    if isinstance(value, (list, tuple)):
        return type(value)(coerce_cell(item) for item in value)
    return value


def resolve_question_column(columns: Iterable[str]) -> str:
    for candidate in QUESTION_ID_CANDIDATES:
        if candidate in columns:
            return candidate
    raise RetryToolError(
        f"Results data must contain one of the identifier columns {QUESTION_ID_CANDIDATES}."
    )


def load_results(path: Path) -> Tuple[pd.DataFrame, str]:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, engine="pyarrow")
        fmt = "parquet"
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path)
        fmt = "json"
    else:
        raise RetryToolError(f"Unsupported results format for {path}")
    return df, fmt


def detect_missing_predictions(
    df: pd.DataFrame,
    *,
    valid_predictions: Sequence[str],
) -> DetectionStats:
    column = resolve_question_column(df.columns)
    normalized = df["predicted"].map(normalize_prediction)
    valid_set = {item.upper() for item in valid_predictions}
    invalid_counts: Dict[str, int] = {}

    mask_missing = normalized.isna()
    mask_invalid = ~normalized.isna() & ~normalized.isin(list(valid_set))
    for value, count in normalized[mask_invalid].value_counts().items():
        invalid_counts[str(value)] = int(count)

    missing_mask = mask_missing | mask_invalid
    question_ids = [str(qid) for qid in df.loc[missing_mask, column].tolist()]
    return DetectionStats(
        total_rows=int(len(df)),
        missing_rows=int(missing_mask.sum()),
        invalid_counts=invalid_counts,
        question_ids=question_ids,
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RetryToolError(f"Dataset not found at {path}")
    return pd.read_parquet(path, engine="pyarrow")


def build_subset_dataset(
    dataset: pd.DataFrame,
    question_column: str,
    missing_ids: Sequence[str],
) -> pd.DataFrame:
    if question_column not in dataset.columns:
        raise RetryToolError(
            f"Dataset is missing identifier column '{question_column}'. Available: {list(dataset.columns)}"
        )
    missing_set = set(str(item) for item in missing_ids)
    subset = dataset[dataset[question_column].astype(str).isin(missing_set)]
    subset = subset.sort_values(by=question_column)
    if subset.empty:
        raise RetryToolError(
            "Filtered subset is empty; none of the missing question IDs were found in the dataset."
        )
    return subset


def prompt_confirmation(count: int) -> None:
    print(f"{count} questions will be retried.")
    response = input("Proceed with harness execution? [y/N]: ").strip().lower()
    if response not in {"y", "yes"}:
        raise RetryToolError("Retry aborted by user.")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_run_config(path: Optional[Path]) -> Mapping[str, object]:
    if path is None:
        return {}
    if not path.exists():
        raise RetryToolError(f"Config file not found at {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RetryToolError(
                "PyYAML is required to read YAML configuration files. Install it or provide JSON."
            )
        return yaml.safe_load(path.read_text())  # type: ignore[no-any-return]
    raise RetryToolError(f"Unsupported config extension for {path}")


def discover_default_config(results_path: Path) -> Optional[Path]:
    parent = results_path.parent
    candidate = parent / "config.json"
    if candidate.exists():
        return candidate
    return None


def parse_run_metadata(
    config: Mapping[str, object],
) -> Tuple[Optional[Mapping[str, object]], Optional[str]]:
    args = config.get("args") if config else None
    args = args if isinstance(args, MutableMapping) else None
    model_id = None
    if args and isinstance(args.get("model"), str):
        model_id = args["model"]
    elif isinstance(config.get("model"), MutableMapping):
        model_entry = config["model"]  # type: ignore[assignment]
        if isinstance(model_entry, Mapping):
            model_id = (
                model_entry.get("id")
                if isinstance(model_entry.get("id"), str)
                else None
            )
    return args, model_id


def extend_with_option(
    cmd: List[str],
    flag: str,
    value: object,
) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if item is None:
                continue
            cmd.append(flag)
            cmd.append(str(item))
        return
    cmd.append(flag)
    cmd.append(str(value))


def build_default_harness_command(
    config_args: Mapping[str, object],
    *,
    dataset_path: Path,
    output_dir: Path,
) -> List[str]:
    cmd = ["uv", "run", "python", "eval_run.py"]

    extend_with_option(cmd, "--dataset", str(dataset_path))
    extend_with_option(cmd, "--output_dir", str(output_dir))

    special_flags = {
        "live_dashboard": ("--live-dashboard", "--no-live-dashboard"),
        "dashboard_refresh_hz": "--dashboard-refresh-hz",
        "events_jsonl": "--events-jsonl",
        "recent_items": "--recent-items",
        "ui_compact": "--ui-compact",
        "text_only": "--text-only",
    }

    for key, value in config_args.items():
        if key in {
            "dataset",
            "output_dir",
            "reasoning",
            "internal_effort",
            "worker_count",
        }:
            continue
        if value is None:
            continue
        if key == "live_dashboard":
            enable_flag, disable_flag = special_flags["live_dashboard"]
            if value is True:
                cmd.append(enable_flag)
            elif value is False:
                cmd.append(disable_flag)
            continue
        flag_name = special_flags.get(key)
        if isinstance(flag_name, tuple):
            # only live_dashboard handled above
            flag = flag_name[0]
        else:
            flag = flag_name if isinstance(flag_name, str) else f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        extend_with_option(cmd, flag, value)
    return cmd


def run_harness_command(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    ensure_directory(stdout_path.parent)
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_file,
        stderr_path.open("w", encoding="utf-8") as stderr_file,
    ):
        result = subprocess.run(
            list(command),
            cwd=str(cwd),
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    if result.returncode != 0:
        raise RetryToolError(
            f"Harness command exited with code {result.returncode}. See logs at {stdout_path} and {stderr_path}."
        )


def locate_retry_run_dir(base_dir: Path, before: Sequence[Path]) -> Path:
    after = [path for path in base_dir.iterdir() if path.is_dir()]
    before_set = {p.resolve() for p in before}
    new_dirs = [p for p in after if p.resolve() not in before_set]
    if not new_dirs:
        raise RetryToolError(f"No new evaluation run directory found under {base_dir}")
    if len(new_dirs) == 1:
        return new_dirs[0]
    return max(new_dirs, key=lambda path: path.stat().st_mtime)


def load_retry_outputs(
    run_dir: Path,
    *,
    expected_ids: Sequence[str],
    valid_predictions: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    parquet_path = run_dir / "results.parquet"
    json_path = run_dir / "results.json"
    if parquet_path.exists():
        retry_df = pd.read_parquet(parquet_path, engine="pyarrow")
    elif json_path.exists():
        retry_df = pd.read_json(json_path)
    else:
        raise RetryToolError(
            f"Retry run at {run_dir} did not produce results.parquet or results.json"
        )

    question_column = resolve_question_column(retry_df.columns)
    retry_df = retry_df.copy()
    retry_df["retry_run_id"] = run_dir.name
    timestamp = run_dir.name.split("_", 1)[0]
    retry_df["retry_timestamp"] = timestamp
    retry_df["retry_attempt"] = 1
    retry_df["retry_source"] = "same_model"

    expected_set = {str(item) for item in expected_ids}
    actual_set = {str(item) for item in retry_df[question_column].astype(str)}
    missing_from_retry = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)

    normalized = retry_df["predicted"].map(normalize_prediction)
    valid_set = {item.upper() for item in valid_predictions}
    unresolved_ids = [
        str(qid)
        for qid, is_valid in zip(
            retry_df[question_column].astype(str), normalized.isin(list(valid_set))
        )
        if not is_valid
    ]

    issues = {
        "missing_from_retry": missing_from_retry,
        "unexpected_retry_ids": unexpected,
        "invalid_retry_predictions": unresolved_ids,
    }
    return retry_df, issues


def merge_results(
    original: pd.DataFrame,
    retry_df: pd.DataFrame,
    *,
    question_column: str,
    missing_ids: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    merged = original.copy()
    merged["predicted_original"] = original["predicted"]
    merged["predicted_filled"] = pd.NA
    merged["filled_by_retry"] = False
    merged["filled_run_id"] = None
    merged["filled_timestamp"] = None
    merged["retry_attempt"] = None
    merged["retry_source"] = None

    retry_index = retry_df.set_index(retry_df[question_column].astype(str))
    missing_order = [str(item) for item in missing_ids]

    unresolved: List[str] = []
    for qid in missing_order:
        if qid not in retry_index.index:
            unresolved.append(qid)
            continue
        retry_row = retry_index.loc[qid]
        if isinstance(retry_row, pd.DataFrame):
            retry_row = retry_row.iloc[0]
        retry_pred = normalize_prediction(retry_row["predicted"])
        if retry_pred is None:
            unresolved.append(qid)
            continue

        row_mask = merged[question_column].astype(str) == qid
        if not row_mask.any():
            unresolved.append(qid)
            continue
        target_indices = merged.index[row_mask]
        for column in retry_df.columns:
            if column == question_column:
                continue
            if column in RETRY_METADATA_COLUMNS:
                continue
            if column in merged.columns:
                for idx in target_indices:
                    merged.at[idx, column] = coerce_cell(retry_row[column])
        for idx in target_indices:
            merged.at[idx, "predicted"] = retry_row["predicted"]
            merged.at[idx, "predicted_filled"] = retry_row["predicted"]
            merged.at[idx, "filled_by_retry"] = True
            merged.at[idx, "filled_run_id"] = retry_row.get("retry_run_id")
            merged.at[idx, "filled_timestamp"] = retry_row.get("retry_timestamp")
            merged.at[idx, "retry_attempt"] = retry_row.get("retry_attempt")
            merged.at[idx, "retry_source"] = retry_row.get("retry_source")
    for column in merged.select_dtypes(include="object").columns:
        merged[column] = merged[column].map(coerce_cell)
    return merged, unresolved


def detect_after_merge(df: pd.DataFrame, valid_predictions: Sequence[str]) -> int:
    stats = detect_missing_predictions(df, valid_predictions=valid_predictions)
    return stats.missing_rows


def write_outputs(
    df: pd.DataFrame,
    *,
    run_dir: Path,
    base_name: str,
    output_format: str,
    force: bool,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    if output_format in {"parquet", "both"}:
        path = run_dir / f"{base_name}{FILLED_SUFFIX}.parquet"
        if path.exists() and not force:
            raise RetryToolError(f"{path} already exists. Use --force to overwrite.")
        df.to_parquet(path, engine="pyarrow", index=False)
        outputs["parquet"] = path
    if output_format in {"json", "both"}:
        path = run_dir / f"{base_name}{FILLED_SUFFIX}.json"
        if path.exists() and not force:
            raise RetryToolError(f"{path} already exists. Use --force to overwrite.")
        records = df.to_dict(orient="records")
        path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        outputs["json"] = path
    return outputs


def write_report(
    *,
    stats_before: DetectionStats,
    missing_after: int,
    retry_run_dir: Path,
    retry_issues: Mapping[str, Sequence[str]],
    output_paths: Mapping[str, Path],
    runtime_seconds: float,
    results_path: Path,
) -> Path:
    ensure_directory(REPORTS_DIR)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"retry_{retry_run_dir.name}_{timestamp}.json"
    payload = {
        "results_path": str(results_path),
        "retry_run_dir": str(retry_run_dir),
        "total_rows": stats_before.total_rows,
        "missing_before": stats_before.missing_rows,
        "missing_after": missing_after,
        "retry_issues": retry_issues,
        "outputs": {fmt: str(path) for fmt, path in output_paths.items()},
        "runtime_seconds": runtime_seconds,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry and merge missing predictions in evaluation results."
    )
    parser.add_argument(
        "--results", required=True, help="Path to results.parquet or results.json"
    )
    parser.add_argument(
        "--dataset",
        help=f"Path to base dataset parquet (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--harness-cmd", help="Shell command template for running the harness."
    )
    parser.add_argument("--config", help="Path to evaluation config (JSON or YAML).")
    parser.add_argument(
        "--output-dir", help="Directory for merged outputs (default: run directory)."
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "json", "both"),
        default="both",
        help="Output formats to produce (default: both).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Analyze missing predictions only."
    )
    parser.add_argument(
        "--retain-temp", action="store_true", help="Keep generated temp files."
    )
    parser.add_argument(
        "--valid-answer",
        action="append",
        dest="valid_answers",
        help="Additional valid answers (defaults include A-E and DECLINED).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing filled outputs."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompt.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)

    results_path = Path(args.results).expanduser().resolve()
    if not results_path.exists():
        raise RetryToolError(f"Results file not found at {results_path}")

    start_time = time.perf_counter()
    results_df, _ = load_results(results_path)
    valid_predictions = list(VALID_PREDICTIONS_DEFAULT)
    if args.valid_answers:
        valid_predictions.extend([item.upper() for item in args.valid_answers])

    stats_before = detect_missing_predictions(
        results_df, valid_predictions=valid_predictions
    )
    question_column = resolve_question_column(results_df.columns)

    print(
        f"Detected {stats_before.missing_rows} missing/invalid predictions out of {stats_before.total_rows} rows."
    )
    if stats_before.invalid_counts:
        print("Invalid prediction breakdown:")
        for value, count in stats_before.invalid_counts.items():
            print(f"  {value}: {count}")

    if stats_before.missing_rows == 0:
        print("No missing predictions detected; nothing to do.")
        return 0

    if args.dataset:
        dataset_path = Path(args.dataset).expanduser().resolve()
        if not dataset_path.exists():
            raise RetryToolError(
                f"Provided dataset path does not exist: {dataset_path}"
            )
    else:
        candidate_paths = [
            REPO_ROOT / DEFAULT_DATASET_PATH,
            results_path.parent / DEFAULT_DATASET_PATH,
        ]
        if len(results_path.parents) > 1:
            candidate_paths.append(results_path.parents[1] / DEFAULT_DATASET_PATH)
        dataset_path = next((path for path in candidate_paths if path.exists()), None)
        if dataset_path is None:
            raise RetryToolError(
                f"Could not locate dataset. Provide --dataset or place it at {REPO_ROOT / DEFAULT_DATASET_PATH}."
            )
    dataset_df = load_dataset(dataset_path)

    subset = build_subset_dataset(
        dataset_df, question_column, stats_before.question_ids
    )

    run_dir = (
        results_path.parent
        if not args.output_dir
        else Path(args.output_dir).expanduser().resolve()
    )
    if not run_dir.exists():
        ensure_directory(run_dir)

    tmp_root = run_dir / TMP_DIR_NAME
    if not args.retain_temp and tmp_root.exists():
        for child in sorted(tmp_root.iterdir()):
            if child.is_dir():
                try:
                    shutil.rmtree(child)
                except Exception:
                    continue
    ensure_directory(tmp_root)
    temp_dir = Path(tempfile.mkdtemp(prefix="retry_", dir=tmp_root))

    subset_path = temp_dir / "retry_subset.parquet"
    subset.to_parquet(subset_path, engine="pyarrow", index=False)
    print(f"Subset dataset written to {subset_path}")

    if args.dry_run:
        print("Dry run requested; skipping harness execution.")
        if not args.retain_temp:
            try:
                shutil.rmtree(temp_dir)
                print("Temporary retry assets removed.")
                if tmp_root.exists() and not any(tmp_root.iterdir()):
                    tmp_root.rmdir()
            except Exception:
                pass
        return 0

    if not args.yes:
        prompt_confirmation(len(subset))

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else discover_default_config(results_path)
    )
    config = load_run_config(config_path) if config_path else {}
    config_args, model_id = parse_run_metadata(config)
    command: List[str]
    output_base = temp_dir / "harness_output"
    ensure_directory(output_base)
    before_dirs = list(output_base.iterdir())

    if args.harness_cmd:
        context = {
            "dataset": subset_path,
            "output_dir": output_base,
            "config": config_path or "",
            "results_dir": results_path.parent,
            "model": model_id or "",
            "repo_root": REPO_ROOT,
        }
        formatted = args.harness_cmd.format(**{k: str(v) for k, v in context.items()})
        command = shlex.split(formatted)
    else:
        if not config_args:
            raise RetryToolError(
                "Unable to determine harness arguments. Provide --config or use --harness-cmd."
            )
        command = build_default_harness_command(
            config_args,
            dataset_path=subset_path,
            output_dir=output_base,
        )

    stdout_log = temp_dir / "harness_stdout.log"
    stderr_log = temp_dir / "harness_stderr.log"
    printable = " ".join(shlex.quote(token) for token in command)
    print(f"Running harness command: {printable}")
    run_harness_command(
        command, cwd=REPO_ROOT, stdout_path=stdout_log, stderr_path=stderr_log
    )

    retry_run_dir = locate_retry_run_dir(output_base, before_dirs)
    print(f"Retry outputs located at {retry_run_dir}")

    retry_df, retry_issues = load_retry_outputs(
        retry_run_dir,
        expected_ids=stats_before.question_ids,
        valid_predictions=valid_predictions,
    )
    merged_df, unresolved = merge_results(
        results_df,
        retry_df,
        question_column=question_column,
        missing_ids=stats_before.question_ids,
    )
    retry_issues = dict(retry_issues)
    if unresolved:
        retry_issues.setdefault("unresolved_after_merge", [])
        retry_issues["unresolved_after_merge"].extend(sorted(set(unresolved)))

    missing_after = detect_after_merge(merged_df, valid_predictions=valid_predictions)

    outputs = write_outputs(
        merged_df,
        run_dir=run_dir,
        base_name=results_path.stem,
        output_format=args.format,
        force=args.force,
    )

    report_path = write_report(
        stats_before=stats_before,
        missing_after=missing_after,
        retry_run_dir=retry_run_dir,
        retry_issues=retry_issues,
        output_paths=outputs,
        runtime_seconds=time.perf_counter() - start_time,
        results_path=results_path,
    )
    print(
        f"Merged outputs written: {', '.join(str(path) for path in outputs.values())}"
    )
    print(f"Summary report written to {report_path}")
    if not args.retain_temp:
        try:
            shutil.rmtree(temp_dir)
            if tmp_root.exists() and not any(tmp_root.iterdir()):
                tmp_root.rmdir()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RetryToolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
