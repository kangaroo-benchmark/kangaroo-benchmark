#!/usr/bin/env python3
"""Känguru dataset sanity checker.

Usage:
  uv run python check_dataset.py /path/to/dataset.parquet [--limit N] [--full-images]

Checks performed (summarized):
  - Required columns match evaluator expectations
  - Row ids present and unique
  - Problem statement non-empty
  - Answer letter in {A..E}
  - Either 5 text options or 5 image options available
  - Multimodal flag consistent with presence of images
  - Option contents not duplicated within a row
  - Types/ranges sanity (year, group, problem_number, points)
  - Optional: image bytes decode to valid images (sampled by default)

Exit codes:
  0 = no errors (warnings may still be present)
  1 = one or more errors detected
"""

from __future__ import annotations

import argparse
import base64
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Reuse schema and helpers from the evaluation runner to avoid divergence
try:
    from eval_run import (
        REQUIRED_COLUMNS,
        LETTER_SET,
        coerce_bytes,
        coerce_list_of_bytes,
    )
except Exception:  # Fallbacks if direct import fails for any reason
    REQUIRED_COLUMNS = [
        "id",
        "year",
        "group",
        "points",
        "problem_number",
        "problem_statement",
        "answer",
        "multimodal",
        "sol_A",
        "sol_B",
        "sol_C",
        "sol_D",
        "sol_E",
        "question_image",
        "sol_A_image_bin",
        "sol_B_image_bin",
        "sol_C_image_bin",
        "sol_D_image_bin",
        "sol_E_image_bin",
        "associated_images_bin",
        "language",
    ]
    LETTER_SET = {"A", "B", "C", "D", "E"}

    def coerce_bytes(x: Any) -> Optional[bytes]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, (bytes, bytearray, memoryview)):
            return bytes(x)
        if isinstance(x, str):
            try:
                if len(x) > 16:
                    return base64.b64decode(x, validate=False)
            except Exception:
                return None
        return None

    def coerce_list_of_bytes(x: Any) -> Optional[List[bytes]]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, (list, tuple)):
            out: List[bytes] = []
            for item in x:
                b = coerce_bytes(item)
                if b is not None:
                    out.append(b)
            return out if out else None
        try:
            from collections.abc import Sequence

            if isinstance(x, Sequence) and not isinstance(x, (bytes, bytearray, str)):
                out: List[bytes] = []
                for item in x:
                    b = coerce_bytes(item)
                    if b is not None:
                        out.append(b)
                return out if out else None
        except Exception:
            pass
        return None


OPTION_LETTERS = ["A", "B", "C", "D", "E"]

ALIAS_COLUMN_MAP = {
    "associated_images": "associated_images_bin",
    **{f"sol_{letter}_image": f"sol_{letter}_image_bin" for letter in OPTION_LETTERS},
}

ALLOWED_EXTRA_COLUMNS = set(ALIAS_COLUMN_MAP.keys()) | {"provenance", "quality"}


DATASET_DIR: Optional[str] = None


# Optional heavy validation
try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False


# ---------- Utilities ----------


def apply_column_aliases(df: pd.DataFrame) -> None:
    for raw, canonical in ALIAS_COLUMN_MAP.items():
        if raw in df.columns and canonical not in df.columns:
            df[canonical] = df[raw]
    if "question_image" not in df.columns:
        df["question_image"] = None


def _boolish(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    return None


def _as_int(x: Any) -> Optional[int]:
    try:
        return int(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _is_nonempty_text(x: Any) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    try:
        return bool(str(x).strip())
    except Exception:
        return False


def _decode_image_ok(img_bytes: Optional[bytes]) -> bool:
    if not img_bytes or not PIL_AVAILABLE:
        return bool(img_bytes)
    try:
        im = Image.open(BytesIO(img_bytes))
        im.load()
        return True
    except Exception:
        return False


def _is_nullish(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _iter_blob_candidates(value: Any) -> Iterable[Any]:
    if _is_nullish(value):
        return
    if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
        if value.shape == ():
            yield from _iter_blob_candidates(value.item())
        else:
            for item in value.tolist():
                yield from _iter_blob_candidates(item)
        return
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if seq and all(isinstance(item, int) and 0 <= item <= 255 for item in seq):
            yield seq
            return
        for item in seq:
            yield from _iter_blob_candidates(item)
        return
    yield value


_BASE64_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r"
)


def _looks_like_base64(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 16:
        return False
    return all(ch in _BASE64_CHARS for ch in stripped)


def _decode_data_url(text: str) -> Optional[bytes]:
    if "," not in text:
        return None
    try:
        _, payload = text.split(",", 1)
    except ValueError:
        return None
    try:
        return base64.b64decode(payload, validate=False)
    except Exception:
        return None


@dataclass
class ImageRef:
    data: Optional[bytes] = None
    path: Optional[str] = None
    url: Optional[str] = None
    missing_path: bool = False

    def has_payload(self) -> bool:
        if self.data is not None:
            return True
        if self.path is not None and not self.missing_path:
            return True
        if self.url is not None:
            return True
        return False

    def ensure_bytes(self) -> Optional[bytes]:
        if self.data is not None:
            return self.data
        if self.path and not self.missing_path:
            try:
                with open(self.path, "rb") as handle:
                    self.data = handle.read()
                return self.data
            except Exception:
                self.missing_path = True
        return None


def _make_image_ref(value: Any) -> Optional[ImageRef]:
    if _is_nullish(value):
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return ImageRef(data=data) if data else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("data:"):
            data = _decode_data_url(text)
            if data:
                return ImageRef(data=data)
        lower_text = text.lower()
        if lower_text.startswith("http://") or lower_text.startswith("https://"):
            return ImageRef(url=text)
        if _looks_like_base64(text):
            try:
                data = base64.b64decode(text, validate=False)
                if data:
                    return ImageRef(data=data)
            except Exception:
                pass
        candidate_path = os.path.expanduser(text)
        if DATASET_DIR and not os.path.isabs(candidate_path):
            candidate_path = os.path.abspath(os.path.join(DATASET_DIR, candidate_path))
        looks_like_path = (
            os.path.sep in text
            or text.startswith(".")
            or text.startswith("~")
            or lower_text.endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".bmp",
                    ".webp",
                    ".tiff",
                    ".svg",
                    ".heic",
                    ".heif",
                )
            )
        )
        if looks_like_path or os.path.exists(candidate_path):
            exists = os.path.exists(candidate_path)
            return ImageRef(path=candidate_path, missing_path=not exists)
        # Attempt base64 decoding as a final fallback.
        try:
            data = base64.b64decode(text, validate=False)
            if data:
                return ImageRef(data=data)
        except Exception:
            pass
        return None
    if isinstance(value, (list, tuple)):
        try:
            data = bytes(value)
            if data:
                return ImageRef(data=data)
        except Exception:
            pass
        # Fall back to first valid entry inside the iterable.
        for item in value:
            ref = _make_image_ref(item)
            if ref:
                return ref
        return None
    if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _make_image_ref(value.item())
        for item in value.tolist():
            ref = _make_image_ref(item)
            if ref:
                return ref
        return None
    return None


def collect_image_refs(value: Any) -> List[ImageRef]:
    refs: List[ImageRef] = []
    for candidate in _iter_blob_candidates(value):
        ref = _make_image_ref(candidate)
        if ref:
            refs.append(ref)
    return refs


# ---------- Issue collection ----------


@dataclass
class Issue:
    severity: str  # "error" or "warning"
    code: str
    row_index: Optional[int]
    row_id: Any
    details: str


class IssueCollector:
    def __init__(self) -> None:
        self.issues: List[Issue] = []

    def error(
        self, code: str, row_index: Optional[int], row_id: Any, details: str
    ) -> None:
        self.issues.append(Issue("error", code, row_index, row_id, details))

    def warn(
        self, code: str, row_index: Optional[int], row_id: Any, details: str
    ) -> None:
        self.issues.append(Issue("warning", code, row_index, row_id, details))

    def counts(self) -> Tuple[int, int]:
        e = sum(1 for i in self.issues if i.severity == "error")
        w = sum(1 for i in self.issues if i.severity == "warning")
        return e, w

    def by_code(self) -> Dict[str, List[Issue]]:
        out: Dict[str, List[Issue]] = {}
        for i in self.issues:
            out.setdefault(f"{i.severity}:{i.code}", []).append(i)
        return out


# ---------- Core checks ----------


def check_columns(df: pd.DataFrame, issues: IssueCollector) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [
        c
        for c in df.columns
        if c not in REQUIRED_COLUMNS and c not in ALLOWED_EXTRA_COLUMNS
    ]
    if missing:
        issues.error(
            "missing_columns", None, None, f"Missing required columns: {missing}"
        )
    # Extras are allowed by the evaluator but we surface them for awareness
    if extra:
        issues.warn(
            "extra_columns",
            None,
            None,
            f"Extra columns present (ignored by evaluator): {extra}",
        )


def check_uniqueness(df: pd.DataFrame, issues: IssueCollector) -> None:
    if "id" in df.columns:
        dup_mask = df["id"].duplicated(keep=False)
        if dup_mask.any():
            dup_ids = df.loc[dup_mask, "id"].astype(str).value_counts()
            show = ", ".join([f"{k}×{v}" for k, v in dup_ids.head(10).items()])
            issues.error(
                "duplicate_ids", None, None, f"Duplicate id values (top 10): {show}"
            )

    # (year, group, problem_number) should generally be unique
    for cols in [("year", "group", "problem_number")]:
        if all(c in df.columns for c in cols):
            dup_mask = df.duplicated(subset=list(cols), keep=False)
            if dup_mask.any():
                sample = df.loc[dup_mask, list(cols)].head(10)
                issues.warn(
                    "duplicate_triplets",
                    None,
                    None,
                    f"Non-unique (year, group, problem_number) examples: {sample.to_dict(orient='records')}",
                )


def _letter(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().upper()
    return s if s in LETTER_SET else None


def check_rows(
    df: pd.DataFrame, issues: IssueCollector, *, image_decode_limit: int = 200
) -> None:
    decoded_images = 0

    for idx, row in df.iterrows():
        rid = row.get("id")

        # Problem statement non-empty
        if not _is_nonempty_text(row.get("problem_statement")):
            issues.error(
                "empty_problem_statement",
                idx,
                rid,
                "Problem text is empty or whitespace only",
            )

        # Answer letter
        ans = _letter(row.get("answer"))
        if ans is None:
            issues.error(
                "invalid_answer", idx, rid, "Answer not in {A,B,C,D,E} or missing"
            )

        # Options: text vs image
        text_ok = all(_is_nonempty_text(row.get(f"sol_{L}")) for L in OPTION_LETTERS)
        option_image_refs: Dict[str, List[ImageRef]] = {
            L: collect_image_refs(row.get(f"sol_{L}_image_bin")) for L in OPTION_LETTERS
        }
        option_has_image = {
            L: any(ref.has_payload() for ref in refs)
            for L, refs in option_image_refs.items()
        }
        image_ok = all(option_has_image.values())

        if not (text_ok or image_ok):
            issues.error(
                "options_missing",
                idx,
                rid,
                "Need either 5 non-empty text options or 5 image options",
            )
        elif text_ok and image_ok:
            issues.warn(
                "mixed_options",
                idx,
                rid,
                "Both text and image options present (allowed, FYI)",
            )

        # Ensure the correct option (if given) actually has content
        if ans:
            if text_ok and not _is_nonempty_text(row.get(f"sol_{ans}")):
                issues.error(
                    "answer_missing_text", idx, rid, f"Solution {ans} text is empty"
                )
            if image_ok and not option_has_image.get(ans, False):
                issues.error(
                    "answer_missing_image",
                    idx,
                    rid,
                    f"Solution {ans} image bytes missing",
                )

        # Duplicate options within a row (text)
        if text_ok:
            texts = [str(row.get(f"sol_{L}")).strip() for L in OPTION_LETTERS]
            if len(set(texts)) < 5:
                issues.warn(
                    "duplicate_option_texts",
                    idx,
                    rid,
                    "Some option texts are identical",
                )

        # Multimodal consistency
        mm = _boolish(row.get("multimodal"))
        question_refs = collect_image_refs(row.get("question_image"))
        assoc_refs = collect_image_refs(row.get("associated_images_bin"))

        missing_path_keys: Set[str] = set()

        def record_missing(ref: ImageRef) -> None:
            if ref.path and ref.missing_path and ref.path not in missing_path_keys:
                missing_path_keys.add(ref.path)
                issues.error(
                    "image_path_missing",
                    idx,
                    rid,
                    f"Image path not found or unreadable: {ref.path}",
                )

        for refs in list(option_image_refs.values()) + [question_refs, assoc_refs]:
            for ref in refs:
                record_missing(ref)

        has_option_images = any(option_has_image.values())
        has_assoc_images = any(ref.has_payload() for ref in assoc_refs)
        has_multimodal_source = has_option_images or has_assoc_images
        if mm is True and not has_multimodal_source:
            issues.error(
                "multimodal_flag_without_images",
                idx,
                rid,
                "multimodal=True but no image options or associated images present",
            )
        if mm is False and has_multimodal_source:
            issues.warn(
                "images_with_multimodal_false",
                idx,
                rid,
                "Image answer options or associated images present while multimodal=False",
            )
        if mm is None:
            issues.warn(
                "multimodal_nonboolean",
                idx,
                rid,
                f"multimodal value not boolean: {row.get('multimodal')!r}",
            )

        # Basic type/range sanity
        year = _as_int(row.get("year"))
        if year is None or year < 1998 or year > 2100:
            issues.warn(
                "year_out_of_range", idx, rid, f"Suspicious year: {row.get('year')!r}"
            )

        pn = _as_int(row.get("problem_number"))
        if pn is None or pn <= 0 or pn > 40:
            issues.warn(
                "problem_number_range",
                idx,
                rid,
                f"Suspicious problem_number: {row.get('problem_number')!r}",
            )

        try:
            points = float(row.get("points"))
        except Exception:
            points = None
        if points is None or not (0.0 < points < 10.0):
            issues.warn(
                "points_range", idx, rid, f"Suspicious points: {row.get('points')!r}"
            )
        elif points not in {3.0, 4.0, 5.0}:
            issues.warn(
                "points_nonstandard", idx, rid, f"Non-standard points value: {points}"
            )

        # Language basic sanity (the evaluator defaults to 'de')
        lang_raw = row.get("language")
        lang = str(lang_raw).strip().lower() if _is_nonempty_text(lang_raw) else None
        if lang not in {"de", "en", None}:
            issues.warn(
                "language_unexpected",
                idx,
                rid,
                f"Unexpected language value: {lang_raw!r}",
            )

        # Image decode checks (capped for performance)
        if decoded_images < image_decode_limit:
            decode_refs: List[ImageRef] = []
            decode_refs.extend(question_refs)
            for refs in option_image_refs.values():
                decode_refs.extend(refs)
            decode_refs.extend(assoc_refs)

            for ref in decode_refs:
                if decoded_images >= image_decode_limit:
                    break
                data = ref.ensure_bytes()
                if data is None:
                    record_missing(ref)
                    continue
                decoded_images += 1
                if not _decode_image_ok(data):
                    issues.error(
                        "image_decode_failed",
                        idx,
                        rid,
                        "Image bytes failed to decode via PIL",
                    )


def check_distribution(df: pd.DataFrame, console: Console) -> None:
    """Heuristic per (year, group) distribution check for points and counts."""
    if not all(c in df.columns for c in ("year", "group", "points")):
        return
    try:
        grp = df.groupby(["year", "group"])  # type: ignore[arg-type]
    except Exception:
        return
    table = Table(title="Per (year, group) distribution (sample)")
    table.add_column("year")
    table.add_column("group")
    table.add_column("count", justify="right")
    table.add_column("3pt", justify="right")
    table.add_column("4pt", justify="right")
    table.add_column("5pt", justify="right")
    for (year, group), gdf in list(grp)[
        :20
    ]:  # only first 20 groups to keep output short
        counts = gdf["points"].value_counts(dropna=False)
        c3, c4, c5 = (
            int(counts.get(3.0, 0) + counts.get(3, 0)),
            int(counts.get(4.0, 0) + counts.get(4, 0)),
            int(counts.get(5.0, 0) + counts.get(5, 0)),
        )
        table.add_row(str(year), str(group), str(len(gdf)), str(c3), str(c4), str(c5))
    console.print(table)


def humanize_issues(
    issues: IssueCollector, console: Console, *, show_examples: int = 5
) -> None:
    error_count, warn_count = issues.counts()
    console.print(
        Panel.fit(
            Text(f"Errors: {error_count}   Warnings: {warn_count}", style="bold"),
            title="Summary",
        )
    )

    by_code = issues.by_code()
    # Sort severity: errors first, then warnings; within, by frequency desc
    items = sorted(
        by_code.items(),
        key=lambda kv: (0 if kv[0].startswith("error:") else 1, -len(kv[1]), kv[0]),
    )
    for key, bucket in items:
        severity, code = key.split(":", 1)
        table = Table(title=f"{severity.upper()}: {code} ({len(bucket)})")
        table.add_column("row_index", justify="right")
        table.add_column("row_id")
        table.add_column("details")
        for i in bucket[:show_examples]:
            table.add_row(
                "-" if i.row_index is None else str(i.row_index),
                str(i.row_id),
                i.details,
            )
        if len(bucket) > show_examples:
            table.add_row("…", "…", f"(+{len(bucket) - show_examples} more)")
        console.print(table)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sanity check a Känguru dataset file (.parquet or .jsonl)"
    )
    parser.add_argument("dataset", help="Path to dataset file (.parquet or .jsonl)")
    parser.add_argument(
        "--limit", type=int, default=None, help="Sample N rows for quick checks"
    )
    parser.add_argument(
        "--full-images",
        action="store_true",
        help="Attempt to decode all images (may be slow). Default decodes a sample only.",
    )
    parser.add_argument(
        "--image-decode-limit",
        type=int,
        default=200,
        help="Maximum number of images to decode when not using --full-images",
    )
    args = parser.parse_args(argv)

    console = Console()

    # Resolve dataset path
    path = os.path.expanduser(args.dataset)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        console.print(f"[red]File not found:[/red] {path}")
        return 1

    global DATASET_DIR
    DATASET_DIR = os.path.dirname(path)

    ext = os.path.splitext(path)[1].lower()
    supported_exts = {".parquet", ".jsonl", ".json"}
    if ext not in supported_exts:
        console.print(
            f"[red]Unsupported dataset format (expected .parquet or .jsonl):[/red] {path}"
        )
        return 1

    # Load
    try:
        if ext == ".parquet":
            df = pd.read_parquet(path)
            source_format = "parquet"
        else:
            df = pd.read_json(path, lines=True)
            source_format = "jsonl"
    except Exception as e:
        console.print(f"[red]Failed to read dataset:[/red] {e}")
        return 1

    df = df.copy()
    apply_column_aliases(df)

    if args.limit and args.limit > 0 and len(df) > args.limit:
        df = df.sample(n=args.limit, random_state=0).reset_index(drop=True)

    console.print(
        Panel.fit(
            Text(
                f"Rows: {len(df)}    Columns: {len(df.columns)}    Format: {source_format}",
                style="bold",
            ),
            title="Dataset",
        )
    )

    issues = IssueCollector()
    check_columns(df, issues)
    if len(df) == 0:
        issues.error("empty_dataset", None, None, "Dataset has 0 rows")
    else:
        check_uniqueness(df, issues)
        decode_limit = args.image_decode_limit if not args.full_images else len(df) * 10
        check_rows(df, issues, image_decode_limit=max(0, decode_limit))
        check_distribution(df, console)

    humanize_issues(issues, console)
    error_count, _ = issues.counts()
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
