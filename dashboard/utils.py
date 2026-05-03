"""Utility helpers for the dashboard backend."""

from __future__ import annotations

import base64
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import orjson

ISO_FORMAT = "%Y%m%d_%H%M%S"


def load_json(path: Path) -> Any:
    """Load JSON via orjson for performance."""
    with path.open("rb") as f:
        return orjson.loads(f.read())


def load_jsonl(path: Path) -> Iterator[Any]:
    """Yield JSON objects from a JSONL file."""
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def dumps_json(data: Any) -> str:
    return orjson.dumps(data).decode("utf-8")


def parse_run_timestamp(run_id: str) -> Optional[dt.datetime]:
    """Extract the leading timestamp from a run id if present."""
    parts = run_id.split("_")
    if len(parts) < 2:
        return None
    ts_part = "_".join(parts[:2])
    try:
        return dt.datetime.strptime(ts_part, ISO_FORMAT)
    except ValueError:
        return None


def _parse_numeric_range(value: str) -> Tuple[Optional[int], Optional[int]]:
    if not value:
        return (None, None)
    parts = value.split("-", 1)
    try:
        start = int(parts[0].strip())
    except (ValueError, AttributeError):
        start = None
    if len(parts) == 1:
        return (start, start)
    try:
        end = int(parts[1].strip())
    except (ValueError, AttributeError):
        end = start
    return (start, end)


def grade_group_sort_key(value: Optional[str]) -> Tuple[int, int, str]:
    """Sort key that respects numeric ranges like ``3-4`` or ``11-13``."""
    if value is None:
        return (10**9, 10**9, "")
    text = str(value)
    start, end = _parse_numeric_range(text)
    if start is None:
        return (10**9, 10**9, text)
    if end is None:
        end = start
    return (start, end, text)


def load_models_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read ``models.json`` once and expose a mapping by id."""
    if not path.exists():
        return {}
    data = load_json(path)
    result: Dict[str, Dict[str, Any]] = {}
    for entry in data.get("models", []):
        model_id = entry.get("id")
        if model_id:
            result[str(model_id)] = entry
    return result


def slug_to_label(model_id: str, registry: Dict[str, Dict[str, Any]]) -> Optional[str]:
    info = registry.get(model_id)
    if not info:
        return None
    label = info.get("label")
    return str(label) if label else None


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def humanize_timedelta(milliseconds: Optional[float]) -> Optional[str]:
    if milliseconds is None:
        return None
    seconds = milliseconds / 1000.0
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def build_data_url(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def guess_mime_from_bytes(data: bytes) -> str:
    # Basic sniffing based on magic numbers used by PNG/JPEG.
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    return "application/octet-stream"


def normalize_list(value: Optional[Iterable[Any]]) -> List[Any]:
    if not value:
        return []
    return [v for v in value if v is not None]


def paginate(sequence: List[Any], page: int, page_size: int) -> List[Any]:
    start = max(page - 1, 0) * page_size
    end = start + page_size
    return sequence[start:end]


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
