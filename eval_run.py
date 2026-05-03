import argparse
import asyncio
import base64
import dataclasses
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from PIL import Image
from tqdm import tqdm

from console.metrics import Aggregator, UsageEvent
from score_utils import score_question, start_points_for_group

try:
    from console.dashboard import Dashboard

    DASHBOARD_AVAILABLE = True
except Exception:  # pragma: no cover - rich not installed
    Dashboard = None  # type: ignore
    DASHBOARD_AVAILABLE = False


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
SELF_DIRECTED_MODE = "self-directed"
DECLINED_TOKEN = "DECLINED"

FINAL_ANSWER_PATTERN = re.compile(
    r"(?im)^\s*(?:final\s*answer|antwort)\s*[:：\-–—]?\s*([^\n]+?)\s*$"
)


def _normalize_phrase(value: str) -> str:
    normalized = value.lower()
    normalized = normalized.replace("’", "'").replace("“", '"').replace("”", '"')
    normalized = re.sub(r"[^a-z0-9äöüß ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


DECLINE_ALIAS_RAW = [
    "declined",
    "i choose not to answer",
    "i choose to not answer",
    "choose not to answer",
    "i decline to answer",
    "i decline",
    "i cannot determine with confidence",
    "cannot determine with confidence",
    "i cannot determine the answer",
    "unable to determine",
    "ich entscheide mich, nicht zu antworten",
    "ich wähle, nicht zu antworten",
    "antwort verweigern",
    "ich verweigere die antwort",
    "nicht mit ausreichender sicherheit bestimmbar",
    "kann nicht mit ausreichender sicherheit bestimmen",
    "ich weiß es nicht",
    "weiß es nicht",
    "keine ahnung",
]

DECLINE_ALIAS_NORMALIZED = {_normalize_phrase(item) for item in DECLINE_ALIAS_RAW}

DECLINE_SEARCH_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bi\s+choose\s+not\s+to\s+answer\b",
        r"\bchoose\s+not\s+to\s+answer\b",
        r"\bi\s+decline\s+to\s+answer\b",
        r"\bi\s+decline\b",
        r"\bcannot\s+determine\s+(?:with\s+)?confidence\b",
        r"\bunable\s+to\s+determine\b",
        r"\bich\s+entscheide\s+mich,\s+nicht\s+zu\s+antworten\b",
        r"\bich\s+wähle,\s+nicht\s+zu\s+antworten\b",
        r"\bantwort\s+verweigern\b",
        r"\bich\s+verweigere\s+die\s+antwort\b",
        r"\bnicht\s+mit\s+ausreichender\s+sicherheit\s+bestimmbar\b",
        r"\bkeine\s+ahnung\b",
        r"\bich\s+weiß\s+es\s+nicht\b",
    ]
]


def _clean_answer_token(raw: str) -> str:
    token = raw.strip().strip("`\"'“”‘’")
    token = re.sub(r"[\s\.\!\?؛؛。,…]+$", "", token).strip()
    return token


def _is_decline_token(candidate: str) -> bool:
    normalized = _normalize_phrase(candidate)
    return bool(normalized) and normalized in DECLINE_ALIAS_NORMALIZED


def _text_contains_decline(text: str) -> bool:
    if _is_decline_token(text):
        return True
    for pattern in DECLINE_SEARCH_PATTERNS:
        if pattern.search(text):
            return True
    return False


DEFAULT_RETRY_MAX_TOKENS = 256

# HTTP statuses considered transient/retryable at either transport- or row-level.
# Includes timeouts and common CDN gateway errors in addition to rate limit and 5xx.
RETRYABLE_STATUS_CODES = {
    408,  # Request Timeout
    425,  # Too Early (ask client to retry)
    429,  # Too Many Requests
    500,
    502,
    503,
    504,  # Typical server/gateway errors
    520,
    521,
    522,
    523,
    524,  # Common CDN edge/gateway errors
}


def default_worker_count() -> int:
    cpu = os.cpu_count() or 4
    return max(2, min(8, cpu))


@dataclass
class ModelInfo:
    id: str
    label: Optional[str] = None
    supports_vision: bool = False
    min_request_interval: Optional[float] = None
    supports_json_response_format: bool = False


@dataclass
class RowRecord:
    id: Any
    year: Any
    group: Any
    problem_number: Any
    language: Any
    multimodal: Any
    points: Any
    answer: Optional[str]
    predicted: Optional[str]
    is_correct: Optional[bool]
    points_earned: Optional[float]
    reasoning_mode: str
    latency_ms: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    # Extra usage details when available from providers (e.g., OpenRouter)
    reasoning_tokens: Optional[int] = None
    explicit_reasoning_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    audio_prompt_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    rationale: Optional[str] = None
    raw_text_response: Optional[str] = None
    generation_id: Optional[str] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None


@dataclass
class WorkerOutcome:
    record: Optional[RowRecord]
    raw_entries: List[Dict[str, Any]]
    failure_entry: Optional[Dict[str, Any]]
    skipped: bool
    fail_fast_trigger: bool
    attempts: int = 1
    status_code: Optional[int] = None
    row_id: Optional[Any] = None


class AdaptiveRateLimiter:
    def __init__(self, initial_interval: float = 0.0) -> None:
        self._min_interval = max(0.0, initial_interval)
        self._last_timestamp = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                target = self._last_timestamp + self._min_interval
                if now >= target:
                    self._last_timestamp = now
                    return
                wait_time = target - now
            await asyncio.sleep(wait_time)

    async def record_throttle(self) -> None:
        async with self._lock:
            if self._min_interval == 0.0:
                self._min_interval = 0.5
            else:
                self._min_interval = min(self._min_interval * 2.0, 30.0)

    async def record_success(self) -> None:
        async with self._lock:
            if self._min_interval == 0.0:
                return
            self._min_interval = max(self._min_interval * 0.8, 0.0)


def read_models_registry(path: str) -> Dict[str, ModelInfo]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    models = {}
    for m in data.get("models", []):
        min_interval: Optional[float] = None
        rate_limit = m.get("rate_limit")
        if isinstance(rate_limit, dict):
            raw_min = rate_limit.get("min_interval_seconds")
            if isinstance(raw_min, (int, float)) and raw_min >= 0:
                min_interval = float(raw_min)
            raw_rps = rate_limit.get("requests_per_second")
            if (
                min_interval is None
                and isinstance(raw_rps, (int, float))
                and raw_rps > 0
            ):
                min_interval = 1.0 / float(raw_rps)
            raw_rpm = rate_limit.get("requests_per_minute")
            if (
                min_interval is None
                and isinstance(raw_rpm, (int, float))
                and raw_rpm > 0
            ):
                min_interval = 60.0 / float(raw_rpm)
        elif isinstance(rate_limit, (int, float)) and rate_limit > 0:
            min_interval = 1.0 / float(rate_limit)

        info = ModelInfo(
            id=m.get("id"),
            label=m.get("label"),
            supports_vision=bool(m.get("supports_vision", False)),
            supports_json_response_format=bool(
                m.get("supports_json_response_format", False)
            ),
            min_request_interval=min_interval,
        )
        models[info.id] = info
    return models


def validate_dataset_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def coerce_bytes(x: Any) -> Optional[bytes]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    # Sometimes parquet may roundtrip as Python "Binary" type; attempt fallback
    if isinstance(x, str):
        # Try base64 decode if it looks like base64; otherwise treat as no-bytes
        try:
            # Heuristic: ignore tiny strings
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
    # Sometimes arrow returns numpy arrays or other sequences
    try:
        from collections.abc import Sequence

        if isinstance(x, Sequence) and not isinstance(x, (bytes, bytearray, str)):
            out = []
            for item in x:
                b = coerce_bytes(item)
                if b is not None:
                    out.append(b)
            return out if out else None
    except Exception:
        pass
    # Explicit numpy.ndarray handling (object arrays of bytes)
    try:
        import numpy as np  # type: ignore

        if isinstance(x, np.ndarray):
            out: List[bytes] = []
            # Convert to list to avoid numpy scalars
            for item in x.tolist():
                b = coerce_bytes(item)
                if b is not None:
                    out.append(b)
            return out if out else None
    except Exception:
        pass
    return None


def pil_from_bytes(img_bytes: bytes) -> Optional[Image.Image]:
    """Open image bytes without forcing colorspace.

    Preserve alpha and original mode; downstream encoders decide the right
    output format and perform conversion only when needed (e.g. JPEG).
    """
    try:
        img = Image.open(BytesIO(img_bytes))
        # Ensure the image is actually loaded to catch decoding errors early
        img.load()
        return img
    except Exception:
        return None


def image_to_data_url(
    img: Image.Image,
    *,
    prefer_format: Optional[str] = None,
    max_dim: int = 1024,
    jpeg_quality: int = 85,
) -> Tuple[str, str]:
    # Downscale while preserving aspect ratio; never upscale.
    if max(img.size) > max_dim:
        img = img.copy()
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    # Choose encoding format based on alpha channel and preference
    fmt = (prefer_format or "").upper().strip()
    has_alpha = getattr(img, "mode", "").upper() in {"RGBA", "LA"}
    if not fmt:
        fmt = "PNG" if has_alpha else "JPEG"
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in {"PNG", "JPEG"}:
        fmt = "PNG" if has_alpha else "JPEG"
    mime = "image/png" if fmt == "PNG" else "image/jpeg"

    # Encode
    buf = BytesIO()
    save_kwargs = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs.update(
            {
                "quality": int(max(50, min(100, jpeg_quality))),
                "optimize": True,
                "progressive": True,
            }
        )
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    else:
        # For PNG, enable optimization; avoid palette conversion to preserve details.
        save_kwargs.update({"optimize": True})
    img.save(buf, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}", fmt


def build_messages(
    row: Dict[str, Any],
    *,
    encoded_images: Dict[str, Optional[str]],
    image_detail: str = "auto",
) -> List[Dict[str, Any]]:
    language = (row.get("language") or "de").lower()
    is_de = language == "de"

    if is_de:
        system_lines = [
            "Du bist ein hilfreicher Assistent für Multiple-Choice-Aufgaben.",
            "Bewertung: Eine korrekte Antwort erhält die vollen Punkte der Aufgabe; eine falsche oder nicht eindeutig auswertbare Endantwort zieht ein Viertel der Aufgabenpunkte ab; 'Ich entscheide mich, nicht zu antworten.' ergibt 0 Punkte ohne Abzug.",
            "Wähle die beste Option (A–E). Wenn die richtige Option nicht mit ausreichender Sicherheit bestimmbar ist oder die Aufgabe mehrdeutig ist, kannst du ausdrücklich mit 'Ich entscheide mich, nicht zu antworten.' antworten. Begründungen sind optional.",
            "Füge am Ende eine einzelne Zeile an: Final answer: A|B|C|D|E|Declined.",
        ]
        final_instruction = "Erinnere dich: Beende deine Antwort mit einer Zeile in diesem Format – Final answer: A|B|C|D|E|Declined."
    else:
        system_lines = [
            "You are a helpful assistant for multiple-choice tasks.",
            "Scoring: a correct answer earns the task's full points; an incorrect or unparseable final answer subtracts one quarter of the task's points; choosing not to answer earns 0 points with no penalty.",
            "Select the best option (A–E). If the correct option cannot be determined with sufficient confidence or the question is ambiguous, you may explicitly reply 'I choose not to answer.' Reasoning is optional.",
            "Add a single line at the end: Final answer: A|B|C|D|E|Declined.",
        ]
        final_instruction = "Remember to finish with a line in this format – Final answer: A|B|C|D|E|Declined."

    sys_text = "\n".join(system_lines)

    content_parts: List[Dict[str, Any]] = []

    # Question text
    q_label = "Frage:" if is_de else "Question:"
    content_parts.append(
        {"type": "text", "text": f"{q_label} {row['problem_statement']}"}
    )

    # Question image
    if encoded_images.get("question"):
        q_img_label = "Fragebild:" if is_de else "Question image:"
        content_parts.append({"type": "text", "text": q_img_label})
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": encoded_images["question"],
                    "detail": image_detail,
                },
            }
        )

    # Associated images
    assoc = encoded_images.get("assoc_list") or []
    for i, url in enumerate(assoc, start=1):
        if url:
            a_label = ("Zusatzbild" if is_de else "Additional image") + f" {i}:"
            content_parts.append({"type": "text", "text": a_label})
            content_parts.append(
                {"type": "image_url", "image_url": {"url": url, "detail": image_detail}}
            )

    # Options A..E
    choice_hdr = "Antwortmöglichkeiten:" if is_de else "Answer choices:"
    content_parts.append({"type": "text", "text": choice_hdr})
    for letter in ["A", "B", "C", "D", "E"]:
        opt_text = row.get(f"sol_{letter}") or ""
        content_parts.append({"type": "text", "text": f"{letter}) {opt_text}"})
        url = encoded_images.get(f"opt_{letter}")
        if url:
            lbl = ("Option" if not is_de else "Option") + f" {letter} Bild:"
            content_parts.append({"type": "text", "text": lbl})
            content_parts.append(
                {"type": "image_url", "image_url": {"url": url, "detail": image_detail}}
            )

    # Instruction last
    content_parts.append({"type": "text", "text": final_instruction})

    messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_text}]
    messages.append({"role": "user", "content": content_parts})
    return messages


def ensure_output_dir(base_dir: str, model_id: str) -> Tuple[str, str]:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_model = model_id.replace("/", "-")
    run_dir = os.path.join(base_dir, f"{ts}_{safe_model}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, ts


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                json_payload = item.get("json")
                if json_payload is not None:
                    try:
                        parts.append(json.dumps(json_payload, ensure_ascii=False))
                    except Exception:
                        parts.append(str(json_payload))
                    continue

                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue

                nested = item.get("content")
                if isinstance(nested, (str, list)):
                    nested_text = normalize_message_content(nested)
                    if nested_text:
                        parts.append(nested_text)
                    continue

                # Fall back to serialising remaining primitive entries for debugging
                for key in ("tool_calls", "arguments"):
                    value = item.get(key)
                    if value is not None:
                        try:
                            parts.append(json.dumps(value, ensure_ascii=False))
                        except Exception:
                            parts.append(str(value))
                        break
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        json_payload = content.get("json")
        if json_payload is not None:
            try:
                return json.dumps(json_payload, ensure_ascii=False)
            except Exception:
                return str(json_payload)
        nested = content.get("content")
        if isinstance(nested, (str, list)):
            return normalize_message_content(nested)
    return str(content)


def write_jsonl_line(handle, obj: Dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    handle.flush()


def load_env_file(path: str = ".env") -> None:
    if not path:
        return
    if not os.path.exists(path) or not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                os.environ.setdefault(key, value)
    except Exception:
        pass


def resolve_dataset_path(raw_path: str) -> str:
    expanded = os.path.expanduser(raw_path)
    if not os.path.isabs(expanded):
        expanded = os.path.abspath(expanded)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Dataset file not found: {expanded}")
    if not os.path.isfile(expanded):
        raise ValueError(f"Dataset path is not a file: {expanded}")
    if not expanded.lower().endswith(".parquet"):
        raise ValueError("Dataset must be a .parquet file")
    return expanded


def parse_answer_from_text(
    text: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # returns (answer, rationale, parse_warning)
    if not text or not text.strip():
        return None, None, "empty_response"

    def interpret_token(raw_value: str) -> Tuple[Optional[str], Optional[str]]:
        cleaned = _clean_answer_token(raw_value)
        upper = cleaned.upper()
        if upper in LETTER_SET:
            return upper, None
        if _is_decline_token(cleaned):
            return DECLINED_TOKEN, "declined_explicit"
        return None, None

    final_matches = list(FINAL_ANSWER_PATTERN.finditer(text))
    if final_matches:
        candidate_raw = final_matches[-1].group(1)
        token, warn = interpret_token(candidate_raw)
        if token:
            return token, None, warn

    def try_json_block(s: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        try:
            data = json.loads(s)
        except Exception:
            return None, None, None
        if not isinstance(data, dict):
            return None, None, None
        ans_value = data.get("answer")
        if not isinstance(ans_value, str):
            return None, None, None
        token, warn = interpret_token(ans_value)
        if token is None:
            return None, None, None
        rat_value = data.get("reason")
        rationale = rat_value if isinstance(rat_value, str) else None
        return token, rationale, warn

    ans_json, rat_json, warn_json = try_json_block(text)
    if ans_json:
        return ans_json, rat_json, warn_json

    json_match = re.search(r"\{[\s\S]*?\}", text)
    if json_match:
        ans_extracted, rat_extracted, warn_extracted = try_json_block(
            json_match.group(0)
        )
        if ans_extracted:
            if warn_extracted is None:
                warn_extracted = "json_extracted"
            return ans_extracted, rat_extracted, warn_extracted

    if _text_contains_decline(text):
        return DECLINED_TOKEN, None, "declined_phrase"

    match = re.search(r"\b([A-Ea-e])\b", text)
    if match:
        return match.group(1).upper(), None, "regex_fallback"

    return None, None, "no_parse"


async def request_with_retries(
    client: httpx.AsyncClient,
    limiter: AdaptiveRateLimiter,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    on_status: Optional[Callable[[int], None]] = None,
    on_throttle: Optional[Callable[[], None]] = None,
) -> Tuple[httpx.Response, float]:
    # Slightly extended deterministic backoff to better absorb rare 429s.
    delays = [0.5, 1.0, 2.0, 4.0]
    last_exc: Optional[Exception] = None
    start = time.perf_counter()
    for attempt in range(len(delays) + 1):
        await limiter.acquire()
        try:
            resp = await client.post(url, headers=headers, json=payload)
        except Exception as exc:
            last_exc = exc
            if attempt < len(delays):
                await asyncio.sleep(delays[attempt])
                continue
            raise

        if on_status is not None:
            try:
                on_status(resp.status_code)
            except Exception:
                pass

        if resp.status_code in RETRYABLE_STATUS_CODES:
            if on_throttle is not None:
                try:
                    on_throttle()
                except Exception:
                    pass
            await limiter.record_throttle()
            if attempt < len(delays):
                await asyncio.sleep(delays[attempt])
                continue
        else:
            await limiter.record_success()

        latency_ms = (time.perf_counter() - start) * 1000.0
        return resp, latency_ms

    raise last_exc if last_exc else RuntimeError("request failed")


def build_failure_record(
    row: Dict[str, Any],
    args: argparse.Namespace,
    latencies: List[float],
    warnings: List[str],
    error_msg: str,
    raw_text_response: Optional[str],
) -> RowRecord:
    answer_value = row.get("answer")
    if pd.isna(answer_value) or answer_value is None:
        normalized_answer = ""
    else:
        normalized_answer = str(answer_value)
    gt = normalized_answer.strip().upper()
    if not gt or gt not in LETTER_SET:
        gt = None

    points_earned, scored_correct = score_question(
        row.get("points"),
        gt,
        None,
        penalize_unanswered=True,
    )

    return RowRecord(
        id=row.get("id"),
        year=row.get("year"),
        group=row.get("group"),
        problem_number=row.get("problem_number"),
        language=row.get("language"),
        multimodal=row.get("multimodal"),
        points=row.get("points"),
        answer=gt,
        predicted=None,
        is_correct=scored_correct,
        points_earned=points_earned,
        reasoning_mode=SELF_DIRECTED_MODE,
        latency_ms=sum(latencies) if latencies else None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost_usd=None,
        rationale=None,
        raw_text_response=raw_text_response,
        generation_id=None,
        error=error_msg,
        warnings=warnings or None,
    )


async def evaluate_single_row(
    row: Dict[str, Any],
    args: argparse.Namespace,
    model_info: ModelInfo,
    client: httpx.AsyncClient,
    limiter: AdaptiveRateLimiter,
    url: str,
    headers: Dict[str, str],
    metrics: Optional[Aggregator] = None,
) -> WorkerOutcome:
    warnings: List[str] = []
    raw_entries: List[Dict[str, Any]] = []

    q_bytes = coerce_bytes(row.get("question_image"))
    opt_bytes = {
        letter: coerce_bytes(row.get(f"sol_{letter}_image_bin"))
        for letter in LETTER_SET
    }
    assoc_bytes = coerce_list_of_bytes(row.get("associated_images_bin")) or []

    has_images = bool(q_bytes or any(opt_bytes.values()) or assoc_bytes)
    if has_images and not model_info.supports_vision:
        return WorkerOutcome(
            record=None,
            raw_entries=[],
            failure_entry=None,
            skipped=True,
            fail_fast_trigger=False,
            attempts=0,
            status_code=None,
            row_id=row.get("id"),
        )

    encoded_images: Dict[str, Any] = {"assoc_list": []}
    if model_info.supports_vision:
        if q_bytes:
            img = pil_from_bytes(q_bytes)
            if img is None:
                warnings.append("question_image_decode_failed")
            else:
                url_data, _ = image_to_data_url(
                    img,
                    max_dim=(args.image_max_dim or 1024),
                    jpeg_quality=(args.image_jpeg_quality or 85),
                )
                encoded_images["question"] = url_data

        for letter in ["A", "B", "C", "D", "E"]:
            b = opt_bytes.get(letter)
            if b:
                img = pil_from_bytes(b)
                if img is None:
                    warnings.append(f"opt_{letter}_image_decode_failed")
                else:
                    url_data, _ = image_to_data_url(
                        img,
                        max_dim=(args.image_max_dim or 1024),
                        jpeg_quality=(args.image_jpeg_quality or 85),
                    )
                    encoded_images[f"opt_{letter}"] = url_data

        for idx, b in enumerate(assoc_bytes):
            img = pil_from_bytes(b)
            if img is None:
                warnings.append(f"assoc_{idx + 1}_image_decode_failed")
                encoded_images["assoc_list"].append(None)
            else:
                url_data, _ = image_to_data_url(
                    img,
                    max_dim=(args.image_max_dim or 1024),
                    jpeg_quality=(args.image_jpeg_quality or 85),
                )
                encoded_images["assoc_list"].append(url_data)

    messages = build_messages(
        row,
        encoded_images=encoded_images,
        image_detail=(args.image_detail or "auto"),
    )

    payload: Dict[str, Any] = {
        "model": model_info.id,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "usage": {"include": True},
    }

    max_tokens_current = args.max_tokens
    attempt = 0
    max_attempts = 5
    combined_usage: Dict[str, float] = {}
    combined_usage_details: Dict[
        str, float
    ] = {}  # reasoning_tokens, cached_prompt_tokens, audio_prompt_tokens, explicit_reasoning_tokens
    last_usage: Optional[Dict[str, Any]] = None
    latencies: List[float] = []
    last_status_code: Optional[int] = None
    ans: Optional[str] = None
    rat: Optional[str] = None
    parse_warn: Optional[str] = None
    content_text = ""
    gen_id = None
    finish_reason = None
    native_finish = None

    parse_retry_performed = False

    while attempt < max_attempts:
        attempt += 1
        if max_tokens_current is not None:
            payload["max_tokens"] = max_tokens_current
        else:
            payload.pop("max_tokens", None)

        try:
            resp, latency_ms = await request_with_retries(
                client,
                limiter,
                url,
                headers,
                payload,
                on_status=(metrics.record_status if metrics else None),
                on_throttle=(metrics.record_throttle if metrics else None),
            )
            latencies.append(latency_ms)
            last_status_code = resp.status_code
        except Exception as exc:
            err_msg = f"request_failed: {exc}"
            failure_entry = {"id": row.get("id"), "error": err_msg}
            record = build_failure_record(row, args, latencies, warnings, err_msg, None)
            return WorkerOutcome(
                record=record,
                raw_entries=raw_entries,
                failure_entry=failure_entry,
                skipped=False,
                fail_fast_trigger=args.fail_fast,
                attempts=attempt,
                status_code=None,
                row_id=row.get("id"),
            )

        if resp.status_code != 200:
            content_text = resp.text or ""
            snippet = re.sub(r"\s+", " ", content_text.strip()) if content_text else ""
            if snippet:
                snippet = snippet[:200]
            err_msg = f"http_error_{resp.status_code}"
            if snippet:
                err_msg = f"{err_msg}: {snippet}"
            warnings.append(f"http_status_{resp.status_code}")
            try:
                error_payload = resp.json()
            except Exception:
                error_payload = {"raw_text": content_text}
            raw_entries.append(
                {
                    "id": row.get("id"),
                    "attempt": attempt,
                    "response": error_payload,
                    "status_code": resp.status_code,
                }
            )
            # If this status is retryable, perform a row-level retry instead of failing immediately
            # to further reduce the chance of a final failure (e.g., rare 429 slipping through).
            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts:
                try:
                    await limiter.record_throttle()
                except Exception:
                    pass
                if metrics is not None:
                    try:
                        metrics.record_throttle()
                    except Exception:
                        pass
                row_retry_delays = [2.0, 4.0, 8.0]
                delay_idx = min(max(0, attempt - 1), len(row_retry_delays) - 1)
                await asyncio.sleep(row_retry_delays[delay_idx])
                continue
            failure_entry = {
                "id": row.get("id"),
                "status_code": resp.status_code,
                "error": err_msg,
            }
            record = build_failure_record(
                row, args, latencies, warnings, err_msg, content_text or None
            )
            return WorkerOutcome(
                record=record,
                raw_entries=raw_entries,
                failure_entry=failure_entry,
                skipped=False,
                fail_fast_trigger=args.fail_fast,
                attempts=attempt,
                status_code=resp.status_code,
                row_id=row.get("id"),
            )

        try:
            data = resp.json()
        except Exception:
            data = None

        content_text = resp.text or ""
        finish_reason = None
        native_finish = None

        if isinstance(data, dict):
            gen_id = data.get("id")
            choices = data.get("choices") or []
            if choices and isinstance(choices, list):
                # Some providers may return None entries; guard accordingly
                sanitized_choices = [
                    choice for choice in choices if isinstance(choice, dict)
                ]
                choice0 = (
                    sanitized_choices[0] if sanitized_choices else (choices[0] or {})
                )
                finish_reason = choice0.get("finish_reason")
                native_finish = choice0.get("native_finish_reason")
                content_raw = (choice0.get("message") or {}).get("content")
                content_text = normalize_message_content(content_raw)
            else:
                choice0 = {}

            usage_key_aliases = {
                "prompt_tokens": ["input_tokens", "prompt_tokens"],
                "completion_tokens": ["output_tokens", "completion_tokens"],
                "total_tokens": ["total_tokens"],
            }
            usage = data.get("usage")
            if isinstance(usage, dict):
                last_usage = usage
                for dst_key, src_keys in usage_key_aliases.items():
                    for src_key in src_keys:
                        val = usage.get(src_key)
                        if isinstance(val, (int, float)):
                            combined_usage[dst_key] = combined_usage.get(
                                dst_key, 0.0
                            ) + float(val)
                            break
                # Details when available
                prompt_details = None
                for detail_key in ("prompt_tokens_details", "input_tokens_details"):
                    details_candidate = usage.get(detail_key)
                    if isinstance(details_candidate, dict):
                        prompt_details = details_candidate
                        break
                if prompt_details is None:
                    prompt_details = {}
                if isinstance(prompt_details, dict):
                    for k_src, k_dst in (
                        ("cached_tokens", "cached_prompt_tokens"),
                        ("audio_tokens", "audio_prompt_tokens"),
                    ):
                        val = prompt_details.get(k_src)
                        if isinstance(val, (int, float)):
                            combined_usage_details[k_dst] = combined_usage_details.get(
                                k_dst, 0.0
                            ) + float(val)
                completion_details = None
                for detail_key in (
                    "completion_tokens_details",
                    "output_tokens_details",
                ):
                    details_candidate = usage.get(detail_key)
                    if isinstance(details_candidate, dict):
                        completion_details = details_candidate
                        break
                if completion_details is None:
                    completion_details = {}
                if isinstance(completion_details, dict):
                    rt = completion_details.get("reasoning_tokens")
                    if isinstance(rt, (int, float)):
                        combined_usage_details["reasoning_tokens"] = (
                            combined_usage_details.get("reasoning_tokens", 0.0)
                            + float(rt)
                        )
                        combined_usage_details["explicit_reasoning_tokens"] = (
                            combined_usage_details.get("explicit_reasoning_tokens", 0.0)
                            + float(rt)
                        )
                    # Some providers report visible reasoning separately (e.g., output_text tokens)
                    explicit_rt = completion_details.get("explicit_reasoning_tokens")
                    if isinstance(explicit_rt, (int, float)):
                        combined_usage_details["explicit_reasoning_tokens"] = (
                            combined_usage_details.get("explicit_reasoning_tokens", 0.0)
                            + float(explicit_rt)
                        )
                cost_val = usage.get("cost") or usage.get("total_cost")
                if isinstance(cost_val, str):
                    try:
                        cost_val = float(cost_val)
                    except Exception:
                        cost_val = None
                if isinstance(cost_val, (int, float)):
                    combined_usage["cost"] = combined_usage.get("cost", 0.0) + float(
                        cost_val
                    )
            # Fallback: some providers put usage into an X-Usage header
            elif hasattr(resp, "headers"):
                hdr = None
                try:
                    hdr = resp.headers.get("x-usage") or resp.headers.get("X-Usage")
                except Exception:
                    hdr = None
                if hdr:
                    parsed: Optional[Dict[str, Any]] = None
                    # First try JSON
                    try:
                        parsed_json = json.loads(hdr)
                        if isinstance(parsed_json, dict):
                            parsed = parsed_json
                    except Exception:
                        parsed = None
                    # Then try a simple key=value parser (comma/semicolon separated)
                    if parsed is None:
                        try:
                            kv: Dict[str, float] = {}
                            for part in re.split(r"[,;]", hdr):
                                if "=" not in part:
                                    continue
                                k, v = part.split("=", 1)
                                k = k.strip()
                                v = v.strip()
                                if not k:
                                    continue
                                try:
                                    kv[k] = float(v)
                                except Exception:
                                    continue
                            if kv:
                                parsed = dict(kv)
                        except Exception:
                            parsed = None
                    # Apply parsed usage fields if any
                    if isinstance(parsed, dict):
                        for dst_key, src_keys in usage_key_aliases.items():
                            for src_key in src_keys:
                                val = parsed.get(src_key)
                                if isinstance(val, (int, float)):
                                    combined_usage[dst_key] = combined_usage.get(
                                        dst_key, 0.0
                                    ) + float(val)
                                    break
                        # Details: either nested objects or dot-keys
                        prompt_details = None
                        for detail_key in (
                            "prompt_tokens_details",
                            "input_tokens_details",
                        ):
                            details_candidate = parsed.get(detail_key)
                            if isinstance(details_candidate, dict):
                                prompt_details = details_candidate
                                break
                        if prompt_details is None and isinstance(
                            parsed.get("prompt_tokens_details"), dict
                        ):
                            prompt_details = parsed.get("prompt_tokens_details")
                        if prompt_details is None:
                            prompt_details = {}
                        completion_details = None
                        for detail_key in (
                            "completion_tokens_details",
                            "output_tokens_details",
                        ):
                            details_candidate = parsed.get(detail_key)
                            if isinstance(details_candidate, dict):
                                completion_details = details_candidate
                                break
                        if completion_details is None and isinstance(
                            parsed.get("completion_tokens_details"), dict
                        ):
                            completion_details = parsed.get("completion_tokens_details")
                        if completion_details is None:
                            completion_details = {}

                        # Also support flattened keys like "prompt_tokens_details.cached_tokens=123"
                        def _maybe_from_flat(src_key: str) -> Optional[float]:
                            val = parsed.get(src_key)
                            try:
                                if isinstance(val, str):
                                    return float(val)
                                if isinstance(val, (int, float)):
                                    return float(val)
                            except Exception:
                                return None
                            return None

                        for k_src, k_dst in (
                            ("cached_tokens", "cached_prompt_tokens"),
                            ("audio_tokens", "audio_prompt_tokens"),
                        ):
                            val = (prompt_details or {}).get(k_src)
                            if not isinstance(val, (int, float)):
                                val = _maybe_from_flat(f"prompt_tokens_details.{k_src}")
                            if isinstance(val, (int, float)):
                                combined_usage_details[k_dst] = (
                                    combined_usage_details.get(k_dst, 0.0) + float(val)
                                )

                        rt = (completion_details or {}).get("reasoning_tokens")
                        if not isinstance(rt, (int, float)):
                            rt = _maybe_from_flat(
                                "completion_tokens_details.reasoning_tokens"
                            )
                        if not isinstance(rt, (int, float)):
                            rt = _maybe_from_flat(
                                "output_tokens_details.reasoning_tokens"
                            )
                        if isinstance(rt, (int, float)):
                            combined_usage_details["reasoning_tokens"] = (
                                combined_usage_details.get("reasoning_tokens", 0.0)
                                + float(rt)
                            )
                            combined_usage_details["explicit_reasoning_tokens"] = (
                                combined_usage_details.get(
                                    "explicit_reasoning_tokens", 0.0
                                )
                                + float(rt)
                            )

                        explicit_rt = (completion_details or {}).get(
                            "explicit_reasoning_tokens"
                        )
                        if not isinstance(explicit_rt, (int, float)):
                            explicit_rt = _maybe_from_flat(
                                "completion_tokens_details.explicit_reasoning_tokens"
                            )
                        if not isinstance(explicit_rt, (int, float)):
                            explicit_rt = _maybe_from_flat(
                                "output_tokens_details.explicit_reasoning_tokens"
                            )
                        if isinstance(explicit_rt, (int, float)):
                            combined_usage_details["explicit_reasoning_tokens"] = (
                                combined_usage_details.get(
                                    "explicit_reasoning_tokens", 0.0
                                )
                                + float(explicit_rt)
                            )

                        cost_val = parsed.get("cost") or parsed.get("total_cost")
                        if isinstance(cost_val, (int, float)):
                            combined_usage["cost"] = combined_usage.get(
                                "cost", 0.0
                            ) + float(cost_val)
            raw_entries.append(
                {
                    "id": row.get("id"),
                    "attempt": attempt,
                    "response": data,
                }
            )
        else:
            raw_entries.append(
                {
                    "id": row.get("id"),
                    "attempt": attempt,
                    "response": {"raw_text": content_text},
                }
            )

        ans, rat, parse_warn = parse_answer_from_text(content_text or "")

        should_retry = False
        finish_values = [finish_reason, native_finish]
        if ans is None and attempt < max_attempts:
            for reason in finish_values:
                if isinstance(reason, str) and reason.lower() in {
                    "length",
                    "max_tokens",
                    "max_tokens_exceeded",
                }:
                    should_retry = True
                    break

        if should_retry:
            if max_tokens_current is None:
                next_tokens = DEFAULT_RETRY_MAX_TOKENS
            else:
                next_tokens = min(max_tokens_current * 2, 4096)
            warnings.append(
                f"max_tokens_retry_{max_tokens_current if max_tokens_current is not None else 'auto'}->{next_tokens}"
            )
            max_tokens_current = next_tokens
            continue

        if parse_warn:
            warnings.append(parse_warn)

        if (
            ans is None
            and parse_warn in {"no_parse", "empty_response"}
            and not parse_retry_performed
            and attempt < max_attempts
        ):
            parse_retry_performed = True
            warnings.append(f"retry_due_to_{parse_warn}")
            continue

        break

    answer_value = row.get("answer")
    if pd.isna(answer_value) or answer_value is None:
        normalized_answer = ""
    else:
        normalized_answer = str(answer_value)

    gt = normalized_answer.strip().upper()
    if not gt or gt not in LETTER_SET:
        gt = None

    if gt is None:
        is_correct = None
        points_earned = 0.0
    else:
        points_earned, scored_correct = score_question(
            row.get("points"),
            gt,
            ans,
            penalize_unanswered=True,
        )
        is_correct = scored_correct

    prompt_tokens = completion_tokens = total_tokens = None
    reasoning_tokens = cached_prompt_tokens = audio_prompt_tokens = None
    explicit_reasoning_tokens = None
    cost_usd = None
    if combined_usage:
        if "prompt_tokens" in combined_usage:
            prompt_tokens = int(combined_usage["prompt_tokens"])
        if "completion_tokens" in combined_usage:
            completion_tokens = int(combined_usage["completion_tokens"])
        if "total_tokens" in combined_usage:
            total_tokens = int(combined_usage["total_tokens"])
        if "cost" in combined_usage:
            cost_usd = float(combined_usage["cost"])
        if "reasoning_tokens" in combined_usage_details:
            reasoning_tokens = int(combined_usage_details["reasoning_tokens"])
        if "explicit_reasoning_tokens" in combined_usage_details:
            explicit_reasoning_tokens = int(
                combined_usage_details["explicit_reasoning_tokens"]
            )
        if "cached_prompt_tokens" in combined_usage_details:
            cached_prompt_tokens = int(combined_usage_details["cached_prompt_tokens"])
        if "audio_prompt_tokens" in combined_usage_details:
            audio_prompt_tokens = int(combined_usage_details["audio_prompt_tokens"])
    elif isinstance(last_usage, dict):

        def _extract_numeric(
            source: Dict[str, Any], keys: Tuple[str, ...]
        ) -> Optional[int]:
            for key in keys:
                val = source.get(key)
                if isinstance(val, (int, float)):
                    return int(val)
                if isinstance(val, str):
                    try:
                        return int(float(val))
                    except Exception:
                        continue
            return None

        prompt_tokens = _extract_numeric(last_usage, ("input_tokens", "prompt_tokens"))
        completion_tokens = _extract_numeric(
            last_usage, ("output_tokens", "completion_tokens")
        )
        total_tokens = _extract_numeric(last_usage, ("total_tokens",))
        cost_usd = last_usage.get("cost") or last_usage.get("total_cost")
        if isinstance(cost_usd, str):
            try:
                cost_usd = float(cost_usd)
            except Exception:
                cost_usd = None
        # Details (single-attempt fallback)
        try:
            prompt_details_map = None
            for detail_key in ("prompt_tokens_details", "input_tokens_details"):
                details_candidate = last_usage.get(detail_key)
                if isinstance(details_candidate, dict):
                    prompt_details_map = details_candidate
                    break
            if prompt_details_map is None:
                prompt_details_map = {}
            if isinstance(prompt_details_map, dict):
                cached_prompt_tokens = (
                    prompt_details_map.get("cached_tokens")
                    if isinstance(prompt_details_map.get("cached_tokens"), int)
                    else cached_prompt_tokens
                )
                audio_prompt_tokens = (
                    prompt_details_map.get("audio_tokens")
                    if isinstance(prompt_details_map.get("audio_tokens"), int)
                    else audio_prompt_tokens
                )
            completion_details_map = None
            for detail_key in ("completion_tokens_details", "output_tokens_details"):
                details_candidate = last_usage.get(detail_key)
                if isinstance(details_candidate, dict):
                    completion_details_map = details_candidate
                    break
            if completion_details_map is None:
                completion_details_map = {}
            if isinstance(completion_details_map, dict):
                reasoning_tokens = (
                    completion_details_map.get("reasoning_tokens")
                    if isinstance(completion_details_map.get("reasoning_tokens"), int)
                    else reasoning_tokens
                )
                if isinstance(
                    completion_details_map.get("explicit_reasoning_tokens"), int
                ):
                    explicit_reasoning_tokens = completion_details_map.get(
                        "explicit_reasoning_tokens"
                    )
        except Exception:
            pass

    if explicit_reasoning_tokens is None and completion_tokens is not None:
        explicit_reasoning_tokens = completion_tokens

    record = RowRecord(
        id=row.get("id"),
        year=row.get("year"),
        group=row.get("group"),
        problem_number=row.get("problem_number"),
        language=row.get("language"),
        multimodal=row.get("multimodal"),
        points=row.get("points"),
        answer=gt,
        predicted=ans,
        is_correct=is_correct,
        points_earned=points_earned,
        reasoning_mode=SELF_DIRECTED_MODE,
        latency_ms=sum(latencies) if latencies else None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        explicit_reasoning_tokens=explicit_reasoning_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        audio_prompt_tokens=audio_prompt_tokens,
        cost_usd=cost_usd,
        rationale=rat,
        raw_text_response=content_text,
        generation_id=gen_id,
        error=None,
        warnings=warnings or None,
    )

    return WorkerOutcome(
        record=record,
        raw_entries=raw_entries,
        failure_entry=None,
        skipped=False,
        fail_fast_trigger=False,
        attempts=attempt,
        status_code=last_status_code or 200,
        row_id=row.get("id"),
    )


async def evaluate_rows_async(
    rows_data: List[Dict[str, Any]],
    args: argparse.Namespace,
    model_info: ModelInfo,
    url: str,
    headers: Dict[str, str],
    results_jsonl,
    raw_responses_file,
    failures_file,
    worker_count: int,
    *,
    dashboard_enabled: bool,
    dashboard_refresh_hz: float,
    dashboard_recent: int,
    dashboard_compact: bool,
    events_path: Optional[Path],
) -> Tuple[List[RowRecord], List[Dict[str, Any]], int]:
    rows: List[RowRecord] = []
    results_records: List[Dict[str, Any]] = []
    skipped = 0

    limiter = AdaptiveRateLimiter(
        initial_interval=model_info.min_request_interval or 0.0
    )
    work_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
    result_queue: asyncio.Queue[Optional[WorkerOutcome]] = asyncio.Queue()
    stop_event = asyncio.Event()

    inflight_lock = asyncio.Lock()
    active_inflight = 0

    aggregator = Aggregator(
        len(rows_data),
        model_id=model_info.id,
        events_path=events_path,
        recent_items=dashboard_recent,
        min_request_interval=model_info.min_request_interval,
    )

    dashboard = None
    progress: Optional[tqdm] = None
    if dashboard_enabled and DASHBOARD_AVAILABLE and Dashboard is not None:
        dashboard = Dashboard(
            aggregator,
            refresh_hz=dashboard_refresh_hz,
            compact=dashboard_compact,
            recent_rows=dashboard_recent,
        )
        dashboard.start()
        dashboard.update(aggregator.snapshot(in_flight=0, worker_count=worker_count))
    else:
        progress = tqdm(total=len(rows_data), desc="Evaluating", unit="q")

    def build_usage_event(outcome: WorkerOutcome) -> UsageEvent:
        record = outcome.record

        def as_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except Exception:
                pass
            try:
                return int(value)
            except Exception:
                return None

        def as_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except Exception:
                pass
            try:
                return float(value)
            except Exception:
                return None

        event_type = "skipped"
        if not outcome.skipped:
            if record is not None and record.error:
                event_type = "failure"
            else:
                event_type = "success"

        group_value: Optional[str] = None
        multimodal_value: Optional[bool] = None
        if record is not None:
            if record.group is not None and record.group != "":
                group_value = str(record.group)
            if isinstance(record.multimodal, bool):
                multimodal_value = record.multimodal
            elif record.multimodal is not None:
                multimodal_value = bool(record.multimodal)

        event = UsageEvent(
            type=event_type,
            row_id=(record.id if record is not None else outcome.row_id),
            year=as_int(record.year) if record is not None else None,
            group=group_value,
            points=as_float(record.points) if record is not None else None,
            problem_number=as_int(record.problem_number)
            if record is not None
            else None,
            multimodal=multimodal_value,
            latency_ms=record.latency_ms if record is not None else None,
            attempts=outcome.attempts or 1,
            prompt_tokens=record.prompt_tokens if record is not None else None,
            completion_tokens=record.completion_tokens if record is not None else None,
            total_tokens=record.total_tokens if record is not None else None,
            reasoning_tokens=getattr(record, "reasoning_tokens", None)
            if record is not None
            else None,
            explicit_reasoning_tokens=getattr(record, "explicit_reasoning_tokens", None)
            if record is not None
            else None,
            cached_prompt_tokens=getattr(record, "cached_prompt_tokens", None)
            if record is not None
            else None,
            audio_prompt_tokens=getattr(record, "audio_prompt_tokens", None)
            if record is not None
            else None,
            cost_usd_known=record.cost_usd if record is not None else None,
            predicted=record.predicted if record is not None else None,
            correct=record.is_correct if record is not None else None,
            status_code=outcome.status_code,
            warnings=record.warnings if record is not None else None,
        )
        return event

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal active_inflight
        while True:
            item = await work_queue.get()
            if item is None:
                work_queue.task_done()
                break
            if args.fail_fast and stop_event.is_set():
                work_queue.task_done()
                continue
            async with inflight_lock:
                active_inflight += 1
            outcome = await evaluate_single_row(
                item,
                args,
                model_info,
                client,
                limiter,
                url,
                headers,
                metrics=aggregator,
            )
            await result_queue.put(outcome)
            async with inflight_lock:
                active_inflight = max(0, active_inflight - 1)
            work_queue.task_done()
            if args.fail_fast and outcome.fail_fast_trigger:
                stop_event.set()

    async def consumer() -> None:
        nonlocal skipped
        while True:
            outcome = await result_queue.get()
            if outcome is None:
                result_queue.task_done()
                break
            if outcome.skipped:
                skipped += 1
            if outcome.failure_entry is not None:
                write_jsonl_line(failures_file, outcome.failure_entry)
            for entry in outcome.raw_entries:
                write_jsonl_line(raw_responses_file, entry)
            if outcome.record is not None:
                rows.append(outcome.record)
                record_dict = asdict(outcome.record)
                results_records.append(record_dict)
                write_jsonl_line(results_jsonl, record_dict)
            event = build_usage_event(outcome)
            aggregator.record_event(event)
            if dashboard is not None:
                async with inflight_lock:
                    inflight_current = active_inflight
                snapshot = aggregator.snapshot(
                    in_flight=inflight_current, worker_count=worker_count
                )
                dashboard.update(snapshot)
            else:
                if progress is not None:
                    progress.update(1)
            result_queue.task_done()

    limits = httpx.Limits(
        max_connections=max(4, worker_count * 2),
        max_keepalive_connections=max(2, worker_count),
    )
    timeout = httpx.Timeout(120.0)

    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            for item in rows_data:
                await work_queue.put(item)
            for _ in range(worker_count):
                await work_queue.put(None)

            workers = [asyncio.create_task(worker(client)) for _ in range(worker_count)]
            consumer_task = asyncio.create_task(consumer())

            await asyncio.gather(*workers)
            await result_queue.put(None)
            await consumer_task
    finally:
        if dashboard is not None:
            dashboard.stop()
        if progress is not None:
            progress.close()
        aggregator.close()

    return rows, results_records, skipped


def main():
    import time

    start_time = time.time()  # Track start time for total run time

    parser = argparse.ArgumentParser(
        description="LLM evaluation runner for Känguru benchmark"
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset .parquet")
    parser.add_argument("--model", required=True, help="Model ID from models.json")
    parser.add_argument("--reasoning", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-effort", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process rows sequentially instead of using the default concurrent worker pool.",
    )
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument(
        "--live-dashboard",
        dest="live_dashboard",
        action="store_true",
        help="Force enable the Rich dashboard even on non-tty outputs.",
    )
    parser.add_argument(
        "--no-live-dashboard",
        dest="live_dashboard",
        action="store_false",
        help="Disable the Rich dashboard even on ttys.",
    )
    parser.set_defaults(live_dashboard=None)
    parser.add_argument(
        "--dashboard-refresh-hz",
        type=float,
        default=5.0,
        help="Refresh rate for the live dashboard (updates per second).",
    )
    parser.add_argument(
        "--events-jsonl",
        default="auto",
        help="Path to usage events JSONL log ('off' to disable, default auto).",
    )
    parser.add_argument(
        "--no-events-jsonl",
        dest="events_jsonl",
        action="store_const",
        const="off",
        help="Disable usage events capture.",
    )
    parser.add_argument(
        "--recent-items",
        type=int,
        default=20,
        help="Number of recent items to show in the dashboard table.",
    )
    parser.add_argument(
        "--ui-compact",
        action="store_true",
        help="Use compact dashboard layout suitable for smaller terminals.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Evaluate only text (non-multimodal) questions; drop multimodal rows before running.",
    )
    # Image controls for multimodal inputs
    parser.add_argument(
        "--image_max_dim",
        type=int,
        default=1024,
        help="Max image dimension (long edge) in pixels; no upscaling",
    )
    parser.add_argument(
        "--image_jpeg_quality", type=int, default=85, help="JPEG quality (50–100)"
    )
    parser.add_argument(
        "--image_detail",
        choices=["auto", "low", "high"],
        default="auto",
        help="Vision detail hint for providers that support it",
    )

    # Categorical Filters
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        help="Filter by one or more specific years.",
    )
    parser.add_argument(
        "--group",
        type=str,
        action="append",
        help="Filter by one or more specific grade groups.",
    )
    parser.add_argument(
        "--language",
        type=str,
        action="append",
        help="Filter by one or more specific languages.",
    )

    # Range Filters
    parser.add_argument(
        "--year-range",
        type=str,
        help="Filter by an inclusive range of years (e.g., '2020-2023').",
    )
    parser.add_argument(
        "--points-range",
        type=str,
        help="Filter by an inclusive range of points (e.g., '3.75-5.0').",
    )

    # Boolean Filter
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Evaluate only vision (multimodal) questions.",
    )

    args = parser.parse_args()

    legacy_reasoning = args.reasoning
    legacy_internal_effort = args.internal_effort
    if legacy_reasoning is not None or legacy_internal_effort is not None:
        print(
            "Warning: --reasoning/--internal-effort are ignored; the evaluator now runs in self-directed mode only.",
            file=sys.stderr,
        )
    args.reasoning = None
    args.internal_effort = None

    stdout_isatty = sys.stdout.isatty()
    if args.live_dashboard is True:
        dashboard_enabled = True
    elif args.live_dashboard is False:
        dashboard_enabled = False
    else:
        dashboard_enabled = stdout_isatty
    if dashboard_enabled and not DASHBOARD_AVAILABLE:
        print(
            "Rich dashboard unavailable; falling back to tqdm progress bar.",
            file=sys.stderr,
        )
        dashboard_enabled = False
    dashboard_refresh = max(1.0, float(args.dashboard_refresh_hz or 5.0))
    dashboard_recent = max(5, int(args.recent_items or 20))
    events_config = args.events_jsonl or "auto"

    if dashboard_enabled and not stdout_isatty:
        print(
            "Warning: live dashboard forced on a non-TTY output; layout may degrade.",
            file=sys.stderr,
        )

    load_env_file()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENROUTER_API_KEY environment variable is required.",
            file=sys.stderr,
        )
        sys.exit(2)

    models = read_models_registry(
        os.path.join(os.path.dirname(__file__), "models.json")
    )
    if args.model not in models:
        print(f"ERROR: model '{args.model}' not found in models.json", file=sys.stderr)
        sys.exit(2)
    model_info = models[args.model]

    try:
        dataset_path = resolve_dataset_path(args.dataset)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    run_dir, ts = ensure_output_dir(args.output_dir, args.model)

    events_path: Optional[Path]
    if isinstance(events_config, str):
        cfg = events_config.strip().lower()
    else:
        cfg = "auto"
    if cfg == "off":
        events_path = None
    elif cfg in {"", "auto"}:
        events_path = Path(run_dir) / "usage_events.jsonl"
    else:
        events_path = Path(events_config).expanduser()

    df = pd.read_parquet(dataset_path, engine="pyarrow")
    validate_dataset_columns(df)

    if args.limit is not None and args.limit < len(df):
        if args.seed is not None:
            df = df.sample(n=args.limit, random_state=args.seed)
        else:
            df = df.head(args.limit)

    # Handle mutual exclusion between text-only and vision-only
    if args.text_only and args.vision_only:
        print(
            "ERROR: --text-only and --vision-only are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Apply filters sequentially
    # Categorical filters
    if args.year:
        df = df[df["year"].isin(args.year)]

    if args.group:
        df = df[df["group"].isin(args.group)]

    if args.language:
        df = df[df["language"].isin(args.language)]

    # Range filters
    if args.year_range:
        try:
            start_str, end_str = args.year_range.split("-")
            start = int(start_str.strip())
            end = int(end_str.strip())
            df = df[(df["year"] >= start) & (df["year"] <= end)]
        except ValueError:
            print(
                f"ERROR: Invalid year range format: '{args.year_range}'. Expected format: 'START-END'",
                file=sys.stderr,
            )
            sys.exit(2)

    if args.points_range:
        try:
            min_str, max_str = args.points_range.split("-")
            min_val = float(min_str.strip())
            max_val = float(max_str.strip())
            df = df[(df["points"] >= min_val) & (df["points"] <= max_val)]
        except ValueError:
            print(
                f"ERROR: Invalid points range format: '{args.points_range}'. Expected format: 'MIN-MAX'",
                file=sys.stderr,
            )
            sys.exit(2)

    if args.vision_only:
        df = df[df["multimodal"].astype(bool)]

    total_rows_loaded = int(len(df))
    cli_text_only = bool(getattr(args, "text_only", False))
    cli_filtered_out_rows = 0
    model_filtered_out_rows = 0
    if cli_text_only:
        mask = ~df["multimodal"].astype(bool)
        cli_filtered_out_rows = int(len(df) - int(mask.sum()))
        df = df.loc[mask]
    elif not model_info.supports_vision:
        mask = ~df["multimodal"].astype(bool)
        model_filtered_out_rows = int(len(df) - int(mask.sum()))
        df = df.loc[mask]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "kaenguru-benchmark-eval",
    }
    url = "https://openrouter.ai/api/v1/chat/completions"

    rows_data = df.to_dict(orient="records")

    worker_count = 1 if args.sequential else default_worker_count()
    if not rows_data:
        worker_count = 1

    results_jsonl_path = os.path.join(run_dir, "results.jsonl")
    raw_responses_path = os.path.join(run_dir, "raw_responses.jsonl")
    failures_jsonl_path = os.path.join(run_dir, "failures.jsonl")

    initial_interval = model_info.min_request_interval or 0.0
    if initial_interval > 0:
        initial_rpm = 60.0 / initial_interval
        limiter_desc = f"seeded at {initial_rpm:.1f} rpm"
    else:
        limiter_desc = "adaptive (no seed)"

    multimodal_count = sum(1 for row in rows_data if bool(row.get("multimodal")))
    avg_points = (
        float(df["points"].dropna().astype(float).mean())
        if not df["points"].dropna().empty
        else None
    )
    year_counts = df["year"].value_counts().to_dict() if "year" in df.columns else {}
    year_desc = (
        ", ".join(f"{k}:{v}" for k, v in sorted(year_counts.items()))
        if year_counts
        else "n/a"
    )

    print("Configuration:")
    print(f"  Model: {model_info.id}")
    if model_info.label:
        print(f"  Label: {model_info.label}")
    print(f"  Dataset rows: {len(rows_data)}")
    print(
        f"  Multimodal rows: {multimodal_count} ({multimodal_count / len(rows_data) * 100:.1f}% )"
        if rows_data
        else "  Multimodal rows: 0"
    )
    print(
        f"  Mode: {'sequential' if worker_count == 1 else f'concurrent x{worker_count}'}"
    )
    print(f"  Rate limiter: {limiter_desc}")
    print(f"  Live dashboard: {'on' if dashboard_enabled else 'off'}")
    if events_path is not None:
        print(f"  Usage events: {str(events_path)}")
    else:
        print("  Usage events: disabled")
    print(f"  Reasoning: {SELF_DIRECTED_MODE}")
    if avg_points is not None:
        print(f"  Avg points: {avg_points:.2f}")
    print(f"  Year distribution: {year_desc}")
    if cli_text_only:
        print("  Text-only filter: on (--text-only)")
        print(f"  Dropped due to --text-only: {cli_filtered_out_rows}")
    elif not model_info.supports_vision:
        print("  Text-only filter: on (model lacks vision)")
        print(f"  Dropped due to no-vision: {model_filtered_out_rows}")
    else:
        print("  Text-only filter: off")
    # Add proper spacing after pre-run summary to separate from dashboard
    print()

    with (
        open(results_jsonl_path, "w", encoding="utf-8") as results_jsonl,
        open(raw_responses_path, "w", encoding="utf-8") as raw_responses_file,
        open(failures_jsonl_path, "w", encoding="utf-8") as failures_file,
    ):
        rows, results_records, skipped = asyncio.run(
            evaluate_rows_async(
                rows_data,
                args,
                model_info,
                url,
                headers,
                results_jsonl,
                raw_responses_file,
                failures_file,
                worker_count,
                dashboard_enabled=dashboard_enabled,
                dashboard_refresh_hz=dashboard_refresh,
                dashboard_recent=dashboard_recent,
                dashboard_compact=bool(args.ui_compact),
                events_path=events_path,
            )
        )

    if rows:
        results_df = pd.DataFrame(results_records)
    else:
        results_df = pd.DataFrame(
            columns=[f.name for f in dataclasses.fields(RowRecord)]
        )
    results_path = os.path.join(run_dir, "results.parquet")
    results_df.to_parquet(results_path, engine="pyarrow", index=False)

    results_json_path = os.path.join(run_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_records, f, ensure_ascii=False, indent=2)

    if len(results_df.columns) > 0:
        answered_mask = results_df["predicted"].notna() & results_df["error"].isna()
        declined_mask = (
            results_df["predicted"].fillna("").astype(str).str.upper() == DECLINED_TOKEN
        ) & results_df["error"].isna()
    else:
        answered_mask = pd.Series([], dtype=bool)
        declined_mask = pd.Series([], dtype=bool)
    answered_count = int(answered_mask.sum())
    declined_count = int(declined_mask.sum()) if len(results_df.columns) > 0 else 0
    if len(results_df.columns) > 0:
        failed_count = int(results_df["error"].notna().sum())
    else:
        failed_count = 0

    skipped_count = skipped

    if len(results_df.columns) > 0:
        correct_mask = results_df["is_correct"].fillna(False).astype(bool)
        accuracy = (
            float((correct_mask & answered_mask).sum() / answered_count)
            if answered_count
            else 0.0
        )
    else:
        accuracy = 0.0

    if not results_df.empty:
        points_series = pd.to_numeric(results_df.get("points"), errors="coerce").fillna(
            0.0
        )
        earned_series = pd.to_numeric(
            results_df.get("points_earned"), errors="coerce"
        ).fillna(0.0)

        start_points_total = 0.0
        if not points_series.empty:
            combos = results_df.loc[points_series > 0, ["year", "group"]]
            seen_keys: set[Tuple[str, str]] = set()
            for year_value, group_value in combos.itertuples(index=False):
                if pd.isna(group_value):
                    continue

                if isinstance(group_value, (int, float)) and not isinstance(
                    group_value, bool
                ):
                    if float(group_value).is_integer():
                        sanitized_group_value: object = int(float(group_value))
                    else:
                        sanitized_group_value = float(group_value)
                else:
                    sanitized_group_value = str(group_value).strip()

                group_key = str(sanitized_group_value).strip()
                if not group_key:
                    continue
                normalized_year = (
                    "None" if pd.isna(year_value) else str(year_value).strip()
                )
                key = (normalized_year, group_key)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                start_bonus = start_points_for_group(sanitized_group_value)
                if start_bonus > 0.0:
                    start_points_total += start_bonus

        raw_total_points = float(earned_series.sum())
        max_points_total = float(points_series.sum() + start_points_total)
        total_points_earned = raw_total_points + start_points_total
        p_weighted_accuracy = (
            total_points_earned / max_points_total if max_points_total > 1e-12 else 0.0
        )
    else:
        p_weighted_accuracy = 0.0
        total_points_earned = 0.0

    if len(results_df.columns) > 0 and not answered_mask.empty:
        lat = results_df.loc[answered_mask, "latency_ms"].dropna().astype(float)
        mean_latency = float(lat.mean()) if not lat.empty else None
        median_latency = float(lat.median()) if not lat.empty else None
    else:
        mean_latency = None
        median_latency = None

    total_tokens_series = pd.Series(dtype="int64")
    prompt_tokens_series = pd.Series(dtype="int64")
    completion_tokens_series = pd.Series(dtype="int64")
    if len(results_df.columns) > 0 and not answered_mask.empty:
        total_tokens_series = (
            results_df.loc[answered_mask, "total_tokens"].dropna().astype(int)
        )
        if "prompt_tokens" in results_df.columns:
            prompt_tokens_series = (
                results_df.loc[answered_mask, "prompt_tokens"].dropna().astype(int)
            )
        if "completion_tokens" in results_df.columns:
            completion_tokens_series = (
                results_df.loc[answered_mask, "completion_tokens"].dropna().astype(int)
            )
        mean_tokens = (
            float(total_tokens_series.mean()) if not total_tokens_series.empty else None
        )
        mean_prompt_tokens = (
            float(prompt_tokens_series.mean())
            if not prompt_tokens_series.empty
            else None
        )
        mean_completion_tokens = (
            float(completion_tokens_series.mean())
            if not completion_tokens_series.empty
            else None
        )
        cost_series = results_df.loc[answered_mask, "cost_usd"].dropna().astype(float)
        total_cost = float(cost_series.sum()) if not cost_series.empty else 0.0
        if "reasoning_tokens" in results_df.columns:
            reasoning_tokens_series = (
                results_df.loc[answered_mask, "reasoning_tokens"].dropna().astype(int)
            )
        else:
            reasoning_tokens_series = pd.Series(dtype="int64")
        if "explicit_reasoning_tokens" in results_df.columns:
            explicit_reasoning_tokens_series = (
                results_df.loc[answered_mask, "explicit_reasoning_tokens"]
                .dropna()
                .astype(int)
            )
        else:
            explicit_reasoning_tokens_series = pd.Series(dtype="int64")
        mean_reasoning_tokens = (
            float(reasoning_tokens_series.mean())
            if not reasoning_tokens_series.empty
            else None
        )
        total_reasoning_tokens = (
            int(reasoning_tokens_series.sum())
            if not reasoning_tokens_series.empty
            else None
        )
        reasoning_tokens_known_count = int(len(reasoning_tokens_series))
        mean_explicit_reasoning_tokens = (
            float(explicit_reasoning_tokens_series.mean())
            if not explicit_reasoning_tokens_series.empty
            else None
        )
        total_explicit_reasoning_tokens = (
            int(explicit_reasoning_tokens_series.sum())
            if not explicit_reasoning_tokens_series.empty
            else None
        )
        explicit_reasoning_tokens_known_count = int(
            len(explicit_reasoning_tokens_series)
        )
        unknown_usage_count = int(answered_count - len(total_tokens_series))
    else:
        mean_tokens = None
        mean_prompt_tokens = None
        mean_completion_tokens = None
        total_cost = 0.0
        unknown_usage_count = 0
        mean_reasoning_tokens = None
        total_reasoning_tokens = None
        reasoning_tokens_known_count = 0
        mean_explicit_reasoning_tokens = None
        total_explicit_reasoning_tokens = None
        explicit_reasoning_tokens_known_count = 0

    total_prompt_tokens_value = (
        int(prompt_tokens_series.sum()) if not prompt_tokens_series.empty else None
    )
    total_completion_tokens_value = (
        int(completion_tokens_series.sum())
        if not completion_tokens_series.empty
        else None
    )
    total_total_tokens_value = (
        int(total_tokens_series.sum()) if not total_tokens_series.empty else None
    )
    prompt_tokens_known = int(len(prompt_tokens_series))
    completion_tokens_known = int(len(completion_tokens_series))

    def breakdown_by(col: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if (
            len(results_df.columns) == 0
            or col not in results_df.columns
            or answered_count == 0
        ):
            return result
        for key, sub in results_df.loc[answered_mask].groupby(col):
            sub_correct = sub["is_correct"].fillna(False).astype(bool).sum()
            sub_count = len(sub)
            sub_points = sub["points"].astype(float)
            sub_earned = sub["points_earned"].astype(float)
            sub_acc = float(sub_correct / sub_count) if sub_count else 0.0
            sub_pwa = (
                float(sub_earned.sum() / sub_points.sum())
                if sub_points.sum() > 0
                else 0.0
            )
            result[str(key)] = {
                "count": int(sub_count),
                "accuracy": sub_acc,
                "points_weighted_accuracy": sub_pwa,
            }
        return result

    warning_counts_counter: Counter[str] = Counter()
    warning_row_count = 0
    if "warnings" in results_df.columns and not results_df.empty:
        for entry in results_df["warnings"]:
            if entry is None:
                continue
            if isinstance(entry, float) and pd.isna(entry):
                continue
            row_warnings: List[str] = []
            if isinstance(entry, str):
                row_warnings = [entry]
            elif isinstance(entry, (list, tuple, set)):
                row_warnings = [str(w) for w in entry if w]
            else:
                row_warnings = [str(entry)]
            if row_warnings:
                warning_row_count += 1
                warning_counts_counter.update(row_warnings)
    warning_counts = (
        {key: count for key, count in warning_counts_counter.most_common()}
        if warning_counts_counter
        else {}
    )

    metrics = {
        "answered_count": answered_count,
        "declined_count": declined_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "accuracy": accuracy,
        "points_weighted_accuracy": p_weighted_accuracy,
        "total_points_earned": total_points_earned,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "mean_total_tokens": mean_tokens,
        "mean_prompt_tokens": mean_prompt_tokens,
        "mean_completion_tokens": mean_completion_tokens,
        "mean_reasoning_tokens": mean_reasoning_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "reasoning_tokens_known_count": reasoning_tokens_known_count,
        "mean_explicit_reasoning_tokens": mean_explicit_reasoning_tokens,
        "total_explicit_reasoning_tokens": total_explicit_reasoning_tokens,
        "explicit_reasoning_tokens_known_count": explicit_reasoning_tokens_known_count,
        "total_prompt_tokens": total_prompt_tokens_value,
        "total_completion_tokens": total_completion_tokens_value,
        "total_tokens_sum": total_total_tokens_value,
        "total_cost_usd_known": total_cost,
        "unknown_usage_count": unknown_usage_count,
        "warning_row_count": int(warning_row_count),
        "warning_counts": warning_counts,
        "breakdown_by_group": breakdown_by("group"),
        "breakdown_by_year": breakdown_by("year"),
        "text_only_evaluation": bool(cli_text_only or (not model_info.supports_vision)),
        "text_only_source": (
            "cli"
            if cli_text_only
            else ("model" if not model_info.supports_vision else "none")
        ),
        "cli_filtered_out_multimodal_rows": int(cli_filtered_out_rows),
        "model_filtered_out_multimodal_rows": int(model_filtered_out_rows),
        "total_rows_loaded": int(total_rows_loaded),
        "rows_after_filters": int(len(results_df)),
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    args_snapshot = vars(args).copy()
    args_snapshot["dataset"] = dataset_path
    args_snapshot["worker_count"] = worker_count
    args_snapshot["reasoning"] = SELF_DIRECTED_MODE
    args_snapshot["internal_effort"] = None
    config = {
        "timestamp_utc": ts,
        "args": args_snapshot,
        "model": dataclasses.asdict(model_info),
        "multimodal_policy": {
            "text_only_cli": cli_text_only,
            "model_supports_vision": bool(model_info.supports_vision),
            "effective_text_only": bool(
                cli_text_only or (not model_info.supports_vision)
            ),
            "total_rows_loaded": int(total_rows_loaded),
            "cli_filtered_out_rows": int(cli_filtered_out_rows),
            "model_filtered_out_rows": int(model_filtered_out_rows),
            "rows_after_filter": int(len(rows_data)),
        },
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Add proper spacing before post-run summary to avoid cramped appearance
    print()
    print("Summary:")
    print(f"  Answered: {answered_count}")
    if declined_count > 0:
        print(f"  Chose not to answer: {declined_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Points-weighted accuracy: {p_weighted_accuracy:.3f}")
    if mean_latency is not None:
        print(f"  Mean latency: {mean_latency:.1f} ms (median {median_latency:.1f} ms)")
    else:
        print("  Mean latency: n/a")
    total_tokens_known = int(len(total_tokens_series))
    if mean_tokens is not None and total_total_tokens_value is not None:
        print(
            "  Total tokens: "
            f"mean {mean_tokens:.1f}, total {total_total_tokens_value} (rows {total_tokens_known})"
        )
    else:
        print("  Total tokens: n/a")
    if mean_prompt_tokens is not None and total_prompt_tokens_value is not None:
        print(
            "  Input tokens: "
            f"mean {mean_prompt_tokens:.1f}, total {total_prompt_tokens_value} (rows {prompt_tokens_known})"
        )
    else:
        print("  Input tokens: n/a")
    if mean_completion_tokens is not None and total_completion_tokens_value is not None:
        print(
            "  Output tokens: "
            f"mean {mean_completion_tokens:.1f}, total {total_completion_tokens_value} (rows {completion_tokens_known})"
        )
    else:
        print("  Output tokens: n/a")
    if total_reasoning_tokens is not None and mean_reasoning_tokens is not None:
        print(
            "  Reasoning tokens: "
            f"mean {mean_reasoning_tokens:.1f}, total {total_reasoning_tokens} (rows {reasoning_tokens_known_count})"
        )
    else:
        print("  Reasoning tokens: n/a")
    if (
        total_explicit_reasoning_tokens is not None
        and mean_explicit_reasoning_tokens is not None
    ):
        print(
            "  Explicit reasoning tokens: "
            f"mean {mean_explicit_reasoning_tokens:.1f}, total {total_explicit_reasoning_tokens} (rows {explicit_reasoning_tokens_known_count})"
        )
    else:
        print("  Explicit reasoning tokens: n/a")
    print(f"  Known total cost: ${total_cost:.4f}")
    print(f"  Unknown usage rows: {unknown_usage_count}")
    print(f"  Rows with warnings: {warning_row_count}")
    if warning_counts_counter:
        top_warnings = ", ".join(
            f"{warn}×{count}" for warn, count in warning_counts_counter.most_common(3)
        )
        if top_warnings:
            print(f"  Top warnings: {top_warnings}")
    print(f"  Worker count: {worker_count}")

    # Calculate and display total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    hours, rem = divmod(int(total_runtime), 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        runtime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        runtime_str = f"{minutes:02d}:{seconds:02d}"

    print(f"  Total runtime: {runtime_str}")


if __name__ == "__main__":
    main()
