import argparse
import base64
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from PIL import Image
from tqdm import tqdm


# Change this to the base URL of your OpenAI-compatible endpoint.
# Using a different API type might need specialised handling of reasoning output, for example.
API_BASE_URL = "https://openrouter.ai/api/v1"


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
]

LETTER_SET = {"A", "B", "C", "D", "E"}
DECLINED_TOKEN = "DECLINED"

FINAL_ANSWER_PATTERN = re.compile(
    r"(?im)^\s*(?:final\s*answer|antwort)\s*[:\uFF1A\-\u2013\u2014]?\s*([^\n]+?)\s*$"
)

IMAGE_MAX_DIM = 1024
IMAGE_JPEG_QUALITY = 85


def validate_dataset_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


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


def coerce_bytes(value: Any) -> Optional[bytes]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        if len(value) <= 16:
            return None
        try:
            return base64.b64decode(value, validate=False)
        except Exception:
            return None
    return None


def coerce_list_of_bytes(value: Any) -> Optional[List[bytes]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        out: List[bytes] = []
        for item in value:
            data = coerce_bytes(item)
            if data is not None:
                out.append(data)
        return out or None
    try:
        from collections.abc import Sequence

        if isinstance(value, Sequence) and not isinstance(
            value, (bytes, bytearray, str)
        ):
            out = []
            for item in value:
                data = coerce_bytes(item)
                if data is not None:
                    out.append(data)
            return out or None
    except Exception:
        return None
    return None


def pil_from_bytes(data: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(BytesIO(data))
        img.load()
        return img
    except Exception:
        return None


def image_to_data_url(img: Image.Image) -> str:
    if max(img.size) > IMAGE_MAX_DIM:
        resize_filter = (
            Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        )
        img = img.copy()
        img.thumbnail((IMAGE_MAX_DIM, IMAGE_MAX_DIM), resize_filter)
    fmt = "PNG" if img.mode in {"RGBA", "LA"} else "JPEG"
    buf = BytesIO()
    if fmt == "JPEG":
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(
            buf,
            format="JPEG",
            quality=IMAGE_JPEG_QUALITY,
            optimize=True,
            progressive=True,
        )
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def encode_row_images(row: Dict[str, Any]) -> Dict[str, Any]:
    encoded: Dict[str, Any] = {"assoc_list": []}
    for letter in ["A", "B", "C", "D", "E"]:
        blob = coerce_bytes(row.get(f"sol_{letter}_image_bin"))
        if not blob:
            continue
        img = pil_from_bytes(blob)
        if img is None:
            continue
        encoded[f"opt_{letter}"] = image_to_data_url(img)
    assoc_bytes = coerce_list_of_bytes(row.get("associated_images_bin")) or []
    for blob in assoc_bytes:
        img = pil_from_bytes(blob)
        if img is None:
            continue
        encoded["assoc_list"].append(image_to_data_url(img))
    return encoded


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            parts.append(normalize_message_content(item))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "json" in content:
            try:
                return json.dumps(content["json"], ensure_ascii=False)
            except Exception:
                return str(content["json"])
        if "content" in content:
            return normalize_message_content(content["content"])
    return str(content)


def parse_answer_from_text(
    text: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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


SYSTEM_PROMPT = "\n".join(
    [
        "Du bist ein hilfreicher Assistent für Multiple-Choice-Aufgaben.",
        "Bewertung: Eine korrekte Antwort erhält die vollen Punkte der Aufgabe; eine falsche oder nicht eindeutig auswertbare Endantwort zieht ein Viertel der Aufgabenpunkte ab; 'Ich entscheide mich, nicht zu antworten.' ergibt 0 Punkte ohne Abzug.",
        "Wähle die beste Option (A-E). Wenn die richtige Option nicht mit ausreichender Sicherheit bestimmbar ist oder die Aufgabe mehrdeutig ist, kannst du ausdrücklich mit 'Ich entscheide mich, nicht zu antworten.' antworten. Begründungen sind optional.",
        "Füge am Ende eine einzelne Zeile an: Final answer: A|B|C|D|E|Declined.",
    ]
)

FINAL_INSTRUCTION = "Erinnere dich: Beende deine Antwort mit einer Zeile in diesem Format - Final answer: A|B|C|D|E|Declined."


def build_messages(
    row: Dict[str, Any],
    *,
    encoded_images: Optional[Dict[str, Any]] = None,
    image_detail: str = "auto",
) -> List[Dict[str, Any]]:
    if encoded_images is None:
        encoded_images = {}

    content_parts: List[Dict[str, Any]] = []

    content_parts.append({"type": "text", "text": f"Frage: {row['problem_statement']}"})

    assoc = encoded_images.get("assoc_list") or []
    for i, url in enumerate(assoc, start=1):
        if url:
            content_parts.append({"type": "text", "text": f"Zusatzbild {i}:"})
            content_parts.append(
                {"type": "image_url", "image_url": {"url": url, "detail": image_detail}}
            )

    content_parts.append({"type": "text", "text": "Antwortmöglichkeiten:"})
    for letter in ["A", "B", "C", "D", "E"]:
        opt_text = row.get(f"sol_{letter}") or ""
        content_parts.append({"type": "text", "text": f"{letter}) {opt_text}"})
        url = encoded_images.get(f"opt_{letter}")
        if url:
            content_parts.append({"type": "text", "text": f"Option {letter} Bild:"})
            content_parts.append(
                {"type": "image_url", "image_url": {"url": url, "detail": image_detail}}
            )

    content_parts.append({"type": "text", "text": FINAL_INSTRUCTION})

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": content_parts})
    return messages


PENALTY_FACTOR = 0.25


def coerce_points(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not (result == result) or result in (float("inf"), float("-inf")):
        return 0.0
    return result


def score_question(
    points: Any,
    correct_answer: Optional[str],
    predicted: Optional[str],
    *,
    penalize_unanswered: bool = True,
) -> Tuple[float, Optional[bool]]:
    question_points = coerce_points(points)
    if correct_answer is None or question_points <= 0.0:
        return 0.0, None

    if predicted is None:
        if penalize_unanswered:
            return -question_points * PENALTY_FACTOR, False
        return 0.0, False

    if predicted.strip().upper() == DECLINED_TOKEN:
        return 0.0, False

    normalized = predicted.strip().upper()
    if normalized == correct_answer:
        return question_points, True
    return -question_points * PENALTY_FACTOR, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple standalone evaluator for the Känguru benchmark."
    )
    parser.add_argument(
        "--dataset",
        default="dataset_full.edited.corrected.parquet",
        help="Path to the dataset parquet file.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID to send to the OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optionally limit the number of rows to evaluate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens for the completion.",
    )
    parser.add_argument(
        "--output",
        default="results.jsonl",
        help="Path to write per-row results as JSONL.",
    )
    parser.add_argument(
        "--no-multimodal",
        action="store_true",
        help="Filter out questions that include any option/associated images (question_image is always present).",
    )

    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    df = pd.read_parquet(dataset_path, engine="pyarrow")
    validate_dataset_columns(df)

    if args.no_multimodal:

        def has_additional_images(row: pd.Series) -> bool:
            for letter in ["A", "B", "C", "D", "E"]:
                val = row.get(f"sol_{letter}_image_bin")
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    return True
            assoc = row.get("associated_images_bin")
            if assoc is not None and not (isinstance(assoc, float) and pd.isna(assoc)):
                try:
                    if isinstance(assoc, (list, tuple)) and len(assoc) > 0:
                        return True
                    if isinstance(assoc, str) and assoc.strip():
                        return True
                except Exception:
                    return True
            return False

        mask = ~df.apply(has_additional_images, axis=1)
        filtered_out = int((~mask).sum())
        df = df.loc[mask].reset_index(drop=True)
        print(
            f"--no-multimodal enabled: filtered out {filtered_out} rows with option/associated images"
        )

    if args.max_rows is not None and args.max_rows < len(df):
        df = df.head(args.max_rows)

    rows = df.to_dict(orient="records")

    print("Configuration:")
    print(f"  API base URL: {API_BASE_URL}")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Rows: {len(rows)}")
    print()

    answered = 0
    declined = 0
    correct = 0
    failed = 0
    total_points_earned = 0.0
    total_points_possible = 0.0

    url = f"{API_BASE_URL.rstrip('/')}/chat/completions"

    with (
        httpx.Client(timeout=60.0) as client,
        open(args.output, "w", encoding="utf-8") as out_f,
    ):
        pbar = tqdm(total=len(rows), desc="Evaluating")
        for row in rows:
            answer_value = row.get("answer")
            if pd.isna(answer_value) or answer_value is None:
                correct_answer = None
            else:
                correct_answer = str(answer_value).strip().upper()
                if correct_answer not in LETTER_SET:
                    correct_answer = None

            if correct_answer is not None:
                total_points_possible += coerce_points(row.get("points"))

            encoded_images = encode_row_images(row)
            messages = build_messages(
                row, encoded_images=encoded_images, image_detail="auto"
            )

            payload: Dict[str, Any] = {
                "model": args.model,
                "messages": messages,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            if args.max_tokens is not None:
                payload["max_tokens"] = args.max_tokens

            try:
                resp = client.post(url, json=payload)
                pbar.update(1)
            except Exception as exc:
                failed += 1
                record = {
                    "id": row.get("id"),
                    "error": f"request_failed: {exc}",
                    "predicted": None,
                    "raw_text_response": None,
                    "is_correct": None,
                    "points_earned": 0.0,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                pbar.update(1)
                continue

            if resp.status_code != 200:
                failed += 1
                text = (resp.text or "").strip()
                snippet = re.sub(r"\s+", " ", text)[:200] if text else ""
                record = {
                    "id": row.get("id"),
                    "error": f"http_error_{resp.status_code}: {snippet}",
                    "predicted": None,
                    "raw_text_response": text or None,
                    "is_correct": None,
                    "points_earned": 0.0,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                continue

            try:
                data = resp.json()
            except Exception:
                failed += 1
                text = (resp.text or "").strip()
                record = {
                    "id": row.get("id"),
                    "error": "invalid_json_response",
                    "predicted": None,
                    "raw_text_response": text or None,
                    "is_correct": None,
                    "points_earned": 0.0,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                continue

            choices = data.get("choices") or []
            if not choices or not isinstance(choices, list):
                failed += 1
                record = {
                    "id": row.get("id"),
                    "error": "no_choices_in_response",
                    "predicted": None,
                    "raw_text_response": None,
                    "is_correct": None,
                    "points_earned": 0.0,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                continue

            choice0 = choices[0] or {}
            message = choice0.get("message") or {}
            content_text = normalize_message_content(message.get("content"))

            predicted, rationale, parse_warning = parse_answer_from_text(content_text)

            if predicted is not None:
                answered += 1
                if predicted == DECLINED_TOKEN:
                    declined += 1

            points_earned, is_correct = score_question(
                row.get("points"),
                correct_answer,
                predicted,
                penalize_unanswered=True,
            )

            if is_correct:
                correct += 1

            total_points_earned += points_earned

            record = {
                "id": row.get("id"),
                "year": row.get("year"),
                "group": row.get("group"),
                "problem_number": row.get("problem_number"),
                "multimodal": row.get("multimodal"),
                "points": row.get("points"),
                "answer": correct_answer,
                "predicted": predicted,
                "is_correct": is_correct,
                "points_earned": points_earned,
                "rationale": rationale,
                "raw_text_response": content_text,
                "parse_warning": parse_warning,
                "error": None,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

        pbar.close()

    print("Summary:")
    print(f"  Answered: {answered}")
    print(f"  Declined: {declined}")
    print(f"  Failed: {failed}")
    if answered > 0:
        accuracy = correct / answered
        print(f"  Accuracy: {accuracy:.3f}")
    else:
        print("  Accuracy: n/a")
    if total_points_possible > 0.0:
        p_weighted = total_points_earned / total_points_possible
        print(f"  Points-weighted accuracy: {p_weighted:.3f}")
    else:
        print("  Points-weighted accuracy: n/a")


if __name__ == "__main__":
    main()
