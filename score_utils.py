"""Shared scoring utilities for Känguru-style evaluations."""

from __future__ import annotations

import math
import re
from typing import Optional, Sequence, Tuple

DECLINED_TOKEN = "DECLINED"
PENALTY_FACTOR = 0.25
START_POINTS_LOWER_GRADES = 24.0
START_POINTS_UPPER_GRADES = 30.0
LOWER_GRADE_MIN = 3
LOWER_GRADE_MAX = 6
UPPER_GRADE_MIN = 7
UPPER_GRADE_MAX = 13


def coerce_points(value: object) -> float:
    """Return ``value`` as a finite float or ``0.0`` if coercion fails."""
    if value is None:
        return 0.0
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 0.0
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def _as_int(value: object) -> Optional[int]:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def extract_grade_numbers(group_value: object) -> Tuple[int, ...]:
    """Extract integer grade numbers from a free-form group label."""
    if group_value is None:
        return ()
    text = str(group_value)
    numbers = [_as_int(match) for match in re.findall(r"\d+", text)]
    return tuple(num for num in numbers if num is not None)


def start_points_for_numbers(numbers: Sequence[int]) -> float:
    """Return the official start capital for the given grade numbers."""
    if not numbers:
        return 0.0
    # Use the smallest grade to decide the bucket.
    smallest = min(numbers)
    if UPPER_GRADE_MIN <= smallest <= UPPER_GRADE_MAX:
        return START_POINTS_UPPER_GRADES
    if LOWER_GRADE_MIN <= smallest <= LOWER_GRADE_MAX:
        return START_POINTS_LOWER_GRADES
    return 0.0


def start_points_for_group(group_value: object) -> float:
    """Return start capital inferred from a dataset ``group`` label."""
    return start_points_for_numbers(extract_grade_numbers(group_value))


def start_points_for_members(members: Sequence[int]) -> float:
    """Return start capital based on explicit grade membership numbers."""
    return start_points_for_numbers(members)


def is_declined(predicted: Optional[str]) -> bool:
    if predicted is None:
        return False
    return predicted.strip().upper() == DECLINED_TOKEN


def score_question(
    points: object,
    correct_answer: Optional[str],
    predicted: Optional[str],
    *,
    penalize_unanswered: bool = True,
) -> Tuple[float, Optional[bool]]:
    """Compute earned points and correctness for a single question.

    Returns a tuple ``(points_earned, is_correct)`` where ``is_correct`` is
    ``None`` when the item cannot be evaluated (e.g. no official solution).
    """

    question_points = coerce_points(points)
    if correct_answer is None or question_points <= 0.0:
        return 0.0, None

    if predicted is None:
        if penalize_unanswered:
            return -question_points * PENALTY_FACTOR, False
        return 0.0, False

    if is_declined(predicted):
        return 0.0, False

    normalized = predicted.strip().upper()
    if normalized == correct_answer:
        return question_points, True
    return -question_points * PENALTY_FACTOR, False


__all__ = [
    "coerce_points",
    "extract_grade_numbers",
    "is_declined",
    "score_question",
    "start_points_for_group",
    "start_points_for_members",
    "start_points_for_numbers",
    "PENALTY_FACTOR",
    "START_POINTS_LOWER_GRADES",
    "START_POINTS_UPPER_GRADES",
]
