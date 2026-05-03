"""Pydantic schemas for the analysis module."""

from pydantic import BaseModel
from typing import Optional


class QuestionAnalysisRecord(BaseModel):
    question_id: str
    avg_llm_score: float
    llm_disagreement: float
    llm_count: int


class YearlyAnalysisRecord(BaseModel):
    year: str
    avg_human_score: Optional[float] = None
    avg_llm_score: Optional[float] = None
    normalized_human_score: Optional[float] = None
