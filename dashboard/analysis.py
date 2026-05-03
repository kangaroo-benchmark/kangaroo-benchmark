"""Core analysis functions for multi-run aggregation."""

from collections import defaultdict
from typing import Dict, List

import numpy as np

from . import analysis_schemas as schemas
from .data_index import RunIndex


def analyze_question_difficulty(
    run_ids: List[str],
    index: RunIndex,
) -> List[schemas.QuestionAnalysisRecord]:
    """Analyzes question difficulty across multiple runs."""
    question_scores: Dict[str, List[float]] = defaultdict(list)

    for run_id in run_ids:
        try:
            results = index.iter_results(run_id)
            for row in results:
                if row.is_correct is not None:
                    score = 1.0 if row.is_correct else 0.0
                    question_scores[row.id].append(score)
        except FileNotFoundError:
            # Handle cases where a run might be deleted or inaccessible
            continue

    analysis_records: List[schemas.QuestionAnalysisRecord] = []
    for question_id, scores in question_scores.items():
        if not scores:
            continue

        np_scores = np.array(scores)
        avg_score = float(np.mean(np_scores))
        disagreement = float(np.std(np_scores))
        count = len(scores)

        record = schemas.QuestionAnalysisRecord(
            question_id=question_id,
            avg_llm_score=avg_score,
            llm_disagreement=disagreement,
            llm_count=count,
        )
        analysis_records.append(record)

    return analysis_records
