import math
from langchain_core.documents import Document
from typing import List


def calculate_evidence_confidence(chunks: List[Document]) -> dict:
    if not chunks:
        return {"score": 0.0, "level": "none", "reason": "No chunks retrieved"}

    rerank_scores = [chunk.metadata.get("rerank_score", 0) for chunk in chunks]
    avg_score = sum(rerank_scores) / len(rerank_scores)
    top_score = max(rerank_scores)

    SCALE = 5
    normalized = 1 / (1 + math.exp(-top_score / SCALE))

    if normalized >= 0.75:
        level = "high"
    elif normalized >= 0.5:
        level = "medium"
    else:
        level = "low"

    return {
        "score": round(normalized, 2),
        "level": level,
        "top_rerank_score": round(top_score, 2),
        "avg_rerank_score": round(avg_score, 2)
    }