import math
from langchain_core.documents import Document
from typing import List


def calculate_evidence_confidence(chunks: List[Document]) -> dict:
    if not chunks:
        return {"score": 0.0, "level": "none", "reason": "No chunks retrieved"}

    # Prefer fused score when available; fall back to raw cross-encoder score.
    scores = [
        chunk.metadata.get("combined_score", chunk.metadata.get("rerank_score", 0))
        for chunk in chunks
    ]
    avg_score = sum(scores) / len(scores)
    top_score = max(scores)

    # combined_score is already 0-1; rerank_score needs sigmoid normalization.
    if all(chunk.metadata.get("combined_score") is not None for chunk in chunks):
        normalized = top_score
    else:
        normalized = 1 / (1 + math.exp(-top_score / 5))

    if normalized >= 0.75:
        level = "high"
    elif normalized >= 0.5:
        level = "medium"
    else:
        level = "low"

    return {
        "score": round(normalized, 2),
        "level": level,
        "top_score": round(top_score, 2),
        "top_rerank_score": round(chunks[0].metadata.get("rerank_score", 0), 2),
        "avg_score": round(avg_score, 2),
    }
