import math
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import TOP_K_RERANK, RERANK_RRF_WEIGHT, RERANK_CE_WEIGHT

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _sigmoid(x: float, scale: float = 5.0) -> float:
    return 1.0 / (1.0 + math.exp(-x / scale))


def rerank(query: str, documents: List[Document]) -> List[Document]:
    """
    Re-rank hybrid-search candidates using a cross-encoder fused with RRF scores.

    Pure cross-encoder reranking often promotes dense summary chunks over the
    section that explicitly answers the question. Fusing with the hybrid RRF
    score keeps retrieval quality while still benefiting from cross-encoder precision.
    """
    if not documents:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    ce_scores = reranker.predict(pairs)

    rrf_scores = [doc.metadata.get("rrf_score", 0.0) for doc in documents]
    max_rrf = max(rrf_scores) if rrf_scores else 1.0
    if max_rrf <= 0:
        max_rrf = 1.0

    scored_docs = []
    for ce_score, doc in zip(ce_scores, documents):
        rrf = doc.metadata.get("rrf_score", 0.0)
        normalized_rrf = rrf / max_rrf
        normalized_ce = _sigmoid(float(ce_score))

        combined = (RERANK_RRF_WEIGHT * normalized_rrf) + (RERANK_CE_WEIGHT * normalized_ce)

        doc.metadata["rerank_score"] = round(float(ce_score), 4)
        doc.metadata["combined_score"] = round(combined, 4)
        scored_docs.append((combined, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:TOP_K_RERANK]]
