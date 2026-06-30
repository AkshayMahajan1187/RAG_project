from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List
from config import TOP_K_RERANK

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, documents: List[Document]) -> List[Document]:
    if not documents:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)

    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

    top_docs = []
    for score, doc in scored_docs[:TOP_K_RERANK]:
        doc.metadata["rerank_score"] = round(float(score), 4)
        top_docs.append(doc)

    return top_docs