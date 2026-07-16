# core/retrieval/hybrid_search.py
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from typing import List, Optional, Dict
from config import TOP_K_RETRIEVAL

# was: single global _bm25_index/_bm25_chunks. now: one BM25 index per user_id,
# so different users' documents never get mixed in keyword search.
_bm25_indexes: Dict[str, BM25Okapi] = {}
_bm25_chunks: Dict[str, List[Document]] = {}


def build_bm25_index(chunks: List[Document], user_id: str) -> None:
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    _bm25_indexes[user_id] = BM25Okapi(tokenized_corpus) if chunks else None
    _bm25_chunks[user_id] = chunks
    print(f"BM25 index built for user {user_id} with {len(chunks)} chunks")


def semantic_search(vector_store: Chroma, query: str, user_id: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
    return vector_store.similarity_search(query, k=k, filter={"user_id": user_id})


def bm25_search(query: str, user_id: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
    index = _bm25_indexes.get(user_id)
    chunks = _bm25_chunks.get(user_id)
    if index is None or not chunks:
        return []  # this user has no docs yet — not an error, just empty
    tokenized_query = query.lower().split()
    scores = index.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_indices]


def reciprocal_rank_fusion(
    semantic_results: List[Document],
    bm25_results: List[Document],
    k: int = 60  # RRF constant, 60 is standard
) -> List[Document]:
    scores = {}

    for rank, doc in enumerate(semantic_results):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, doc in enumerate(bm25_results):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    id_to_doc = {}
    for doc in semantic_results + bm25_results:
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        if doc_id not in id_to_doc:
            id_to_doc[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    result = []
    for doc_id in sorted_ids:
        doc = id_to_doc[doc_id]
        doc.metadata["rrf_score"] = round(scores[doc_id], 4)
        result.append(doc)

    return result


def hybrid_search(
    vector_store: Chroma,
    query: str,
    user_id: str,
    k: int = TOP_K_RETRIEVAL
) -> List[Document]:
    semantic_results = semantic_search(vector_store, query, user_id, k=k)
    bm25_results = bm25_search(query, user_id, k=k)
    fused = reciprocal_rank_fusion(semantic_results, bm25_results)
    return fused[:k]


def get_chunks_for_source(vector_store: Chroma, source: str, user_id: str) -> List[Document]:
    """Fetch all chunks belonging to one specific document for THIS user."""
    results = vector_store.get(where={"$and": [{"filename": source}, {"user_id": user_id}]})
    chunks = []
    if results and results.get("documents"):
        for doc_text, metadata in zip(results["documents"], results["metadatas"]):
            chunks.append(Document(page_content=doc_text, metadata=metadata))
    return chunks


def hybrid_search_scoped(vector_store: Chroma, query: str, source: str, user_id: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
    """
    Same idea as hybrid_search(), but scoped to a single document instead of
    the whole knowledge base — used for document comparison. Now also scoped
    to user_id so comparison can't pull another user's document.
    """
    semantic_results = vector_store.similarity_search(
        query, k=k, filter={"$and": [{"filename": source}, {"user_id": user_id}]}
    )

    doc_chunks = get_chunks_for_source(vector_store, source, user_id)
    if not doc_chunks:
        return semantic_results  # nothing to build a BM25 index from, fall back to semantic-only

    tokenized_corpus = [c.page_content.lower().split() for c in doc_chunks]
    temp_bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = temp_bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_results = [doc_chunks[i] for i in top_indices]

    fused = reciprocal_rank_fusion(semantic_results, bm25_results)
    return fused[:k]


def get_all_chunks(vector_store: Chroma, user_id: Optional[str] = None) -> List[Document]:
    """Fetch chunks currently in the vector store. Pass user_id to scope to one
    user (used to build their BM25 index); omit only for admin/debug purposes."""
    results = vector_store.get(where={"user_id": user_id}) if user_id else vector_store.get()
    chunks = []
    if results and results.get("documents"):
        for doc_text, metadata in zip(results["documents"], results["metadatas"]):
            chunks.append(Document(page_content=doc_text, metadata=metadata))
    return chunks