from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from typing import List, Optional
from config import TOP_K_RETRIEVAL

# BM25 singleton — built once, reused
_bm25_index: Optional[BM25Okapi] = None
_bm25_chunks: Optional[List[Document]] = None


def build_bm25_index(chunks: List[Document]) -> None:
    global _bm25_index, _bm25_chunks
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_chunks = chunks
    print(f"BM25 index built with {len(chunks)} chunks")


def semantic_search(vector_store: Chroma, query: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
    return vector_store.similarity_search(query, k=k)


def bm25_search(query: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
    if _bm25_index is None or _bm25_chunks is None:
        raise RuntimeError("BM25 index not built. Call build_bm25_index() first.")
    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [_bm25_chunks[i] for i in top_indices]


def reciprocal_rank_fusion(
    semantic_results: List[Document],
    bm25_results: List[Document],
    k: int = 60  # RRF constant, 60 is standard
) -> List[Document]:
    scores = {}

    # score each doc from semantic ranking
    for rank, doc in enumerate(semantic_results):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # score each doc from BM25 ranking
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # map chunk_id back to document
    id_to_doc = {}
    for doc in semantic_results + bm25_results:
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        if doc_id not in id_to_doc:
            id_to_doc[doc_id] = doc

    # sort by combined RRF score
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    # attach RRF score to metadata for visibility
    result = []
    for doc_id in sorted_ids:
        doc = id_to_doc[doc_id]
        doc.metadata["rrf_score"] = round(scores[doc_id], 4)
        result.append(doc)

    return result


def hybrid_search(
    vector_store: Chroma,
    query: str,
    k: int = TOP_K_RETRIEVAL
) -> List[Document]:
    semantic_results = semantic_search(vector_store, query, k=k)
    bm25_results = bm25_search(query, k=k)
    fused = reciprocal_rank_fusion(semantic_results, bm25_results)
    return fused[:k]

def get_chunks_for_source(vector_store: Chroma, source: str) -> List[Document]:
    """Fetch all chunks belonging to one specific document (for building a temporary per-doc BM25 index)."""
    results = vector_store.get(where={"filename": source})
    chunks = []
    if results and results.get("documents"):
        for doc_text, metadata in zip(results["documents"], results["metadatas"]):
            chunks.append(Document(page_content=doc_text, metadata=metadata))
    return chunks


def hybrid_search_scoped(vector_store: Chroma, query: str, source: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
    """
    Same idea as hybrid_search(), but scoped to a single document instead of
    the whole knowledge base — used for document comparison, where each side
    needs its own independent retrieval, not a mix of both documents.
    """
    semantic_results = vector_store.similarity_search(query, k=k, filter={"filename": source})

    doc_chunks = get_chunks_for_source(vector_store, source)
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

def get_all_chunks(vector_store: Chroma) -> List[Document]:
    """Fetch every chunk currently in the vector store — used to bootstrap the
    BM25 index in a fresh process that didn't run ingestion itself."""
    results = vector_store.get()
    chunks = []
    if results and results.get("documents"):
        for doc_text, metadata in zip(results["documents"], results["metadatas"]):
            chunks.append(Document(page_content=doc_text, metadata=metadata))
    return chunks