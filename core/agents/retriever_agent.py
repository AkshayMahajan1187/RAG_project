from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
from core.retrieval.hybrid_search import hybrid_search
from core.retrieval.reranker import rerank


def retrieve(query: str, vector_store: Chroma, k: int = 10) -> List[Document]:
    candidates = hybrid_search(vector_store, query, k=k)
    return rerank(query, candidates)