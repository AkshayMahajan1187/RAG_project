from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from config import CHROMA_DIR, CHROMA_COLLECTION_NAME
from core.retrieval.embedder import get_embedder


def get_vector_store() -> Chroma:
    embedder = get_embedder()
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=str(CHROMA_DIR)
    )
    return vector_store


def get_existing_hashes(vector_store: Chroma, file_path: str, user_id: str) -> set:
    try:
        results = vector_store.get(where={"$and": [{"source": file_path}, {"user_id": user_id}]})
        if results and results["metadatas"]:
            return {m["file_hash"] for m in results["metadatas"] if m and "file_hash" in m}
    except Exception:
        pass
    return set()

def delete_documents(vector_store: Chroma, file_path: str, user_id: str) -> None:
    try:
        results = vector_store.get(where={"$and": [{"source": file_path}, {"user_id": user_id}]})
        if results and results["ids"]:
            vector_store.delete(ids=results["ids"])
    except Exception as e:
        print(f"Failed to delete old chunks: {e}")

def get_chunks_by_hash(vector_store: Chroma, file_hash: str, user_id: str) -> set:
    try:
        results = vector_store.get(where={"$and": [{"file_hash": file_hash}, {"user_id": user_id}]})
        if results and results["ids"]:
            return set(results["ids"])
    except Exception:
        pass
    return set()

def add_documents(chunks: List[Document]) -> Chroma:
    # chunks must already carry metadata["user_id"] (set at upload time)
    vector_store = get_vector_store()
    if not chunks:
        return vector_store
    files = {}
    for chunk in chunks:
        files.setdefault(chunk.metadata["source"], []).append(chunk)

    new_chunks = []
    for source, file_chunks in files.items():
        file_hash = file_chunks[0].metadata["file_hash"]
        user_id = file_chunks[0].metadata["user_id"]
        filename = file_chunks[0].metadata["filename"]

        if get_chunks_by_hash(vector_store, file_hash, user_id):
            print(f"Skipping {filename} — identical content already ingested for this user")
            continue
        if get_existing_hashes(vector_store, source, user_id):
            delete_documents(vector_store, source, user_id)

        new_chunks.extend(file_chunks)

    if new_chunks:
        vector_store.add_documents(new_chunks)
    return vector_store