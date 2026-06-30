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


def get_existing_hashes(vector_store: Chroma, file_path: str) -> set:
    try:
        results = vector_store.get(where={"source": file_path})
        if results and results["metadatas"]:
            return {m["file_hash"] for m in results["metadatas"] if m and "file_hash" in m}
    except Exception:
        pass
    return set()

def delete_documents(vector_store: Chroma, file_path: str) -> None:
    try:
        results = vector_store.get(where={"source": file_path})
        if results and results["ids"]:
            vector_store.delete(ids=results["ids"])
            print(f"Deleted {len(results['ids'])} old chunks for {file_path}")
    except Exception as e:
        print(f"Failed to delete old chunks: {e}")

def get_chunks_by_hash(vector_store: Chroma, file_hash: str) -> set:
    """Check if this exact content (by hash) exists anywhere, regardless of filename."""
    try:
        results = vector_store.get(where={"file_hash": file_hash})
        if results and results["ids"]:
            return set(results["ids"])
    except Exception:
        pass
    return set()


def add_documents(chunks: List[Document]) -> Chroma:
    vector_store = get_vector_store()

    if not chunks:
        print("No chunks to add.")
        return vector_store

    files = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        files.setdefault(source, []).append(chunk)

    new_chunks = []
    for source, file_chunks in files.items():
        file_hash = file_chunks[0].metadata["file_hash"]
        filename = file_chunks[0].metadata["filename"]

        # Case 1: identical content already exists somewhere (same or different filename)
        if get_chunks_by_hash(vector_store, file_hash):
            print(f"Skipping {filename} — identical content already ingested")
            continue

        # Case 2: this exact file path exists but with OLD/different content — replace it
        existing_at_path = get_existing_hashes(vector_store, source)
        if existing_at_path:
            print(f"{filename} changed — replacing old version")
            delete_documents(vector_store, source)

        new_chunks.extend(file_chunks)

    if not new_chunks:
        return vector_store

    vector_store.add_documents(new_chunks)
    print(f"Added {len(new_chunks)} new chunks to vector store")
    return vector_store