"""
Ingest documents into the vector store.

Usage:
    python ingest_documents.py data/uploaded_docs/quantum_notes.pdf
    python ingest_documents.py data/uploaded_docs/nlp_assignment.pdf data/uploaded_docs/cn_notes.pdf

You can pass one or more file paths. Each file goes through:
load -> chunk -> add to ChromaDB (with dedup, handled automatically by add_documents).
"""

import sys
from core.ingestion.loader import load_document
from core.ingestion.chunker import chunk_documents
from core.retrieval.vector_store import add_documents


def ingest_file(file_path: str):
    print(f"\nIngesting: {file_path}")
    docs = load_document(file_path)
    print(f"  Loaded {len(docs)} pages")

    chunks = chunk_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    add_documents(chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_documents.py <file1.pdf> [file2.pdf] ...")
        sys.exit(1)

    for file_path in sys.argv[1:]:
        ingest_file(file_path)

    print("\nDone. Run generate_benchmark.py next to include these in the benchmark.")