import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_docs"
CHROMA_DIR = DATA_DIR / "chroma_db"

# === LLM ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.2
VERIFIER_LLM_MODEL = "llama-3.3-70b-versatile"

# === Embeddings ===
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# === Retrieval ===
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5
# Fuse hybrid RRF rank with cross-encoder score (sum should be 1.0).
# RRF weight prevents the reranker from overriding good hybrid results.
RERANK_RRF_WEIGHT = 0.45
RERANK_CE_WEIGHT = 0.55

# === Chunking ===
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# === ChromaDB ===
CHROMA_COLLECTION_NAME = "knowledge_base"