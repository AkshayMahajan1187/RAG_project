from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, List

from core.retrieval.vector_store import get_vector_store
from core.retrieval.hybrid_search import build_bm25_index, get_all_chunks
from core.agents.planner import route_and_execute
from core.memory.session_manager import SessionMemoryManager

from fastapi import UploadFile, File
from pathlib import Path
from core.ingestion.loader import load_document
from core.ingestion.chunker import chunk_documents
from core.retrieval.vector_store import add_documents

app = FastAPI(title="AI Knowledge Assistant API")

# --- Bootstrap on startup: load vector store + BM25 once, not per-request ---
vector_store = get_vector_store()
DEFAULT_USER_ID = "default"
all_chunks = get_all_chunks(vector_store, DEFAULT_USER_ID)
build_bm25_index(all_chunks, DEFAULT_USER_ID)


# --- Request/response schemas ---

class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: str = "default"
    file_a: Optional[str] = None
    file_b: Optional[str] = None


class ChatResponse(BaseModel):
    intent: Literal["qa", "compare", "smalltalk"]
    answer: str
    planner_reasoning: Optional[str] = None
    # QA-specific
    citations: Optional[list] = None
    confidence: Optional[dict] = None
    hallucination: Optional[dict] = None
    trust: Optional[dict] = None
    retries: Optional[int] = None
    # Compare-specific
    chunks_a: Optional[list] = None
    chunks_b: Optional[list] = None
    confidence_a: Optional[dict] = None
    confidence_b: Optional[dict] = None


# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    memory = SessionMemoryManager.get(request.session_id)

    try:
        result = route_and_execute(
            user_input=request.message,
            vector_store=vector_store,
            user_id=request.user_id,
            file_a=request.file_a,
            file_b=request.file_b,
            memory=memory
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return ChatResponse(**result)


@app.get("/history/{session_id}")
def get_history(session_id: str):
    memory = SessionMemoryManager.get(session_id)
    return {"session_id": session_id, "history": memory.history}


@app.get("/")
def root():
    return {"status": "AI Knowledge Assistant API is running"}

UPLOAD_DIR = Path("data/uploaded_docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
def upload_document(file: UploadFile = File(...), user_id: str = "default"):
    # Step 1: save the uploaded file to disk
    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            f.write(file.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Step 2: load + chunk + add to vector store (reuses your existing pipeline)
    try:
        raw_docs = load_document(str(save_path))
        chunks = chunk_documents(raw_docs)
        for c in chunks:
            c.metadata["user_id"] = user_id
        add_documents(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    # Step 3: rebuild BM25 with the COMPLETE chunk set (old + new)
    global vector_store
    try:
        all_chunks = get_all_chunks(vector_store, user_id)
        build_bm25_index(all_chunks, user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BM25 rebuild failed: {e}")

    return {
        "filename": file.filename,
        "chunks_added": len(chunks),
        "total_chunks_in_system": len(all_chunks)
    }

@app.get("/documents")
def list_documents():
    global vector_store
    try:
        results = vector_store.get()
        filenames = sorted(set(
            m["filename"] for m in results["metadatas"] if m and "filename" in m
        ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

    return {"documents": filenames}

@app.get("/sessions")
def list_sessions():
    from pathlib import Path
    session_dir = Path("data/sessions")
    if not session_dir.exists():
        return {"sessions": []}

    sessions = []
    for file in session_dir.glob("*.json"):
        session_id = file.stem
        memory = SessionMemoryManager.get(session_id)
        if memory.history:
            first_question = memory.history[0]["query"]
            last_timestamp = memory.history[-1]["timestamp"]
            sessions.append({
                "session_id": session_id,
                "preview": first_question[:60],
                "last_active": last_timestamp,
                "turn_count": len(memory.history)
            })

    sessions.sort(key=lambda s: s["last_active"], reverse=True)
    return {"sessions": sessions}