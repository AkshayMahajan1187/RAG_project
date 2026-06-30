import streamlit as st
import uuid
from pathlib import Path

from core.retrieval.vector_store import get_vector_store, add_documents
from core.retrieval.hybrid_search import build_bm25_index, get_all_chunks
from core.agents.planner import route_and_execute
from core.memory.session_manager import SessionMemoryManager
from core.ingestion.loader import load_document
from core.ingestion.chunker import chunk_documents

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🧠", layout="wide")


# --- One-time setup: load models + build BM25, cached across reruns ---
@st.cache_resource
def init_pipeline():
    vector_store = get_vector_store()
    all_chunks = get_all_chunks(vector_store)
    build_bm25_index(all_chunks)
    return vector_store


vector_store = init_pipeline()

UPLOAD_DIR = Path("data/uploaded_docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --- Session setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


def new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


def get_documents():
    try:
        results = vector_store.get()
        return sorted(set(
            m["filename"] for m in results["metadatas"] if m and "filename" in m
        ))
    except Exception as e:
        st.error(f"Could not list documents: {e}")
        return []


TRUST_COLORS = {
    "high": "#16a34a",
    "medium": "#ca8a04",
    "low": "#dc2626",
}


def render_trust_badge(trust_dict):
    if not trust_dict:
        return
    level = trust_dict.get("trust", "unknown")
    color = TRUST_COLORS.get(level, "#6b7280")
    st.markdown(
        f"""<span style="background-color:{color}20; color:{color};
        padding:3px 10px; border-radius:12px; font-size:0.8em; font-weight:600;
        border:1px solid {color}40;">● TRUST: {level.upper()}</span>""",
        unsafe_allow_html=True
    )


# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🧠 Knowledge Assistant")
    st.caption("Agentic RAG system")

    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()

    st.markdown("### 📄 Documents")
    docs = get_documents()
    if docs:
        for d in docs:
            st.markdown(f"- `{d}`")
    else:
        st.caption("No documents ingested yet.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("📤 Ingest document", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    save_path = UPLOAD_DIR / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    raw_docs = load_document(str(save_path))
                    chunks = chunk_documents(raw_docs)
                    add_documents(chunks)

                    all_chunks = get_all_chunks(vector_store)
                    build_bm25_index(all_chunks)

                    st.success(f"Added {len(chunks)} chunks from {uploaded_file.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.divider()

    st.markdown("### 🔍 Compare Mode")
    compare_mode = st.toggle("Enable comparison")
    file_a, file_b = None, None
    if compare_mode:
        if len(docs) >= 2:
            file_a = st.selectbox("Document A", docs, key="doc_a")
            file_b = st.selectbox("Document B", docs, key="doc_b")
        else:
            st.warning("Need at least 2 documents to compare.")


# --- Main chat area ---
st.markdown("### Ask anything about your documents")

if not st.session_state.messages:
    st.info("👋 Start by asking a question, or enable Compare Mode in the sidebar.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("intent") != "smalltalk":
            if msg.get("trust"):
                render_trust_badge(msg["trust"])

            if msg.get("citations"):
                with st.expander("📚 Citations"):
                    for c in msg["citations"]:
                        st.markdown(f"**[{c['ref']}]** {c['filename']} — page {c['page']}")
                        st.caption(c['snippet'])


user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    memory = SessionMemoryManager.get(st.session_state.session_id)

    with st.spinner("Thinking..."):
        try:
            result = route_and_execute(
                user_input=user_input,
                vector_store=vector_store,
                file_a=file_a if compare_mode else None,
                file_b=file_b if compare_mode else None,
                memory=memory
            )
        except Exception as e:
            result = {"answer": f"Error: {e}", "intent": "qa"}

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "intent": result.get("intent"),
        "trust": result.get("trust"),
        "citations": result.get("citations"),
    })

    st.rerun()