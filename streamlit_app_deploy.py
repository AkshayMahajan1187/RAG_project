import streamlit as st
import uuid
from pathlib import Path

from core.retrieval.vector_store import get_vector_store, add_documents
from core.retrieval.hybrid_search import build_bm25_index, get_all_chunks
from core.agents.planner import route_and_execute
from core.memory.session_manager import SessionMemoryManager
from core.ingestion.loader import load_document
from core.ingestion.chunker import chunk_documents
import extra_streamlit_components as stx


@st.cache_resource
def get_cookie_manager():
    return stx.CookieManager()


def get_user_id():
    cookie_manager = get_cookie_manager()
    user_id = cookie_manager.get("rag_user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        cookie_manager.set("rag_user_id", user_id, expires_at=None)
    return user_id


st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🧠", layout="wide")


@st.cache_resource
def init_pipeline():
    return get_vector_store()


vector_store = init_pipeline()

UPLOAD_DIR = Path("data/uploaded_docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --- Session setup ---

if "user_id" not in st.session_state:
    st.session_state.user_id = get_user_id()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "bm25_built_for" not in st.session_state or st.session_state.bm25_built_for != st.session_state.user_id:
    build_bm25_index(get_all_chunks(vector_store, st.session_state.user_id), st.session_state.user_id)
    st.session_state.bm25_built_for = st.session_state.user_id


def new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


def load_chat(session_id: str):
    """Switch to a previously saved chat and rebuild the screen from its saved history."""
    st.session_state.session_id = session_id
    memory = SessionMemoryManager.get(session_id)
    messages = []
    for turn in memory.history:
        messages.append({"role": "user", "content": turn["query"]})
        messages.append({
            "role": "assistant",
            "content": turn.get("answer", ""),
            "intent": turn.get("intent"),
            "trust": turn.get("trust"),
            "citations": turn.get("citations"),
            "query_reformulated": False,
            "reformulated_query": None,
        })
    st.session_state.messages = messages


def get_documents():
    try:
        results = vector_store.get(where={"user_id": st.session_state.user_id})
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

    # --- Chat history list ---
    st.markdown("### 💬 Chats")
    chats = SessionMemoryManager.list_chats(st.session_state.user_id)

    def render_chat_button(chat):
        label = chat["title"][:30] + ("..." if len(chat["title"]) > 30 else "")
        is_current = chat["session_id"] == st.session_state.session_id
        prefix = "🟢 " if is_current else ""
        if st.button(prefix + label, key=f"chat_{chat['session_id']}", use_container_width=True):
            load_chat(chat["session_id"])
            st.rerun()

    if chats:
        recent, older = chats[:10], chats[10:]
        for chat in recent:
            render_chat_button(chat)
        if older:
            with st.expander(f"Show {len(older)} older chats"):
                for chat in older:
                    render_chat_button(chat)
    else:
        st.caption("No past chats yet.")

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

                    for c in chunks:
                        c.metadata["user_id"] = st.session_state.user_id

                    add_documents(chunks)

                    all_chunks = get_all_chunks(vector_store, st.session_state.user_id)
                    build_bm25_index(all_chunks, st.session_state.user_id)

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

            if msg.get("query_reformulated"):
                st.info(f" Query reformulated for better retrieval: *\"{msg.get('reformulated_query')}\"*")

            if msg.get("citations"):
                with st.expander(" Citations"):
                    for c in msg["citations"]:
                        st.markdown(f"**[{c['ref']}]** {c['filename']} — page {c['page']}")
                        st.caption(c['snippet'])


user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    memory = SessionMemoryManager.get(st.session_state.session_id)
    is_first_turn = len(memory.history) == 0  # check BEFORE this turn gets added

    with st.spinner("Thinking..."):
        try:
            result = route_and_execute(
                user_input=user_input,
                vector_store=vector_store,
                user_id=st.session_state.user_id,
                file_a=file_a if compare_mode else None,
                file_b=file_b if compare_mode else None,
                memory=memory
            )
        except Exception as e:
            result = {"answer": f"Error: {e}", "intent": "qa"}

    if is_first_turn:
        SessionMemoryManager.register_chat(
            st.session_state.user_id,
            st.session_state.session_id,
            title=user_input
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "intent": result.get("intent"),
        "trust": result.get("trust"),
        "citations": result.get("citations"),
        "query_reformulated": result.get("query_reformulated"),
        "reformulated_query": result.get("reformulated_query"),
    })

    st.rerun()