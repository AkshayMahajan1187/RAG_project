import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🧠", layout="wide")

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
        resp = requests.get(f"{API_URL}/documents")
        resp.raise_for_status()
        return resp.json()["documents"]
    except Exception as e:
        st.error(f"Could not fetch document list: {e}")
        return []


TRUST_COLORS = {
    "high": "#16a34a",    # green
    "medium": "#ca8a04",  # amber
    "low": "#dc2626",     # red
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

    st.divider()
    st.markdown("### 🕐 Recent Chats")

    try:
        resp = requests.get(f"{API_URL}/sessions")
        sessions = resp.json().get("sessions", [])
    except Exception:
        sessions = []

    for s in sessions[:10]:  # show last 10
        label = f"{s['preview']}..." if len(s['preview']) == 60 else s['preview']
        if st.button(label, key=f"session_{s['session_id']}", use_container_width=True):
            st.session_state.session_id = s['session_id']
            hist_resp = requests.get(f"{API_URL}/history/{s['session_id']}")
            history = hist_resp.json().get("history", [])

            # rebuild displayed messages from saved history
            st.session_state.messages = []
            for turn in history:
                st.session_state.messages.append({"role": "user", "content": turn["query"]})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": turn["answer"],
                    "intent": turn["intent"],
                    "trust": turn["trust"],
                    "citations": turn["citations"],
                    "planner_reasoning": None  # not stored historically, that's fine
                })
            st.rerun()

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
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    resp = requests.post(f"{API_URL}/upload", files=files)
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(f"Added {result['chunks_added']} chunks from {result['filename']}")
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

    payload = {
        "session_id": st.session_state.session_id,
        "message": user_input,
        "file_a": file_a if compare_mode else None,
        "file_b": file_b if compare_mode else None
    }

    with st.spinner("Thinking..."):
        try:
            resp = requests.post(f"{API_URL}/chat", json=payload)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            result = {"answer": f"Error: {e}", "intent": "qa"}

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "intent": result.get("intent"),
        "trust": result.get("trust"),
        "citations": result.get("citations"),
        "planner_reasoning": result.get("planner_reasoning")
    })

    st.rerun()