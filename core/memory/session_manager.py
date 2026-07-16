import json
from pathlib import Path
from typing import Dict, List
from core.memory.conversation import ConversationMemory

INDEX_DIR = Path("data/sessions")
MAX_STORED_CHATS = 30  # oldest chats drop off past this


class SessionMemoryManager:
    _sessions: Dict[str, ConversationMemory] = {}

    @classmethod
    def get(cls, session_id: str) -> ConversationMemory:
        if session_id not in cls._sessions:
            path = f"data/sessions/{session_id}.json"
            cls._sessions[session_id] = ConversationMemory(history_file=path)
        return cls._sessions[session_id]

    @classmethod
    def clear_session(cls, session_id: str) -> None:
        if session_id in cls._sessions:
            cls._sessions[session_id].clear()

    @classmethod
    def active_session_count(cls) -> int:
        return len(cls._sessions)

    @classmethod
    def register_chat(cls, user_id: str, session_id: str, title: str) -> None:
        index_path = INDEX_DIR / f"index_{user_id}.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)

        entries = []
        if index_path.exists():
            try:
                entries = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                entries = []

        if not any(e["session_id"] == session_id for e in entries):
            entries.insert(0, {"session_id": session_id, "title": title})
            entries = entries[:MAX_STORED_CHATS]  # cap it — oldest silently dropped
            index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    @classmethod
    def list_chats(cls, user_id: str) -> List[dict]:
        index_path = INDEX_DIR / f"index_{user_id}.json"
        if not index_path.exists():
            return []
        try:
            return json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return []