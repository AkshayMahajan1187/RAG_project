from typing import Dict
from core.memory.conversation import ConversationMemory


class SessionMemoryManager:
    """
    Routes each session_id to its own isolated ConversationMemory instance,
    so multiple users/browser tabs never share or corrupt each other's history.
    """
    _sessions: Dict[str, ConversationMemory] = {}

    @classmethod
    def get(cls, session_id: str) -> ConversationMemory:
        if session_id not in cls._sessions:
            path = f"data/sessions/{session_id}.json"
            cls._sessions[session_id] = ConversationMemory(history_file=path)
        return cls._sessions[session_id]

    @classmethod
    def clear_session(cls, session_id: str) -> None:
        """Clear a specific session's history (used for 'new chat' in the UI later)."""
        if session_id in cls._sessions:
            cls._sessions[session_id].clear()

    @classmethod
    def active_session_count(cls) -> int:
        """How many sessions are currently loaded in memory (useful for debugging)."""
        return len(cls._sessions)