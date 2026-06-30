import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class ConversationMemory:
    def __init__(self, history_file: str = "data/conversation_history.json",
                 max_turns: int = 3, max_stored_turns: int = 50):
        self.history_file = Path(history_file)
        self.max_turns = max_turns          # how many turns get sent to the LLM as context
        self.max_stored_turns = max_stored_turns  # how many turns we keep on disk overall
        self.history = self._load_history()

    def _load_history(self) -> List[dict]:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, default=str)

    def add_turn(self, query: str, intent: str, answer: str, trust: Optional[dict] = None,
            citations: Optional[List[dict]] = None, grounded: Optional[bool] = None,
            file_a: Optional[str] = None, file_b: Optional[str] = None,
            confidence_a: Optional[dict] = None, confidence_b: Optional[dict] = None) -> None:
            """Record a single user query + system response."""
            turn = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "intent": intent,
                "answer": answer,
                "trust": trust,
                "grounded": grounded,
                "citations": citations,
                "file_a": file_a,
                "file_b": file_b,
                "confidence_a": confidence_a,
                "confidence_b": confidence_b
            }
            self.history.append(turn)

            if len(self.history) > self.max_stored_turns:
                self.history = self.history[-self.max_stored_turns:]

            self._save_history()

    def get_context(self) -> str:
        """Format recent conversation history as context string for the planner."""
        if not self.history:
            return ""

        recent = self.history[-self.max_turns:]
        lines = ["=== Recent Conversation ==="]
        for turn in recent:
            lines.append(f"Q: {turn['query']}")
            lines.append(f"A: {turn['answer'][:200]}...")  # truncate for brevity
            trust_val = turn.get('trust') or {}
            lines.append(f"Trust: {trust_val.get('trust', 'unknown')}\n")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all conversation history."""
        self.history = []
        self._save_history()