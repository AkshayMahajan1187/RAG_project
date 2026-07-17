import json
import logging
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from core.agents.generator import get_llm

# Follow-ups that need conversation context to become standalone search queries.
_VAGUE_PATTERN = re.compile(
    r"\b(both|it|its|them|their|this|that|these|those|one|other|second|first|"
    r"third|last|why|how so|explain more|tell me more|go on|continue|elaborate)\b",
    re.IGNORECASE,
)


def needs_contextual_rewrite(user_query: str, planner_topic: str) -> bool:
    """Return True when the query/topic is too vague to retrieve on its own."""
    if _VAGUE_PATTERN.search(user_query):
        return True
    if len(user_query.split()) <= 4 and _VAGUE_PATTERN.search(planner_topic or ""):
        return True
    # Planner echoed the vague query instead of resolving it.
    if planner_topic and user_query.strip().lower() == planner_topic.strip().lower():
        if _VAGUE_PATTERN.search(user_query):
            return True
    return False


def rewrite_for_retrieval(
    user_query: str,
    conversation_context: str,
    planner_topic: Optional[str] = None,
) -> str:
    """
    Turn a conversational follow-up into a standalone search query.
    Example: 'Explain both.' + history -> 'Explain micro-optimization and macro-optimization'
    """
    if not conversation_context:
        return planner_topic or user_query

    llm = get_llm()
    system_message = SystemMessage(content=(
        "You rewrite follow-up questions into standalone search queries for a document "
        "retrieval system.\n\n"
        "Rules:\n"
        "- Use the conversation history to resolve pronouns and vague references "
        "('both', 'it', 'the second one', 'why') into the actual concepts.\n"
        "- Output ONE search query that would find the relevant passage in a textbook.\n"
        "- Do NOT answer the question — only rewrite it for search.\n"
        "- Return ONLY the rewritten query as plain text, no quotes or JSON."
    ))

    topic_hint = f"\nPlanner topic hint: {planner_topic}" if planner_topic else ""
    human_message = HumanMessage(content=(
        f"{conversation_context}\n\n"
        f"Follow-up question: {user_query}"
        f"{topic_hint}\n\n"
        "Standalone search query:"
    ))

    try:
        response = llm.invoke([system_message, human_message])
        rewritten = response.content.strip().strip('"').strip("'")
        if rewritten and len(rewritten) > 3:
            return rewritten
    except Exception as e:
        logging.warning(f"Query rewrite failed: {e}")

    return planner_topic or user_query


def resolve_retrieval_query(
    user_query: str,
    conversation_context: str,
    planner_topic: str,
) -> str:
    """Pick the best query string to send to hybrid search."""
    topic = (planner_topic or user_query).strip()
    if not conversation_context:
        return topic

    if needs_contextual_rewrite(user_query, topic):
        return rewrite_for_retrieval(user_query, conversation_context, topic)

    # Planner resolved the topic but it may still be too narrow for follow-ups.
    if _VAGUE_PATTERN.search(user_query) and topic.lower() not in user_query.lower():
        return rewrite_for_retrieval(user_query, conversation_context, topic)

    return topic
