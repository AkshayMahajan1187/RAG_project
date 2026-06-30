import time
import logging
from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain_chroma import Chroma
from core.agents.generator import get_llm
from core.agents.qa_agent import answer_question
from core.agents.comparator import compare_documents
from core.memory.conversation import ConversationMemory


class RouteDecision(BaseModel):
    reasoning: str = Field(
        description="brief explanation: is the user comparing two DOCUMENTS/FILES, "
                    "asking a normal QUESTION, or just making SMALL TALK with no real "
                    "question (greeting, thanks, acknowledgment)?"
    )
    intent: Literal["qa", "compare", "smalltalk"] = Field(
        description="'qa' for a normal question about the documents, 'compare' only if "
                    "comparing two separate documents/files, 'smalltalk' for greetings, "
                    "thanks, or any message with no real question (e.g. 'hi', 'thanks', 'ok', 'bye')"
    )
    topic: str = Field(
        description="the core question to search for, with filler words removed AND any "
                    "pronouns/references ('it', 'that', 'this') resolved using conversation "
                    "history into the actual concept name. NEVER return null or empty — if "
                    "the request is vague with no specific subject (e.g. 'compare these two "
                    "documents' with no topic named), use 'general comparison' or 'general "
                    "overview' instead. For smalltalk, this field can just repeat the original message."
    )


COMPARE_KEYWORDS = ["compare", "difference between", " vs ", "versus"]


def classify_intent(user_input: str, fallback_topic: str = None) -> RouteDecision:
    llm = get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    system_prompt = (
        "You are a routing assistant for a document Q&A system.\n\n"
        "If conversation history is provided above the new question, use it to resolve "
        "any pronouns or vague references in the new question ('it', 'that', 'this one') "
        "into the actual concept being discussed. The topic field must never contain an "
        "unresolved pronoun.\n\n"
        "First, check if the new question is actually a real question at all. Greetings, "
        "thanks, acknowledgments, or filler messages ('hi', 'hello', 'thanks', 'ok', 'bye') "
        "are 'smalltalk' — do NOT try to resolve these against conversation history, "
        "even if earlier turns were technical.\n\n"
        "If it IS a real question, decide: are they pointing at two separate DOCUMENTS or "
        "FILES they want compared against each other ('compare'), or are they asking about "
        "one or more CONCEPTS or TOPICS that could be explained from a single source ('qa')? "
        "The words 'compare' or 'difference' alone do NOT decide this — the actual target of "
        "the comparison does.\n\n"
        "Write your reasoning first, then give the intent and a clean topic string."
    )

    messages = [
        ("system", system_prompt),
        ("human", user_input)
    ]

    for attempt in range(2):
        try:
            decision = structured_llm.invoke(messages)
            return decision
        except Exception as e:
            logging.warning(f"classify_intent attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(0.5)
                continue

    # Both attempts failed — fall back safely.
    # Guess compare vs qa from keywords only because no LLM judgment is available at all.
    safe_topic = fallback_topic or user_input
    likely_compare = any(kw in user_input.lower() for kw in COMPARE_KEYWORDS)

    return RouteDecision(
        intent="compare" if likely_compare else "qa",
        topic=safe_topic,
        reasoning="fallback: structured parsing failed after retry"
    )


def route_and_execute(
    user_input: str,
    vector_store: Chroma,
    file_a: Optional[str] = None,
    file_b: Optional[str] = None,
    memory: Optional[ConversationMemory] = None
) -> dict:
    memory = memory or ConversationMemory()

    context_prompt = memory.get_context()
    augmented_input = f"{context_prompt}\n\nNew question: {user_input}" if context_prompt else user_input

    decision = classify_intent(augmented_input, fallback_topic=user_input)
    if not decision.topic:
        decision.topic = user_input
    print(f"[DEBUG] Extracted topic: '{decision.topic}' | reasoning: {decision.reasoning}")

    if decision.intent == "smalltalk":
        return {
            "intent": "smalltalk",
            "answer": "Hey! Ask me anything about your documents.",
            "planner_reasoning": decision.reasoning
        }

    if decision.intent == "compare":
        if not file_a or not file_b:
            return {
                "intent": "compare",
                "answer": "Comparison requires two documents to be selected first.",
                "planner_reasoning": decision.reasoning
            }
        result = compare_documents(decision.topic, file_a, file_b, vector_store)
        result["intent"] = "compare"
    else:
        result = answer_question(decision.topic, vector_store)
        result["intent"] = "qa"

    memory.add_turn(
        query=user_input,
        intent=result["intent"],
        answer=result.get("answer", ""),
        trust=result.get("trust"),
        citations=result.get("citations"),
        grounded=result.get("hallucination", {}).get("grounded"),
        file_a=file_a if result["intent"] == "compare" else None,
        file_b=file_b if result["intent"] == "compare" else None,
        confidence_a=result.get("confidence_a"),
        confidence_b=result.get("confidence_b")
    )

    result["planner_reasoning"] = decision.reasoning
    return result