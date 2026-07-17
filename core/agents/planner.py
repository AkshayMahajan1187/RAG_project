import json
import logging
import time
from typing import Literal, Optional

from langchain_chroma import Chroma
from pydantic import BaseModel, Field, ValidationError

from core.agents.comparator import compare_documents
from core.agents.generator import get_llm
from core.agents.qa_agent import answer_question
from core.memory.conversation import ConversationMemory

logger = logging.getLogger(__name__)


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
                    "pronouns/references ('it', 'that', 'this', 'both', 'the second one') "
                    "resolved using conversation history into the actual concept name. "
                    "NEVER return null or empty — if the request is vague with no specific "
                    "subject, use 'general comparison' or 'general overview' instead. "
                    "For smalltalk, this field can just repeat the original message."
    )


COMPARE_KEYWORDS = ["compare", "difference between", " vs ", "versus"]

SYSTEM_PROMPT = (
    "You are a routing assistant for a document Q&A system.\n\n"
    "Return a JSON object with exactly these fields: reasoning, intent, topic.\n"
    "The intent field MUST be one of: qa, compare, smalltalk.\n\n"
    "If conversation history is provided above the new question, use it to resolve "
    "any pronouns or vague references in the new question ('it', 'that', 'both', "
    "'the second one') into the actual concept being discussed. The topic field must "
    "never contain an unresolved pronoun.\n\n"
    "First, check if the new question is actually a real question at all. Greetings, "
    "thanks, acknowledgments, or filler messages ('hi', 'hello', 'thanks', 'ok', 'bye') "
    "are 'smalltalk' — do NOT try to resolve these against conversation history, "
    "even if earlier turns were technical.\n\n"
    "If it IS a real question, decide: are they pointing at two separate DOCUMENTS or "
    "FILES they want compared against each other ('compare'), or are they asking about "
    "one or more CONCEPTS or TOPICS that could be explained from a single source ('qa')? "
    "The words 'compare' or 'difference' alone do NOT decide this — the actual target of "
    "the comparison does. Comparing two concepts from the same document is still 'qa'.\n\n"
    "Write your reasoning first, then give the intent and a clean topic string."
)


def _normalize_route_args(raw: dict) -> dict:
    """
    Groq/Llama sometimes puts the intent value in a 'name' field instead of 'intent'
    when using tool-calling mode. Normalize before Pydantic validation.
    """
    data = dict(raw)
    if "intent" not in data and "name" in data:
        candidate = str(data["name"]).lower()
        if candidate in ("qa", "compare", "smalltalk"):
            data["intent"] = candidate
    # Drop spurious keys the model adds.
    data.pop("name", None)
    return data


def _parse_route_decision(raw_content) -> RouteDecision:
    """Parse model output whether it came from json_schema, tool call, or plain JSON."""
    if isinstance(raw_content, RouteDecision):
        return raw_content

    if isinstance(raw_content, dict):
        if "parsed" in raw_content and raw_content["parsed"] is not None:
            return raw_content["parsed"]
        if "intent" in raw_content or "name" in raw_content:
            return RouteDecision(**_normalize_route_args(raw_content))

    # Tool-call arguments string
    if isinstance(raw_content, str):
        data = json.loads(raw_content)
        return RouteDecision(**_normalize_route_args(data))

    raise ValueError(f"Unrecognized route decision format: {type(raw_content)}")


def _extract_tool_args(response) -> Optional[dict]:
    tool_calls = getattr(response, "tool_calls", None) or response.additional_kwargs.get("tool_calls")
    if not tool_calls:
        return None
    tc = tool_calls[0]
    args = tc["function"]["arguments"] if isinstance(tc, dict) else tc.get("args", tc.get("arguments"))
    if isinstance(args, str):
        return json.loads(args)
    return args


def classify_intent(user_input: str, fallback_topic: str = None) -> RouteDecision:
    llm = get_llm()
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", user_input),
    ]

    # Prefer json_schema — avoids the tool-calling 'name' vs 'intent' bug on Groq/Llama.
    methods = ["json_schema", "function_calling"]
    last_error = None

    for method in methods:
        structured_llm = llm.with_structured_output(
            RouteDecision,
            method=method,
            include_raw=True,
        )
        for attempt in range(2):
            try:
                result = structured_llm.invoke(messages)
                parsed = result.get("parsed") if isinstance(result, dict) else result
                if parsed is not None:
                    return parsed

                raw = result.get("raw") if isinstance(result, dict) else None
                if raw is not None:
                    args = _extract_tool_args(raw)
                    if args:
                        return RouteDecision(**_normalize_route_args(args))
                    if raw.content:
                        return RouteDecision(**_normalize_route_args(json.loads(raw.content)))

            except (ValidationError, json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                logger.warning(f"classify_intent [{method}] attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    time.sleep(0.3)
            except Exception as e:
                last_error = e
                logger.warning(f"classify_intent [{method}] attempt {attempt + 1} error: {e}")
                if attempt == 0:
                    time.sleep(0.3)

    safe_topic = fallback_topic or user_input
    likely_compare = any(kw in user_input.lower() for kw in COMPARE_KEYWORDS)
    logger.warning(f"classify_intent falling back after error: {last_error}")

    return RouteDecision(
        intent="compare" if likely_compare else "qa",
        topic=safe_topic,
        reasoning="fallback: structured parsing failed after retry",
    )


def route_and_execute(
    user_input: str,
    vector_store: Chroma,
    user_id: str = "default",
    file_a: Optional[str] = None,
    file_b: Optional[str] = None,
    memory: Optional[ConversationMemory] = None,
) -> dict:
    memory = memory or ConversationMemory()

    context_prompt = memory.get_context()
    augmented_input = f"{context_prompt}\n\nNew question: {user_input}" if context_prompt else user_input

    decision = classify_intent(augmented_input, fallback_topic=user_input)
    if not decision.topic:
        decision.topic = user_input

    logger.info("Planner topic='%s' intent=%s", decision.topic, decision.intent)

    if decision.intent == "smalltalk":
        return {
            "intent": "smalltalk",
            "answer": "Hey! Ask me anything about your documents.",
            "planner_reasoning": decision.reasoning,
        }

    if decision.intent == "compare":
        if not file_a or not file_b:
            return {
                "intent": "compare",
                "answer": "Comparison requires two documents to be selected first.",
                "planner_reasoning": decision.reasoning,
            }
        result = compare_documents(decision.topic, file_a, file_b, vector_store, user_id)
        result["intent"] = "compare"
    else:
        result = answer_question(
            search_query=decision.topic,
            display_query=user_input,
            vector_store=vector_store,
            user_id=user_id,
            conversation_context=context_prompt or None,
            planner_topic=decision.topic,
        )
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
        confidence_b=result.get("confidence_b"),
    )

    result["planner_reasoning"] = decision.reasoning
    result["planner_topic"] = decision.topic
    return result
