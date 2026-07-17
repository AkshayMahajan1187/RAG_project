from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from core.agents.generator import generate_answer, get_llm
from core.agents.hallucination_checker import check_hallucination
from core.agents.query_rewriter import resolve_retrieval_query
from core.agents.retriever_agent import retrieve
from core.utils.confidence import calculate_evidence_confidence
from core.utils.trust import assess_trust

RETRY_INSTRUCTION = (
    "Your previous answer included claims that were not clearly supported "
    "by the context. Re-answer using ONLY what is explicitly stated above, "
    "and drop anything you can't directly point to."
)


def reformulate_query(original_query: str) -> str:
    llm = get_llm()
    system_message = SystemMessage(content=(
        "You rewrite search queries to improve retrieval from a knowledge base. "
        "Given a query that returned weak/low-confidence results, rewrite it to "
        "be broader or use different phrasing that might match the source "
        "material better. Return ONLY the rewritten query, nothing else."
    ))
    human_message = HumanMessage(content=f"Original query: {original_query}\n\nRewritten query:")

    try:
        response = llm.invoke([system_message, human_message])
        rewritten = response.content.strip().strip('"')
        return rewritten if rewritten else original_query
    except Exception:
        return original_query


def answer_question(
    search_query: str,
    vector_store: Chroma,
    user_id: str,
    display_query: str = None,
    conversation_context: str = None,
    planner_topic: str = None,
    k: int = 10,
    max_retries: int = 1,
) -> dict:
    """
    Args:
        search_query: topic from planner (may still be vague on follow-ups)
        display_query: original user message shown to the generator
        conversation_context: formatted history for query rewriting
        planner_topic: planner-resolved topic (for rewriter heuristics)
    """
    display_query = display_query or search_query
    planner_topic = planner_topic or search_query

    retrieval_query = resolve_retrieval_query(
        user_query=display_query,
        conversation_context=conversation_context or "",
        planner_topic=planner_topic,
    )

    is_follow_up = bool(conversation_context)
    chunks = retrieve(retrieval_query, vector_store, user_id, k=k)
    confidence = calculate_evidence_confidence(chunks)

    reformulated = False
    reformulated_query = None
    if confidence["level"] in ("low", "none"):
        reformulated_query = reformulate_query(retrieval_query)
        if reformulated_query != retrieval_query:
            new_chunks = retrieve(reformulated_query, vector_store, user_id, k=k)
            new_confidence = calculate_evidence_confidence(new_chunks)
            if new_confidence["level"] not in ("low", "none") or (
                new_confidence.get("top_score", -999) > confidence.get("top_score", -999)
            ):
                chunks = new_chunks
                confidence = new_confidence
                reformulated = True

    result = generate_answer(
        query=display_query,
        chunks=chunks,
        is_follow_up=is_follow_up,
    )
    hallucination = check_hallucination(result["answer"], chunks)

    retries = 0
    while hallucination.get("grounded") is False and retries < max_retries:
        retries += 1

        if confidence["level"] != "high":
            k *= 2
            chunks = retrieve(retrieval_query, vector_store, user_id, k=k)
            confidence = calculate_evidence_confidence(chunks)

        result = generate_answer(
            query=display_query,
            chunks=chunks,
            is_follow_up=is_follow_up,
            extra_instruction=RETRY_INSTRUCTION,
        )
        hallucination = check_hallucination(result["answer"], chunks)

    trust = assess_trust(confidence, hallucination)

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "confidence": confidence,
        "hallucination": hallucination,
        "trust": trust,
        "retries": retries,
        "query_reformulated": reformulated,
        "reformulated_query": reformulated_query if reformulated else None,
        "retrieval_query": retrieval_query,
    }
