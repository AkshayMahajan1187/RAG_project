from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from core.agents.retriever_agent import retrieve
from core.agents.generator import generate_answer, get_llm
from core.utils.confidence import calculate_evidence_confidence
from core.agents.hallucination_checker import check_hallucination
from core.utils.trust import assess_trust

RETRY_INSTRUCTION = (
    "Your previous answer included claims that were not clearly supported "
    "by the context. Re-answer using ONLY what is explicitly stated above, "
    "and drop anything you can't directly point to."
)


def reformulate_query(original_query: str) -> str:
    """
    Agentic pre-retrieval step: if initial retrieval evidence is weak, the
    system rewrites its own query (broadening/rephrasing it) instead of
    proceeding with weak context — a decision made autonomously based on
    retrieval quality, before generation ever runs.
    """
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


def answer_question(query: str, vector_store: Chroma, k: int = 10, max_retries: int = 1) -> dict:
    chunks = retrieve(query, vector_store, k=k)
    confidence = calculate_evidence_confidence(chunks)

    # --- Agentic pre-generation step: reformulate query if evidence is weak ---
    reformulated = False
    reformulated_query = None
    if confidence["level"] in ("low", "none"):
        reformulated_query = reformulate_query(query)
        if reformulated_query != query:
            new_chunks = retrieve(reformulated_query, vector_store, k=k)
            new_confidence = calculate_evidence_confidence(new_chunks)
            if new_confidence["level"] not in ("low", "none") or (
                new_confidence.get("top_rerank_score", -999) > confidence.get("top_rerank_score", -999)
            ):
                chunks = new_chunks
                confidence = new_confidence
                reformulated = True

    result = generate_answer(query, chunks)
    hallucination = check_hallucination(result["answer"], chunks)

    retries = 0
    while hallucination.get("grounded") is False and retries < max_retries:
        retries += 1

        if confidence["level"] != "high":
            # evidence itself was weak — pull in more chunks before retrying
            k *= 2
            chunks = retrieve(query, vector_store, k=k)
            confidence = calculate_evidence_confidence(chunks)

        result = generate_answer(query, chunks, extra_instruction=RETRY_INSTRUCTION)
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
    }