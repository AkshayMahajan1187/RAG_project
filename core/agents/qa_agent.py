from langchain_chroma import Chroma
from core.agents.retriever_agent import retrieve
from core.agents.generator import generate_answer
from core.utils.confidence import calculate_evidence_confidence
from core.agents.hallucination_checker import check_hallucination
from core.utils.trust import assess_trust

RETRY_INSTRUCTION = (
    "Your previous answer included claims that were not clearly supported "
    "by the context. Re-answer using ONLY what is explicitly stated above, "
    "and drop anything you can't directly point to."
)


def answer_question(query: str, vector_store: Chroma, k: int = 10, max_retries: int = 1) -> dict:
    chunks = retrieve(query, vector_store, k=k)
    result = generate_answer(query, chunks)
    confidence = calculate_evidence_confidence(chunks)
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
        "retries": retries
    }