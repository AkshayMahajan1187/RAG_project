from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from core.agents.generator import get_llm, build_context
from core.retrieval.reranker import rerank
from core.utils.confidence import calculate_evidence_confidence
from core.retrieval.hybrid_search import hybrid_search_scoped


def compare_documents(query: str, file_a: str, file_b: str, vector_store: Chroma, user_id: str, k: int = 10) -> dict:
    candidates_a = hybrid_search_scoped(vector_store, query, file_a, user_id, k=k)
    candidates_b = hybrid_search_scoped(vector_store, query, file_b, user_id, k=k)

    if not candidates_a or not candidates_b:
        missing = "Document A" if not candidates_a else "Document B"
        return {
            "answer": f"{missing} has no content at all for this query.",
            "chunks_a": [], "chunks_b": [],
            "confidence_a": None, "confidence_b": None
        }

    chunks_a = rerank(query, candidates_a)
    chunks_b = rerank(query, candidates_b)

    confidence_a = calculate_evidence_confidence(chunks_a)
    confidence_b = calculate_evidence_confidence(chunks_b)

    if confidence_a["level"] in ("low", "none"):
        return {
            "answer": f"Document A does not contain sufficient information on this topic to compare (confidence: {confidence_a['level']}).",
            "chunks_a": chunks_a, "chunks_b": chunks_b,
            "confidence_a": confidence_a, "confidence_b": confidence_b
        }

    if confidence_b["level"] in ("low", "none"):
        return {
            "answer": f"Document B does not contain sufficient information on this topic to compare (confidence: {confidence_b['level']}).",
            "chunks_a": chunks_a, "chunks_b": chunks_b,
            "confidence_a": confidence_a, "confidence_b": confidence_b
        }

    context_a = build_context(chunks_a)
    context_b = build_context(chunks_b)

    system_message = SystemMessage(content=(
        "You are a helpful assistant that compares content from two different documents. "
        "You will be shown excerpts from Document A and Document B on the same topic. "
        "Identify the key similarities and differences between how the two documents "
        "address this topic, using only the provided excerpts. If one document doesn't "
        "cover something the other does, point that out."
    ))

    human_message = HumanMessage(content=f"""Topic: {query}

Document A excerpts:
{context_a}

Document B excerpts:
{context_b}

Compare how these two documents address this topic.""")

    llm = get_llm()

    try:
        response = llm.invoke([system_message, human_message])
        answer = response.content
    except Exception as e:
        answer = f"Failed to generate comparison: {e}"

    return {
        "answer": answer,
        "chunks_a": chunks_a, "chunks_b": chunks_b,
        "confidence_a": confidence_a, "confidence_b": confidence_b
    }