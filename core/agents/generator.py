from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from typing import List
import re
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE

_llm = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )
    return _llm


def build_context(chunks: List[Document]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[{i}] (Source: {chunk.metadata['filename']}, Page: {chunk.metadata['page']})\n"
            f"{chunk.page_content}\n"
        )
    return "\n".join(context_parts)


def extract_citation_numbers(answer_text: str) -> List[int]:
    match = re.search(r"Sources:\s*(.+)", answer_text)
    if not match:
        return []
    numbers = re.findall(r"\[(\d+)\]", match.group(1))
    return [int(n) for n in numbers]


def clean_answer_text(answer_text: str) -> str:
    return re.sub(r"\n?Sources:.*", "", answer_text).strip()


def build_citations(chunks: List[Document], cited_numbers: List[int]) -> List[dict]:
    """
    Build citations from ALL retrieved chunks, not just what LLM claims it used.
    If LLM cited specific numbers we trust those for ordering, but we always
    fall back to showing all chunks that were sent as context — this way
    we never lose source transparency even if the LLM's self-reported
    citations are wrong or missing.
    """
    citations = []

    # if LLM gave valid citation numbers, use those chunks first
    numbers_to_use = cited_numbers if cited_numbers else list(range(1, len(chunks) + 1))

    for num in numbers_to_use:
        if 1 <= num <= len(chunks):
            chunk = chunks[num - 1]
            citations.append({
                "ref": num,
                "filename": chunk.metadata["filename"],
                "page": chunk.metadata["page"],
                "chunk_id": chunk.metadata["chunk_id"],
                "snippet": chunk.page_content[:150].strip() + "..."
            })

    return citations


def generate_answer(query: str, chunks: List[Document], extra_instruction: str = "") -> dict:
    if not chunks:
        return {
            "answer": "I cannot find this information in the provided documents.",
            "citations": [],
            "chunks_used": []
        }

    context = build_context(chunks)

    system_message = SystemMessage(content=(
        "You are a helpful assistant that answers questions using ONLY the provided context. "
        "If the answer is not in the context, say so clearly. "
        "After your answer, on a new line, write 'Sources: ' followed by the bracket "
        "numbers you used, e.g. Sources: [1], [2]"
        + (f"\n\n{extra_instruction}" if extra_instruction else "")
    ))

    human_message = HumanMessage(content=f"""Context:
{context}

Question: {query}

Answer:""")

    llm = get_llm()

    try:
        response = llm.invoke([system_message, human_message])
        raw_answer = response.content
    except Exception as e:
        return {
            "answer": f"Failed to generate answer: {e}",
            "citations": [],
            "chunks_used": chunks
        }

    cited_numbers = extract_citation_numbers(raw_answer)
    clean_answer = clean_answer_text(raw_answer)

    # fallback — if LLM forgot citations entirely, we still show source chunks
    citations = build_citations(chunks, cited_numbers)

    return {
        "answer": clean_answer,
        "citations": citations,
        "chunks_used": chunks
    }