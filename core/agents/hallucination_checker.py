from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from typing import List
import re
from config import GROQ_API_KEY, VERIFIER_LLM_MODEL
from core.agents.generator import build_context

_verifier_llm = None


def get_verifier_llm() -> ChatGroq:
    global _verifier_llm
    if _verifier_llm is None:
        _verifier_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=VERIFIER_LLM_MODEL,
            temperature=0  # judge should be consistent, not creative
        )
    return _verifier_llm


def split_into_claims(answer: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [s.strip() for s in sentences if s.strip()]


def parse_claim_verdicts(raw_output: str) -> dict:
    pattern = re.compile(r"Claim\s*(\d+):\s*(SUPPORTED|NOT SUPPORTED)", re.IGNORECASE)
    matches = pattern.findall(raw_output)
    return {int(num): verdict.upper() == "SUPPORTED" for num, verdict in matches}


def check_hallucination(answer: str, chunks: List[Document]) -> dict:
    if not chunks:
        return {
            "grounded": None,
            "coverage": 0.0,
            "claims": [],
            "reason": "No source chunks were provided to verify against."
        }

    claims = split_into_claims(answer)

    if not claims:
        return {
            "grounded": None,
            "coverage": 0.0,
            "claims": [],
            "reason": "Answer had no checkable claims."
        }

    context = build_context(chunks)
    numbered_claims = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

    system_message = SystemMessage(content=(
        "You are a strict fact-checker. You will be shown source text and a list of "
        "numbered claims that supposedly came from that source text. For EACH claim, "
        "decide if it is directly supported by the source text. Do not use any outside "
        "knowledge of your own — only judge based on the source text given. "
        "Respond with one line per claim, in exactly this format:\n"
        "Claim 1: SUPPORTED or NOT SUPPORTED\n"
        "Claim 2: SUPPORTED or NOT SUPPORTED\n"
        "(and so on for every claim, in order, with no extra commentary)"
    ))

    human_message = HumanMessage(content=f"""Source text:
{context}

Claims to check:
{numbered_claims}""")

    llm = get_verifier_llm()

    try:
        response = llm.invoke([system_message, human_message])
        raw_output = response.content
    except Exception as e:
        return {
            "grounded": None,
            "coverage": 0.0,
            "claims": [],
            "reason": f"Could not run hallucination check: {e}"
        }

    verdicts = parse_claim_verdicts(raw_output)

    claim_results = []
    for i, claim in enumerate(claims, start=1):
        claim_results.append({"claim": claim, "supported": verdicts.get(i)})

    judged = [c for c in claim_results if c["supported"] is not None]
    supported_count = sum(1 for c in judged if c["supported"])
    coverage = round(supported_count / len(judged), 2) if judged else 0.0
    unsupported = [c["claim"] for c in claim_results if c["supported"] is False]

    if not judged:
        grounded = None
        reason = "Could not parse the verifier's response into per-claim verdicts."
    elif unsupported:
        grounded = False
        reason = f"{len(unsupported)} of {len(judged)} claim(s) not supported, e.g.: \"{unsupported[0]}\""
    else:
        grounded = True
        reason = f"All {len(judged)} claim(s) are supported by the source text."

    return {
        "grounded": grounded,
        "coverage": coverage,
        "claims": claim_results,
        "reason": reason
    }