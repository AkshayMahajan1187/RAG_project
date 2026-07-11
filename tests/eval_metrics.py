"""
Eval metrics — scoring functions used by run_pipeline_eval.py

Design notes (read before extending):
- The benchmark has exactly ONE known-correct chunk per question, not a
  full relevance-graded list. So:
  - "Context Precision" is implemented as Mean Reciprocal Rank (MRR) —
    how high the correct chunk ranked in the retrieved list — instead of
    "% of retrieved chunks that are relevant" (not computable with only
    one known-relevant chunk).
  - "Context Recall" and "Retrieval Recall@K" collapse into the same
    computation at different K values, so we just report Recall@K for
    a few K values instead of two separately-named identical numbers.
"""

import json
import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, VERIFIER_LLM_MODEL, LLM_MODEL

_judge_llm = None


def get_judge_llm() -> ChatGroq:
    """Uses the smaller/cheaper model for accuracy grading — a simpler task
    than hallucination checking — to conserve the 70b model's daily token
    quota for the more critical faithfulness check."""
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=0
        )
    return _judge_llm


def get_chunk_id(doc: Document) -> str:
    return doc.metadata.get("chunk_id", doc.page_content[:50])


# ---------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------

def reciprocal_rank(retrieved_chunks: List[Document], expected_chunk_id: str) -> float:
    """MRR contribution for one question: 1/rank if found, else 0."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if get_chunk_id(chunk) == expected_chunk_id:
            return round(1.0 / rank, 4)
    return 0.0


def recall_at_k(retrieved_chunks: List[Document], expected_chunk_id: str, k: int) -> bool:
    """Was the expected chunk present in the top-k retrieved chunks?"""
    top_k = retrieved_chunks[:k]
    return any(get_chunk_id(c) == expected_chunk_id for c in top_k)


# ---------------------------------------------------------------------
# Answer accuracy (LLM-as-judge)
# ---------------------------------------------------------------------

ACCURACY_JUDGE_PROMPT = """You are grading a student's answer against a reference answer.

Question: {question}

Reference answer: {expected_answer}

Student's answer: {generated_answer}

Judge whether the student's answer is factually consistent with the reference answer.
Respond with EXACTLY ONE WORD, no punctuation, no explanation:
CORRECT   - if the student's answer conveys the same key information as the reference
PARTIAL   - if the student's answer is partially right but missing or slightly wrong on some key point
INCORRECT - if the student's answer contradicts the reference or is substantially wrong/missing
"""


def judge_answer_accuracy(question: str, expected_answer: str, generated_answer: str) -> str:
    """Returns 'correct', 'partial', or 'incorrect' (lowercase)."""
    llm = get_judge_llm()
    try:
        response = llm.invoke([
            SystemMessage(content="You are a strict, consistent grader."),
            HumanMessage(content=ACCURACY_JUDGE_PROMPT.format(
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated_answer
            ))
        ])
        verdict = response.content.strip().upper()
        if "CORRECT" in verdict and "INCORRECT" not in verdict:
            return "correct"
        elif "PARTIAL" in verdict:
            return "partial"
        elif "INCORRECT" in verdict:
            return "incorrect"
        else:
            return "unparseable"
    except Exception as e:
        return f"error: {e}"


def accuracy_score(verdict: str) -> Optional[float]:
    """Convert verdict to a numeric score for averaging. None = excluded from average."""
    return {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(verdict)