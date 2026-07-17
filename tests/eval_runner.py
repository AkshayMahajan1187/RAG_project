import json
import time
from pathlib import Path
from core.retrieval.vector_store import get_vector_store
from core.agents.planner import route_and_execute
from core.memory.conversation import ConversationMemory

memory = ConversationMemory()

vector_store = get_vector_store()

from core.retrieval.hybrid_search import build_bm25_index, get_all_chunks

all_chunks = get_all_chunks(vector_store, "default")
build_bm25_index(all_chunks, "default")

FILE_A = "data/uploaded_docs/test.pdf"    # unit 4 - code optimization & generation
FILE_B = "data/uploaded_docs/test2.pdf"   # unit 2 - syntax analysis

EVAL_CASES = [
    {"id": 1, "query": "What is code optimization, and why is it needed?", "expected_intent": "qa"},
    {"id": 2, "query": "What are the two types of code optimization?", "expected_intent": "qa"},
    {"id": 3, "query": "What is constant folding?", "expected_intent": "qa"},
    {"id": 4, "query": "What is dead code elimination?", "expected_intent": "qa"},
    {"id": 5, "query": "What is a basic block?", "expected_intent": "qa"},
    {"id": 6, "query": "What is register allocation, and what does 'spilling' mean?", "expected_intent": "qa"},
    {"id": 7, "query": "What's the difference between code optimization and code generation?", "expected_intent": "qa"},
    {"id": 8, "query": "What are the main challenges in code generation?", "expected_intent": "qa"},
    {"id": 9, "query": "What is syntax analysis (parsing)?", "expected_intent": "qa"},
    {"id": 10, "query": "What is a context-free grammar, and what are its four components?", "expected_intent": "qa"},
    {"id": 11, "query": "What is a parse tree?", "expected_intent": "qa"},
    {"id": 12, "query": "What does it mean for a grammar to be ambiguous?", "expected_intent": "qa"},
    {"id": 13, "query": "What is LL(1) parsing, and what does the '1' mean?", "expected_intent": "qa"},
    {"id": 14, "query": "What is recursive descent parsing?", "expected_intent": "qa"},
    {"id": 15, "query": "What's the difference between top-down and bottom-up parsing?", "expected_intent": "qa"},
    {"id": 16, "query": "What is panic mode error recovery?", "expected_intent": "qa"},
    {"id": 17, "query": "What is database normalization?", "expected_intent": "qa",
     "note": "out-of-corpus, should refuse gracefully, not hallucinate"},
    {"id": 18, "query": "What is the time complexity of quicksort?", "expected_intent": "qa",
     "note": "out-of-corpus, should refuse gracefully, not hallucinate"},
    {"id": 19, "query": "Compare how document A and document B discuss code optimization.",
     "expected_intent": "compare", "file_a": FILE_A, "file_b": FILE_B,
     "note": "doc B has nothing on this topic, should trigger relevance gate"},
    {"id": 20, "query": "Compare these two documents.",
     "expected_intent": "compare", "file_a": FILE_A, "file_b": FILE_B,
     "note": "vague topic, tests planner's topic extraction"},
]


def run_eval():
    results = []

    for case in EVAL_CASES:
        print(f"\n--- Case {case['id']}: {case['query']} ---")
        try:
            output = route_and_execute(
            case["query"],
            vector_store,
            file_a=case.get("file_a"),
            file_b=case.get("file_b"),
            memory=memory
        )
        except Exception as e:
            output = {"intent": "error", "answer": str(e)}

        record = {
            "id": case["id"],
            "query": case["query"],
            "expected_intent": case["expected_intent"],
            "actual_intent": output.get("intent"),
            "answer": output.get("answer"),
            "note": case.get("note", "")
        }

        # qa and compare paths return differently shaped dicts — branch accordingly
        if record["actual_intent"] == "compare":
            record["confidence_a"] = output.get("confidence_a")
            record["confidence_b"] = output.get("confidence_b")
        else:
            record["confidence"] = output.get("confidence")
            record["hallucination"] = output.get("hallucination")
            record["trust"] = output.get("trust")
            record["retries"] = output.get("retries")

        results.append(record)
        print(f"Intent: {record['actual_intent']} | Trust: {record.get('trust') or '(n/a for compare)'}")

        time.sleep(2)  # be gentle on Groq's free-tier rate limit across 20 back-to-back cases

    output_path = Path("eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {output_path}")

    qa_results = [r for r in results if r["actual_intent"] == "qa"]
    high_trust = sum(1 for r in qa_results if (r.get("trust") or {}).get("trust") == "high")
    grounded = sum(1 for r in qa_results if (r.get("hallucination") or {}).get("grounded") is True)
    intent_mismatches = sum(1 for r in results if r["actual_intent"] != r["expected_intent"])

    print(f"\n=== SUMMARY ===")
    print(f"Total cases: {len(results)}")
    print(f"QA cases: {len(qa_results)}")
    print(f"High trust: {high_trust}/{len(qa_results)}")
    print(f"Grounded: {grounded}/{len(qa_results)}")
    print(f"Intent mismatches (planner picked the wrong route): {intent_mismatches}")


if __name__ == "__main__":
    run_eval()


