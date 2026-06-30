from core.ingestion.loader import load_document
from core.ingestion.chunker import chunk_documents
from core.retrieval.vector_store import add_documents, get_vector_store
from core.retrieval.hybrid_search import build_bm25_index, hybrid_search

docs = load_document("data/uploaded_docs/test.pdf")
chunks = chunk_documents(docs)

vector_store = get_vector_store()
add_documents(chunks)

# build BM25 index once
build_bm25_index(chunks)

results = hybrid_search(vector_store, "what is code optimization", k=5)

print(f"\nHybrid search results:")
for r in results:
    print(f"RRF Score: {r.metadata['rrf_score']} | Page {r.metadata['page']} | chunk_id: {r.metadata['chunk_id']}")
    print(f"Preview: {r.page_content[:100]}")
    print("---")

from core.retrieval.reranker import rerank

query = "what is code optimization"
hybrid_results = hybrid_search(vector_store, query, k=10)
reranked = rerank(query, hybrid_results)

print(f"\nReranked top {len(reranked)} results:")
for r in reranked:
    print(f"Rerank Score: {r.metadata['rerank_score']} | RRF: {r.metadata['rrf_score']} | chunk_id: {r.metadata['chunk_id']}")
    print(f"Preview: {r.page_content[:100]}")
    print("---")

from core.agents.generator import generate_answer

result = generate_answer(query, reranked)

print(f"\n=== FINAL ANSWER ===")
print(result["answer"])

print(f"\n=== CITATIONS ===")
for c in result["citations"]:
    print(f"[{c['ref']}] {c['filename']} — Page {c['page']}")


from core.agents.hallucination_checker import check_hallucination

from core.utils.confidence import calculate_evidence_confidence
from core.utils.trust import assess_trust

confidence = calculate_evidence_confidence(reranked)
print(f"\n=== EVIDENCE CONFIDENCE ===")
print(confidence)

verdict = check_hallucination(result["answer"], reranked)
print(f"\n=== HALLUCINATION CHECK ===")
print(verdict)

trust = assess_trust(confidence, verdict)
print(f"\n=== TRUST ASSESSMENT ===")
print(trust)

# ingest the "second" document
docs2 = load_document("data/uploaded_docs/test2.pdf")
chunks2 = chunk_documents(docs2)
add_documents(chunks2)

from core.agents.comparator import compare_documents

comparison = compare_documents(
    "what is code optimization",
    "data/uploaded_docs/test.pdf",
    "data/uploaded_docs/test2.pdf",
    vector_store
)

print(f"\n=== DOCUMENT COMPARISON ===")
print(comparison["answer"])

from core.agents.planner import route_and_execute

# should route to "qa"
result1 = route_and_execute("what is code optimization", vector_store)
print(f"\n=== ORCHESTRATOR TEST 1 (expect qa) ===")
print(f"Intent: {result1['intent']}")
print(result1)

# should route to "compare"
result2 = route_and_execute(
    "compare how these two documents define code optimization",
    vector_store,
    file_a="data/uploaded_docs/test.pdf",
    file_b="data/uploaded_docs/test2.pdf"
)
print(f"\n=== ORCHESTRATOR TEST 2 (expect compare) ===")
print(f"Intent: {result2['intent']}")
print(result2)