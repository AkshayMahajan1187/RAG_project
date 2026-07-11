"""
Run the benchmark dataset through 4 retrieval pipeline variants
(Vector-only, BM25-only, Hybrid, Hybrid+Reranker) and produce a
comparison table with: Answer Accuracy, Faithfulness, Context Precision
(MRR), Retrieval Recall@1/3/5, Latency.

Usage:
    python run_pipeline_eval.py
    python run_pipeline_eval.py --limit 15   # quick test on first 15 questions

Takes a while (4 variants x N questions x several API calls each) — free-tier
rate limits mean expect roughly 1-2 minutes per 10 questions per variant.
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict

from config import TOP_K_RETRIEVAL, TOP_K_RERANK
from core.retrieval.vector_store import get_vector_store
from core.retrieval.hybrid_search import (
    semantic_search, bm25_search, hybrid_search,
    build_bm25_index, get_all_chunks
)
from core.retrieval.reranker import rerank
from core.agents.generator import generate_answer
from core.agents.hallucination_checker import check_hallucination

from eval_metrics import (
    reciprocal_rank, recall_at_k, judge_answer_accuracy, accuracy_score, get_chunk_id
)


def vector_only_pipeline(vector_store, query):
    return semantic_search(vector_store, query, k=TOP_K_RETRIEVAL)


def bm25_only_pipeline(vector_store, query):
    return bm25_search(query, k=TOP_K_RETRIEVAL)


def hybrid_pipeline(vector_store, query):
    return hybrid_search(vector_store, query, k=TOP_K_RETRIEVAL)


def hybrid_rerank_pipeline(vector_store, query):
    hybrid_results = hybrid_search(vector_store, query, k=TOP_K_RETRIEVAL)
    return rerank(query, hybrid_results)[:TOP_K_RERANK]


PIPELINES = {
    "Vector-only": vector_only_pipeline,
    "BM25-only": bm25_only_pipeline,
    "Hybrid": hybrid_pipeline,
    "Hybrid+Reranker": hybrid_rerank_pipeline,
}


def run_variant(variant_name: str, pipeline_fn, vector_store, benchmark: List[Dict]) -> List[Dict]:
    print(f"\n{'='*60}\nRunning variant: {variant_name}\n{'='*60}")
    results = []

    for i, item in enumerate(benchmark, 1):
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_chunk_id = item["expected_chunk_id"]

        print(f"  [{i}/{len(benchmark)}] {question[:60]}...")

        start = time.time()
        try:
            retrieved_chunks = pipeline_fn(vector_store, question)
            gen_result = generate_answer(question, retrieved_chunks)
            generated_answer = gen_result["answer"]
        except Exception as e:
            results.append({
                "id": item["id"], "question": question, "error": str(e)
            })
            continue
        latency = round(time.time() - start, 3)

        # retrieval metrics
        mrr = reciprocal_rank(retrieved_chunks, expected_chunk_id)
        recall_1 = recall_at_k(retrieved_chunks, expected_chunk_id, 1)
        recall_3 = recall_at_k(retrieved_chunks, expected_chunk_id, 3)
        recall_5 = recall_at_k(retrieved_chunks, expected_chunk_id, 5)

        # faithfulness (reuses your existing hallucination checker)
        halluc = check_hallucination(generated_answer, retrieved_chunks)

        # answer accuracy (LLM judge)
        verdict = judge_answer_accuracy(question, expected_answer, generated_answer)

        results.append({
            "id": item["id"],
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "expected_chunk_id": expected_chunk_id,
            "retrieved_chunk_ids": [get_chunk_id(c) for c in retrieved_chunks],
            "mrr": mrr,
            "recall@1": recall_1,
            "recall@3": recall_3,
            "recall@5": recall_5,
            "faithfulness_grounded": halluc.get("grounded"),
            "faithfulness_coverage": halluc.get("coverage"),
            "faithfulness_reason": halluc.get("reason"),
            "accuracy_verdict": verdict,
            "accuracy_score": accuracy_score(verdict),
            "latency_seconds": latency,
        })

        time.sleep(2.5)  # gentle on free-tier rate limits (3 LLM calls per question)

    return results


def summarize(variant_name: str, results: List[Dict]) -> Dict:
    valid = [r for r in results if "error" not in r]
    n = len(valid)
    if n == 0:
        return {"variant": variant_name, "error": "all questions failed"}

    def avg(key):
        vals = [r[key] for r in valid if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def pct(key):
        vals = [r[key] for r in valid if r.get(key) is not None]
        return round(100 * sum(vals) / len(vals), 1) if vals else None

    accuracy_vals = [r["accuracy_score"] for r in valid if r.get("accuracy_score") is not None]

    return {
        "variant": variant_name,
        "n_questions": n,
        "answer_accuracy_pct": round(100 * sum(accuracy_vals) / len(accuracy_vals), 1) if accuracy_vals else None,
        "faithfulness_grounded_pct": pct("faithfulness_grounded"),
        "avg_faithfulness_coverage": avg("faithfulness_coverage"),
        "context_precision_mrr": avg("mrr"),
        "recall@1_pct": pct("recall@1"),
        "recall@3_pct": pct("recall@3"),
        "recall@5_pct": pct("recall@5"),
        "avg_latency_seconds": avg("latency_seconds"),
    }


def print_comparison_table(summaries: List[Dict]):
    headers = ["Variant", "Accuracy", "Faithful", "Precision(MRR)", "Recall@1", "Recall@3", "Recall@5", "Latency(s)"]
    print("\n" + "=" * 100)
    print("PIPELINE COMPARISON")
    print("=" * 100)
    print(f"{headers[0]:<18}{headers[1]:<11}{headers[2]:<11}{headers[3]:<16}{headers[4]:<11}{headers[5]:<11}{headers[6]:<11}{headers[7]:<11}")
    print("-" * 100)
    for s in summaries:
        if "error" in s:
            print(f"{s['variant']:<18} FAILED: {s['error']}")
            continue
        print(f"{s['variant']:<18}"
              f"{str(s['answer_accuracy_pct'])+'%':<11}"
              f"{str(s['faithfulness_grounded_pct'])+'%':<11}"
              f"{str(s['context_precision_mrr']):<16}"
              f"{str(s['recall@1_pct'])+'%':<11}"
              f"{str(s['recall@3_pct'])+'%':<11}"
              f"{str(s['recall@5_pct'])+'%':<11}"
              f"{str(s['avg_latency_seconds']):<11}")
    print("=" * 100)


def main(limit: int = None, variant: str = None):
    print("Loading benchmark dataset...")
    with open("tests/benchmark_dataset.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    if limit:
        benchmark = benchmark[:limit]
    print(f"Loaded {len(benchmark)} benchmark questions")

    print("Loading vector store and building BM25 index...")
    vector_store = get_vector_store()
    all_chunks = get_all_chunks(vector_store)
    build_bm25_index(all_chunks)

    pipelines_to_run = PIPELINES
    if variant:
        if variant not in PIPELINES:
            print(f"Unknown variant '{variant}'. Choose from: {list(PIPELINES.keys())}")
            return
        pipelines_to_run = {variant: PIPELINES[variant]}

    all_results = {}
    summaries = []

    for variant_name, pipeline_fn in pipelines_to_run.items():
        variant_results = run_variant(variant_name, pipeline_fn, vector_store, benchmark)
        all_results[variant_name] = variant_results
        summaries.append(summarize(variant_name, variant_results))

    print_comparison_table(summaries)

    # merge with existing results file instead of overwriting, so re-running
    # one variant doesn't wipe out the others
    results_path = Path("tests/eval_comparison_results.json")
    if results_path.exists() and variant:
        existing = json.loads(results_path.read_text(encoding="utf-8"))
        existing["detailed_results"][variant] = all_results[variant]
        existing["summary"] = [s for s in existing["summary"] if s["variant"] != variant] + summaries
        results_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    else:
        results_path.write_text(
            json.dumps({"detailed_results": all_results, "summary": summaries}, indent=2, default=str),
            encoding="utf-8"
        )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions (for a quick test)")
    parser.add_argument("--variant", type=str, default=None,
                         help="Only run one named variant (e.g. 'Hybrid+Reranker') instead of all 4 — useful to re-run a single variant that got cut off by a rate limit")
    args = parser.parse_args()
    main(limit=args.limit, variant=args.variant)