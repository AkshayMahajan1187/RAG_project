"""
Benchmark Dataset Generator
----------------------------
Pulls chunks from your ChromaDB vector store (via your existing
get_all_chunks()), filters out thin/low-content chunks, and uses your
VERIFIER_LLM_MODEL (llama-3.3-70b-versatile) to generate one question +
expected answer per chunk. Output is a JSON file you spot-check before
feeding into the eval pipeline.

Run from your project root:
    python generate_benchmark.py --sample-size 70
"""

import json
import os
import random
import time
import argparse
from pathlib import Path
from typing import List, Dict

from groq import Groq
from langchain_core.documents import Document

from config import GROQ_API_KEY, VERIFIER_LLM_MODEL
from core.retrieval.vector_store import get_vector_store
from core.retrieval.hybrid_search import get_all_chunks

MIN_WORDS_PER_CHUNK = 50


def get_chunk_id(doc: Document) -> str:
    """Same fallback logic hybrid_search.py's RRF uses — keeps benchmark
    chunk IDs consistent with what retrieval will actually return."""
    return doc.metadata.get("chunk_id", doc.page_content[:50])


def filter_thin_chunks(chunks: List[Document], min_words: int = MIN_WORDS_PER_CHUNK) -> List[Document]:
    filtered = [c for c in chunks if len(c.page_content.split()) >= min_words]
    print(f"Filtered {len(chunks)} -> {len(filtered)} chunks (removed {len(chunks) - len(filtered)} thin chunks)")
    return filtered


def sample_chunks_proportionally(chunks: List[Document], sample_size: int) -> List[Document]:
    """Sample evenly across source documents so no single PDF dominates
    the benchmark just because it has more chunks."""
    by_source: Dict[str, List[Document]] = {}
    for c in chunks:
        source = c.metadata.get("filename", "unknown")
        by_source.setdefault(source, []).append(c)

    num_sources = len(by_source)
    per_source = max(1, sample_size // num_sources)

    sampled = []
    for source, source_chunks in by_source.items():
        random.shuffle(source_chunks)
        take = min(per_source, len(source_chunks))
        sampled.extend(source_chunks[:take])
        print(f"  {source}: {take} chunks sampled (of {len(source_chunks)} available)")

    random.shuffle(sampled)
    return sampled[:sample_size]


QUESTION_GEN_PROMPT = """You are creating benchmark test data for a RAG (retrieval-augmented generation) system.

Given the following text chunk from a study document, generate ONE clear, specific question that:
- Can be answered using ONLY the information in this chunk
- Is the kind of question a student would realistically ask while studying this topic
- Is NOT a yes/no question
- Is NOT overly generic (avoid "what is this about")
- Does NOT reference "the text", "the passage", "the document", "the provided content", or similar meta-references — ask about the actual concept directly, as if you already know the material and are asking a real question about it
- Does NOT reference incidental/stray details (e.g. random numbers, dates, or examples used only as illustrations) — focus on the core concept being taught

Also provide the expected answer, using ONLY information present in the chunk.

Text chunk:
\"\"\"
{chunk_text}
\"\"\"

Respond ONLY in this exact JSON format, no preamble, no markdown fences:
{{"question": "...", "expected_answer": "..."}}
"""


def generate_qa_pair(client: Groq, chunk_text: str, retries: int = 2) -> Dict:
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=VERIFIER_LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": QUESTION_GEN_PROMPT.format(chunk_text=chunk_text[:2000])
                }],
                temperature=0.3,
                max_tokens=300
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            if "question" in parsed and "expected_answer" in parsed:
                return parsed
        except Exception as e:
            if attempt == retries:
                print(f"  [SKIP] Failed after {retries + 1} attempts: {e}")
                return None
            time.sleep(1)
    return None


def build_benchmark_dataset(sample_size: int, output_path: str):
    print("Step 1: Loading vector store and pulling all chunks...")
    vector_store = get_vector_store()
    all_chunks = get_all_chunks(vector_store)
    print(f"  Found {len(all_chunks)} total chunks")

    print("\nStep 2: Filtering out thin/low-content chunks...")
    good_chunks = filter_thin_chunks(all_chunks)

    print(f"\nStep 3: Sampling {sample_size} chunks proportionally across sources...")
    sampled = sample_chunks_proportionally(good_chunks, sample_size)

    print(f"\nStep 4: Generating Q&A pairs via {VERIFIER_LLM_MODEL}...")
    groq_client = Groq(api_key=GROQ_API_KEY)

    benchmark_dataset = []
    for i, chunk in enumerate(sampled, 1):
        filename = chunk.metadata.get("filename", "?")
        page = chunk.metadata.get("page", "?")
        print(f"  [{i}/{len(sampled)}] {filename} (page {page})")

        qa = generate_qa_pair(groq_client, chunk.page_content)
        if qa is None:
            continue

        # skip cases where the source chunk was too garbled/unrelated for the
        # AI to generate a real question from (e.g. corrupted scanned PDFs) —
        # these aren't valid test cases, just noise
        bad_answer_markers = ["no answerable information", "cannot generate", "not enough information", "no information"]
        if any(marker in qa["expected_answer"].lower() for marker in bad_answer_markers):
            print(f"    [SKIP] Chunk produced no real answer (likely garbled source content)")
            continue

        benchmark_dataset.append({
            "id": f"bq_{i:03d}",
            "question": qa["question"],
            "expected_answer": qa["expected_answer"],
            "expected_chunk_id": get_chunk_id(chunk),
            "expected_source": chunk.metadata.get("source"),
            "expected_filename": filename,
            "expected_page": page,
        })

        time.sleep(0.5)  # gentle on free-tier rate limits

    print(f"\nStep 5: Writing {len(benchmark_dataset)} Q&A pairs to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_dataset, f, indent=2, ensure_ascii=False)

    print("\nDone. Next: open the file and spot-check ~20% of the questions")
    print("for quality before running them through the eval pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tests/benchmark_dataset.json")
    parser.add_argument("--sample-size", type=int, default=70)
    args = parser.parse_args()

    build_benchmark_dataset(sample_size=args.sample_size, output_path=args.output)