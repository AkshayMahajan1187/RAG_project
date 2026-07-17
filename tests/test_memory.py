from core.retrieval.vector_store import get_vector_store
from core.retrieval.hybrid_search import build_bm25_index, get_all_chunks
from core.agents.planner import route_and_execute
from core.memory.conversation import ConversationMemory

vector_store = get_vector_store()

# Bootstrap BM25
all_chunks = get_all_chunks(vector_store)
build_bm25_index(all_chunks)

# Start fresh memory for this test
memory = ConversationMemory()
memory.clear()

print("=== MEMORY TEST ===\n")

# Turn 1: Ask something
query1 = "What is register allocation?"
print(f"Q1: {query1}")
result1 = route_and_execute(query1, vector_store, user_id="default", memory=memory)
print(f"A1: {result1['answer'][:150]}...")
print(f"Trust: {result1.get('trust', {}).get('trust')}\n")

# Turn 2: Follow-up that references the first answer
query2 = "What problem does it try to solve?"
print(f"Q2: {query2}")
result2 = route_and_execute(query2, vector_store, user_id="default", memory=memory)
print(f"A2: {result2['answer'][:150]}...")
print(f"Trust: {result2.get('trust', {}).get('trust')}\n")

# Show what's in memory
print("=== SAVED HISTORY ===")
for i, turn in enumerate(memory.history, 1):
    print(f"Turn {i}: {turn['query']}")
    print(f"  Answer: {turn['answer'][:80]}...")
    print()