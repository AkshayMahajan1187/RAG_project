from core.memory.session_manager import SessionMemoryManager

# Simulate two different users/sessions
mem_a = SessionMemoryManager.get("user_a")
mem_a.clear()
mem_a.add_turn(query="What is register allocation?", intent="qa", answer="Some answer A")

mem_b = SessionMemoryManager.get("user_b")
mem_b.clear()
mem_b.add_turn(query="What is parsing?", intent="qa", answer="Some answer B")

print("=== Session A history ===")
for turn in mem_a.history:
    print(turn["query"])

print("\n=== Session B history ===")
for turn in mem_b.history:
    print(turn["query"])

print(f"\nActive sessions: {SessionMemoryManager.active_session_count()}")