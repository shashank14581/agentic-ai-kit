"""Example 10 — RAG agent that answers questions from ingested documents."""
from agentic_ai.rag.rag_agent import RAGAgent

agent = RAGAgent(
    "DocBot",
    "You answer questions using only the provided context. If unsure, say so.",
    top_k=3,
)

document = """
Agentic AI refers to AI systems that can autonomously plan, reason, and execute
multi-step tasks. Unlike a simple chatbot, an agentic system decides WHAT to do
next based on a goal, uses tools to take actions, and adapts when things go wrong.

Key components of an agentic system:
1. A planner that breaks goals into steps
2. Tools the agent can invoke (search, code execution, APIs)
3. Memory — short-term (context window) and long-term (vector stores, databases)
4. A feedback loop to observe results and adjust

Popular frameworks include LangChain, CrewAI, AutoGen, and custom setups with
Gemini or OpenAI function-calling.
"""

chunks_added = agent.ingest(document)
print(f"Ingested {chunks_added} chunks.\n")

agent.think("What are the key components of an agentic AI system?")
agent.think("Which frameworks are mentioned for building agents?")
