# agentic-ai

A progressive Python framework for building **agentic AI systems with Google Gemini**.

`agentic-ai` turns common agentic AI patterns into reusable Python components: talking agents, memory, tool use, structured JSON output, reasoning workflows, supervisor delegation, parallel execution, debate, RAG, MCP tool servers, and applied data science agents.

This project started with a simple question: what happens if two agents talk and one of them is a dog that can bark? From that playful starting point, the patterns evolved into a reusable framework for building composable agentic systems.

---

## Installation

```bash
pip install agentic-ai

For the latest GitHub version:

pip install --upgrade --force-reinstall git+https://github.com/shashank14581/agentic-ai-kit.git

Set your Gemini API key:

export GEMINI_API_KEY="your-key-here"

In Google Colab, store GEMINI_API_KEY in Colab Secrets and load it into os.environ.

Quick start
from agentic_ai.agents import BaseAgent

agent = BaseAgent(
    name="Alfred",
    sys_prompt="You are a witty British butler.",
    model="gemini-1.5-flash",
    extract_memory=False,
)

agent.think("Good morning! What should I do today?", stream=False)
What it supports
Capability	Component
Simple conversational agents	BaseAgent
Persona and short-term memory	BaseAgent, ShortTermMemory
LLM-extracted durable facts	BaseAgent.facts_store
SQLite-backed long-term memory	LongTermMemory
Shared state across agents	SharedMemory
Gemini function-calling tools	ToolAgent, ToolRegistry
Structured JSON output	JsonAgent
Reasoning / planning workflows	ReasoningAgent
Round-robin conversations	run_conversation
Supervisor delegation	run_supervisor
Parallel fan-out execution	run_parallel
Debate and judge pattern	run_debate
Chunking, embeddings, vector search	chunk_text, embed_texts, VectorStore
Retrieval-augmented generation	RAGAgent
MCP-style tool server and client	MCPServer, MCPClient
DataFrame analysis	AnalystAgent
Machine learning model training	MLEAgent
End-to-end auto modeling workflow	AutoModelAgent
Package layout
agentic_ai/
├── agents/
│   ├── base.py              # BaseAgent — memory, streaming, durable facts
│   ├── tool_agent.py        # ToolAgent — Gemini function-calling
│   ├── json_agent.py        # JsonAgent — structured JSON responses
│   ├── reasoning_agent.py   # ReasoningAgent — planning-style workflows
│   ├── analyst_agent.py     # AnalystAgent — dataframe profiling and analysis
│   ├── mle_agent.py         # MLEAgent — train and compare sklearn models
│   └── auto_model_agent.py  # AutoModelAgent — profile data, train model, summarize results
├── memory/
│   ├── short_term.py        # Sliding-window turn memory
│   ├── long_term.py         # SQLite-backed persistent memory
│   └── shared.py            # Thread-safe shared state across agents
├── tools/
│   ├── registry.py          # @tool decorator and ToolRegistry
│   └── builtins.py          # get_weather, add_numbers, search_wikipedia
├── patterns/
│   ├── orchestrator.py      # run_conversation, run_supervisor
│   ├── parallel.py          # run_parallel fan-out execution
│   └── debate.py            # run_debate with debaters and judge
├── rag/
│   ├── chunker.py           # chunk_text
│   ├── embedder.py          # embed_texts using Gemini embeddings
│   ├── vector_store.py      # In-memory cosine-similarity vector store
│   └── rag_agent.py         # RAGAgent ingest → retrieve → generate
└── mcp/
    ├── server.py            # MCPServer — HTTP tool server
    └── client.py            # MCPClient — calls a running MCPServer
Examples

All examples live in agentic_ai/examples/ and can be run directly:

File	Concept
01_talking_agents.py	Two agents in a round-robin conversation
02_personality_and_memory.py	Agent with persona and short-term memory
03_tool_agent_weather.py	Real weather tool through Gemini function-calling
04_json_agent.py	Structured JSON output
05_reasoning_agent.py	Planning-style reasoning workflow
06_supervisor.py	Supervisor delegates to specialist workers
07_parallel_agents.py	Fan-out: same task, multiple agents in parallel
08_debate.py	Two agents debate; a judge picks a winner
09_long_term_memory.py	SQLite-backed memory across sessions
10_rag_agent.py	Document ingestion and retrieval-augmented generation
11_mcp_demo.py	MCP tool server and agent integration
Data science agents

The kit also includes applied analytics and modeling agents.

from agentic_ai.agents import AutoModelAgent

agent = AutoModelAgent(
    project_context="Marketing audience selection and customer conversion modeling.",
    model="gemini-1.5-flash",
)

result = agent.run(
    df=df,
    target_col="converted",
    objective="Predict customer conversion.",
    interpret=False,
)

print(result["best_model"])
print(result["best_score"])

Use interpret=False when you want local sklearn training without an additional Gemini interpretation call.

Design philosophy

The framework separates reasoning from execution.

Agents can reason, plan, route, critique, and summarize.
Utilities can execute deterministic work such as memory lookup, chunking, vector search, dataframe profiling, and model training.
Pattern functions provide lightweight orchestration.
Specialist agents provide higher-level user-facing workflows.

This makes the library useful for both learning agentic AI patterns and building practical data science or automation workflows.

License

MIT