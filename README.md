# agentic-ai-kit

**agentic-ai-kit** is an open-source Python framework for building
emotion-aware AI agents, adaptive episodic memory systems, SQL AI
workflows, tool-using agents, RAG pipelines, MCP integrations, and
Gemini-powered multi-agent applications.

> **Naming**
>
> - GitHub repository: `agentic-ai-kit`
> - PyPI package: `agentic-ai-kit`
> - Installation: `pip install agentic-ai-kit`
> - Python import package: `agentic_ai`

---

## AAK 0.5.5

AAK 0.5.5 includes the command-line interface, Agentic SQL AI, the Antigravity execution runtime, and documented optional installation extras.

### CLI

```bash
aak --help

aak run "Explain gradient descent"

aak run --adapter antigravity "Investigate this problem"

aak chat
```

### Agentic SQL AI

```bash
aak sql inspect database.db

aak sql ask database.db "Which category generated the most revenue?" --show-sql

aak sql agent database.db "Investigate which category leads revenue" --trace
```

The SQL agent can inspect database schema, generate read-only SQLite queries,
execute them, observe validation or execution failures, repair the SQL,
retry, and interpret the successful result.

### Antigravity Runtime

```python
from agentic_ai import BaseAgent
from agentic_ai.adapters import AntigravityAdapter

agent = BaseAgent(
    name="AAK",
    sys_prompt="Be concise.",
    extract_memory=False,
)

runtime = AntigravityAdapter(agent)

print(runtime.run("Hello"))
```

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Layout](#package-layout)
- [Agents](#agents)
  - [BaseAgent](#baseagent)
  - [ToolAgent](#toolagent)
  - [JsonAgent](#jsonagent)
  - [ReasoningAgent](#reasoningagent)
  - [AnalystAgent](#analystagent)
  - [MLEAgent](#mleagent)
  - [AutoModelAgent](#automodelagent)
  - [RAGAgent](#ragagent)
- [Memory](#memory)
  - [Short-Term Memory](#short-term-memory)
  - [Long-Term Memory](#long-term-memory)
  - [Shared Memory](#shared-memory)
  - [Emotion-Aware Episodic Memory](#emotion-aware-episodic-memory)
- [Tools](#tools)
- [SQL AI](#sql-ai)
- [Multi-Agent Patterns](#multi-agent-patterns)
- [MCP](#mcp-model-context-protocol)
- [Examples](#examples)
- [License](#license)

---

## Installation

Choose the installation that matches the features you need.

### Core

Lightweight AAK installation:

```bash
pip install agentic-ai-kit
```

### Memory

Adds transformer-backed semantic memory:

```bash
pip install "agentic-ai-kit[memory]"
```

### Antigravity

Adds the Google Antigravity execution runtime:

```bash
pip install "agentic-ai-kit[antigravity]"
```

### Full

Installs all optional AAK capabilities:

```bash
pip install "agentic-ai-kit[full]"
```

Install the latest GitHub version:

```bash
pip install --upgrade --force-reinstall \
  git+https://github.com/shashank14581/agentic-ai-kit.git
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

In Google Colab:

```python
import os
from google.colab import userdata

os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```

---

## Quick Start

```python
from agentic_ai.agents import BaseAgent

agent = BaseAgent(
    name="Alfred",
    sys_prompt="You are a witty British butler.",
    model="gemini-2.5-flash-lite",
    extract_memory=False,
)

response = agent.think(
    "Good morning! What should I do today?",
    stream=False,
)

print(response)
```

---

## Package Layout

```text
agentic_ai/
├── agents/
│   ├── base.py
│   ├── tool_agent.py
│   ├── json_agent.py
│   ├── reasoning_agent.py
│   ├── analyst_agent.py
│   ├── mle_agent.py
│   └── auto_model_agent.py
├── memory/
│   ├── short_term.py
│   ├── long_term.py
│   ├── shared.py
│   └── emotion/
│       ├── appraisal.py
│       ├── ledgers.py
│       ├── topology.py
│       ├── retention.py
│       ├── retrieval.py
│       ├── tree_attention.py
│       ├── system.py
│       └── rl.py
├── tools/
│   ├── registry.py
│   └── builtins.py
├── patterns/
│   ├── orchestrator.py
│   ├── parallel.py
│   └── debate.py
├── rag/
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── rag_agent.py
├── mcp/
│   ├── server.py
│   └── client.py
├── sql_ai/
│   ├── __init__.py
│   └── runner.py
└── examples/
```

---

## Agents

### BaseAgent

**File:** `agentic_ai/agents/base.py`

The foundation for all other agents. It supports:

- persona and system instructions,
- recent-turn conversation memory,
- optional durable-fact extraction,
- streaming and non-streaming generation,
- selectable `generate_content` and `interactions` transports,
- configurable Gemini models.

#### Constructor

```python
BaseAgent(
    name: str,
    sys_prompt: str,
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
    memory_window: int = 3,
    max_turns: int | None = None,
    max_facts: int = 50,
    extract_memory: bool = True,
    thinking_budget: int = 0,
    transport: str = "generate_content",
    thinking_level: str | None = None,
    client: object | None = None,
)
```

#### Runnable Example

```python
import os

from agentic_ai.agents import BaseAgent


if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("Set GEMINI_API_KEY before running this example.")


agent = BaseAgent(
    name="Sage",
    sys_prompt="You are a wise and concise assistant.",
    model="gemini-2.5-flash-lite",
    memory_window=3,
    max_turns=10,
    extract_memory=True,
)

response = agent.think(
    "I am learning Python. Explain list comprehensions simply.",
    stream=False,
)

print(response)
print(agent.memory)
print(agent.facts_store)

agent.clear_memory()
```

---

#### Interactions Transport

`generate_content` remains the default for compatibility. Use `transport="interactions"` to route requests through `client.interactions.create()`.

```python
agent = BaseAgent(
    name="Agent",
    sys_prompt="Be concise.",
    transport="interactions",
)
```

Thinking controls are model-specific. With the default
`gemini-2.5-flash-lite`, leave `thinking_level` unset because the API maps
`low` to a budget below the model minimum. Use a compatible Gemini 3
model when configuring `thinking_level`.

Streaming and non-streaming requests are supported. Failed interactions and stream errors raise `RuntimeError`.

The initial integration is stateless and sends `store=False`, so existing local BaseAgent memory remains the source of conversational context. Server-side continuation with `previous_interaction_id` is not yet enabled.

Response metadata is exposed through `last_interaction`, `last_interaction_id`, `last_steps`, and `last_usage`.

With `store=False`, `last_interaction_id` may be `None`. Streaming completion
events contain a partial interaction snapshot, so `last_steps` may be empty
even when step deltas produced the returned text. Use non-streaming generation
when a complete final steps list is required.

---

### ToolAgent

**File:** `agentic_ai/agents/tool_agent.py`

`ToolAgent` extends `BaseAgent` with Gemini function calling. Register Python functions and let the model decide when to call them.

#### Constructor

```python
ToolAgent(
    name: str,
    sys_prompt: str,
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
)
```

#### Runnable Example

```python
import os

from agentic_ai.agents import ToolAgent


if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("Set GEMINI_API_KEY before running this example.")


def calculate_rectangle_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width


agent = ToolAgent(
    name="Calculator",
    sys_prompt="Use available tools whenever calculation is required.",
    model="gemini-2.5-flash",
)

agent.register_tool(calculate_rectangle_area)

response = agent.think(
    "What is the area of a rectangle that is 12.5 metres long and 8 metres wide?"
)

print(response)
```

You may explicitly describe parameters:

```python
agent.register_tool(
    calculate_rectangle_area,
    description="Calculate the area of a rectangle.",
    params={
        "length": "NUMBER",
        "width": "NUMBER",
    },
    required=["length", "width"],
)
```

---

### JsonAgent

**File:** `agentic_ai/agents/json_agent.py`

`JsonAgent` returns a parsed Python dictionary instead of free-form text.

#### Constructor

```python
JsonAgent(
    name: str,
    sys_prompt: str,
    schema: dict | None = None,
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
)
```

#### Runnable Example

```python
import os

from agentic_ai.agents import JsonAgent


if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("Set GEMINI_API_KEY before running this example.")


agent = JsonAgent(
    name="Extractor",
    sys_prompt="Extract structured entities from the supplied text.",
    schema={
        "people": ["string"],
        "places": ["string"],
        "dates": ["string"],
    },
    model="gemini-2.5-flash-lite",
)

result = agent.think(
    "Alice visited Paris on 14 July with Bob."
)

print(result)
print(result["people"])
print(result["places"])
```

`JsonAgent.think()` raises `ValueError` if the model response cannot be parsed as JSON.

---

### ReasoningAgent

**File:** `agentic_ai/agents/reasoning_agent.py`

`ReasoningAgent` returns a `ReasoningOutput` object with:

- `reasoning`: model-generated planning text,
- `answer`: the final response,
- `context_for_next`: a combined string suitable for passing to another agent.

#### Constructor

```python
ReasoningAgent(
    name: str,
    sys_prompt: str,
    show_reasoning: bool = True,
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
)
```

#### Runnable Example

```python
import os

from agentic_ai.agents import ReasoningAgent


if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("Set GEMINI_API_KEY before running this example.")


planner = ReasoningAgent(
    name="Planner",
    sys_prompt="Break complex tasks into practical steps.",
    model="gemini-2.5-flash-lite",
    show_reasoning=True,
)

result = planner.think(
    "Plan a surprise birthday party for 30 people on a 500 dollar budget.",
    stream=False,
)

print("REASONING")
print(result.reasoning)

print("\nANSWER")
print(result.answer)
```

#### Planner-to-Executor Example

```python
import os

from agentic_ai.agents import BaseAgent, ReasoningAgent


if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("Set GEMINI_API_KEY before running this example.")


planner = ReasoningAgent(
    name="Planner",
    sys_prompt="Create a concrete implementation plan.",
    model="gemini-2.5-flash-lite",
)

executor = BaseAgent(
    name="Executor",
    sys_prompt="Execute the supplied plan carefully.",
    model="gemini-2.5-flash-lite",
)

plan = planner.think(
    "Create a release checklist for a Python package.",
    stream=False,
)

result = executor.think(
    plan.context_for_next,
    stream=False,
)

print(result)
```

---

### AnalystAgent

**File:** `agentic_ai/agents/analyst_agent.py`

`AnalystAgent` is a dataframe-aware analytics specialist. It can profile data, summarize numeric and categorical columns, calculate grouped metrics, inspect correlations, and ask Gemini to interpret the results.

#### Constructor

```python
AnalystAgent(
    name: str = "Analyst",
    domain_context: str | None = None,
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
    memory_window: int = 3,
    max_turns: int | None = None,
    thinking_budget: int = 0,
)
```

#### Key Methods

| Method | Returns | Description |
|---|---|---|
| `profile_dataframe(df)` | `dict` | Rows, columns, data types, missingness, and duplicates |
| `numeric_summary(df)` | `dict` | Descriptive statistics for numeric columns |
| `categorical_summary(df, top_n=10)` | `dict` | Top values for categorical columns |
| `groupby_summary(df, group_col, metric_col, agg)` | `list[dict]` | Group-level aggregation |
| `correlation_summary(df)` | `dict` | Numeric correlation matrix |
| `analyze_dataframe(df, question)` | `str` | Gemini-generated interpretation |

#### Runnable Example

```python
import os

import pandas as pd

from agentic_ai.agents import AnalystAgent


if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("Set GEMINI_API_KEY before running this example.")


df = pd.DataFrame(
    {
        "department": [
            "Engineering",
            "Sales",
            "Engineering",
            "Support",
            "Sales",
            "Support",
            "Engineering",
            "Sales",
        ],
        "employees": [32, 25, 35, 18, 28, 20, 40, 30],
        "monthly_cost": [280000, 190000, 310000, 120000, 205000, 135000, 350000, 220000],
        "satisfaction_score": [8.3, 7.1, 8.5, 6.9, 7.4, 7.2, 8.7, 7.6],
    }
)


agent = AnalystAgent(
    domain_context="Workforce planning and departmental performance.",
    model="gemini-2.5-flash-lite",
)


print("\n=== DATAFRAME PROFILE ===")
print(agent.profile_dataframe(df))


print("\n=== NUMERIC SUMMARY ===")
print(agent.numeric_summary(df))


print("\n=== CATEGORICAL SUMMARY ===")
print(agent.categorical_summary(df))


print("\n=== COST BY DEPARTMENT ===")
print(
    agent.groupby_summary(
        df=df,
        group_col="department",
        metric_col="monthly_cost",
        agg="sum",
    )
)


print("\n=== CORRELATION SUMMARY ===")
print(agent.correlation_summary(df))


print("\n=== GEMINI ANALYSIS ===")
analysis = agent.analyze_dataframe(
    df=df,
    question=(
        "Which departments appear most expensive, how does satisfaction vary, "
        "and what should management investigate next?"
    ),
    stream=False,
)

print(analysis)
```

The dataframe calculations are performed locally with pandas. Only `analyze_dataframe()` sends the generated summary context to Gemini.

---

### MLEAgent

**File:** `agentic_ai/agents/mle_agent.py`

`MLEAgent` profiles a dataframe, examines the target and features, performs a heuristic leakage scan, trains baseline scikit-learn pipelines, compares metrics, and returns the best fitted pipeline.

The `model` parameter controls the **Gemini model used for interpretation**. The predictive scikit-learn models are selected internally.

Current baseline candidates:

- Classification: logistic regression and random forest classifier
- Regression: linear regression and random forest regressor

#### Constructor

```python
MLEAgent(
    name: str = "MLE",
    project_context: str | None = None,
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
    memory_window: int = 3,
    max_turns: int | None = None,
    thinking_budget: int = 0,
)
```

#### Runnable Classification Example

```python
import os

import numpy as np
import pandas as pd

from agentic_ai.agents import MLEAgent


if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("Set GEMINI_API_KEY before running this example.")


rng = np.random.default_rng(42)
row_count = 250

study_hours = rng.normal(5, 2, size=row_count).clip(0.5)
attendance = rng.normal(82, 10, size=row_count).clip(40, 100)
previous_score = rng.normal(68, 12, size=row_count).clip(20, 100)
course_type = rng.choice(
    ["Online", "Classroom", "Hybrid"],
    size=row_count,
)

course_effect = np.select(
    [
        course_type == "Online",
        course_type == "Classroom",
        course_type == "Hybrid",
    ],
    [
        -0.3,
        0.2,
        0.6,
    ],
)

pass_score = (
    0.45 * study_hours
    + 0.05 * attendance
    + 0.04 * previous_score
    + course_effect
    + rng.normal(0, 1.5, size=row_count)
)

passed = (
    pass_score > np.median(pass_score)
).astype(int)


df = pd.DataFrame(
    {
        "study_hours": study_hours.round(2),
        "attendance": attendance.round(2),
        "previous_score": previous_score.round(2),
        "course_type": course_type,
        "passed": passed,
    }
)


agent = MLEAgent(
    project_context="Student course-completion prediction.",
    model="gemini-2.5-flash-lite",
)


print("\n=== TARGET SUMMARY ===")
print(agent.target_summary(df, "passed"))


print("\n=== FEATURE SUMMARY ===")
print(agent.feature_summary(df, "passed"))


print("\n=== LEAKAGE SCAN ===")
print(agent.leakage_scan(df, "passed"))


result = agent.create_model(
    df=df,
    target_col="passed",
    objective="Predict whether a student will pass the course.",
    test_size=0.25,
    random_state=42,
    interpret=False,
)


print("\n=== MODEL RESULT ===")
print("Problem type:", result["problem_type"])
print("Best model:", result["best_model"])
print("Best score:", result["best_score"])
print("Model comparison:", result["model_comparison"])


best_pipeline = result["best_pipeline"]

new_students = pd.DataFrame(
    {
        "study_hours": [2.0, 6.5, 9.0],
        "attendance": [58.0, 84.0, 96.0],
        "previous_score": [45.0, 72.0, 91.0],
        "course_type": ["Online", "Hybrid", "Classroom"],
    }
)

predictions = best_pipeline.predict(new_students)

print("\n=== NEW PREDICTIONS ===")
print(predictions)
```

Set `interpret=True` for a Gemini explanation after training:

```python
result = agent.create_model(
    df=df,
    target_col="passed",
    objective="Predict whether a student will pass the course.",
    interpret=True,
    stream=False,
)

print(result["llm_interpretation"])
```

> `interpret=False` avoids the extra interpretation call, but the current class still requires a Gemini API key during construction because `MLEAgent` inherits from `BaseAgent`.

---

### AutoModelAgent

**File:** `agentic_ai/agents/auto_model_agent.py`

`AutoModelAgent` extends `MLEAgent` into an end-to-end dataframe-to-model workflow.

It:

1. profiles the dataframe,
2. summarizes the target,
3. summarizes candidate features,
4. scans for leakage-prone columns,
5. optionally creates a modeling policy with Gemini,
6. trains baseline scikit-learn pipelines,
7. selects the best model,
8. optionally creates a final interpretation.

The `model` parameter controls the Gemini model used for policy generation and interpretation.

#### Main Method

```python
agent.run(
    df: pd.DataFrame,
    target_col: str,
    objective: str,
    drop_columns: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stream: bool = True,
    interpret: bool = True,
) -> dict
```

#### Runnable Property-Price Example

```python
import os

import numpy as np
import pandas as pd

from agentic_ai.agents import AutoModelAgent


if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("Set GEMINI_API_KEY before running this example.")


rng = np.random.default_rng(42)
row_count = 350

square_feet = rng.integers(600, 3500, size=row_count)
bedrooms = rng.integers(1, 6, size=row_count)
bathrooms = rng.integers(1, 5, size=row_count)
property_age = rng.integers(0, 50, size=row_count)

location = rng.choice(
    ["Central", "Suburban", "Outer"],
    size=row_count,
    p=[0.30, 0.45, 0.25],
)

location_effect = np.select(
    [
        location == "Central",
        location == "Suburban",
        location == "Outer",
    ],
    [
        150000,
        60000,
        0,
    ],
)

sale_price = (
    180 * square_feet
    + 25000 * bedrooms
    + 18000 * bathrooms
    - 2500 * property_age
    + location_effect
    + rng.normal(0, 35000, size=row_count)
)


df = pd.DataFrame(
    {
        "property_id": [
            f"PROP_{index:04d}"
            for index in range(row_count)
        ],
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "property_age": property_age,
        "location": location,
        "sale_price": sale_price.round(2),
    }
)


agent = AutoModelAgent(
    project_context="Residential property price prediction.",
    model="gemini-2.5-flash-lite",
)


result = agent.run(
    df=df,
    target_col="sale_price",
    objective="Predict the sale price of a residential property.",
    drop_columns=["property_id"],
    test_size=0.25,
    random_state=42,
    interpret=False,
)


print("\n=== DATA PROFILE ===")
print(result["data_profile"])


print("\n=== TARGET SUMMARY ===")
print(result["target_summary"])


print("\n=== FEATURE SUMMARY ===")
print(result["feature_summary"])


print("\n=== LEAKAGE SCAN ===")
print(result["leakage_scan"])


print("\n=== BEST MODEL ===")
print(result["best_model"])


print("\n=== BEST SCORE ===")
print(result["best_score"])


print("\n=== MODEL COMPARISON ===")
print(result["model_result"]["model_comparison"])


best_pipeline = result["best_pipeline"]

new_properties = pd.DataFrame(
    {
        "square_feet": [850, 1650, 2800],
        "bedrooms": [2, 3, 5],
        "bathrooms": [1, 2, 4],
        "property_age": [20, 8, 2],
        "location": ["Outer", "Suburban", "Central"],
    }
)

predictions = best_pipeline.predict(new_properties)

print("\n=== PRICE PREDICTIONS ===")
print(predictions)
```

#### Full Gemini-Assisted Workflow

```python
result = agent.run(
    df=df,
    target_col="sale_price",
    objective="Predict the sale price of a residential property.",
    drop_columns=["property_id"],
    test_size=0.25,
    random_state=42,
    interpret=True,
    stream=False,
)

print("\n=== MODELING POLICY ===")
print(result["modeling_policy"])

print("\n=== INTERPRETATION ===")
print(result["interpretation"])
```

With `interpret=True`, `AutoModelAgent` makes two Gemini calls:

1. one for modeling-policy generation,
2. one for final result interpretation.

With `interpret=False`, local scikit-learn training still runs and both interpretation fields are returned as `None`.

---

### RAGAgent

**File:** `agentic_ai/rag/rag_agent.py`

`RAGAgent` combines text chunking, Gemini embeddings, in-memory vector retrieval, and grounded response generation.

#### Constructor

```python
RAGAgent(
    name: str,
    sys_prompt: str,
    top_k: int = 3,
    **base_agent_kwargs,
)
```

#### Runnable Example

```python
import os

from agentic_ai.rag.rag_agent import RAGAgent


if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("Set GEMINI_API_KEY before running this example.")


document = """
The Acme Learning Platform allows learners to cancel a course purchase
within 14 days. Refunds are processed to the original payment method.
Completed courses are not eligible for refunds.
"""


agent = RAGAgent(
    name="Policy Assistant",
    sys_prompt=(
        "Answer only from the retrieved context. "
        "Say that the answer is unavailable when the context does not contain it."
    ),
    top_k=3,
    model="gemini-2.5-flash-lite",
)


chunk_count = agent.ingest(
    document,
    chunk_size=200,
    overlap=20,
)

print("Chunks ingested:", chunk_count)


answer = agent.think(
    "How long does a learner have to cancel a course purchase?",
    stream=False,
)

print(answer)
```

The default vector store is in memory and does not persist across Python sessions.

---

## Memory

### Short-Term Memory

**File:** `agentic_ai/memory/short_term.py`

A fixed-window store of `(user_input, agent_output)` pairs.

#### Runnable Example

```python
from agentic_ai.memory.short_term import ShortTermMemory


memory = ShortTermMemory(window=3)

memory.add(
    "What is Python?",
    "Python is a general-purpose programming language.",
)

memory.add(
    "Who created it?",
    "Python was created by Guido van Rossum.",
)

print(memory.as_text())
print(list(memory))
print("Stored turns:", len(memory))

memory.clear()
```

---

### Long-Term Memory

**File:** `agentic_ai/memory/long_term.py`

SQLite-backed persistent memory for logging, retrieving, and searching entries.

#### Runnable Example

```python
from agentic_ai.memory.long_term import LongTermMemory


memory = LongTermMemory("agent_memory.db")

entry_id = memory.log(
    content="The deployment target is Python 3.12.",
    agent="ReleaseManager",
)

print("Stored entry:", entry_id)

memory.log(
    content="The package should be published after tests pass.",
    agent="ReleaseManager",
)

print("\n=== RECENT ===")
print(memory.recent(n=10, agent="ReleaseManager"))

print("\n=== SEARCH ===")
print(memory.search("Python", agent="ReleaseManager"))

memory.close()
```

---

### Shared Memory

**File:** `agentic_ai/memory/shared.py`

A thread-safe in-process key-value store that multiple agents or workers can share.

#### Runnable Example

```python
from agentic_ai.memory.shared import SharedMemory


shared = SharedMemory()

shared.set("plan", "Build, test, and publish the package.")
shared.set("status", "in_progress")

print(shared.get("plan"))
print(shared.all())

shared.delete("status")
shared.clear()
```

---

### Emotion-Aware Episodic Memory

**Package:** `agentic_ai.memory.emotion`

The emotion-memory package implements deterministic, operational affective
state. It does not claim subjective emotion. Its five labels—`ordinary`,
`success`, `failure`, `wound`, and `trauma`—are derived from continuous
appraisal and state-transition values.

The implementation includes:

- identity-conditioned appraisal and forgetting,
- linked timestep memory whose roots contain event trees,
- separate identity and action-policy ledgers,
- outcome-dependent retention floors,
- relevance-first retrieval and trajectory deduplication,
- relevance-gated NumPy tree attention,
- correction, expiry, deletion, deterministic replay, and audit records,
- an RL state encoder and linear Q-learner that use environmental reward.

Outcome salience protects memory survival and fidelity; it never bypasses the
semantic relevance gate. Operational labels are not used as RL rewards.

#### Runnable Example

The complete demonstration needs no Gemini API key:

```bash
python agentic_ai/examples/12_emotion_memory.py
```

It ingests three recurring failures, demonstrates the
`failure -> wound -> trauma` transition, performs tree-attention retrieval,
constructs an RL state, applies an environment-reward Q update, then deletes
one event and deterministically replays the remaining history.

The primary public components are:

| Component | Responsibility |
|---|---|
| `EmotionMemorySystem` | Ingestion, correction, replay, retrieval, and audit |
| `AppraisalEngine` | Identity-sensitive appraisal and operational state |
| `EpisodicTimeline` | Linked timesteps and rooted event trees |
| `IdentityLedger` | Positive and negative contrastive self-facts |
| `PolicyLedger` | Repeat and avoidance evidence |
| `RetentionEngine` | Identity decay and outcome-dependent floors |
| `RelevanceFirstRetriever` | Eligibility, deduplication, and bounded ranking |
| `TreeAttentionEngine` | Attention within an eligible event tree |
| `RLStateEncoder` | Environment, identity, memory, and policy features |
| `LinearQLearner` | Deterministic Q-learning with external reward |

The mathematical contract and falsification criteria are documented in
`docs/emotion_architecture.md`. Default parameters are frozen in
`configs/emotion_architecture_v0.1.yaml`.

---

## Tools

`ToolAgent.register_tool()` can expose ordinary Python callables to Gemini.

```python
from agentic_ai.agents import ToolAgent


def convert_celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature to Fahrenheit."""
    return celsius * 9 / 5 + 32


agent = ToolAgent(
    name="Converter",
    sys_prompt="Use tools for unit conversions.",
)

agent.register_tool(convert_celsius_to_fahrenheit)

print(
    agent.think(
        "Convert 25 degrees Celsius to Fahrenheit."
    )
)
```

Functions should have clear names, type hints, and concise docstrings because Gemini uses this information to decide when and how to call them.

---
---

---

## SQL AI

`SQL AI` lets you enrich local SQLite query results with Gemini-generated columns.

It is useful when you have structured data in SQLite and want to create summaries, labels, classifications, keywords, or retrieval-friendly text fields directly from SQL-style queries.

### Runnable Example

```python
import os
import sqlite3

import pandas as pd

from agentic_ai.sql_ai import run_ai_sql


if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("Set GEMINI_API_KEY before running this example.")


conn = sqlite3.connect(":memory:")

records = pd.DataFrame(
    {
        "record_id": [1, 2, 3],
        "topic": ["Access", "Documentation", "Automation"],
        "priority": ["High", "Medium", "Low"],
        "notes": [
            "User cannot access their account after changing password.",
            "Setup instructions are unclear and need examples.",
            "Incoming messages should be classified automatically.",
        ],
    }
)

records.to_sql("records", conn, if_exists="replace", index=False)

sql = """
SELECT
    record_id,
    topic,
    priority,
    notes,
    ai_generate(
        prompt='Write a short retrieval-friendly summary for this row',
        model='gemini-2.5-flash'
    ) AS retrieval_summary
FROM records
"""

result = run_ai_sql(
    sql=sql,
    conn=conn,
)

print(result)
```

### Supported SQL AI Pseudo-Functions

```sql
ai_generate(prompt='...', model='...') AS generated_text

ai_summarize(model='...') AS summary

ai_classify(labels='A | B | C', model='...') AS label

ai_extract(prompt='Extract retrieval keywords', count=5, model='...') AS keywords
```

### Write Results Back to SQLite

`SQL AI` also supports a notebook-friendly `CREATE OR REPLACE TABLE` pattern. SQLite does not support this syntax natively, but `run_ai_sql()` handles it by writing the final dataframe back to SQLite.

```python
sql = """
CREATE OR REPLACE TABLE enriched_records AS
SELECT
    record_id,
    notes,
    ai_summarize(
        model='gemini-2.5-flash'
    ) AS summary,
    ai_extract(
        prompt='Extract retrieval keywords from this row',
        count=5,
        model='gemini-2.5-flash'
    ) AS retrieval_keywords
FROM records
"""

result = run_ai_sql(
    sql=sql,
    conn=conn,
)

saved = pd.read_sql_query("SELECT * FROM enriched_records", conn)

print(saved)
```

## Multi-Agent Patterns

The package includes lightweight orchestration patterns:

- `run_conversation()` for round-robin conversations,
- `run_supervisor()` for delegation,
- `run_parallel()` for fan-out execution,
- `run_debate()` for opposing arguments and judging.

See the runnable files in `agentic_ai/examples/`.

---

## MCP (Model Context Protocol)

The MCP-style package includes:

- `MCPServer` for exposing tools over HTTP,
- `MCPClient` for discovering and calling remote tools.

See:

```text
agentic_ai/examples/11_mcp_demo.py
```

for the current runnable implementation.

---

## Examples

| File | Concept |
|---|---|
| `01_talking_agents.py` | Round-robin agent conversation |
| `02_personality_and_memory.py` | Persona and memory |
| `03_tool_agent_weather.py` | Gemini function calling |
| `04_json_agent.py` | Structured JSON output |
| `05_reasoning_agent.py` | Planner-to-executor workflow |
| `06_supervisor.py` | Supervisor delegation |
| `07_parallel_agents.py` | Parallel fan-out |
| `08_debate.py` | Debate and judging |
| `09_long_term_memory.py` | Persistent SQLite memory |
| `10_rag_agent.py` | Retrieval-augmented generation |
| `11_mcp_demo.py` | MCP-style tool integration |
| `12_emotion_memory.py` | Emotion-aware memory, replay, tree attention, and RL |

Run an example from the repository root:

```bash
python agentic_ai/examples/01_talking_agents.py
```

---

## Design Philosophy

`agentic-ai-kit` separates reasoning from deterministic execution:

- Agents reason, plan, summarize, route, and communicate.
- Python functions perform calculations and tool execution.
- pandas and scikit-learn perform deterministic data analysis and model training.
- Lightweight orchestration functions coordinate multiple agents.
- Specialist agents combine deterministic operations with optional LLM interpretation.
- Emotion-aware memory keeps relevance, retention, identity, policy, and reward as separate mechanisms.

---

## Important Notes

- The `model` argument on agents refers to the Gemini model.
- `MLEAgent` and `AutoModelAgent` select their predictive scikit-learn candidates internally.
- `interpret=False` avoids optional Gemini interpretation calls but does not currently remove the API-key requirement during agent construction.
- Leakage checks are heuristic and must be validated using real feature timing and domain knowledge.
- Built-in machine-learning workflows are baselines, not replacements for production validation, fairness analysis, monitoring, or expert review.
- The default RAG vector store is in memory.

---

## License

MIT
