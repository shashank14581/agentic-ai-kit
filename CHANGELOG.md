# Changelog

All notable changes to this project are documented here.

## Unreleased

## 0.5.0 - 2026-08-26

### Added

- `aak` command-line interface with `run`, `chat`, and SQL workflows.
- Antigravity runtime adapter with synchronous and asynchronous execution.
- Google Antigravity as a built-in AAK runtime dependency.
- SQLite schema inspection through `aak sql inspect`.
- Natural-language database querying through `aak sql ask`.
- Agentic SQL execution through `aak sql agent`.
- Read-only SQL validation and bounded result execution.
- Autonomous SQL plan, execute, observe, repair, and retry loop.
- Optional SQL execution traces.

### Changed

- One-shot `aak run` disables conversational memory extraction.
- `aak chat` retains transformer-backed conversational memory.
- Agents can run through either the native Gemini runtime or Antigravity.

## 0.4.0 - 2026-08-25

### Added

- Opt-in stateless Interactions API transport for `BaseAgent`.
- Streaming, non-streaming, metadata, completion-status, and stream-error handling.
- Contract tests for Interactions request construction and lifecycle events.

### Changed

- Raised the minimum `google-genai` dependency to 2.19.0.
- Preserved `generate_content` as the default transport for backward compatibility.
- Normalized empty stateless interaction IDs to `None`.
- Clarified model-specific thinking controls and partial streaming metadata.

## 0.3.0 - 2026-07-25

### Added

- Local transformer-backed conversational memory retrieval.
- Shared Sentence Transformer model caching across agent instances.
- Semantic top-k retrieval using normalized sentence embeddings.

### Changed

- Replaced Gemini-based fact extraction with local memory ingestion.
- Preserved the existing `extract_facts` interface and `facts_store` schema.
- Updated context construction to retrieve relevant memories instead of including every stored fact.
- Added pruning and clearing of transformer embedding state.

## 0.2.1 - 2026-07-17

### Changed

- Improved PyPI description and search metadata.
- Clarified the repository, PyPI distribution, installation, and Python import names.
- Added documentation, issue tracker, and changelog project links.
- Removed generated egg-info metadata from source control.

## 0.2.0 - 2026-07-16

### Added

- Deterministic identity-sensitive appraisal with operational outcome regions.
- Linked timestep memory with rooted event trees.
- Separate identity and action-policy ledgers.
- Identity-conditioned forgetting and outcome-dependent retention floors.
- Relevance-first retrieval, trajectory deduplication, and bounded salience.
- Relevance-gated NumPy tree attention with traceable node weights.
- Correction, expiry, deletion, deterministic replay, and audit journaling.
- Memory-augmented RL state encoding and deterministic linear Q-learning.
- Falsification fixtures, integration documentation, and a no-API example.

### Evidence boundary

The outcome labels are operational states. They are not claims of subjective
emotion, and they are not used as reinforcement-learning rewards. Environmental
reward remains the learning signal; identity and memory are policy inputs.
