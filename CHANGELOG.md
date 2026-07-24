# Changelog

All notable changes to this project are documented here.

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
