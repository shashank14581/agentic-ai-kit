"""
SharedMemory: a simple in-process key-value store that multiple agents can read/write.

Useful for fan-out / fan-in patterns where agents collaborate on a shared state.

Usage::

    mem = SharedMemory()

    agent_a.shared_mem = mem
    agent_b.shared_mem = mem

    mem.set("plan", "Build a rocket.")
    print(mem.get("plan"))
"""

from __future__ import annotations
from threading import Lock


class SharedMemory:
    """Thread-safe in-process key-value store.

    All agents that hold a reference to the same instance share state.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._lock = Lock()

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._store[key] = value

    def get(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            return self._store.get(key, default)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def all(self) -> dict[str, str]:
        with self._lock:
            return dict(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __repr__(self) -> str:
        return f"<SharedMemory keys={list(self._store.keys())}>"
