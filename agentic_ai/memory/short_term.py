"""
ShortTermMemory: a fixed-window list of (input, output) pairs.

Can be attached to any BaseAgent or used standalone.
"""

from __future__ import annotations
from collections import deque
from typing import Iterator


class ShortTermMemory:
    """Sliding window memory.

    Args:
        window (int): Maximum number of turns to retain.
    """

    def __init__(self, window: int = 5):
        self.window = window
        self._store: deque[tuple[str, str]] = deque(maxlen=window)

    def add(self, user_input: str, agent_output: str) -> None:
        self._store.append((user_input, agent_output))

    def as_text(self) -> str:
        """Format memory as a plain-text history block."""
        lines = []
        for inp, out in self._store:
            lines.append(f"user: {inp}\nassistant: {out}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._store.clear()

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"<ShortTermMemory window={self.window} turns={len(self)}>"
