from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RuntimeAdapter(Protocol):
    """Runtime contract for external AAK execution runtimes."""

    async def arun(self, prompt: str) -> str:
        ...

    def run(self, prompt: str) -> str:
        ...
