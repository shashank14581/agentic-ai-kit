"""
Parallel pattern: fan-out a task to multiple agents simultaneously, then collect results.
"""

from __future__ import annotations
import concurrent.futures
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_ai.agents.base import BaseAgent


def run_parallel(
    agents: list["BaseAgent"],
    task: str,
    timeout: float = 60.0,
) -> dict[str, str]:
    """Send the same task to all agents in parallel and return their responses.

    Args:
        agents: List of agents to run concurrently.
        task: The prompt sent to every agent.
        timeout: Max seconds to wait for all agents.

    Returns:
        Dict mapping agent name to response string.
    """
    results: dict[str, str] = {}

    def _run(agent: "BaseAgent") -> tuple[str, str]:
        return agent.name, str(agent.think(task, stream=False))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as pool:
        futures = {pool.submit(_run, a): a.name for a in agents}
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            name, response = future.result()
            results[name] = response

    return results
