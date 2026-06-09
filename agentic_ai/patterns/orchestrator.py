"""
Orchestration patterns: sequential conversation and supervisor-delegate.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_ai.agents.base import BaseAgent


def run_conversation(
    agents: list["BaseAgent"],
    opening_message: str,
    turns: int = 5,
    verbose: bool = True,
) -> list[str]:
    """Run a round-robin conversation between a list of agents.

    Each agent receives the previous agent's output as its input.

    Args:
        agents: List of agents to involve (min 2).
        opening_message: The first message sent to ``agents[0]``.
        turns: Total number of turns (one turn = one agent speaks).
        verbose: Print agent names as they speak.

    Returns:
        List of all responses in order.
    """
    if len(agents) < 2:
        raise ValueError("run_conversation requires at least 2 agents.")

    responses: list[str] = []
    message = opening_message

    for i in range(turns):
        agent = agents[i % len(agents)]
        if verbose:
            print(f"\n{'─' * 40}\n{agent.name}:")
        response = agent.think(message)
        responses.append(response)
        message = str(response)  # handles ReasoningOutput too

    return responses


def run_supervisor(
    supervisor: "BaseAgent",
    workers: dict[str, "BaseAgent"],
    task: str,
    verbose: bool = True,
) -> dict[str, str]:
    """Supervisor pattern: the supervisor assigns tasks to named worker agents.

    The supervisor receives the original task and the names of available workers,
    and produces instructions for each one. Each worker then executes its task.

    Args:
        supervisor: The orchestrating agent.
        workers: Dict mapping worker names to agent instances.
        task: The high-level task to delegate.
        verbose: Print agent names as they work.

    Returns:
        Dict mapping worker names to their outputs.
    """
    worker_list = ", ".join(workers.keys())
    delegation_prompt = (
        f"You are a supervisor. Available workers: {worker_list}.\n"
        f"Task: {task}\n\n"
        f"For each worker, write exactly one line in the format:\n"
        f"<WorkerName>: <specific instruction>\n"
        f"Cover all workers."
    )

    if verbose:
        print(f"\n{'─' * 40}\nSupervisor ({supervisor.name}) delegating...")

    delegation = supervisor.think(delegation_prompt)
    results: dict[str, str] = {}

    for line in str(delegation).splitlines():
        for name, agent in workers.items():
            if line.strip().startswith(f"{name}:"):
                instruction = line.split(":", 1)[1].strip()
                if verbose:
                    print(f"\n{'─' * 40}\nWorker ({name}):")
                results[name] = str(agent.think(instruction))

    return results
