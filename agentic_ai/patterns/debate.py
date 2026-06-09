"""
Debate pattern: two agents argue opposing sides; a judge picks a winner.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_ai.agents.base import BaseAgent


def run_debate(
    agent_a: "BaseAgent",
    agent_b: "BaseAgent",
    judge: "BaseAgent",
    topic: str,
    rounds: int = 2,
) -> dict:
    """Run a structured debate and return the judge's verdict.

    Args:
        agent_a: First debater.
        agent_b: Second debater.
        judge: Neutral agent that evaluates the debate.
        topic: The motion to debate.
        rounds: Number of back-and-forth rounds.

    Returns:
        Dict with keys ``transcript`` (list of turns) and ``verdict`` (str).
    """
    transcript: list[dict] = []
    message = f"Debate topic: {topic}\nMake your opening argument."

    for _ in range(rounds):
        for agent in (agent_a, agent_b):
            response = str(agent.think(message, stream=False))
            transcript.append({"agent": agent.name, "text": response})
            message = f"Your opponent said:\n{response}\nNow respond."

    full_debate = "\n\n".join(
        f"{t['agent']}: {t['text']}" for t in transcript
    )
    verdict = str(judge.think(
        f"Here is a debate on '{topic}':\n\n{full_debate}\n\n"
        "Who made the stronger argument and why? Give a concise verdict.",
        stream=False,
    ))

    return {"transcript": transcript, "verdict": verdict}
