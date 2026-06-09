"""Example 1 — Two talking agents having a conversation."""
import os
from agentic_ai.agents.base import BaseAgent
from agentic_ai.patterns.orchestrator import run_conversation

human = BaseAgent("Shashank", "You are a curious human who asks short questions.")
dog   = BaseAgent("Bruno",    "You are a smart, friendly dog who speaks in short sentences.")

run_conversation([human, dog], opening_message="What's your favourite thing to do?", turns=4)
