"""Example 11 — Start an MCP server and call it from an agent."""
import time
from agentic_ai.mcp.server import MCPServer
from agentic_ai.mcp.client import MCPClient
from agentic_ai.agents.tool_agent import ToolAgent

# --- Start the server in a background thread ---
server = MCPServer(port=8765)

@server.tool
def reverse_string(text: str) -> str:
    """Reverse the characters in a string."""
    return text[::-1]

@server.tool
def word_count(text: str) -> int:
    """Count the number of words in a string."""
    return len(text.split())

server.run_in_thread()
time.sleep(0.5)  # give it a moment to start

# --- Explore the server via the client ---
client = MCPClient("http://localhost:8765")
print("Available tools:", client.list_tools())
print("reverse_string('hello'):", client.call("reverse_string", text="hello"))
print("word_count(...):", client.call("word_count", text="agentic AI is really cool"))

# --- Wire the MCP tools into a ToolAgent ---
agent = ToolAgent("MCP Agent", "You are a helpful assistant with text utilities.")

def reverse_string_tool(text: str) -> str:
    return client.call("reverse_string", text=text)

def word_count_tool(text: str) -> int:
    return client.call("word_count", text=text)

agent.register_tool(reverse_string_tool, description="Reverse a string via MCP server")
agent.register_tool(word_count_tool,     description="Count words in a string via MCP server")

agent.think("How many words are in 'The quick brown fox jumps over the lazy dog'?")
