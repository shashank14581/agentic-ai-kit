"""
MCPClient: HTTP client for a running MCPServer.

Usage::

    client = MCPClient("http://localhost:8765")
    print(client.list_tools())
    result = client.call("echo", message="hello")
"""

from __future__ import annotations
import json
import urllib.request


class MCPClient:
    """Thin HTTP client for :class:`~agentic_ai.mcp.server.MCPServer`.

    Args:
        base_url: Server base URL, e.g. ``"http://localhost:8765"``.
    """

    def __init__(self, base_url: str = "http://localhost:8765"):
        self.base_url = base_url.rstrip("/")

    def list_tools(self) -> list[dict]:
        """Fetch available tools from the server."""
        with urllib.request.urlopen(f"{self.base_url}/tools") as resp:
            return json.loads(resp.read())

    def call(self, tool: str, **kwargs) -> object:
        """Call a named tool with keyword arguments.

        Returns:
            The ``result`` field from the server response.

        Raises:
            RuntimeError: If the server returns an error.
        """
        payload = json.dumps({"tool": tool, "args": kwargs}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        if "error" in data:
            raise RuntimeError(f"MCPServer error: {data['error']}")
        return data.get("result")

    def __repr__(self) -> str:
        return f"<MCPClient base_url={self.base_url!r}>"
