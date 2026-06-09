"""
MCPServer: a minimal Model Context Protocol-style tool server.

Exposes registered Python functions over a simple JSON-RPC-like HTTP interface
using Python's built-in ``http.server``. Suitable for local development and
agent-to-agent tool sharing.

Usage::

    server = MCPServer(port=8765)

    @server.tool
    def echo(message: str) -> str:
        return message

    server.run()  # blocks; use server.run_in_thread() for background use
"""

from __future__ import annotations
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Any


class MCPServer:
    """Minimal MCP-compatible tool server.

    Args:
        host: Bind address. Defaults to ``"localhost"``.
        port: TCP port to listen on. Defaults to ``8765``.
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._tools: dict[str, Callable] = {}
        self._thread: threading.Thread | None = None
        self._httpd: HTTPServer | None = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def tool(self, fn: Callable) -> Callable:
        """Decorator — register a function as an MCP tool."""
        self._tools[fn.__name__] = fn
        return fn

    def register(self, fn: Callable, name: str | None = None) -> None:
        self._tools[name or fn.__name__] = fn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict]:
        """Return a JSON-serialisable list of tool descriptors."""
        import inspect
        result = []
        for name, fn in self._tools.items():
            hints = inspect.get_annotations(fn)
            _map = {str: "string", int: "number", float: "number", bool: "boolean"}
            params = {
                k: {"type": _map.get(v, "string")}
                for k, v in hints.items()
                if k != "return"
            }
            result.append({
                "name": name,
                "description": (fn.__doc__ or "").strip().split("\n")[0],
                "parameters": {"type": "object", "properties": params},
            })
        return result

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------

    def _make_handler(self) -> type:
        tools = self._tools
        list_tools = self.list_tools

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # silence default access log

            def do_GET(self):
                if self.path == "/tools":
                    self._json(200, list_tools())
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                name = body.get("tool")
                kwargs = body.get("args", {})
                if name not in tools:
                    self._json(404, {"error": f"unknown tool '{name}'"})
                    return
                try:
                    result = tools[name](**kwargs)
                    self._json(200, {"result": result})
                except Exception as exc:
                    self._json(500, {"error": str(exc)})

            def _json(self, code: int, payload: Any):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    def run(self) -> None:
        """Start the server (blocks the calling thread)."""
        self._httpd = HTTPServer((self.host, self.port), self._make_handler())
        print(f"MCPServer listening on http://{self.host}:{self.port}")
        self._httpd.serve_forever()

    def run_in_thread(self) -> threading.Thread:
        """Start the server in a daemon thread and return the thread."""
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        self._thread = t
        return t

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
