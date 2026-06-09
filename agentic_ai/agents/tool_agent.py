"""
ToolAgent: an agent that exposes Python functions to Gemini as callable tools.

Usage::

    from agentic_ai.agents.tool_agent import ToolAgent
    from agentic_ai.tools.registry import tool

    @tool(description="Get weather for a city", params={"city": "STRING"})
    def get_weather(city: str) -> str:
        ...

    agent = ToolAgent("WeatherBot", "You are a helpful weather assistant.")
    agent.register_tool(get_weather)
    agent.think("What's the weather in Chennai?")
"""

import os
from google import genai
from google.genai import types
from agentic_ai.agents.base import BaseAgent


class ToolAgent(BaseAgent):
    """An agent that can invoke registered Python callables via Gemini function-calling.

    Args:
        name: Display name.
        sys_prompt: System instruction.
        model: Gemini model. Defaults to ``gemini-2.5-flash``.
        api_key: Optional API key (falls back to ``GEMINI_API_KEY``).
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ):
        super().__init__(name, sys_prompt, model=model, api_key=api_key)
        self._tool_fns: dict[str, callable] = {}
        self._tool_declarations: list[types.FunctionDeclaration] = []

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        fn: callable,
        description: str | None = None,
        params: dict | None = None,
        required: list[str] | None = None,
    ) -> "ToolAgent":
        """Register a Python callable as a Gemini tool.

        Args:
            fn: The Python function to expose.
            description: Human-readable description (uses docstring if omitted).
            params: Dict mapping param names to Gemini type strings
                    e.g. ``{"city": "STRING", "units": "STRING"}``.
                    If ``None``, inferred from type hints.
            required: List of required param names. Defaults to all params.

        Returns:
            Self, for chaining.
        """
        name = fn.__name__
        desc = description or (fn.__doc__ or "").strip().split("\n")[0]

        if params is None:
            import inspect
            hints = inspect.get_annotations(fn)
            _type_map = {str: "STRING", int: "NUMBER", float: "NUMBER", bool: "BOOLEAN"}
            params = {
                k: _type_map.get(v, "STRING")
                for k, v in hints.items()
                if k != "return"
            }

        properties = {k: {"type": v} for k, v in params.items()}
        req = required if required is not None else list(properties.keys())

        self._tool_fns[name] = fn
        self._tool_declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=desc,
                parameters={
                    "type": "OBJECT",
                    "properties": properties,
                    "required": req,
                },
            )
        )
        return self

    # ------------------------------------------------------------------
    # Override think() to handle tool calls
    # ------------------------------------------------------------------

    def think(self, input_text: str, use_memory: bool = True, stream: bool = False) -> str:
        """Process input, invoke tools if needed, and return the final response."""
        self.extract_facts(input_text)
        prompt = self.build_context(input_text) if use_memory else input_text

        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        ]
        tools_config = (
            [types.Tool(function_declarations=self._tool_declarations)]
            if self._tool_declarations
            else []
        )
        config = types.GenerateContentConfig(
            system_instruction=[types.Part.from_text(text=self.sys_prompt)],
            tools=tools_config,
        )

        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=config
        )

        # Check for a function call
        fc = None
        for part in response.candidates[0].content.parts:
            if part.function_call:
                fc = part.function_call
                break

        if fc:
            fn = self._tool_fns.get(fc.name)
            if fn is None:
                tool_result = f"Error: unknown tool '{fc.name}'"
            else:
                print(f"⚙️  {self.name} → {fc.name}({dict(fc.args)})")
                try:
                    tool_result = str(fn(**fc.args))
                except Exception as exc:
                    tool_result = f"Error: {exc}"
                print(f"🔧  {tool_result}")

            tool_response = types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=fc.name, response={"result": tool_result}
                    )
                ],
            )

            final = self.client.models.generate_content(
                model=self.model,
                contents=[
                    contents[0],
                    types.Content(
                        role="model",
                        parts=response.candidates[0].content.parts,
                    ),
                    tool_response,
                ],
                config=config,
            )
            output = final.text or ""
        else:
            output = response.text or ""

        print(output)
        self.memory.append((input_text, output))
        return output
