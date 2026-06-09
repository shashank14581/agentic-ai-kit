"""
Built-in tools included with agentic_ai.

These are plain Python functions; register them with a ToolAgent via
``agent.register_tool(get_weather)``.
"""

from __future__ import annotations


def get_weather(city: str) -> str:
    """Get the current temperature for a city using the Open-Meteo API (no key needed)."""
    import requests

    geo = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    ).json()
    if "results" not in geo:
        return f"City not found: {city}"

    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]

    weather = requests.get(
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    ).json()
    data = weather.get("current_weather", {})
    temp = data.get("temperature", "?")
    code = data.get("weathercode", "?")
    return f"{city}: {temp}°C (weather code {code})"


def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


def search_wikipedia(query: str) -> str:
    """Return the first paragraph of a Wikipedia article for the given query."""
    import requests

    resp = requests.get(
        "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
    ).json()
    return resp.get("extract", "No result found.")
