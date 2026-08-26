from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import sqlglot
from google import genai
from google.genai import types

from .schema import format_schema, inspect_schema


FORBIDDEN_SQL_NODES = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "CREATE",
    "DROP",
    "ALTER",
    "MERGE",
    "COMMAND",
    "TRANSACTION",
    "ATTACH",
    "DETACH",
    "PRAGMA",
}


@dataclass
class SQLAskResult:
    question: str
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    answer: str
    truncated: bool = False


def _clean_model_sql(text: str) -> str:
    """Remove markdown fences and surrounding whitespace."""

    text = (text or "").strip()

    fenced = re.match(
        r"^```(?:sql)?\s*(.*?)\s*```$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced:
        text = fenced.group(1).strip()

    return text.rstrip(";").strip()


def validate_read_only_sql(sql: str) -> str:
    """Require exactly one read-only SELECT-style SQLite statement."""

    sql = _clean_model_sql(sql)

    if not sql:
        raise ValueError("Generated SQL is empty.")

    statements = sqlglot.parse(sql, read="sqlite")

    if len(statements) != 1:
        raise ValueError(
            "SQL AI generated multiple statements. "
            "Only one read-only query is allowed."
        )

    statement = statements[0]

    root_key = statement.key.upper()

    if root_key not in {
        "SELECT",
        "UNION",
        "INTERSECT",
        "EXCEPT",
    }:
        raise ValueError(
            f"Only read-only SELECT queries are allowed; got {root_key}."
        )

    for node in statement.walk():
        node_key = node.key.upper()

        if node_key in FORBIDDEN_SQL_NODES:
            raise ValueError(
                f"Unsafe SQL operation detected: {node_key}."
            )

    return sql


def execute_read_only_sql(
    conn: sqlite3.Connection,
    sql: str,
    max_rows: int = 100,
) -> tuple[list[str], list[tuple[Any, ...]], bool]:
    """Execute validated SQL and return a bounded result set."""

    if max_rows <= 0:
        raise ValueError("max_rows must be greater than 0.")

    sql = validate_read_only_sql(sql)

    cursor = conn.execute(sql)

    columns = [
        item[0]
        for item in (cursor.description or [])
    ]

    fetched = cursor.fetchmany(max_rows + 1)

    truncated = len(fetched) > max_rows

    rows = fetched[:max_rows]

    return columns, rows, truncated


def _format_result_for_model(
    columns: list[str],
    rows: list[tuple[Any, ...]],
    truncated: bool,
) -> str:
    if not columns:
        return "(query returned no columns)"

    lines = ["\t".join(columns)]

    for row in rows:
        lines.append(
            "\t".join(
                "NULL" if value is None else str(value)
                for value in row
            )
        )

    if truncated:
        lines.append("[RESULT TRUNCATED]")

    return "\n".join(lines)


def ask_sql(
    question: str,
    conn: sqlite3.Connection,
    *,
    client: Any | None = None,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    max_rows: int = 100,
) -> SQLAskResult:
    """
    Answer a natural-language question against a SQLite database.

    Flow:
        schema inspection
        -> SQL generation
        -> safety validation
        -> execution
        -> result interpretation
    """

    if not question.strip():
        raise ValueError("question cannot be empty.")

    if conn is None:
        raise ValueError("Pass a valid SQLite connection.")

    schema_text = format_schema(
        inspect_schema(conn)
    )

    if client is None:
        key = api_key or os.getenv("GEMINI_API_KEY")

        if not key:
            raise ValueError(
                "No Gemini API key found. "
                "Pass api_key= or set GEMINI_API_KEY."
            )

        client = genai.Client(api_key=key)

    sql_system = """
You are the SQL planner for Agentic AI Kit.

Generate exactly one SQLite read-only query that answers the user's question.

Rules:
- Return SQL only.
- Do not use markdown.
- Do not explain.
- Use only tables and columns present in the supplied schema.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA,
  ATTACH, DETACH, or any other mutating operation.
- Prefer concise queries.
""".strip()

    sql_prompt = f"""
DATABASE SCHEMA

{schema_text}

USER QUESTION

{question}
""".strip()

    sql_response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=sql_prompt
                    )
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=[
                types.Part.from_text(
                    text=sql_system
                )
            ],
        ),
    )

    generated_sql = validate_read_only_sql(
        sql_response.text or ""
    )

    columns, rows, truncated = execute_read_only_sql(
        conn,
        generated_sql,
        max_rows=max_rows,
    )

    result_text = _format_result_for_model(
        columns,
        rows,
        truncated,
    )

    answer_system = """
You are a data analyst.

Answer the user's question using only the supplied SQL result.

Rules:
- Be concise.
- Do not invent facts not present in the result.
- Mention when the result is empty.
- If the result was truncated, say so.
""".strip()

    answer_prompt = f"""
USER QUESTION

{question}

SQL EXECUTED

{generated_sql}

QUERY RESULT

{result_text}
""".strip()

    answer_response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=answer_prompt
                    )
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=[
                types.Part.from_text(
                    text=answer_system
                )
            ],
        ),
    )

    return SQLAskResult(
        question=question,
        sql=generated_sql,
        columns=columns,
        rows=rows,
        answer=(answer_response.text or "").strip(),
        truncated=truncated,
    )


@dataclass
class SQLAgentAttempt:
    """One planning/execution attempt made by the SQL agent."""

    attempt: int
    sql: str
    error: str | None = None


@dataclass
class SQLAgentResult:
    """Final result from the autonomous SQL agent."""

    question: str
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    answer: str
    attempts: list[SQLAgentAttempt]
    truncated: bool = False


def run_sql_agent(
    question: str,
    conn: sqlite3.Connection,
    *,
    client: Any | None = None,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    max_rows: int = 100,
    max_attempts: int = 3,
) -> SQLAgentResult:
    """
    Autonomously plan, execute, observe, and repair SQLite queries.

    Loop:
        inspect schema
        -> generate SQL
        -> validate
        -> execute
        -> observe error
        -> repair SQL
        -> retry
        -> interpret result
    """

    if not question.strip():
        raise ValueError("question cannot be empty.")

    if conn is None:
        raise ValueError("Pass a valid SQLite connection.")

    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than 0.")

    if max_rows <= 0:
        raise ValueError("max_rows must be greater than 0.")

    schema_text = format_schema(
        inspect_schema(conn)
    )

    if client is None:
        key = api_key or os.getenv("GEMINI_API_KEY")

        if not key:
            raise ValueError(
                "No Gemini API key found. "
                "Pass api_key= or set GEMINI_API_KEY."
            )

        client = genai.Client(api_key=key)

    planner_system = """
You are an autonomous SQLite agent inside Agentic AI Kit.

Your job is to answer the user's question by generating one safe,
read-only SQLite query.

Rules:
- Return SQL only.
- Do not use markdown.
- Do not explain.
- Use only tables and columns from the supplied schema.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA,
  ATTACH, DETACH, or other mutating operations.
- If previous SQL failed, study the error and repair the query.
- Do not repeat a query that already failed.
- Prefer the smallest query that answers the question.
""".strip()

    planner_prompt = f"""
DATABASE SCHEMA

{schema_text}

USER QUESTION

{question}
""".strip()

    attempts: list[SQLAgentAttempt] = []

    successful_sql: str | None = None
    columns: list[str] = []
    rows: list[tuple[Any, ...]] = []
    truncated = False

    for attempt_number in range(1, max_attempts + 1):
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=planner_prompt
                        )
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=[
                    types.Part.from_text(
                        text=planner_system
                    )
                ],
            ),
        )

        candidate_sql = _clean_model_sql(
            response.text or ""
        )

        try:
            safe_sql = validate_read_only_sql(
                candidate_sql
            )

            columns, rows, truncated = (
                execute_read_only_sql(
                    conn,
                    safe_sql,
                    max_rows=max_rows,
                )
            )

            attempts.append(
                SQLAgentAttempt(
                    attempt=attempt_number,
                    sql=safe_sql,
                    error=None,
                )
            )

            successful_sql = safe_sql
            break

        except Exception as exc:
            error_text = str(exc)

            attempts.append(
                SQLAgentAttempt(
                    attempt=attempt_number,
                    sql=candidate_sql,
                    error=error_text,
                )
            )

            if attempt_number >= max_attempts:
                trace = "\n".join(
                    (
                        f"attempt {item.attempt}: "
                        f"{item.sql} -> "
                        f"{item.error or 'success'}"
                    )
                    for item in attempts
                )

                raise RuntimeError(
                    "SQL agent exhausted all repair attempts.\n"
                    f"{trace}"
                ) from exc

            planner_prompt = f"""
DATABASE SCHEMA

{schema_text}

USER QUESTION

{question}

PREVIOUS SQL

{candidate_sql}

EXECUTION OR VALIDATION ERROR

{error_text}

Repair the SQL using the schema and error above.

Return only the corrected SQLite query.
""".strip()

    if successful_sql is None:
        raise RuntimeError(
            "SQL agent failed without producing a query."
        )

    result_text = _format_result_for_model(
        columns,
        rows,
        truncated,
    )

    answer_system = """
You are a data analyst completing an autonomous SQL investigation.

Answer the user's question using only the successful SQL result.

Rules:
- Be concise.
- State the important result directly.
- Do not invent facts.
- Mention if the query returned no rows.
- Mention if the result was truncated.
""".strip()

    answer_prompt = f"""
USER QUESTION

{question}

SUCCESSFUL SQL

{successful_sql}

QUERY RESULT

{result_text}
""".strip()

    answer_response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=answer_prompt
                    )
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=[
                types.Part.from_text(
                    text=answer_system
                )
            ],
        ),
    )

    return SQLAgentResult(
        question=question,
        sql=successful_sql,
        columns=columns,
        rows=rows,
        answer=(
            answer_response.text or ""
        ).strip(),
        attempts=attempts,
        truncated=truncated,
    )
