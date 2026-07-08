from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
import sqlglot
from google import genai
from google.genai import types


INTERNAL_ROW_ID = "__ai_sql_row_id"

AI_SQL_FUNCTION_NAMES = {
    "ai_generate",
    "ai_summarize",
    "ai_classify",
    "ai_extract",
}


# ------------------------------------------------------------
# SQL parsing helpers
# ------------------------------------------------------------
def _clean_sql(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def _strip_identifier_quotes(name: str) -> str:
    name = name.strip()

    if (
        (name.startswith("`") and name.endswith("`"))
        or (name.startswith('"') and name.endswith('"'))
        or (name.startswith("[") and name.endswith("]"))
    ):
        return name[1:-1]

    return name


def _extract_create_or_replace(sql: str) -> tuple[str | None, str]:
    """
    Supports:

        CREATE OR REPLACE TABLE output_table AS SELECT ...

    SQLite does not support CREATE OR REPLACE TABLE natively.
    This helper extracts the target table name and returns the SELECT SQL.

    Returns:
        (output_table, select_sql)

    If no CREATE OR REPLACE TABLE is present:
        (None, original_sql)
    """

    sql = _clean_sql(sql)

    pattern = re.compile(
        r"""
        ^\s*
        CREATE\s+OR\s+REPLACE\s+TABLE\s+
        (?P<table>`[^`]+`|"[^"]+"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)
        \s+AS\s+
        (?P<select>SELECT\s+.*)
        \s*$
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )

    match = pattern.match(sql)

    if not match:
        return None, sql

    output_table = _strip_identifier_quotes(match.group("table"))
    select_sql = match.group("select").strip()

    return output_table, select_sql


def _extract_string_kwarg(expr_raw: str, name: str) -> str | None:
    """
    Extracts string keyword arguments from a pseudo-function expression.

    Examples:
        prompt='Summarize this row'
        model="gemini-2.5-flash"
        labels='A | B | C'
    """

    pattern = re.compile(
        rf"""{name}\s*=\s*(['"])(.*?)\1""",
        re.IGNORECASE | re.DOTALL,
    )

    match = pattern.search(expr_raw)

    if not match:
        return None

    return match.group(2)


def _extract_number_kwarg(expr_raw: str, name: str) -> int | None:
    """
    Extracts numeric keyword arguments.

    Examples:
        count=5
        max_items=10
    """

    pattern = re.compile(
        rf"""{name}\s*=\s*(\d+)""",
        re.IGNORECASE,
    )

    match = pattern.search(expr_raw)

    if not match:
        return None

    return int(match.group(1))


def _detect_ai_sql_function(expr_raw: str) -> str | None:
    """
    Detect supported SQL AI pseudo-functions.

    Handles:
        ai_generate(...)
        AI_GENERATE(...)
        AI_GENERATE (...)
    """

    expr_lower = expr_raw.lower()

    for function_name in AI_SQL_FUNCTION_NAMES:
        pattern = rf"\b{function_name}\s*\("

        if re.search(pattern, expr_lower):
            return function_name

    return None


def _build_ai_prompt(function_name: str, expr_raw: str) -> str:
    """
    Convert a SQL AI pseudo-function call into the actual Gemini prompt.
    """

    if function_name == "ai_generate":
        prompt = _extract_string_kwarg(expr_raw, "prompt")

        if not prompt:
            raise ValueError(
                f"ai_generate requires prompt='...'. Problem expression: {expr_raw}"
            )

        return prompt

    if function_name == "ai_summarize":
        prompt = _extract_string_kwarg(expr_raw, "prompt")

        if prompt:
            return prompt

        return "Summarize this row clearly and concisely."

    if function_name == "ai_classify":
        prompt = _extract_string_kwarg(expr_raw, "prompt")
        labels = (
            _extract_string_kwarg(expr_raw, "labels")
            or _extract_string_kwarg(expr_raw, "classes")
            or _extract_string_kwarg(expr_raw, "categories")
        )

        if not prompt:
            prompt = "Classify this row into the most appropriate category."

        if labels:
            return f"""
{prompt}

Allowed labels:
{labels}

Return only one label from the allowed labels.
""".strip()

        return f"""
{prompt}

Return only the final class label. Do not explain.
""".strip()

    if function_name == "ai_extract":
        prompt = _extract_string_kwarg(expr_raw, "prompt")
        count = (
            _extract_number_kwarg(expr_raw, "count")
            or _extract_number_kwarg(expr_raw, "max_items")
            or 5
        )

        if prompt:
            return f"""
{prompt}

Return at most {count} compact items.
Return the result as a comma-separated list.
""".strip()

        return f"""
Extract the most important retrieval signals from this row.

Include keywords, entities, intent, category interest, and useful facts.

Return at most {count} compact items.
Return the result as a comma-separated list.
""".strip()

    raise ValueError(f"Unsupported SQL AI function: {function_name}")


def parse_ai_sql(sql: str, default_model: str = "gemini-2.5-flash") -> dict[str, Any]:
    """
    Parse local SQL AI pseudo-functions from a SQLite SELECT query.

    Supported pseudo-functions:

        ai_generate(
            prompt='Write a short summary',
            model='gemini-2.5-flash'
        ) AS generated_text

        ai_summarize(
            model='gemini-2.5-flash'
        ) AS summary

        ai_classify(
            labels='A | B | C',
            model='gemini-2.5-flash'
        ) AS label

        ai_extract(
            prompt='Extract retrieval keywords',
            count=5,
            model='gemini-2.5-flash'
        ) AS keywords

    Also supports:

        CREATE OR REPLACE TABLE output_table AS SELECT ...

    Returns:
        A dictionary with:
            - output_table
            - base_sql
            - ai_functions
    """

    output_table, select_sql = _extract_create_or_replace(sql)

    ast = sqlglot.parse_one(select_sql, read="sqlite")

    if ast.key.upper() != "SELECT":
        raise ValueError("SQL AI currently supports SELECT queries only.")

    clean_expressions = []
    ai_functions = []

    for expr in ast.expressions:
        expr_raw = expr.sql(dialect="sqlite")
        function_name = _detect_ai_sql_function(expr_raw)

        if function_name:
            alias = expr.alias_or_name

            if not alias:
                raise ValueError(
                    f"SQL AI expression must have an alias using AS. "
                    f"Problem expression: {expr_raw}"
                )

            model = _extract_string_kwarg(expr_raw, "model") or default_model
            prompt = _build_ai_prompt(function_name, expr_raw)

            ai_functions.append(
                {
                    "function_name": function_name,
                    "alias": alias,
                    "model": model,
                    "prompt": prompt,
                    "raw_expression": expr_raw,
                }
            )

        else:
            clean_expressions.append(expr)

    # Add stable row id only when AI enrichment is needed.
    if ai_functions:
        row_id_expr = sqlglot.parse_one(
            f"SELECT ROW_NUMBER() OVER () AS {INTERNAL_ROW_ID}",
            read="sqlite",
        ).expressions[0]

        clean_expressions.append(row_id_expr)

    ast.set("expressions", clean_expressions)

    base_sql = ast.sql(dialect="sqlite")

    return {
        "output_table": output_table,
        "base_sql": base_sql,
        "ai_functions": ai_functions,
    }


# ------------------------------------------------------------
# Batch helpers
# ------------------------------------------------------------
def _safe_cell(value: Any) -> str:
    if value is None:
        return "NA"

    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass

    return (
        str(value)
        .replace("\t", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def _split_batches(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size].copy()


def _batch_to_tsv(df_batch: pd.DataFrame) -> str:
    visible_cols = [col for col in df_batch.columns if col != INTERNAL_ROW_ID]

    header = "\t".join([INTERNAL_ROW_ID] + visible_cols)
    lines = [header]

    for _, row in df_batch.iterrows():
        row_id = _safe_cell(row[INTERNAL_ROW_ID])
        values = [_safe_cell(row[col]) for col in visible_cols]
        lines.append("\t".join([row_id] + values))

    return "\n".join(lines)


def _parse_ai_response_lines(text: str, alias: str) -> list[dict[str, str]]:
    """
    Expected model output:

        1: some text
        2: some text
        3: some text
    """

    rows = []

    for line in text.strip().splitlines():
        line = line.strip()

        if not line:
            continue

        match = re.match(r"^\s*(\d+)\s*:\s*(.*)\s*$", line)

        if not match:
            continue

        row_id = match.group(1).strip()
        content = match.group(2).strip()

        rows.append(
            {
                INTERNAL_ROW_ID: row_id,
                alias: content,
            }
        )

    return rows


def _run_ai_batch(
    df_batch: pd.DataFrame,
    func: dict[str, Any],
    api_key: str,
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_output_tokens: int = 8192,
    retries: int = 2,
) -> list[dict[str, str]]:
    alias = func["alias"]
    prompt = func["prompt"]
    model = func["model"]

    rows_tsv = _batch_to_tsv(df_batch)

    system_instruction = f"""
You are a local SQL AI enrichment function.

You will receive rows in TSV format.
Each row has an internal id column named {INTERNAL_ROW_ID}.

Task:
{prompt}

Return exactly one line per input row.

Required output format:
<{INTERNAL_ROW_ID}>: <result>

Rules:
- Do not add commentary.
- Do not use markdown.
- Do not skip rows.
- Do not change the row id.
- Keep each result on one line.
""".strip()

    user_prompt = f"""
Rows:
{rows_tsv}
""".strip()

    last_error = None

    for attempt in range(retries + 1):
        try:
            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_prompt)],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    response_modalities=["TEXT"],
                    system_instruction=[
                        types.Part.from_text(text=system_instruction)
                    ],
                ),
            )

            text = response.text or ""
            return _parse_ai_response_lines(text, alias)

        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt)

    raise RuntimeError(f"AI batch failed for alias `{alias}`: {last_error}")


# ------------------------------------------------------------
# Main runner
# ------------------------------------------------------------
def run_ai_sql(
    sql: str,
    conn: Any,
    api_key: str | None = None,
    batch_size: int = 15,
    workers: int = 4,
    default_model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_output_tokens: int = 8192,
    write_if_create: bool = True,
) -> pd.DataFrame:
    """
    Run local SQL AI against a SQLite connection.

    Args:
        sql:
            SQLite SELECT query, optionally with SQL AI pseudo-functions.

        conn:
            SQLite connection object.

        api_key:
            Gemini API key. If omitted, reads GEMINI_API_KEY.

        batch_size:
            Number of rows sent to Gemini per request.

        workers:
            Number of parallel Gemini requests.

        default_model:
            Gemini model used when a pseudo-function does not specify model='...'.

        temperature:
            Gemini generation temperature.

        top_p:
            Gemini top-p setting.

        max_output_tokens:
            Max output tokens for each Gemini batch call.

        write_if_create:
            If SQL uses CREATE OR REPLACE TABLE, write final dataframe back
            to SQLite using pandas.to_sql(..., if_exists='replace').

    Returns:
        pandas DataFrame with generated AI columns.
    """

    if conn is None:
        raise ValueError("Pass a valid SQLite connection as conn=.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if workers <= 0:
        raise ValueError("workers must be greater than 0.")

    api_key = api_key or os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Pass api_key= or set GEMINI_API_KEY.")

    parsed = parse_ai_sql(sql, default_model=default_model)

    base_sql = parsed["base_sql"]
    output_table = parsed["output_table"]
    ai_functions = parsed["ai_functions"]

    base_df = pd.read_sql_query(base_sql, conn)

    if not ai_functions:
        final_df = base_df

        if output_table and write_if_create:
            final_df.to_sql(output_table, conn, if_exists="replace", index=False)

        return final_df

    if INTERNAL_ROW_ID not in base_df.columns:
        raise RuntimeError(f"Internal row id column `{INTERNAL_ROW_ID}` was not created.")

    base_df[INTERNAL_ROW_ID] = base_df[INTERNAL_ROW_ID].astype(str)

    result_df = base_df.copy()
    ai_input_df = base_df.copy()

    batches = list(_split_batches(ai_input_df, batch_size=batch_size))

    for func in ai_functions:
        alias = func["alias"]
        all_rows = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_ai_batch,
                    df_batch=batch,
                    func=func,
                    api_key=api_key,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                )
                for batch in batches
            ]

            for future in as_completed(futures):
                all_rows.extend(future.result())

        func_df = pd.DataFrame(all_rows)

        if func_df.empty:
            func_df = pd.DataFrame(columns=[INTERNAL_ROW_ID, alias])

        func_df[INTERNAL_ROW_ID] = func_df[INTERNAL_ROW_ID].astype(str)

        result_df = result_df.merge(func_df, on=INTERNAL_ROW_ID, how="left")

    final_df = result_df.drop(columns=[INTERNAL_ROW_ID])

    if output_table and write_if_create:
        final_df.to_sql(output_table, conn, if_exists="replace", index=False)

    return final_df


__all__ = [
    "parse_ai_sql",
    "run_ai_sql",
]