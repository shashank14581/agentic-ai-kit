from __future__ import annotations

import sqlite3
from typing import Any


def inspect_schema(conn: sqlite3.Connection) -> dict[str, Any]:
    """Inspect user-defined tables and columns in a SQLite database."""

    if conn is None:
        raise ValueError("Pass a valid SQLite connection.")

    cursor = conn.cursor()

    tables = cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()

    schema: dict[str, Any] = {}

    for (table_name,) in tables:
        escaped_table = table_name.replace('"', '""')

        columns = cursor.execute(
            f'PRAGMA table_info("{escaped_table}")'
        ).fetchall()

        schema[table_name] = [
            {
                "name": column[1],
                "type": column[2] or "",
                "nullable": not bool(column[3]),
                "default": column[4],
                "primary_key": bool(column[5]),
            }
            for column in columns
        ]

    return schema


def format_schema(schema: dict[str, Any]) -> str:
    """Format an inspected SQLite schema for humans or model prompts."""

    if not schema:
        return "(no tables found)"

    blocks: list[str] = []

    for table_name, columns in schema.items():
        blocks.append(f"TABLE {table_name}")

        for column in columns:
            type_name = column["type"] or "UNKNOWN"

            flags: list[str] = []

            if column["primary_key"]:
                flags.append("PRIMARY KEY")

            if not column["nullable"]:
                flags.append("NOT NULL")

            suffix = f" [{' '.join(flags)}]" if flags else ""

            blocks.append(
                f"  - {column['name']}: {type_name}{suffix}"
            )

        blocks.append("")

    return "\n".join(blocks).rstrip()
