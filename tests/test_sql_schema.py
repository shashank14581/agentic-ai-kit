import sqlite3

from agentic_ai.sql_ai import format_schema, inspect_schema


def test_inspect_schema():
    conn = sqlite3.connect(":memory:")

    conn.execute(
        """
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            revenue REAL
        )
        """
    )

    schema = inspect_schema(conn)

    assert "sales" in schema

    columns = schema["sales"]

    assert [column["name"] for column in columns] == [
        "id",
        "category",
        "revenue",
    ]

    assert columns[0]["primary_key"] is True
    assert columns[1]["nullable"] is False


def test_format_schema():
    conn = sqlite3.connect(":memory:")

    conn.execute(
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT,
            state TEXT
        )
        """
    )

    output = format_schema(inspect_schema(conn))

    assert "TABLE customers" in output
    assert "customer_id: INTEGER" in output
    assert "name: TEXT" in output
    assert "state: TEXT" in output


def test_empty_schema():
    conn = sqlite3.connect(":memory:")

    assert inspect_schema(conn) == {}
    assert format_schema({}) == "(no tables found)"
