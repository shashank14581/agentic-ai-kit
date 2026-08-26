import sqlite3

import pytest

from agentic_ai.sql_ai import (
    ask_sql,
    execute_read_only_sql,
    validate_read_only_sql,
)


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1

        if self.calls == 1:
            return FakeResponse(
                """
                SELECT category, SUM(revenue) AS total_revenue
                FROM sales
                GROUP BY category
                ORDER BY total_revenue DESC
                """
            )

        return FakeResponse(
            "Tools generated the highest total revenue."
        )


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_validate_read_only_sql():
    sql = validate_read_only_sql(
        "SELECT category FROM sales"
    )

    assert sql == "SELECT category FROM sales"


def test_validate_read_only_sql_rejects_delete():
    with pytest.raises(ValueError):
        validate_read_only_sql(
            "DELETE FROM sales"
        )


def test_execute_read_only_sql():
    conn = sqlite3.connect(":memory:")

    conn.execute(
        """
        CREATE TABLE sales (
            category TEXT,
            revenue REAL
        )
        """
    )

    conn.executemany(
        "INSERT INTO sales VALUES (?, ?)",
        [
            ("Tools", 100.0),
            ("Paint", 75.0),
        ],
    )

    columns, rows, truncated = execute_read_only_sql(
        conn,
        "SELECT category, revenue FROM sales ORDER BY revenue DESC",
    )

    assert columns == ["category", "revenue"]
    assert rows[0] == ("Tools", 100.0)
    assert truncated is False


def test_ask_sql():
    conn = sqlite3.connect(":memory:")

    conn.execute(
        """
        CREATE TABLE sales (
            category TEXT,
            revenue REAL
        )
        """
    )

    conn.executemany(
        "INSERT INTO sales VALUES (?, ?)",
        [
            ("Tools", 120.0),
            ("Paint", 80.0),
            ("Tools", 150.0),
        ],
    )

    result = ask_sql(
        "Which category generated the most revenue?",
        conn,
        client=FakeClient(),
    )

    assert "SELECT category" in result.sql
    assert result.columns == [
        "category",
        "total_revenue",
    ]
    assert result.rows[0] == ("Tools", 270.0)
    assert result.answer == (
        "Tools generated the highest total revenue."
    )
