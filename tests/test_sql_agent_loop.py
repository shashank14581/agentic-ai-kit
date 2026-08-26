import sqlite3

import pytest

from agentic_ai.sql_ai import run_sql_agent


class FakeResponse:
    def __init__(self, text):
        self.text = text


class RepairModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1

        if self.calls == 1:
            return FakeResponse(
                "SELECT missing_column FROM sales"
            )

        if self.calls == 2:
            return FakeResponse(
                """
                SELECT category,
                       SUM(revenue) AS total_revenue
                FROM sales
                GROUP BY category
                ORDER BY total_revenue DESC
                LIMIT 1
                """
            )

        return FakeResponse(
            "Tools generated the most revenue."
        )


class RepairClient:
    def __init__(self):
        self.models = RepairModels()


class UnsafeThenRepairModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1

        if self.calls == 1:
            return FakeResponse(
                "DELETE FROM sales"
            )

        if self.calls == 2:
            return FakeResponse(
                "SELECT COUNT(*) AS row_count FROM sales"
            )

        return FakeResponse(
            "The table contains 2 rows."
        )


class UnsafeThenRepairClient:
    def __init__(self):
        self.models = UnsafeThenRepairModels()


class AlwaysBrokenModels:
    def generate_content(self, **kwargs):
        return FakeResponse(
            "SELECT nonexistent FROM sales"
        )


class AlwaysBrokenClient:
    def __init__(self):
        self.models = AlwaysBrokenModels()


def make_sales_db():
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

    return conn


def test_sql_agent_repairs_failed_query():
    conn = make_sales_db()

    result = run_sql_agent(
        "Which category generated the most revenue?",
        conn,
        client=RepairClient(),
    )

    assert len(result.attempts) == 2

    assert (
        result.attempts[0].error
        is not None
    )

    assert (
        result.attempts[1].error
        is None
    )

    assert result.rows[0] == (
        "Tools",
        270.0,
    )

    assert result.answer == (
        "Tools generated the most revenue."
    )


def test_sql_agent_blocks_unsafe_sql_then_repairs():
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

    result = run_sql_agent(
        "How many sales rows exist?",
        conn,
        client=UnsafeThenRepairClient(),
    )

    assert len(result.attempts) == 2
    assert result.attempts[0].error is not None

    count = conn.execute(
        "SELECT COUNT(*) FROM sales"
    ).fetchone()[0]

    assert count == 2

    assert result.rows[0][0] == 2


def test_sql_agent_stops_after_max_attempts():
    conn = make_sales_db()

    with pytest.raises(
        RuntimeError,
        match="exhausted all repair attempts",
    ):
        run_sql_agent(
            "Do something impossible.",
            conn,
            client=AlwaysBrokenClient(),
            max_attempts=2,
        )
