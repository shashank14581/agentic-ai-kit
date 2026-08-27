import sqlite3
from types import SimpleNamespace

from agentic_ai.cli.main import (
    _build_parser,
    _sql_ask_command,
    _sql_inspect_command,
)


def test_parser_supports_sql_inspect():
    parser = _build_parser()

    args = parser.parse_args(
        [
            "sql",
            "inspect",
            "example.db",
        ]
    )

    assert args.command == "sql"
    assert args.sql_command == "inspect"
    assert args.database == "example.db"


def test_parser_supports_sql_ask():
    parser = _build_parser()

    args = parser.parse_args(
        [
            "sql",
            "ask",
            "example.db",
            "Which category has the most revenue?",
            "--show-sql",
        ]
    )

    assert args.command == "sql"
    assert args.sql_command == "ask"
    assert args.database == "example.db"
    assert args.question == (
        "Which category has the most revenue?"
    )
    assert args.show_sql is True


def test_sql_inspect_command(tmp_path, capsys):
    database = tmp_path / "sales.db"

    conn = sqlite3.connect(database)

    conn.execute(
        """
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            revenue REAL
        )
        """
    )

    conn.commit()
    conn.close()

    args = SimpleNamespace(
        database=str(database)
    )

    result = _sql_inspect_command(args)

    captured = capsys.readouterr()

    assert result == 0
    assert "TABLE sales" in captured.out
    assert "category: TEXT" in captured.out
    assert "revenue: REAL" in captured.out


def test_sql_inspect_missing_database(
    tmp_path,
    capsys,
):
    missing = tmp_path / "missing.db"

    args = SimpleNamespace(
        database=str(missing)
    )

    result = _sql_inspect_command(args)

    captured = capsys.readouterr()

    assert result == 2
    assert "database not found" in captured.err


def test_sql_ask_command(
    tmp_path,
    capsys,
    monkeypatch,
):
    database = tmp_path / "sales.db"

    conn = sqlite3.connect(database)

    conn.execute(
        """
        CREATE TABLE sales (
            category TEXT,
            revenue REAL
        )
        """
    )

    conn.commit()
    conn.close()

    def fake_ask_sql(
        question,
        conn,
        model,
        max_rows,
    ):
        return SimpleNamespace(
            sql=(
                "SELECT category, SUM(revenue) "
                "FROM sales GROUP BY category"
            ),
            answer=(
                "Tools generated the most revenue."
            ),
        )

    monkeypatch.setattr(
        "agentic_ai.sql_ai.ask_sql",
        fake_ask_sql,
    )

    args = SimpleNamespace(
        database=str(database),
        question=(
            "Which category generated the most revenue?"
        ),
        model="gemini-2.5-flash",
        max_rows=100,
        show_sql=True,
    )

    result = _sql_ask_command(args)

    captured = capsys.readouterr()

    assert result == 0
    assert "SQL:" in captured.out
    assert "SELECT category" in captured.out
    assert (
        "Tools generated the most revenue."
        in captured.out
    )
