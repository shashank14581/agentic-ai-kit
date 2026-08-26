from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from agentic_ai import BaseAgent, __version__
from agentic_ai.adapters import AntigravityAdapter
from agentic_ai.sql_ai import (
    ask_sql,
    format_schema,
    inspect_schema,
    run_sql_agent,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aak",
        description="Agentic AI Kit command-line interface.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"agentic-ai-kit {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ---------------------------------------------------------
    # aak run
    # ---------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run a single prompt through an AAK agent.",
    )

    run_parser.add_argument(
        "prompt",
        help="Prompt to send to the agent.",
    )

    run_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
    )

    run_parser.add_argument(
        "--transport",
        choices=["generate_content", "interactions"],
        default="generate_content",
    )

    run_parser.add_argument(
        "--adapter",
        choices=["native", "antigravity"],
        default="native",
        help="Execution runtime. Default: native.",
    )

    run_parser.add_argument(
        "--name",
        default="AAK",
    )

    run_parser.add_argument(
        "--system",
        default="You are a concise and useful AI assistant.",
    )

    # ---------------------------------------------------------
    # aak chat
    # ---------------------------------------------------------
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive AAK chat session.",
    )

    chat_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
    )

    chat_parser.add_argument(
        "--transport",
        choices=["generate_content", "interactions"],
        default="generate_content",
    )

    chat_parser.add_argument(
        "--name",
        default="AAK",
    )

    chat_parser.add_argument(
        "--system",
        default="You are a concise and useful AI assistant.",
    )

    # ---------------------------------------------------------
    # aak sql
    # ---------------------------------------------------------
    sql_parser = subparsers.add_parser(
        "sql",
        help="Inspect and interact with databases using AAK SQL AI.",
    )

    sql_subparsers = sql_parser.add_subparsers(
        dest="sql_command",
    )

    # ---------------------------------------------------------
    # aak sql inspect
    # ---------------------------------------------------------
    inspect_parser = sql_subparsers.add_parser(
        "inspect",
        help="Inspect the schema of a SQLite database.",
    )

    inspect_parser.add_argument(
        "database",
        help="Path to the SQLite database.",
    )

    # ---------------------------------------------------------
    # aak sql ask
    # ---------------------------------------------------------
    ask_parser = sql_subparsers.add_parser(
        "ask",
        help="Ask a natural-language question about a SQLite database.",
    )

    ask_parser.add_argument(
        "database",
        help="Path to the SQLite database.",
    )

    ask_parser.add_argument(
        "question",
        help="Natural-language question to answer.",
    )

    ask_parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model used for SQL planning and interpretation.",
    )

    ask_parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Maximum number of query rows supplied to the model.",
    )

    ask_parser.add_argument(
        "--show-sql",
        action="store_true",
        help="Print the generated SQL before the answer.",
    )

    # ---------------------------------------------------------
    # aak sql agent
    # ---------------------------------------------------------
    agent_parser = sql_subparsers.add_parser(
        "agent",
        help="Run an autonomous SQL planning and repair loop.",
    )

    agent_parser.add_argument(
        "database",
        help="Path to the SQLite database.",
    )

    agent_parser.add_argument(
        "question",
        help="Natural-language investigation or question.",
    )

    agent_parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model used by the SQL agent.",
    )

    agent_parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Maximum number of result rows supplied to the model.",
    )

    agent_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum SQL planning/repair attempts.",
    )

    agent_parser.add_argument(
        "--trace",
        action="store_true",
        help="Show SQL attempts, errors, and repairs.",
    )

    return parser


def _create_agent(args: argparse.Namespace) -> BaseAgent:
    return BaseAgent(
        name=args.name,
        sys_prompt=args.system,
        model=args.model,
        transport=args.transport,
        extract_memory=(args.command == "chat"),
    )


def _run_command(args: argparse.Namespace) -> int:
    agent = _create_agent(args)

    if args.adapter == "antigravity":
        runtime = AntigravityAdapter(agent)
        response = runtime.run(args.prompt)

    else:
        response = agent.think(
            args.prompt,
            stream=False,
        )

    print(response)
    return 0


def _chat_command(args: argparse.Namespace) -> int:
    agent = _create_agent(args)

    print(
        f"AAK {__version__} | "
        f"{args.model} | {args.transport}"
    )

    print("Type /exit to quit.\n")

    while True:
        try:
            user_input = input("you> ").strip()

        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue

        if user_input.lower() in {
            "/exit",
            "/quit",
            "exit",
            "quit",
        }:
            return 0

        try:
            response = agent.think(
                user_input,
                stream=False,
            )

            print(f"\naak> {response}\n")

        except Exception as exc:
            print(f"\nerror> {exc}\n")


def _resolve_database_path(database: str) -> Path | None:
    database_path = Path(database).expanduser()

    if not database_path.exists():
        print(
            f"error: database not found: {database_path}",
            file=sys.stderr,
        )
        return None

    if not database_path.is_file():
        print(
            f"error: not a file: {database_path}",
            file=sys.stderr,
        )
        return None

    return database_path


def _connect_read_only(
    database_path: Path,
) -> sqlite3.Connection:
    uri = (
        f"file:{database_path.resolve().as_posix()}"
        "?mode=ro"
    )

    return sqlite3.connect(
        uri,
        uri=True,
    )


def _sql_inspect_command(args: argparse.Namespace) -> int:
    database_path = _resolve_database_path(
        args.database
    )

    if database_path is None:
        return 2

    conn = _connect_read_only(database_path)

    try:
        schema = inspect_schema(conn)

    finally:
        conn.close()

    print(format_schema(schema))
    return 0


def _sql_ask_command(args: argparse.Namespace) -> int:
    database_path = _resolve_database_path(
        args.database
    )

    if database_path is None:
        return 2

    if args.max_rows <= 0:
        print(
            "error: --max-rows must be greater than 0",
            file=sys.stderr,
        )
        return 2

    conn = _connect_read_only(database_path)

    try:
        result = ask_sql(
            args.question,
            conn,
            model=args.model,
            max_rows=args.max_rows,
        )

    except Exception as exc:
        print(
            f"error: {exc}",
            file=sys.stderr,
        )
        return 1

    finally:
        conn.close()

    if args.show_sql:
        print("SQL:")
        print(result.sql)
        print()

    print(result.answer)

    return 0


def _sql_agent_command(args: argparse.Namespace) -> int:
    database_path = _resolve_database_path(
        args.database
    )

    if database_path is None:
        return 2

    if args.max_rows <= 0:
        print(
            "error: --max-rows must be greater than 0",
            file=sys.stderr,
        )
        return 2

    if args.max_attempts <= 0:
        print(
            "error: --max-attempts must be greater than 0",
            file=sys.stderr,
        )
        return 2

    conn = _connect_read_only(database_path)

    try:
        result = run_sql_agent(
            args.question,
            conn,
            model=args.model,
            max_rows=args.max_rows,
            max_attempts=args.max_attempts,
        )

    except Exception as exc:
        print(
            f"error: {exc}",
            file=sys.stderr,
        )
        return 1

    finally:
        conn.close()

    if args.trace:
        print("TRACE:")
        print()

        for attempt in result.attempts:
            status = (
                "SUCCESS"
                if attempt.error is None
                else "FAILED"
            )

            print(
                f"Attempt {attempt.attempt} [{status}]"
            )
            print(attempt.sql)

            if attempt.error:
                print(f"Error: {attempt.error}")

            print()

    print("SQL:")
    print(result.sql)
    print()

    print("ANSWER:")
    print(result.answer)

    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return _run_command(args)

    if args.command == "chat":
        return _chat_command(args)

    if args.command == "sql":
        if args.sql_command == "inspect":
            return _sql_inspect_command(args)

        if args.sql_command == "ask":
            return _sql_ask_command(args)

        if args.sql_command == "agent":
            return _sql_agent_command(args)

        print(
            "Use `aak sql --help` to see SQL commands."
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
