"""
AnalystAgent: dataframe-aware analytics specialist agent.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from agentic_ai.agents.base import BaseAgent


_ANALYST_PROMPT = """
You are an elite Decision Scientist.

You help users analyze pandas DataFrames and business datasets.

You are expected to reason using:
- dataframe schema
- row counts
- missing values
- numeric summaries
- categorical distributions
- group-level metrics
- data quality risks
- business interpretation

Always separate:
1. what the data says
2. what it might mean
3. what should be checked next

Preferred response structure:

BUSINESS QUESTION
DATA READ
KEY OBSERVATIONS
DATA QUALITY RISKS
RECOMMENDED ANALYSIS
DECISION SUMMARY

Do not invent numbers.
If dataframe context is provided, use it explicitly.
""".strip()


class AnalystAgent(BaseAgent):
    """Analytics-focused specialist agent with dataframe tools."""

    def __init__(
        self,
        name: str = "Analyst",
        domain_context: str | None = None,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        memory_window: int = 3,
        max_turns: int | None = None,
        thinking_budget: int = 0,
    ):
        prompt = _ANALYST_PROMPT

        if domain_context:
            prompt += f"\n\nDOMAIN CONTEXT:\n{domain_context.strip()}"

        super().__init__(
            name=name,
            sys_prompt=prompt,
            model=model,
            api_key=api_key,
            memory_window=memory_window,
            max_turns=max_turns,
            thinking_budget=thinking_budget,
        )

    # ------------------------------------------------------------------
    # DataFrame tools
    # ------------------------------------------------------------------

    def profile_dataframe(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return a consistent dataframe profile."""

        self._validate_dataframe(df)

        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isna().sum().to_dict(),
            "missing_percentage": (
                df.isna().mean().mul(100).round(2).to_dict()
            ),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": list(df.select_dtypes(include="number").columns),
            "categorical_columns": list(
                df.select_dtypes(include=["object", "category", "bool"]).columns
            ),
            "datetime_columns": list(
                df.select_dtypes(include=["datetime", "datetimetz"]).columns
            ),
        }

    def numeric_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return numeric column summary."""

        self._validate_dataframe(df)

        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            return {"message": "No numeric columns found."}

        return numeric_df.describe().round(4).to_dict()

    def categorical_summary(
        self,
        df: pd.DataFrame,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Return top category counts for categorical columns."""

        self._validate_dataframe(df)

        categorical_df = df.select_dtypes(include=["object", "category", "bool"])

        if categorical_df.empty:
            return {"message": "No categorical columns found."}

        summary = {}

        for col in categorical_df.columns:
            summary[col] = (
                categorical_df[col]
                .value_counts(dropna=False)
                .head(top_n)
                .to_dict()
            )

        return summary

    def groupby_summary(
        self,
        df: pd.DataFrame,
        group_col: str,
        metric_col: str,
        agg: str = "mean",
    ) -> dict[str, Any]:
        """Return grouped metric summary."""

        self._validate_dataframe(df)
        self._validate_columns(df, [group_col, metric_col])

        allowed_aggs = {"mean", "sum", "count", "median", "min", "max"}

        if agg not in allowed_aggs:
            raise ValueError(f"agg must be one of {sorted(allowed_aggs)}")

        result = (
            df.groupby(group_col, dropna=False)[metric_col]
            .agg(agg)
            .reset_index()
            .sort_values(metric_col, ascending=False)
        )

        return result.to_dict(orient="records")

    def correlation_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return numeric correlation matrix."""

        self._validate_dataframe(df)

        numeric_df = df.select_dtypes(include="number")

        if numeric_df.shape[1] < 2:
            return {"message": "At least two numeric columns are needed."}

        return numeric_df.corr().round(4).to_dict()

    def dataframe_context(
        self,
        df: pd.DataFrame,
        include_numeric: bool = True,
        include_categorical: bool = True,
    ) -> str:
        """Create dataframe context for the LLM."""

        context = {
            "profile": self.profile_dataframe(df),
        }

        if include_numeric:
            context["numeric_summary"] = self.numeric_summary(df)

        if include_categorical:
            context["categorical_summary"] = self.categorical_summary(df)

        return str(context)

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        question: str,
        stream: bool = True,
    ) -> str:
        """Analyze a dataframe using generated dataframe context."""

        context = self.dataframe_context(df)

        prompt = f"""
Analyze the dataframe for the following question.

QUESTION:
{question}

DATAFRAME CONTEXT:
{context}
""".strip()

        return self.think(prompt, stream=stream)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        if df.empty:
            raise ValueError("DataFrame is empty.")

    @staticmethod
    def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [col for col in columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing}")