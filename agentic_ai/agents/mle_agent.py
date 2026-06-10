"""
MLEAgent: dataframe-aware machine learning engineering specialist agent.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from agentic_ai.agents.base import BaseAgent


_MLE_PROMPT = """
You are an elite Machine Learning Engineer.

You help users design ML systems from pandas DataFrames.

You are expected to reason using:
- target definition
- feature columns
- label quality
- leakage risks
- train/test strategy
- baseline models
- evaluation metrics
- deployment constraints
- monitoring and drift

Preferred response structure:

PROBLEM DEFINITION
DATA READ
TARGET AND FEATURES
LEAKAGE RISKS
MODELING STRATEGY
EVALUATION PLAN
DEPLOYMENT CONSIDERATIONS
MONITORING PLAN
IMPLEMENTATION STEPS

Do not invent model performance.
If dataframe context is provided, use it explicitly.
""".strip()


class MLEAgent(BaseAgent):
    """ML-engineering specialist agent with dataframe tools."""

    def __init__(
        self,
        name: str = "MLE",
        project_context: str | None = None,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        memory_window: int = 3,
        max_turns: int | None = None,
        thinking_budget: int = 0,
    ):
        prompt = _MLE_PROMPT

        if project_context:
            prompt += f"\n\nPROJECT CONTEXT:\n{project_context.strip()}"

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

    def target_summary(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> dict[str, Any]:
        """Summarize the target variable."""

        self._validate_dataframe(df)
        self._validate_columns(df, [target_col])

        target = df[target_col]

        summary = {
            "target_column": target_col,
            "dtype": str(target.dtype),
            "missing_values": int(target.isna().sum()),
            "missing_percentage": round(float(target.isna().mean() * 100), 2),
            "unique_values": int(target.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(target):
            summary["type_guess"] = "regression_or_numeric_classification"
            summary["describe"] = target.describe().round(4).to_dict()
        else:
            summary["type_guess"] = "classification"
            summary["class_distribution"] = (
                target.value_counts(dropna=False).to_dict()
            )
            summary["class_percentage"] = (
                target.value_counts(normalize=True, dropna=False)
                .mul(100)
                .round(2)
                .to_dict()
            )

        return summary

    def feature_summary(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
    ) -> dict[str, Any]:
        """Summarize candidate feature columns."""

        self._validate_dataframe(df)

        feature_df = df.drop(columns=[target_col]) if target_col else df

        return {
            "feature_count": int(feature_df.shape[1]),
            "numeric_features": list(feature_df.select_dtypes(include="number").columns),
            "categorical_features": list(
                feature_df.select_dtypes(include=["object", "category", "bool"]).columns
            ),
            "datetime_features": list(
                feature_df.select_dtypes(include=["datetime", "datetimetz"]).columns
            ),
            "high_missing_features": self.high_missing_features(feature_df),
            "constant_features": self.constant_features(feature_df),
        }

    def high_missing_features(
        self,
        df: pd.DataFrame,
        threshold: float = 50.0,
    ) -> dict[str, float]:
        """Return columns with missing percentage above threshold."""

        self._validate_dataframe(df)

        missing_pct = df.isna().mean().mul(100)

        return (
            missing_pct[missing_pct >= threshold]
            .round(2)
            .to_dict()
        )

    def constant_features(self, df: pd.DataFrame) -> list[str]:
        """Return columns with one or zero unique non-null values."""

        self._validate_dataframe(df)

        return [
            col for col in df.columns
            if df[col].nunique(dropna=True) <= 1
        ]

    def leakage_scan(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> dict[str, Any]:
        """Heuristic scan for possible leakage-prone columns."""

        self._validate_dataframe(df)
        self._validate_columns(df, [target_col])

        leakage_terms = [
            "target",
            "label",
            "outcome",
            "conversion",
            "converted",
            "purchase",
            "purchased",
            "revenue",
            "sales",
            "post",
            "after",
            "future",
            "next",
        ]

        suspicious = []

        for col in df.columns:
            if col == target_col:
                continue

            col_lower = col.lower()

            if any(term in col_lower for term in leakage_terms):
                suspicious.append(col)

        return {
            "target_column": target_col,
            "suspicious_columns": suspicious,
            "note": "This is a heuristic scan. Confirm leakage using business timing.",
        }

    def ml_dataframe_context(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> str:
        """Create ML-ready dataframe context for the LLM."""

        context = {
            "profile": self.profile_dataframe(df),
            "target_summary": self.target_summary(df, target_col),
            "feature_summary": self.feature_summary(df, target_col),
            "leakage_scan": self.leakage_scan(df, target_col),
        }

        return str(context)

    def design_modeling_plan(
        self,
        df: pd.DataFrame,
        target_col: str,
        objective: str,
        stream: bool = True,
    ) -> str:
        """Design an ML plan using dataframe context."""

        context = self.ml_dataframe_context(df, target_col)

        prompt = f"""
Design a machine learning plan.

OBJECTIVE:
{objective}

TARGET COLUMN:
{target_col}

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