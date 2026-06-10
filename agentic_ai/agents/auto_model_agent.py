"""
AutoModelAgent: end-to-end dataframe-to-model agent.

The user gives:
- dataframe
- target column
- business objective

The agent:
- profiles the data
- designs modeling policy
- trains models using MLEAgent
- returns results and interpretation
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from agentic_ai.agents.mle_agent import MLEAgent


class AutoModelAgent(MLEAgent):
    """End-to-end modeling agent built on top of MLEAgent."""

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        objective: str,
        drop_columns: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Look at data, design policy, train model, and explain results."""

        self._validate_dataframe(df)
        self._validate_columns(df, [target_col])

        profile = self.profile_dataframe(df)
        target = self.target_summary(df, target_col)
        features = self.feature_summary(df, target_col)
        leakage = self.leakage_scan(df, target_col)

        policy_prompt = f"""
You are about to create a machine learning model.

BUSINESS OBJECTIVE:
{objective}

TARGET COLUMN:
{target_col}

DATA PROFILE:
{profile}

TARGET SUMMARY:
{target}

FEATURE SUMMARY:
{features}

LEAKAGE SCAN:
{leakage}

Design a concise modeling policy before training.

Include:
1. problem type
2. usable features
3. columns to drop, if any
4. leakage risks
5. train/test approach
6. model candidates
7. evaluation metric
8. expected limitations

Return a practical modeling policy.
""".strip()

        modeling_policy = self.think(policy_prompt, stream=stream)

        model_result = self.train_model(
            df=df,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state,
            drop_columns=drop_columns,
        )

        result_prompt = f"""
A model has now been trained.

BUSINESS OBJECTIVE:
{objective}

MODELING POLICY:
{modeling_policy}

MODEL RESULT:
{self._safe_model_result_for_llm(model_result)}

Explain:
1. what model was created
2. which model performed best
3. what the metrics mean
4. whether the result is usable
5. what should be improved next
6. how this could be deployed
""".strip()

        interpretation = self.think(result_prompt, stream=stream)

        return {
            "objective": objective,
            "target_column": target_col,
            "data_profile": profile,
            "target_summary": target,
            "feature_summary": features,
            "leakage_scan": leakage,
            "modeling_policy": modeling_policy,
            "model_result": model_result,
            "interpretation": interpretation,
            "best_model": model_result["best_model"],
            "best_score": model_result["best_score"],
            "best_pipeline": model_result["best_pipeline"],
        }