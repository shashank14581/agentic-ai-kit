"""
MLEAgent: dataframe-aware machine learning engineering specialist agent.

This agent can:
- profile a dataframe
- summarize target and features
- scan for leakage-prone columns
- train baseline ML models
- evaluate models
- return the best fitted pipeline
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agentic_ai.agents.base import BaseAgent


_MLE_PROMPT = """
You are an elite Machine Learning Engineer.

You help users build, evaluate, and interpret ML models from pandas DataFrames.

You reason using:
- target definition
- feature columns
- label quality
- leakage risks
- train/test strategy
- baseline models
- evaluation metrics
- model comparison
- deployment constraints
- monitoring and drift

Preferred response structure:

PROBLEM DEFINITION
DATA READ
TARGET AND FEATURES
LEAKAGE RISKS
MODEL RESULTS
BEST MODEL
INTERPRETATION
DEPLOYMENT CONSIDERATIONS
MONITORING PLAN
NEXT STEPS

Do not invent model performance.
If model results are provided, use them explicitly.
""".strip()


class MLEAgent(BaseAgent):
    """ML-engineering specialist agent with dataframe and modeling tools."""

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
            "missing_percentage": df.isna().mean().mul(100).round(2).to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": list(df.select_dtypes(include="number").columns),
            "categorical_columns": list(
                df.select_dtypes(include=["object", "category", "bool"]).columns
            ),
            "datetime_columns": list(
                df.select_dtypes(include=["datetime", "datetimetz"]).columns
            ),
        }

    def target_summary(self, df: pd.DataFrame, target_col: str) -> dict[str, Any]:
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

        if self._is_classification_target(target):
            summary["type_guess"] = "classification"
            summary["class_distribution"] = target.value_counts(dropna=False).to_dict()
            summary["class_percentage"] = (
                target.value_counts(normalize=True, dropna=False)
                .mul(100)
                .round(2)
                .to_dict()
            )
        else:
            summary["type_guess"] = "regression"
            summary["describe"] = target.describe().round(4).to_dict()

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

        return missing_pct[missing_pct >= threshold].round(2).to_dict()

    def constant_features(self, df: pd.DataFrame) -> list[str]:
        """Return columns with one or zero unique non-null values."""

        self._validate_dataframe(df)

        return [col for col in df.columns if df[col].nunique(dropna=True) <= 1]

    def leakage_scan(self, df: pd.DataFrame, target_col: str) -> dict[str, Any]:
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

    # ------------------------------------------------------------------
    # Modeling tools
    # ------------------------------------------------------------------

    def train_model(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        drop_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Train baseline ML models and return the best fitted pipeline.

        Returns:
            A dictionary containing problem type, model comparison,
            best model name, best pipeline, and test predictions.
        """

        self._validate_dataframe(df)
        self._validate_columns(df, [target_col])

        data = df.copy()

        if drop_columns:
            self._validate_columns(data, drop_columns)
            data = data.drop(columns=drop_columns)

        data = data.dropna(subset=[target_col])

        y = data[target_col]
        X = data.drop(columns=[target_col])

        X = self._drop_unusable_features(X)

        if X.empty:
            raise ValueError("No usable feature columns found after preprocessing.")

        problem_type = self.detect_problem_type(y)

        stratify = y if problem_type == "classification" and y.nunique() > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        preprocessor = self._build_preprocessor(X_train)

        models = self._get_baseline_models(problem_type, random_state)

        results = []

        best_model_name = None
        best_pipeline = None
        best_score = None
        best_predictions = None
        best_probabilities = None

        for model_name, estimator in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            y_proba = None
            if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                except Exception:
                    y_proba = None

            metrics = self._evaluate_model(
                problem_type=problem_type,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
            )

            score = self._selection_score(problem_type, metrics)

            results.append(
                {
                    "model_name": model_name,
                    "metrics": metrics,
                    "selection_score": score,
                }
            )

            if best_score is None or score > best_score:
                best_score = score
                best_model_name = model_name
                best_pipeline = pipeline
                best_predictions = y_pred
                best_probabilities = y_proba

        return {
            "problem_type": problem_type,
            "target_column": target_col,
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "features_used": list(X.columns),
            "model_comparison": results,
            "best_model": best_model_name,
            "best_score": best_score,
            "best_pipeline": best_pipeline,
            "test_predictions": best_predictions,
            "test_probabilities": best_probabilities,
        }

    def create_model(
        self,
        df: pd.DataFrame,
        target_col: str,
        objective: str | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        drop_columns: list[str] | None = None,
        stream: bool = True,
        interpret: bool = True,
    ) -> dict[str, Any]:
        """Train models, evaluate them, and generate an MLE interpretation."""

        model_result = self.train_model(
            df=df,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state,
            drop_columns=drop_columns,
        )
        if not interpret:
            model_result["llm_interpretation"] = None
            return model_result

        interpretation_prompt = f"""
A model has been trained.

OBJECTIVE:
{objective or "Not provided"}

DATAFRAME PROFILE:
{self.profile_dataframe(df)}

TARGET SUMMARY:
{self.target_summary(df, target_col)}

FEATURE SUMMARY:
{self.feature_summary(df, target_col)}

LEAKAGE SCAN:
{self.leakage_scan(df, target_col)}

MODEL RESULT:
{self._safe_model_result_for_llm(model_result)}

Explain the model result and recommend next steps.
""".strip()

        model_result["llm_interpretation"] = self.think(
            interpretation_prompt,
            stream=stream,
        )

        return model_result

    def detect_problem_type(self, y: pd.Series) -> str:
        """Detect whether the task is classification or regression."""

        if self._is_classification_target(y):
            return "classification"

        return "regression"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build a preprocessing pipeline for numeric and categorical features."""

        numeric_features = list(X.select_dtypes(include="number").columns)

        categorical_features = list(
            X.select_dtypes(include=["object", "category", "bool"]).columns
        )

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )

    def _get_baseline_models(
        self,
        problem_type: str,
        random_state: int,
    ) -> dict[str, Any]:
        """Return baseline models for classification or regression."""

        if problem_type == "classification":
            return {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "random_forest_classifier": RandomForestClassifier(
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            }

        return {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            ),
        }

    def _evaluate_model(
        self,
        problem_type: str,
        y_true: pd.Series,
        y_pred: Any,
        y_proba: Any = None,
    ) -> dict[str, float | None]:
        """Evaluate classification or regression models."""

        if problem_type == "classification":
            metrics = {
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "roc_auc": None,
            }

            if y_proba is not None and y_true.nunique() == 2:
                metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)

            return metrics

        mse = mean_squared_error(y_true, y_pred)

        return {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(mse ** 0.5), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }

    def _selection_score(
        self,
        problem_type: str,
        metrics: dict[str, float | None],
    ) -> float:
        """Return model-selection score where higher is better."""

        if problem_type == "classification":
            if metrics.get("roc_auc") is not None:
                return float(metrics["roc_auc"])
            return float(metrics["f1"] or 0.0)

        return float(metrics["r2"])

    def _safe_model_result_for_llm(self, result: dict[str, Any]) -> dict[str, Any]:
        """Remove non-serializable fitted model objects before LLM interpretation."""

        return {
            key: value
            for key, value in result.items()
            if key
            not in {
                "best_pipeline",
                "test_predictions",
                "test_probabilities",
            }
        }

    def _drop_unusable_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop datetime and constant columns before modeling."""

        X = X.copy()

        datetime_cols = list(
            X.select_dtypes(include=["datetime", "datetimetz"]).columns
        )

        constant_cols = self.constant_features(X)

        drop_cols = sorted(set(datetime_cols + constant_cols))

        if drop_cols:
            X = X.drop(columns=drop_cols)

        return X

    def _is_classification_target(self, y: pd.Series) -> bool:
        """Heuristic classification target detection."""

        unique_values = y.nunique(dropna=True)

        if y.dtype == "object" or str(y.dtype) == "category" or y.dtype == "bool":
            return True

        if unique_values <= 20 and pd.api.types.is_integer_dtype(y):
            return True

        return False

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

