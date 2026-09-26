"""
LightGBM quantile forecaster with direct multi-step horizon support.

Trains three separate LightGBM regressors (alpha 0.1, 0.5, 0.9) using
the native ``quantile`` objective. Uses direct multi-step forecasting:
horizon step is added as a feature so each model predicts all 30 days
from current features without recursive error compounding.
"""

import os
from typing import Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


class QuantileForecaster:
    """Three-quantile LightGBM forecaster with direct multi-step support.

    Trains one LightGBM model per quantile (P10, P50, P90) using the
    native ``objective='quantile'`` parameter. Horizon step is injected
    as a feature column so a single model per quantile covers all 30
    forecast days.

    Attributes:
        horizon: Number of forecast steps (default 30).
        alphas: Quantile levels to train [0.1, 0.5, 0.9].
        models: Dict mapping alpha -> trained LGBMRegressor.
    """

    DEFAULT_ALPHAS: List[float] = [0.1, 0.5, 0.9]

    def __init__(
        self,
        horizon: int = 30,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"Horizon must be >= 1, got {horizon}")

        self.horizon = horizon
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.kwargs = kwargs
        self.alphas = list(self.DEFAULT_ALPHAS)
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self._feature_names: Optional[List[str]] = None

    @property
    def is_trained(self) -> bool:
        """Return True if all quantile models have been trained."""
        return len(self.models) == len(self.alphas)

    def _prepare_training_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple:
        """Expand feature matrix for direct multi-step forecasting.

        For each row *i* and horizon step *h* (1..horizon), creates a
        training example where features are ``X.iloc[i]`` plus a
        ``horizon_step`` column set to *h*, and the target is ``y.iloc[i + h]``.

        Args:
            X: Feature matrix (one row per day).
            y: Target series aligned with X.

        Returns:
            Tuple of (X_expanded, y_expanded).
        """
        X_parts: List[pd.DataFrame] = []
        y_parts: List[pd.Series] = []

        for h in range(1, self.horizon + 1):
            n_valid = len(X) - h
            if n_valid <= 0:
                continue
            X_h = X.iloc[:n_valid].copy()
            X_h = X_h.reset_index(drop=True)
            X_h["horizon_step"] = h
            y_h = y.iloc[h : h + n_valid].reset_index(drop=True)
            X_parts.append(X_h)
            y_parts.append(y_h)

        X_expanded = pd.concat(X_parts, ignore_index=True)
        y_expanded = pd.concat(y_parts, ignore_index=True)
        return X_expanded, y_expanded

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train all three quantile models.

        Args:
            X: Feature matrix (one row per day, no horizon_step column).
            y: Target series aligned with X.

        Raises:
            ValueError: If dataset is too small for the configured horizon.
        """
        if len(X) <= self.horizon:
            raise ValueError(
                f"Need at least {self.horizon + 1} rows to train with "
                f"horizon={self.horizon}, got {len(X)}"
            )

        X_exp, y_exp = self._prepare_training_data(X, y)
        self._feature_names = list(X_exp.columns)

        for alpha in self.alphas:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                verbose=-1,
                **self.kwargs,
            )
            model.fit(X_exp, y_exp)
            self.models[alpha] = model

    def predict(self, X_current: pd.DataFrame) -> pd.DataFrame:
        """Predict P10, P50, P90 for all horizon steps.

        Args:
            X_current: Feature row(s) representing the current state.
                Typically a single row (the latest observation). If
                multiple rows are provided, predictions are made for
                each row independently.

        Returns:
            DataFrame with columns ``horizon_step``, ``p10``, ``p50``,
            ``p90`` and one row per (input_row x horizon_step).

        Raises:
            RuntimeError: If models have not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Models have not been trained. Call train() first.")

        results: List[Dict] = []
        for idx in range(len(X_current)):
            row = X_current.iloc[[idx]]
            for h in range(1, self.horizon + 1):
                row_h = row.copy()
                row_h["horizon_step"] = h
                # Ensure column order matches training
                row_h = row_h[self._feature_names]
                p10 = float(self.models[0.1].predict(row_h)[0])
                p50 = float(self.models[0.5].predict(row_h)[0])
                p90 = float(self.models[0.9].predict(row_h)[0])
                results.append({
                    "horizon_step": h,
                    "p10": round(p10, 2),
                    "p50": round(p50, 2),
                    "p90": round(p90, 2),
                })
        return pd.DataFrame(results)

    def save(self, directory: str) -> None:
        """Save all trained models and metadata to a directory.

        Args:
            directory: Target directory (created if it does not exist).

        Raises:
            RuntimeError: If models have not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Models have not been trained. Call train() first.")

        os.makedirs(directory, exist_ok=True)
        for alpha, model in self.models.items():
            label = str(alpha).replace(".", "")
            filepath = os.path.join(directory, f"lgb_q{label}.pkl")
            joblib.dump(model, filepath)

        # Save metadata
        meta = {
            "horizon": self.horizon,
            "alphas": self.alphas,
            "feature_names": self._feature_names,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "random_state": self.random_state,
        }
        joblib.dump(meta, os.path.join(directory, "meta.pkl"))

    def load(self, directory: str) -> None:
        """Load trained models and metadata from a directory.

        Args:
            directory: Directory containing saved model files.

        Raises:
            FileNotFoundError: If the directory or required files are missing.
        """
        meta_path = os.path.join(directory, "meta.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        meta = joblib.load(meta_path)
        self.horizon = meta["horizon"]
        self.alphas = meta["alphas"]
        self._feature_names = meta["feature_names"]
        self.n_estimators = meta["n_estimators"]
        self.learning_rate = meta["learning_rate"]
        self.num_leaves = meta["num_leaves"]
        self.random_state = meta["random_state"]

        self.models = {}
        for alpha in self.alphas:
            label = str(alpha).replace(".", "")
            filepath = os.path.join(directory, f"lgb_q{label}.pkl")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            self.models[alpha] = joblib.load(filepath)
