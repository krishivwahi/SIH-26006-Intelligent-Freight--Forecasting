"""
AutoARIMA econometric baseline model for dry bulk freight forecasting.

Grounded in Paper 1 (econometric time-series analysis of the Baltic Dry Index).
Provides:
1. Classical statistical baseline benchmark against which ML is evaluated.
2. In-sample and out-of-sample point forecasts feeding into LightGBM as an
   ensemble feature, allowing tree models to learn statistical-ML interactions.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pmdarima as pm


class AutoARIMABaseline:
    """Wrapper around pmdarima.auto_arima for statistical time-series forecasting."""

    def __init__(
        self,
        max_p: int = 3,
        max_q: int = 3,
        max_d: int = 2,
        seasonal: bool = False,
        stepwise: bool = True,
        random_state: int = 42,
    ) -> None:
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.seasonal = seasonal
        self.stepwise = stepwise
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.order: Optional[Tuple[int, int, int]] = None
        self.last_train_length: int = 0

    def fit(self, y: pd.Series | np.ndarray) -> AutoARIMABaseline:
        """Fit the AutoARIMA model on a univariate time series.

        Args:
            y: Freight rate target series (1D array or Series).

        Returns:
            self
        """
        if isinstance(y, pd.Series):
            y_arr = y.dropna().values.astype(float)
        else:
            y_arr = np.asarray(y, dtype=float)
            y_arr = y_arr[~np.isnan(y_arr)]

        if len(y_arr) < 10:
            raise ValueError(f"Time series too short for AutoARIMA fitting (need >= 10, got {len(y_arr)})")

        self.last_train_length = len(y_arr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = pm.auto_arima(
                y_arr,
                start_p=1,
                start_q=1,
                max_p=self.max_p,
                max_q=self.max_q,
                max_d=self.max_d,
                seasonal=self.seasonal,
                stepwise=self.stepwise,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
                random_state=self.random_state,
            )

        self.order = self.model.order
        return self

    def predict(self, n_periods: int = 30) -> np.ndarray:
        """Generate forward point forecasts.

        Args:
            n_periods: Number of steps ahead to predict (default 30).

        Returns:
            1D numpy array of point forecasts for horizons 1..n_periods.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict() is called.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecasts = self.model.predict(n_periods=n_periods)

        return np.asarray(forecasts, dtype=float)

    def predict_in_sample(self) -> np.ndarray:
        """Return in-sample fitted values.

        Returns:
            1D numpy array of in-sample fitted values.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict_in_sample() is called.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = self.model.predict_in_sample()

        return np.asarray(fitted, dtype=float)

    def get_order(self) -> Tuple[int, int, int]:
        """Return the selected (p, d, q) order."""
        if self.order is None:
            raise RuntimeError("Model not yet fitted.")
        return self.order

    def save(self, filepath: str) -> str:
        """Serialize fitted model to disk.

        Args:
            filepath: Destination file path.

        Returns:
            Absolute path to saved model.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "order": self.order,
                "max_p": self.max_p,
                "max_q": self.max_q,
                "max_d": self.max_d,
                "seasonal": self.seasonal,
                "stepwise": self.stepwise,
                "last_train_length": self.last_train_length,
            },
            filepath,
        )
        return os.path.abspath(filepath)

    @classmethod
    def load(cls, filepath: str) -> AutoARIMABaseline:
        """Load serialized AutoARIMA instance from disk."""
        data = joblib.load(filepath)
        instance = cls(
            max_p=data["max_p"],
            max_q=data["max_q"],
            max_d=data["max_d"],
            seasonal=data["seasonal"],
            stepwise=data["stepwise"],
        )
        instance.model = data["model"]
        instance.order = data["order"]
        instance.last_train_length = data["last_train_length"]
        return instance
