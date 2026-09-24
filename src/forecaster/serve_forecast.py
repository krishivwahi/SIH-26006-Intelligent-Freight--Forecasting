"""
src/forecaster/serve_forecast.py

Lightweight inference-only module. Loads pre-trained models from disk
and generates fresh 30-day forecasts WITHOUT retraining.
"""
import os
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from src.data.feature_engineering import build_feature_matrix
from src.data.load_real_data import load_and_merge_all
from src.forecaster.baseline_arima import AutoARIMABaseline
from src.forecaster.quantile_model import QuantileForecaster
from src.forecaster.write_forecast import (
    BDI_TO_USD_PER_MT_SCALE,
    create_forecast_records,
    write_forecast_json,
)

DEFAULT_MODEL_DIR = "models"
DEFAULT_FORECAST_PATH = os.path.join("data", "interim", "freight_forecast_30d.json")


class ForecastServer:
    """Stateful forecast server — loads models once, serves many requests."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.forecaster: Optional[QuantileForecaster] = None
        self.arima: Optional[AutoARIMABaseline] = None
        self.model_version: Optional[str] = None
        self._loaded = False

    def load_models(self, version: Optional[str] = None) -> None:
        """Load trained models from disk. Call once at startup."""
        if version:
            versioned_dir = os.path.join(self.model_dir, version)
        else:
            versioned_dir = self._find_latest_version()

        # Load LightGBM quantile models
        self.forecaster = QuantileForecaster()
        self.forecaster.load(versioned_dir)

        # Load AutoARIMA baseline
        arima_path = os.path.join(versioned_dir, "arima_baseline.joblib")
        if os.path.exists(arima_path):
            self.arima = AutoARIMABaseline.load(arima_path)

        self.model_version = version or os.path.basename(versioned_dir)
        self._loaded = True

    def generate_forecast(
        self,
        data_path: Optional[str] = None,
        base_date: Optional[date] = None,
        output_path: str = DEFAULT_FORECAST_PATH,
    ) -> Dict:
        """Generate a fresh 30-day forecast using loaded models.

        This does NOT retrain — it runs inference only (~2-5 seconds).
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if base_date is None:
            base_date = date.today()

        # Load latest market data and build features
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df["date"] = pd.to_datetime(df["date"])
        else:
            df = load_and_merge_all()

        X, y = build_feature_matrix(df, target_column="freight_rate", date_column="date")

        # Inject ARIMA bridge feature
        if self.arima is not None:
            try:
                arima_pred = self.arima.predict(n_periods=1)
                X["arima_pred"] = arima_pred[0]
            except Exception:
                X["arima_pred"] = float(y.iloc[-1])
        else:
            X["arima_pred"] = float(y.iloc[-1])

        # Run inference (the fast part — no training)
        latest_row = X.iloc[[-1]]
        predictions = self.forecaster.predict(latest_row)

        # Scale and write
        mean_p50 = float(predictions["p50"].mean())
        scale = BDI_TO_USD_PER_MT_SCALE if mean_p50 > 100.0 else 1.0
        records = create_forecast_records(predictions, start_date=base_date, rate_scale=scale)
        written_path = write_forecast_json(records, filepath=output_path)

        return {
            "status": "Success",
            "model_version": self.model_version,
            "base_date": base_date.isoformat(),
            "records_written": len(records),
            "forecast_path": written_path,
            "rate_scale": scale,
        }

    def _find_latest_version(self) -> str:
        """Find the latest versioned model directory."""
        if not os.path.isdir(self.model_dir):
            return self.model_dir

        # Look for versioned subdirs like v001, v002, etc.
        versions = [
            d for d in os.listdir(self.model_dir)
            if os.path.isdir(os.path.join(self.model_dir, d)) and d.startswith("v")
        ]
        if versions:
            latest = sorted(versions)[-1]
            return os.path.join(self.model_dir, latest)

        # Fallback: models are directly in model_dir (Alpha layout)
        return self.model_dir
