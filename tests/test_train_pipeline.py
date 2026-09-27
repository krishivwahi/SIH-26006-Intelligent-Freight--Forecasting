"""Unit tests for src.forecaster.train_pipeline."""

import json
import os
import pandas as pd
import pytest

from src.data.synthetic_data import generate_all_synthetic_data
from src.forecaster.train_pipeline import run_training_pipeline
from src.solver.solver import solve


@pytest.fixture
def synthetic_market_data(tmp_path):
    """Generate a small merged synthetic market dataset."""
    from src.data.feature_engineering import merge_data_sources
    from src.data.synthetic_data import (
        generate_synthetic_bunker_fuel,
        generate_synthetic_freight_rates,
    )
    freight = generate_synthetic_freight_rates("2024-01-01", 200, seed=42)
    bunker = generate_synthetic_bunker_fuel("2024-01-01", 200, seed=43)
    merged = merge_data_sources(freight, bunker)

    out_csv = str(tmp_path / "test_market_data.csv")
    merged.to_csv(out_csv, index=False)
    return out_csv


class TestTrainPipeline:

    def test_run_training_pipeline(self, synthetic_market_data, tmp_path):
        model_dir = str(tmp_path / "models")
        forecast_path = str(tmp_path / "freight_forecast_30d.json")

        summary = run_training_pipeline(
            data_path=synthetic_market_data,
            model_dir=model_dir,
            forecast_path=forecast_path,
            n_estimators=10,  # fast test
        )

        assert summary["status"] == "Success"
        assert os.path.exists(summary["model_path"])
        assert os.path.exists(summary["arima_path"])
        assert os.path.exists(summary["shap_path"])
        assert os.path.exists(summary["forecast_path"])

        # Check JSON contract
        with open(summary["forecast_path"], "r", encoding="utf-8") as f:
            records = json.load(f)

        assert len(records) == 30 * 4 * 10  # 30 days x 4 feasible vessels (V-001 excluded) x 10 routes
        required_keys = {"date_index", "vessel_id", "route_id", "p10_rate", "p50_rate", "p90_rate"}
        assert required_keys.issubset(records[0].keys())

        # Test solver runs on this real model forecast
        result = solve(lam=0.5, forecast_path=summary["forecast_path"])
        assert result.solver_status == "Optimal"
        assert len(result.assignments) > 0
