"""Tests for src.forecaster.write_forecast."""

import json
import os
from datetime import date

import pandas as pd
import pytest

from src.forecaster.write_forecast import (
    DEFAULT_FILEPATH,
    DEFAULT_ROUTES,
    DEFAULT_ROUTE_MULTIPLIERS,
    DEFAULT_VESSELS,
    DEFAULT_VESSEL_MULTIPLIERS,
    create_forecast_records,
    generate_forecast,
    write_forecast_json,
)


@pytest.fixture
def sample_predictions():
    """Minimal predictions DataFrame (5 horizon steps)."""
    return pd.DataFrame({
        "horizon_step": [1, 2, 3, 4, 5],
        "p10": [1000.0, 1010.0, 1020.0, 1030.0, 1040.0],
        "p50": [1200.0, 1210.0, 1220.0, 1230.0, 1240.0],
        "p90": [1400.0, 1410.0, 1420.0, 1430.0, 1440.0],
    })


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_five_vessels(self):
        assert len(DEFAULT_VESSELS) == 5

    def test_ten_routes(self):
        assert len(DEFAULT_ROUTES) == 10

    def test_route_multipliers_match_routes(self):
        assert set(DEFAULT_ROUTE_MULTIPLIERS.keys()) == set(DEFAULT_ROUTES)

    def test_vessel_multipliers_match_vessels(self):
        assert set(DEFAULT_VESSEL_MULTIPLIERS.keys()) == set(DEFAULT_VESSELS)


# ---------------------------------------------------------------------------
# create_forecast_records
# ---------------------------------------------------------------------------

class TestCreateForecastRecords:

    def test_record_count(self, sample_predictions):
        records = create_forecast_records(sample_predictions)
        # 5 steps x 5 vessels x 10 routes = 250
        assert len(records) == 5 * 5 * 10

    def test_record_schema(self, sample_predictions):
        records = create_forecast_records(sample_predictions)
        required_keys = {"date_index", "vessel_id", "route_id",
                         "p10_rate", "p50_rate", "p90_rate"}
        for rec in records:
            assert set(rec.keys()) == required_keys

    def test_custom_vessels_and_routes(self, sample_predictions):
        records = create_forecast_records(
            sample_predictions,
            vessels=["V-X"],
            routes=["R-A", "R-B"],
        )
        assert len(records) == 5 * 1 * 2

    def test_custom_start_date(self, sample_predictions):
        records = create_forecast_records(
            sample_predictions,
            start_date=date(2025, 6, 1),
        )
        dates = {r["date_index"] for r in records}
        assert "2025-06-02" in dates  # horizon_step=1 -> June 2

    def test_multipliers_applied(self):
        preds = pd.DataFrame({
            "horizon_step": [1],
            "p10": [1000.0],
            "p50": [1200.0],
            "p90": [1400.0],
        })
        records = create_forecast_records(
            preds,
            vessels=["V-001"],
            routes=["R-01"],
            route_multipliers={"R-01": 2.0},
            vessel_multipliers={"V-001": 0.5},
        )
        # combined = 2.0 * 0.5 = 1.0, so rates unchanged
        assert records[0]["p50_rate"] == 1200.0

    def test_unknown_vessel_uses_default_multiplier(self):
        preds = pd.DataFrame({
            "horizon_step": [1],
            "p10": [1000.0],
            "p50": [1200.0],
            "p90": [1400.0],
        })
        records = create_forecast_records(
            preds,
            vessels=["V-UNKNOWN"],
            routes=["R-01"],
            route_multipliers={"R-01": 1.0},
            vessel_multipliers={},  # no multiplier -> defaults to 1.0
        )
        assert records[0]["p50_rate"] == 1200.0

    def test_default_start_date_is_today(self, sample_predictions):
        records = create_forecast_records(sample_predictions, vessels=["V-001"], routes=["R-01"])
        from datetime import timedelta
        expected = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert records[0]["date_index"] == expected


# ---------------------------------------------------------------------------
# write_forecast_json
# ---------------------------------------------------------------------------

class TestWriteForecastJson:

    def test_creates_file(self, sample_predictions, tmp_path):
        records = create_forecast_records(
            sample_predictions, vessels=["V-001"], routes=["R-01"],
        )
        filepath = str(tmp_path / "forecast.json")
        result_path = write_forecast_json(records, filepath=filepath)
        assert os.path.exists(result_path)

    def test_default_filepath(self, sample_predictions, monkeypatch, tmp_path):
        """Covers the filepath=None branch using DEFAULT_FILEPATH."""
        import src.forecaster.write_forecast as wf
        monkeypatch.setattr(wf, "DEFAULT_FILEPATH", str(tmp_path / "data" / "interim" / "forecast.json"))
        records = create_forecast_records(
            sample_predictions, vessels=["V-001"], routes=["R-01"],
        )
        result = write_forecast_json(records, filepath=None)
        assert os.path.exists(result)

    def test_valid_json(self, sample_predictions, tmp_path):
        records = create_forecast_records(
            sample_predictions, vessels=["V-001"], routes=["R-01"],
        )
        filepath = str(tmp_path / "forecast.json")
        write_forecast_json(records, filepath=filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 5

    def test_creates_parent_directories(self, sample_predictions, tmp_path):
        filepath = str(tmp_path / "deep" / "nested" / "forecast.json")
        records = create_forecast_records(
            sample_predictions, vessels=["V-001"], routes=["R-01"],
        )
        write_forecast_json(records, filepath=filepath)
        assert os.path.exists(filepath)

    def test_returns_absolute_path(self, sample_predictions, tmp_path):
        filepath = str(tmp_path / "forecast.json")
        records = create_forecast_records(
            sample_predictions, vessels=["V-001"], routes=["R-01"],
        )
        result = write_forecast_json(records, filepath=filepath)
        assert os.path.isabs(result)


# ---------------------------------------------------------------------------
# generate_forecast (end-to-end)
# ---------------------------------------------------------------------------

class TestGenerateForecast:

    def test_end_to_end(self, tmp_path):
        """Train a small model and run the full pipeline."""
        from src.data.feature_engineering import build_feature_matrix, merge_data_sources
        from src.data.synthetic_data import (
            generate_synthetic_bunker_fuel,
            generate_synthetic_freight_rates,
        )
        from src.forecaster.quantile_model import QuantileForecaster

        freight = generate_synthetic_freight_rates("2022-01-01", 200, seed=50)
        bunker = generate_synthetic_bunker_fuel("2022-01-01", 200, seed=51)
        merged = merge_data_sources(freight, bunker)
        X, y = build_feature_matrix(merged)

        fc = QuantileForecaster(horizon=5, n_estimators=10, random_state=42)
        fc.train(X, y)

        filepath = str(tmp_path / "forecast.json")
        result = generate_forecast(
            fc, X.iloc[[-1]],
            start_date=date(2025, 1, 1),
            vessels=["V-001", "V-002"],
            routes=["R-01"],
            filepath=filepath,
        )
        assert os.path.exists(result)
        with open(result) as f:
            data = json.load(f)
        # 5 steps x 2 vessels x 1 route = 10
        assert len(data) == 10
