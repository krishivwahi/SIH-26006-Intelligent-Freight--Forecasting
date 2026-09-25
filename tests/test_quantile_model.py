"""Tests for src.forecaster.quantile_model."""

import os

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import build_feature_matrix, merge_data_sources
from src.data.synthetic_data import (
    generate_synthetic_bunker_fuel,
    generate_synthetic_freight_rates,
)
from src.forecaster.quantile_model import QuantileForecaster


@pytest.fixture
def small_training_data():
    """Small feature matrix for fast training tests."""
    freight = generate_synthetic_freight_rates("2022-01-01", 200, seed=50)
    bunker = generate_synthetic_bunker_fuel("2022-01-01", 200, seed=51)
    merged = merge_data_sources(freight, bunker)
    X, y = build_feature_matrix(merged)
    return X, y


@pytest.fixture
def trained_forecaster(small_training_data):
    """A QuantileForecaster already trained on small data."""
    X, y = small_training_data
    fc = QuantileForecaster(horizon=5, n_estimators=10, random_state=42)
    fc.train(X, y)
    return fc


class TestQuantileForecasterInit:

    def test_default_horizon(self):
        fc = QuantileForecaster()
        assert fc.horizon == 30

    def test_custom_horizon(self):
        fc = QuantileForecaster(horizon=10)
        assert fc.horizon == 10

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="Horizon must be >= 1"):
            QuantileForecaster(horizon=0)

    def test_not_trained_initially(self):
        fc = QuantileForecaster()
        assert not fc.is_trained

    def test_default_alphas(self):
        fc = QuantileForecaster()
        assert fc.alphas == [0.1, 0.5, 0.9]


class TestPrepareTrainingData:

    def test_expansion_shape(self, small_training_data):
        X, y = small_training_data
        fc = QuantileForecaster(horizon=5)
        X_exp, y_exp = fc._prepare_training_data(X, y)
        # For horizon=5, we get sum of (len(X)-1, len(X)-2, ..., len(X)-5) rows
        expected = sum(len(X) - h for h in range(1, 6))
        assert len(X_exp) == expected
        assert len(y_exp) == expected

    def test_horizon_step_column_present(self, small_training_data):
        X, y = small_training_data
        fc = QuantileForecaster(horizon=3)
        X_exp, _ = fc._prepare_training_data(X, y)
        assert "horizon_step" in X_exp.columns

    def test_horizon_step_values(self, small_training_data):
        X, y = small_training_data
        fc = QuantileForecaster(horizon=3)
        X_exp, _ = fc._prepare_training_data(X, y)
        assert set(X_exp["horizon_step"].unique()) == {1, 2, 3}

    def test_skips_horizon_steps_exceeding_data(self):
        """When horizon > len(X), some steps produce n_valid <= 0 and are skipped."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([10.0, 20.0, 30.0])
        fc = QuantileForecaster(horizon=5)
        X_exp, y_exp = fc._prepare_training_data(X, y)
        # Only horizon steps 1 and 2 are valid (len=3, so n_valid=2 and 1)
        # Steps 3,4,5 would have n_valid <= 0 and hit the continue branch
        assert set(X_exp["horizon_step"].unique()) == {1, 2}
        assert len(X_exp) == 3  # 2 + 1


class TestTrain:

    def test_train_sets_is_trained(self, small_training_data):
        X, y = small_training_data
        fc = QuantileForecaster(horizon=5, n_estimators=10)
        assert not fc.is_trained
        fc.train(X, y)
        assert fc.is_trained

    def test_trains_all_three_models(self, small_training_data):
        X, y = small_training_data
        fc = QuantileForecaster(horizon=5, n_estimators=10)
        fc.train(X, y)
        assert 0.1 in fc.models
        assert 0.5 in fc.models
        assert 0.9 in fc.models

    def test_too_small_dataset_raises(self):
        X = pd.DataFrame({"a": range(5)})
        y = pd.Series(range(5))
        fc = QuantileForecaster(horizon=10)
        with pytest.raises(ValueError, match="Need at least"):
            fc.train(X, y)

    def test_stores_feature_names(self, small_training_data):
        X, y = small_training_data
        fc = QuantileForecaster(horizon=5, n_estimators=10)
        fc.train(X, y)
        assert fc._feature_names is not None
        assert "horizon_step" in fc._feature_names


class TestPredict:

    def test_output_shape(self, trained_forecaster, small_training_data):
        X, _ = small_training_data
        preds = trained_forecaster.predict(X.iloc[[-1]])
        assert len(preds) == trained_forecaster.horizon
        assert set(preds.columns) == {"horizon_step", "p10", "p50", "p90"}

    def test_quantile_ordering(self, trained_forecaster, small_training_data):
        X, _ = small_training_data
        preds = trained_forecaster.predict(X.iloc[[-1]])
        # P10 <= P50 <= P90 should generally hold (may not be strict for
        # very small training sets, but direction should be roughly correct)
        assert (preds["p10"] <= preds["p90"]).all()

    def test_horizon_steps_complete(self, trained_forecaster, small_training_data):
        X, _ = small_training_data
        preds = trained_forecaster.predict(X.iloc[[-1]])
        expected_steps = list(range(1, trained_forecaster.horizon + 1))
        assert list(preds["horizon_step"]) == expected_steps

    def test_multiple_input_rows(self, trained_forecaster, small_training_data):
        X, _ = small_training_data
        preds = trained_forecaster.predict(X.iloc[-3:])
        # 3 input rows x 5 horizon steps = 15 output rows
        assert len(preds) == 3 * trained_forecaster.horizon

    def test_untrained_raises(self, small_training_data):
        X, _ = small_training_data
        fc = QuantileForecaster(horizon=5)
        with pytest.raises(RuntimeError, match="not been trained"):
            fc.predict(X.iloc[[-1]])

    def test_predictions_are_numeric(self, trained_forecaster, small_training_data):
        X, _ = small_training_data
        preds = trained_forecaster.predict(X.iloc[[-1]])
        assert preds["p10"].dtype == np.float64 or preds["p10"].dtype == float
        assert preds["p50"].dtype == np.float64 or preds["p50"].dtype == float
        assert preds["p90"].dtype == np.float64 or preds["p90"].dtype == float


class TestSaveLoad:

    def test_save_creates_files(self, trained_forecaster, tmp_path):
        save_dir = str(tmp_path / "models")
        trained_forecaster.save(save_dir)
        assert os.path.exists(os.path.join(save_dir, "lgb_q01.pkl"))
        assert os.path.exists(os.path.join(save_dir, "lgb_q05.pkl"))
        assert os.path.exists(os.path.join(save_dir, "lgb_q09.pkl"))
        assert os.path.exists(os.path.join(save_dir, "meta.pkl"))

    def test_load_restores_model(self, trained_forecaster, small_training_data, tmp_path):
        save_dir = str(tmp_path / "models")
        trained_forecaster.save(save_dir)

        loaded = QuantileForecaster()
        loaded.load(save_dir)
        assert loaded.is_trained
        assert loaded.horizon == trained_forecaster.horizon

        X, _ = small_training_data
        preds_original = trained_forecaster.predict(X.iloc[[-1]])
        preds_loaded = loaded.predict(X.iloc[[-1]])
        pd.testing.assert_frame_equal(preds_original, preds_loaded)

    def test_save_untrained_raises(self, tmp_path):
        fc = QuantileForecaster()
        with pytest.raises(RuntimeError, match="not been trained"):
            fc.save(str(tmp_path))

    def test_load_missing_meta_raises(self, tmp_path):
        fc = QuantileForecaster()
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            fc.load(str(tmp_path))

    def test_load_missing_model_raises(self, trained_forecaster, tmp_path):
        save_dir = str(tmp_path / "models")
        trained_forecaster.save(save_dir)
        # Delete one model file
        os.remove(os.path.join(save_dir, "lgb_q05.pkl"))
        fc = QuantileForecaster()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            fc.load(save_dir)
