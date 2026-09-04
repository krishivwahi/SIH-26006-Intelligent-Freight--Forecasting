"""Unit tests for src.forecaster.baseline_arima."""

import os
import numpy as np
import pandas as pd
import pytest

from src.forecaster.baseline_arima import AutoARIMABaseline


@pytest.fixture
def sample_series():
    """Generate synthetic stationary-trending freight rate series."""
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    noise = np.random.normal(0, 0.5, n)
    rates = 20.0 + 0.05 * t + noise
    return pd.Series(rates)


class TestAutoARIMABaseline:

    def test_fit_and_predict(self, sample_series):
        model = AutoARIMABaseline(max_p=1, max_q=1, max_d=1)
        model.fit(sample_series)
        assert model.order is not None
        assert len(model.order) == 3

        preds = model.predict(n_periods=10)
        assert len(preds) == 10
        assert not np.isnan(preds).any()
        # Predictions should be in plausible freight rate range
        assert (preds > 15.0).all() and (preds < 35.0).all()

    def test_predict_in_sample(self, sample_series):
        model = AutoARIMABaseline(max_p=1, max_q=1, max_d=1)
        model.fit(sample_series)
        fitted = model.predict_in_sample()
        assert len(fitted) == len(sample_series)
        assert not np.isnan(fitted).any()

    def test_predict_before_fit_raises(self):
        model = AutoARIMABaseline()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(5)

    def test_short_series_raises(self):
        model = AutoARIMABaseline()
        with pytest.raises(ValueError, match="too short"):
            model.fit(np.array([10.0, 11.0, 12.0]))

    def test_save_and_load(self, sample_series, tmp_path):
        model = AutoARIMABaseline(max_p=1, max_q=1, max_d=1)
        model.fit(sample_series)
        save_path = str(tmp_path / "arima.joblib")
        model.save(save_path)

        assert os.path.exists(save_path)
        loaded = AutoARIMABaseline.load(save_path)
        assert loaded.order == model.order

        preds_orig = model.predict(5)
        preds_loaded = loaded.predict(5)
        np.testing.assert_allclose(preds_orig, preds_loaded)
