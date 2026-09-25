"""Unit tests for src.forecaster.explainability."""

import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.forecaster.explainability import (
    compute_feature_importance_shap,
    compute_prediction_attribution,
    export_shap_summary,
)


@pytest.fixture
def trained_toy_model():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "bunker_price": np.random.uniform(400, 600, n),
        "crude_oil": np.random.uniform(60, 90, n),
        "dxy": np.random.uniform(95, 105, n),
        "seasonal_pressure": np.random.uniform(0.1, 0.9, n),
    })
    # Target strongly correlates with bunker_price and crude_oil
    y = 0.03 * X["bunker_price"] + 0.1 * X["crude_oil"] + np.random.normal(0, 0.5, n)

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        n_estimators=15,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    return model, X


class TestExplainability:

    def test_global_importance(self, trained_toy_model):
        model, X = trained_toy_model
        importance = compute_feature_importance_shap(model, X)
        assert len(importance) == X.shape[1]
        # Verify descending order
        scores = [score for _, score in importance]
        assert scores == sorted(scores, reverse=True)
        # Features should include bunker_price
        feature_names = [feat for feat, _ in importance]
        assert "bunker_price" in feature_names

    def test_local_attribution(self, trained_toy_model):
        model, X = trained_toy_model
        single_row = X.iloc[[0]]
        attr = compute_prediction_attribution(model, single_row, top_n=3)

        assert "base_value" in attr
        assert "predicted_value" in attr
        assert "top_contributions" in attr
        assert len(attr["top_contributions"]) == 3
        for c in attr["top_contributions"]:
            assert "feature" in c
            assert "shap_value" in c

    def test_export_shap_summary(self, trained_toy_model, tmp_path):
        model, X = trained_toy_model
        out_path = str(tmp_path / "shap_summary.json")
        result = export_shap_summary(model, X, X.iloc[[-1]], output_path=out_path)

        assert os.path.exists(result)
        with open(result, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "global_importance" in data
        assert "local_attribution" in data
        assert len(data["global_importance"]) > 0
