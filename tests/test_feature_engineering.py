"""Tests for src.data.feature_engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    add_calendar_features,
    add_lag_features,
    add_momentum_features,
    add_rolling_features,
    build_feature_matrix,
    merge_data_sources,
)


# ---------------------------------------------------------------------------
# add_lag_features
# ---------------------------------------------------------------------------

class TestAddLagFeatures:

    def test_default_lags(self, sample_daily_df):
        result = add_lag_features(sample_daily_df, "value")
        assert "value_lag_7" in result.columns
        assert "value_lag_14" in result.columns
        assert "value_lag_30" in result.columns

    def test_custom_lags(self, sample_daily_df):
        result = add_lag_features(sample_daily_df, "value", lags=[1, 3])
        assert "value_lag_1" in result.columns
        assert "value_lag_3" in result.columns
        assert "value_lag_7" not in result.columns

    def test_lag_values_correct(self):
        df = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=5), "v": [10, 20, 30, 40, 50]})
        result = add_lag_features(df, "v", lags=[1, 2])
        assert np.isnan(result["v_lag_1"].iloc[0])
        assert result["v_lag_1"].iloc[1] == 10
        assert result["v_lag_2"].iloc[2] == 10

    def test_does_not_mutate_input(self, sample_daily_df):
        cols_before = list(sample_daily_df.columns)
        add_lag_features(sample_daily_df, "value")
        assert list(sample_daily_df.columns) == cols_before

    def test_missing_column_raises(self, sample_daily_df):
        with pytest.raises(KeyError, match="not found"):
            add_lag_features(sample_daily_df, "nonexistent")


# ---------------------------------------------------------------------------
# add_rolling_features
# ---------------------------------------------------------------------------

class TestAddRollingFeatures:

    def test_default_windows(self, sample_daily_df):
        result = add_rolling_features(sample_daily_df, "value")
        for w in [7, 14, 30]:
            assert f"value_rmean_{w}" in result.columns
            assert f"value_rstd_{w}" in result.columns

    def test_custom_windows(self, sample_daily_df):
        result = add_rolling_features(sample_daily_df, "value", windows=[5])
        assert "value_rmean_5" in result.columns
        assert "value_rstd_5" in result.columns

    def test_no_nan_in_output(self, sample_daily_df):
        result = add_rolling_features(sample_daily_df, "value", windows=[3])
        assert not result["value_rmean_3"].isna().any()
        assert not result["value_rstd_3"].isna().any()

    def test_rolling_mean_correctness(self):
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = add_rolling_features(df, "v", windows=[3])
        # Window 3, min_periods=1: first value is mean of [1] = 1
        assert result["v_rmean_3"].iloc[0] == pytest.approx(1.0)
        # Third value is mean of [1,2,3] = 2
        assert result["v_rmean_3"].iloc[2] == pytest.approx(2.0)

    def test_does_not_mutate_input(self, sample_daily_df):
        cols_before = list(sample_daily_df.columns)
        add_rolling_features(sample_daily_df, "value")
        assert list(sample_daily_df.columns) == cols_before

    def test_missing_column_raises(self, sample_daily_df):
        with pytest.raises(KeyError, match="not found"):
            add_rolling_features(sample_daily_df, "missing")


# ---------------------------------------------------------------------------
# add_momentum_features
# ---------------------------------------------------------------------------

class TestAddMomentumFeatures:

    def test_default_periods(self, sample_daily_df):
        result = add_momentum_features(sample_daily_df, "value")
        assert "value_mom_7" in result.columns
        assert "value_mom_14" in result.columns

    def test_custom_periods(self, sample_daily_df):
        result = add_momentum_features(sample_daily_df, "value", periods=[3])
        assert "value_mom_3" in result.columns
        assert "value_mom_7" not in result.columns

    def test_no_nan_in_output(self, sample_daily_df):
        result = add_momentum_features(sample_daily_df, "value", periods=[1])
        assert not result["value_mom_1"].isna().any()

    def test_does_not_mutate_input(self, sample_daily_df):
        cols_before = list(sample_daily_df.columns)
        add_momentum_features(sample_daily_df, "value")
        assert list(sample_daily_df.columns) == cols_before

    def test_missing_column_raises(self, sample_daily_df):
        with pytest.raises(KeyError, match="not found"):
            add_momentum_features(sample_daily_df, "nope")


# ---------------------------------------------------------------------------
# add_calendar_features
# ---------------------------------------------------------------------------

class TestAddCalendarFeatures:

    def test_adds_all_columns(self, sample_daily_df):
        result = add_calendar_features(sample_daily_df, "date")
        assert "month" in result.columns
        assert "day_of_week" in result.columns
        assert "day_of_year" in result.columns

    def test_correct_values(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2022-03-15"])})
        result = add_calendar_features(df, "date")
        assert result["month"].iloc[0] == 3
        assert result["day_of_week"].iloc[0] == 1  # Tuesday
        assert result["day_of_year"].iloc[0] == 74

    def test_does_not_mutate_input(self, sample_daily_df):
        cols_before = list(sample_daily_df.columns)
        add_calendar_features(sample_daily_df, "date")
        assert list(sample_daily_df.columns) == cols_before

    def test_missing_column_raises(self):
        df = pd.DataFrame({"ts": ["2022-01-01"]})
        with pytest.raises(KeyError, match="not found"):
            add_calendar_features(df, "date")


# ---------------------------------------------------------------------------
# merge_data_sources
# ---------------------------------------------------------------------------

class TestMergeDataSources:

    def test_freight_only(self, freight_rates):
        result = merge_data_sources(freight_rates)
        assert "freight_rate" in result.columns
        assert len(result) == len(freight_rates)

    def test_with_all_sources(self, freight_rates, bunker_fuel, crude_oil,
                               sp500, dxy, gscpi, commodities):
        result = merge_data_sources(
            freight_rates, bunker_fuel, crude_oil,
            sp500, dxy, gscpi, commodities,
        )
        expected_cols = {"date", "freight_rate", "ifo380_price",
                         "brent", "wti", "sp500", "dxy", "gscpi",
                         "iron_ore", "coal", "grain"}
        assert expected_cols.issubset(set(result.columns))

    def test_sorted_by_date(self, freight_rates, bunker_fuel):
        result = merge_data_sources(freight_rates, bunker_fuel)
        dates = pd.to_datetime(result["date"])
        assert dates.is_monotonic_increasing

    def test_no_nans_after_fill(self, freight_rates, gscpi):
        result = merge_data_sources(freight_rates, gscpi=gscpi)
        assert not result.isna().any().any()

    def test_missing_date_column_raises(self):
        df = pd.DataFrame({"rate": [100]})
        with pytest.raises(KeyError, match="not found"):
            merge_data_sources(df)

    def test_partial_sources(self, freight_rates, sp500, dxy):
        result = merge_data_sources(freight_rates, sp500=sp500, dxy=dxy)
        assert "sp500" in result.columns
        assert "dxy" in result.columns
        assert "brent" not in result.columns


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:

    def test_returns_x_and_y(self, freight_rates, bunker_fuel, crude_oil):
        merged = merge_data_sources(freight_rates, bunker_fuel, crude_oil)
        X, y = build_feature_matrix(merged)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_no_nans_in_output(self, freight_rates, bunker_fuel):
        merged = merge_data_sources(freight_rates, bunker_fuel)
        X, y = build_feature_matrix(merged)
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_target_not_in_features(self, freight_rates):
        merged = merge_data_sources(freight_rates)
        X, y = build_feature_matrix(merged)
        assert "freight_rate" not in X.columns

    def test_date_not_in_features(self, freight_rates):
        merged = merge_data_sources(freight_rates)
        X, y = build_feature_matrix(merged)
        assert "date" not in X.columns

    def test_seasonal_label_not_in_features(self, freight_rates):
        merged = merge_data_sources(freight_rates)
        X, y = build_feature_matrix(merged)
        assert "seasonal_label" not in X.columns

    def test_has_calendar_features(self, freight_rates):
        merged = merge_data_sources(freight_rates)
        X, _ = build_feature_matrix(merged)
        assert "month" in X.columns
        assert "day_of_week" in X.columns
        assert "day_of_year" in X.columns

    def test_has_seasonal_pressure(self, freight_rates):
        merged = merge_data_sources(freight_rates)
        X, _ = build_feature_matrix(merged)
        assert "seasonal_pressure" in X.columns

    def test_has_lag_features(self, freight_rates):
        merged = merge_data_sources(freight_rates)
        X, _ = build_feature_matrix(merged)
        assert "freight_rate_lag_7" in X.columns

    def test_has_rolling_features(self, freight_rates):
        merged = merge_data_sources(freight_rates)
        X, _ = build_feature_matrix(merged)
        assert "freight_rate_rmean_7" in X.columns
        assert "freight_rate_rstd_7" in X.columns

    def test_has_momentum_features(self, freight_rates):
        merged = merge_data_sources(freight_rates)
        X, _ = build_feature_matrix(merged)
        assert "freight_rate_mom_7" in X.columns

    def test_x_and_y_aligned(self, freight_rates):
        merged = merge_data_sources(freight_rates)
        X, y = build_feature_matrix(merged)
        assert len(X) == len(y)

    def test_missing_target_raises(self, freight_rates):
        merged = merge_data_sources(freight_rates)
        with pytest.raises(KeyError, match="not found"):
            build_feature_matrix(merged, target_column="nonexistent")

    def test_missing_date_raises(self, freight_rates):
        merged = merge_data_sources(freight_rates)
        merged = merged.rename(columns={"date": "ts"})
        with pytest.raises(KeyError, match="not found"):
            build_feature_matrix(merged)
