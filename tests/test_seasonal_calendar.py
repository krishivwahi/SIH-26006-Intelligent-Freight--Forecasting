"""Tests for src.data.seasonal_calendar."""

import pandas as pd
import pytest

from src.data.seasonal_calendar import (
    SEASONAL_PRESSURE,
    add_seasonal_features,
    get_seasonal_label,
    get_seasonal_pressure,
)


class TestSeasonalPressureConstant:
    """Validate the SEASONAL_PRESSURE lookup table."""

    def test_has_all_twelve_months(self):
        assert set(SEASONAL_PRESSURE.keys()) == set(range(1, 13))

    def test_all_scores_in_range(self):
        for month, (label, score) in SEASONAL_PRESSURE.items():
            assert 0.0 <= score <= 1.0, f"Month {month} score {score} out of range"

    def test_all_labels_valid(self):
        valid = {"High", "Medium", "Low"}
        for month, (label, _) in SEASONAL_PRESSURE.items():
            assert label in valid, f"Month {month} label '{label}' invalid"


class TestGetSeasonalPressure:
    """Tests for get_seasonal_pressure()."""

    @pytest.mark.parametrize("month", range(1, 13))
    def test_valid_months(self, month):
        score = get_seasonal_pressure(month)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_january_is_high(self):
        assert get_seasonal_pressure(1) == 0.80

    def test_may_is_lowest(self):
        assert get_seasonal_pressure(5) == 0.20

    def test_september_is_highest(self):
        assert get_seasonal_pressure(9) == 0.90

    def test_invalid_month_zero(self):
        with pytest.raises(ValueError, match="between 1 and 12"):
            get_seasonal_pressure(0)

    def test_invalid_month_thirteen(self):
        with pytest.raises(ValueError, match="between 1 and 12"):
            get_seasonal_pressure(13)

    def test_invalid_month_negative(self):
        with pytest.raises(ValueError, match="between 1 and 12"):
            get_seasonal_pressure(-1)

    def test_invalid_month_float(self):
        with pytest.raises(ValueError, match="between 1 and 12"):
            get_seasonal_pressure(1.5)  # type: ignore

    def test_invalid_month_string(self):
        with pytest.raises(ValueError, match="between 1 and 12"):
            get_seasonal_pressure("Jan")  # type: ignore


class TestGetSeasonalLabel:
    """Tests for get_seasonal_label()."""

    @pytest.mark.parametrize("month", range(1, 13))
    def test_valid_months(self, month):
        label = get_seasonal_label(month)
        assert label in {"High", "Medium", "Low"}

    def test_april_is_low(self):
        assert get_seasonal_label(4) == "Low"

    def test_july_is_high(self):
        assert get_seasonal_label(7) == "High"

    def test_november_is_medium(self):
        assert get_seasonal_label(11) == "Medium"

    def test_invalid_month_zero(self):
        with pytest.raises(ValueError):
            get_seasonal_label(0)

    def test_invalid_month_thirteen(self):
        with pytest.raises(ValueError):
            get_seasonal_label(13)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            get_seasonal_label(3.0)  # type: ignore


class TestAddSeasonalFeatures:
    """Tests for add_seasonal_features()."""

    def test_adds_columns(self, sample_dates):
        df = pd.DataFrame({"date": sample_dates, "value": range(len(sample_dates))})
        result = add_seasonal_features(df, "date")
        assert "seasonal_pressure" in result.columns
        assert "seasonal_label" in result.columns

    def test_does_not_mutate_input(self, sample_dates):
        df = pd.DataFrame({"date": sample_dates})
        original_cols = list(df.columns)
        add_seasonal_features(df, "date")
        assert list(df.columns) == original_cols

    def test_correct_pressure_values(self):
        dates = pd.to_datetime(["2022-01-15", "2022-05-15", "2022-09-15"])
        df = pd.DataFrame({"date": dates})
        result = add_seasonal_features(df, "date")
        assert result["seasonal_pressure"].iloc[0] == 0.80  # Jan
        assert result["seasonal_pressure"].iloc[1] == 0.20  # May
        assert result["seasonal_pressure"].iloc[2] == 0.90  # Sep

    def test_correct_labels(self):
        dates = pd.to_datetime(["2022-04-01", "2022-06-01", "2022-10-01"])
        df = pd.DataFrame({"date": dates})
        result = add_seasonal_features(df, "date")
        assert result["seasonal_label"].iloc[0] == "Low"
        assert result["seasonal_label"].iloc[1] == "Medium"
        assert result["seasonal_label"].iloc[2] == "High"

    def test_missing_column_raises(self):
        df = pd.DataFrame({"timestamp": ["2022-01-01"]})
        with pytest.raises(KeyError, match="not found"):
            add_seasonal_features(df, "date")

    def test_custom_date_column(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2022-07-01"])})
        result = add_seasonal_features(df, "ts")
        assert result["seasonal_pressure"].iloc[0] == 0.85

    def test_preserves_existing_columns(self, sample_dates):
        df = pd.DataFrame({
            "date": sample_dates,
            "value": range(len(sample_dates)),
            "extra": "keep",
        })
        result = add_seasonal_features(df, "date")
        assert "value" in result.columns
        assert "extra" in result.columns
