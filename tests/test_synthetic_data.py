"""Tests for src.data.synthetic_data."""

import os

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_data import (
    generate_all_synthetic_data,
    generate_synthetic_bunker_fuel,
    generate_synthetic_commodities,
    generate_synthetic_crude_oil,
    generate_synthetic_dxy,
    generate_synthetic_freight_rates,
    generate_synthetic_gscpi,
    generate_synthetic_sp500,
)


class TestGenerateSyntheticFreightRates:

    def test_returns_dataframe(self):
        df = generate_synthetic_freight_rates(periods=100)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = generate_synthetic_freight_rates(periods=100)
        assert "date" in df.columns
        assert "freight_rate" in df.columns

    def test_correct_length(self):
        df = generate_synthetic_freight_rates(periods=200)
        assert len(df) == 200

    def test_rates_in_expected_range(self):
        df = generate_synthetic_freight_rates(periods=1000, seed=42)
        assert df["freight_rate"].min() >= 500
        assert df["freight_rate"].max() <= 3000

    def test_reproducible_with_seed(self):
        df1 = generate_synthetic_freight_rates(periods=100, seed=99)
        df2 = generate_synthetic_freight_rates(periods=100, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_synthetic_freight_rates(periods=100, seed=1)
        df2 = generate_synthetic_freight_rates(periods=100, seed=2)
        assert not df1["freight_rate"].equals(df2["freight_rate"])


class TestGenerateSyntheticBunkerFuel:

    def test_has_required_columns(self):
        df = generate_synthetic_bunker_fuel(periods=100)
        assert "date" in df.columns
        assert "ifo380_price" in df.columns

    def test_prices_in_range(self):
        df = generate_synthetic_bunker_fuel(periods=1000, seed=42)
        assert df["ifo380_price"].min() >= 250
        assert df["ifo380_price"].max() <= 800


class TestGenerateSyntheticCrudeOil:

    def test_has_required_columns(self):
        df = generate_synthetic_crude_oil(periods=100)
        assert {"date", "brent", "wti"}.issubset(set(df.columns))

    def test_brent_above_wti(self):
        df = generate_synthetic_crude_oil(periods=500, seed=42)
        assert (df["brent"] > df["wti"]).all()

    def test_prices_in_range(self):
        df = generate_synthetic_crude_oil(periods=1000, seed=42)
        assert df["brent"].min() >= 40
        assert df["brent"].max() <= 140


class TestGenerateSyntheticSP500:

    def test_has_required_columns(self):
        df = generate_synthetic_sp500(periods=100)
        assert {"date", "sp500"}.issubset(set(df.columns))

    def test_values_in_range(self):
        df = generate_synthetic_sp500(periods=1000, seed=42)
        assert df["sp500"].min() >= 2000
        assert df["sp500"].max() <= 6000


class TestGenerateSyntheticDXY:

    def test_has_required_columns(self):
        df = generate_synthetic_dxy(periods=100)
        assert {"date", "dxy"}.issubset(set(df.columns))

    def test_values_in_range(self):
        df = generate_synthetic_dxy(periods=1000, seed=42)
        assert df["dxy"].min() >= 85
        assert df["dxy"].max() <= 120


class TestGenerateSyntheticGSCPI:

    def test_has_required_columns(self):
        df = generate_synthetic_gscpi(periods=365)
        assert {"date", "gscpi"}.issubset(set(df.columns))

    def test_monthly_frequency(self):
        df = generate_synthetic_gscpi(periods=365)
        # ~12 months for 365 days
        assert 10 <= len(df) <= 14

    def test_values_in_range(self):
        df = generate_synthetic_gscpi(periods=3000, seed=42)
        assert df["gscpi"].min() >= -3.0
        assert df["gscpi"].max() <= 5.0


class TestGenerateSyntheticCommodities:

    def test_has_required_columns(self):
        df = generate_synthetic_commodities(periods=365)
        assert {"date", "iron_ore", "coal", "grain"}.issubset(set(df.columns))

    def test_monthly_frequency(self):
        df = generate_synthetic_commodities(periods=365)
        assert 10 <= len(df) <= 14

    def test_iron_ore_in_range(self):
        df = generate_synthetic_commodities(periods=3000, seed=42)
        assert df["iron_ore"].min() >= 70
        assert df["iron_ore"].max() <= 200


class TestGenerateAllSyntheticData:

    def test_returns_all_sources(self):
        data = generate_all_synthetic_data(periods=100)
        expected_keys = {
            "freight_rates", "bunker_fuel", "crude_oil",
            "sp500", "dxy", "gscpi", "commodities",
        }
        assert set(data.keys()) == expected_keys

    def test_all_values_are_dataframes(self):
        data = generate_all_synthetic_data(periods=100)
        for name, df in data.items():
            assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"

    def test_saves_to_disk(self, tmp_path):
        output_dir = str(tmp_path / "synthetic")
        generate_all_synthetic_data(periods=50, output_dir=output_dir)
        expected_files = [
            "synthetic_freight_rates.csv",
            "synthetic_bunker_fuel.csv",
            "synthetic_crude_oil.csv",
            "synthetic_sp500.csv",
            "synthetic_dxy.csv",
            "synthetic_gscpi.csv",
            "synthetic_commodities.csv",
        ]
        for fname in expected_files:
            assert os.path.exists(os.path.join(output_dir, fname)), f"{fname} not saved"

    def test_no_save_when_no_dir(self):
        # Should not raise when output_dir is None
        data = generate_all_synthetic_data(periods=50, output_dir=None)
        assert len(data) == 7

    def test_reproducible_with_seed(self):
        d1 = generate_all_synthetic_data(periods=50, seed=77)
        d2 = generate_all_synthetic_data(periods=50, seed=77)
        for key in d1:
            pd.testing.assert_frame_equal(d1[key], d2[key])
