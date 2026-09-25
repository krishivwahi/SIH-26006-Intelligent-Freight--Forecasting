"""Tests for src.data.load_real_data."""

import os
from datetime import datetime
import pandas as pd
import pytest

from src.data.load_real_data import (
    load_sp500,
    load_dxy,
    load_crude_oil,
    load_bunker_fuel,
    load_gscpi,
    load_commodities,
    load_and_merge_all,
)

RAW_DATA_DIR = os.path.join("data", "raw")


class TestLoadRealData:

    def test_load_sp500(self):
        filepath = os.path.join(RAW_DATA_DIR, "SP500.csv")
        if not os.path.exists(filepath):
            pytest.skip("SP500.csv not found")
        df = load_sp500(filepath)
        assert "date" in df.columns
        assert "sp500" in df.columns
        assert len(df) > 100
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_load_dxy(self):
        filepath = os.path.join(RAW_DATA_DIR, "DXYUSDollar Index.csv")
        if not os.path.exists(filepath):
            pytest.skip("DXYUSDollar Index.csv not found")
        df = load_dxy(filepath)
        assert "date" in df.columns
        assert "dxy" in df.columns
        assert len(df) > 100
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_load_crude_oil(self):
        filepath = os.path.join(RAW_DATA_DIR, "Crude_Oil_Combined.xlsx")
        if not os.path.exists(filepath):
            pytest.skip("Crude_Oil_Combined.xlsx not found")
        df = load_crude_oil(filepath)
        assert "date" in df.columns
        assert "brent" in df.columns
        assert "wti" in df.columns
        assert len(df) > 100
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_load_bunker_fuel(self):
        filepath = os.path.join(RAW_DATA_DIR, "Daily_Bunker_Fuel_Prices_20260904.csv")
        if not os.path.exists(filepath):
            pytest.skip("Daily_Bunker_Fuel_Prices_20260904.csv not found")
        df = load_bunker_fuel(filepath)
        assert "date" in df.columns
        assert "ifo380_price" in df.columns
        assert len(df) > 100
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_load_gscpi(self):
        filepath = os.path.join(RAW_DATA_DIR, "gscpi_data.xls")
        if not os.path.exists(filepath):
            pytest.skip("gscpi_data.xls not found")
        df = load_gscpi(filepath)
        assert "date" in df.columns
        assert "gscpi" in df.columns
        assert len(df) > 50
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_load_commodities(self):
        filepath = os.path.join(RAW_DATA_DIR, "CMO-Historical-Data-Monthly.xlsx")
        if not os.path.exists(filepath):
            pytest.skip("CMO-Historical-Data-Monthly.xlsx not found")
        df = load_commodities(filepath)
        assert "date" in df.columns
        assert "coal" in df.columns
        assert "iron_ore" in df.columns
        assert "grain" in df.columns
        assert len(df) > 100
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_load_missing_files_raises(self, tmp_path):
        nonexistent = str(tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            load_sp500(nonexistent)
        with pytest.raises(FileNotFoundError):
            load_dxy(nonexistent)
        with pytest.raises(FileNotFoundError):
            load_crude_oil(nonexistent)
        with pytest.raises(FileNotFoundError):
            load_bunker_fuel(nonexistent)
        with pytest.raises(FileNotFoundError):
            load_gscpi(nonexistent)
        with pytest.raises(FileNotFoundError):
            load_commodities(nonexistent)

    def test_load_and_merge_all(self, tmp_path):
        out_csv = str(tmp_path / "test_merged.csv")
        merged = load_and_merge_all(raw_dir=RAW_DATA_DIR, output_path=out_csv)
        assert "date" in merged.columns
        assert "freight_rate" in merged.columns
        assert os.path.exists(out_csv)
