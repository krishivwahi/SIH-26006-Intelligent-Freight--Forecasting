"""Shared pytest fixtures for the test suite."""

import pandas as pd
import pytest

from src.data.synthetic_data import (
    generate_synthetic_bunker_fuel,
    generate_synthetic_commodities,
    generate_synthetic_crude_oil,
    generate_synthetic_dxy,
    generate_synthetic_freight_rates,
    generate_synthetic_gscpi,
    generate_synthetic_sp500,
)


@pytest.fixture
def sample_dates():
    """365 daily dates starting 2022-01-01."""
    return pd.date_range("2022-01-01", periods=365, freq="D")


@pytest.fixture
def sample_daily_df(sample_dates):
    """Simple daily DataFrame with date and value columns."""
    return pd.DataFrame({
        "date": sample_dates,
        "value": range(len(sample_dates)),
    })


@pytest.fixture
def freight_rates():
    """Synthetic freight rates, 365 days."""
    return generate_synthetic_freight_rates("2022-01-01", 365, seed=100)


@pytest.fixture
def bunker_fuel():
    """Synthetic bunker fuel, 365 days."""
    return generate_synthetic_bunker_fuel("2022-01-01", 365, seed=101)


@pytest.fixture
def crude_oil():
    """Synthetic crude oil, 365 days."""
    return generate_synthetic_crude_oil("2022-01-01", 365, seed=102)


@pytest.fixture
def sp500():
    """Synthetic S&P 500, 365 days."""
    return generate_synthetic_sp500("2022-01-01", 365, seed=103)


@pytest.fixture
def dxy():
    """Synthetic DXY, 365 days."""
    return generate_synthetic_dxy("2022-01-01", 365, seed=104)


@pytest.fixture
def gscpi():
    """Synthetic GSCPI, monthly for ~365 days."""
    return generate_synthetic_gscpi("2022-01-01", 365, seed=105)


@pytest.fixture
def commodities():
    """Synthetic commodities, monthly for ~365 days."""
    return generate_synthetic_commodities("2022-01-01", 365, seed=106)
