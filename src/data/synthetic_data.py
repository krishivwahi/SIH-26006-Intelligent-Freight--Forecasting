"""
Synthetic data generator for development and testing.

Produces realistic-looking time-series data for all six data sources
and a calibrated synthetic freight-rate target variable. The synthetic
freight series is shaped to match publicly known BDI ranges (500-3000
post-2010) and is labeled as synthetic everywhere.

NEVER present this data as real. It is a transparent development proxy.
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _random_walk(
    n: int,
    start: float,
    drift: float,
    volatility: float,
    mean_reversion: float,
    long_run_mean: float,
    rng: np.random.Generator,
    floor: Optional[float] = None,
    ceiling: Optional[float] = None,
) -> np.ndarray:
    """Ornstein-Uhlenbeck-style mean-reverting random walk.

    Args:
        n: Number of steps to generate.
        start: Starting value.
        drift: Constant drift per step.
        volatility: Standard deviation of daily shocks.
        mean_reversion: Speed of reversion to *long_run_mean* (0-1).
        long_run_mean: Long-run equilibrium level.
        rng: NumPy random generator.
        floor: Optional minimum value (hard clamp).
        ceiling: Optional maximum value (hard clamp).

    Returns:
        Array of length *n*.
    """
    values = np.empty(n)
    values[0] = start
    for i in range(1, n):
        shock = rng.normal(0, volatility)
        reversion = mean_reversion * (long_run_mean - values[i - 1])
        values[i] = values[i - 1] + drift + reversion + shock
        if floor is not None:
            values[i] = max(values[i], floor)
        if ceiling is not None:
            values[i] = min(values[i], ceiling)
    return values


def generate_synthetic_freight_rates(
    start_date: str = "2019-01-01",
    periods: int = 1826,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a calibrated synthetic freight-rate series.

    Shaped to match publicly known BDI ranges: 500-3000 post-2010,
    centered around 1500 with seasonal variation.

    Args:
        start_date: First date in the series (ISO format).
        periods: Number of daily observations.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``date`` and ``freight_rate``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    base = _random_walk(
        n=periods, start=1500.0, drift=0.0, volatility=25.0,
        mean_reversion=0.02, long_run_mean=1500.0, rng=rng,
        floor=400.0, ceiling=3500.0,
    )
    # Add seasonal component
    months = dates.month
    seasonal_amp = 200.0 * np.sin(2 * np.pi * (months - 3) / 12)
    rates = base + seasonal_amp
    rates = np.clip(rates, 500, 3000)
    return pd.DataFrame({"date": dates, "freight_rate": np.round(rates, 2)})


def generate_synthetic_bunker_fuel(
    start_date: str = "2019-01-01",
    periods: int = 1826,
    seed: int = 43,
) -> pd.DataFrame:
    """Generate synthetic IFO380 bunker fuel prices (USD/ton).

    Typical range: 300-700 USD/ton.

    Args:
        start_date: First date (ISO format).
        periods: Number of daily observations.
        seed: Random seed.

    Returns:
        DataFrame with columns ``date`` and ``ifo380_price``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    prices = _random_walk(
        n=periods, start=450.0, drift=0.0, volatility=8.0,
        mean_reversion=0.01, long_run_mean=500.0, rng=rng,
        floor=250.0, ceiling=800.0,
    )
    return pd.DataFrame({"date": dates, "ifo380_price": np.round(prices, 2)})


def generate_synthetic_crude_oil(
    start_date: str = "2019-01-01",
    periods: int = 1826,
    seed: int = 44,
) -> pd.DataFrame:
    """Generate synthetic Brent and WTI crude oil prices (USD/bbl).

    Brent range: 50-120, WTI tracks 2-5 below Brent.

    Args:
        start_date: First date (ISO format).
        periods: Number of daily observations.
        seed: Random seed.

    Returns:
        DataFrame with columns ``date``, ``brent``, ``wti``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    brent = _random_walk(
        n=periods, start=70.0, drift=0.0, volatility=1.5,
        mean_reversion=0.005, long_run_mean=75.0, rng=rng,
        floor=40.0, ceiling=140.0,
    )
    spread = rng.uniform(2.0, 5.0, size=periods)
    wti = brent - spread
    return pd.DataFrame({
        "date": dates,
        "brent": np.round(brent, 2),
        "wti": np.round(wti, 2),
    })


def generate_synthetic_sp500(
    start_date: str = "2019-01-01",
    periods: int = 1826,
    seed: int = 45,
) -> pd.DataFrame:
    """Generate synthetic S&P 500 index values.

    Range: 2500-5500 with upward drift.

    Args:
        start_date: First date (ISO format).
        periods: Number of daily observations.
        seed: Random seed.

    Returns:
        DataFrame with columns ``date`` and ``sp500``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    values = _random_walk(
        n=periods, start=2900.0, drift=0.5, volatility=30.0,
        mean_reversion=0.001, long_run_mean=4000.0, rng=rng,
        floor=2000.0, ceiling=6000.0,
    )
    return pd.DataFrame({"date": dates, "sp500": np.round(values, 2)})


def generate_synthetic_dxy(
    start_date: str = "2019-01-01",
    periods: int = 1826,
    seed: int = 46,
) -> pd.DataFrame:
    """Generate synthetic US Dollar Index values.

    Range: 88-115, mean-reverting around 100.

    Args:
        start_date: First date (ISO format).
        periods: Number of daily observations.
        seed: Random seed.

    Returns:
        DataFrame with columns ``date`` and ``dxy``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    values = _random_walk(
        n=periods, start=97.0, drift=0.0, volatility=0.3,
        mean_reversion=0.01, long_run_mean=100.0, rng=rng,
        floor=85.0, ceiling=120.0,
    )
    return pd.DataFrame({"date": dates, "dxy": np.round(values, 2)})


def generate_synthetic_gscpi(
    start_date: str = "2019-01-01",
    periods: int = 1826,
    seed: int = 47,
) -> pd.DataFrame:
    """Generate synthetic Global Supply Chain Pressure Index (monthly).

    Range: -2 to 4, centered around 0. Resampled to monthly frequency
    and then expanded to daily via forward-fill for merging convenience.

    Args:
        start_date: First date (ISO format).
        periods: Number of daily observations (used to compute month count).
        seed: Random seed.

    Returns:
        DataFrame with columns ``date`` and ``gscpi`` at monthly frequency.
    """
    rng = np.random.default_rng(seed)
    n_months = max(1, periods // 30)
    dates = pd.date_range(start=start_date, periods=n_months, freq="MS")
    values = _random_walk(
        n=n_months, start=0.0, drift=0.0, volatility=0.4,
        mean_reversion=0.1, long_run_mean=0.0, rng=rng,
        floor=-3.0, ceiling=5.0,
    )
    return pd.DataFrame({"date": dates, "gscpi": np.round(values, 2)})


def generate_synthetic_commodities(
    start_date: str = "2019-01-01",
    periods: int = 1826,
    seed: int = 48,
) -> pd.DataFrame:
    """Generate synthetic commodity price proxies (monthly).

    Iron ore (USD/ton): 80-180. Coal (USD/ton): 100-400.
    Grain index: 80-200.

    Args:
        start_date: First date (ISO format).
        periods: Number of daily observations (used to compute month count).
        seed: Random seed.

    Returns:
        DataFrame with columns ``date``, ``iron_ore``, ``coal``, ``grain``
        at monthly frequency.
    """
    rng = np.random.default_rng(seed)
    n_months = max(1, periods // 30)
    dates = pd.date_range(start=start_date, periods=n_months, freq="MS")

    iron_ore = _random_walk(
        n=n_months, start=120.0, drift=0.0, volatility=5.0,
        mean_reversion=0.05, long_run_mean=120.0, rng=rng,
        floor=70.0, ceiling=200.0,
    )
    coal = _random_walk(
        n=n_months, start=180.0, drift=0.0, volatility=10.0,
        mean_reversion=0.03, long_run_mean=200.0, rng=rng,
        floor=80.0, ceiling=450.0,
    )
    grain = _random_walk(
        n=n_months, start=120.0, drift=0.0, volatility=4.0,
        mean_reversion=0.05, long_run_mean=130.0, rng=rng,
        floor=70.0, ceiling=220.0,
    )
    return pd.DataFrame({
        "date": dates,
        "iron_ore": np.round(iron_ore, 2),
        "coal": np.round(coal, 2),
        "grain": np.round(grain, 2),
    })


def generate_all_synthetic_data(
    start_date: str = "2019-01-01",
    periods: int = 1826,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Generate all synthetic data sources and optionally save to disk.

    Args:
        start_date: First date (ISO format).
        periods: Number of daily observations (~5 years = 1826).
        output_dir: If provided, save each source as a CSV in this directory.
        seed: Base random seed (each source offsets by +1).

    Returns:
        Dictionary mapping source name to its DataFrame.
    """
    data: Dict[str, pd.DataFrame] = {
        "freight_rates": generate_synthetic_freight_rates(start_date, periods, seed),
        "bunker_fuel": generate_synthetic_bunker_fuel(start_date, periods, seed + 1),
        "crude_oil": generate_synthetic_crude_oil(start_date, periods, seed + 2),
        "sp500": generate_synthetic_sp500(start_date, periods, seed + 3),
        "dxy": generate_synthetic_dxy(start_date, periods, seed + 4),
        "gscpi": generate_synthetic_gscpi(start_date, periods, seed + 5),
        "commodities": generate_synthetic_commodities(start_date, periods, seed + 6),
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for name, df in data.items():
            df.to_csv(os.path.join(output_dir, f"synthetic_{name}.csv"), index=False)

    return data
