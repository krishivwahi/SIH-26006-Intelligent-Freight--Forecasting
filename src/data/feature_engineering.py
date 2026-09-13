"""
Feature engineering pipeline for the freight forecasting model.

Transforms raw time-series data from multiple sources into a single
LightGBM-ready feature matrix. Each function is a composable transformation
that can be applied independently or chained via ``build_feature_matrix``.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.seasonal_calendar import add_seasonal_features


# ---------------------------------------------------------------------------
# Individual feature transformations
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Create lagged versions of a column.

    Args:
        df: Input DataFrame.
        column: Column to lag.
        lags: List of lag periods in days. Defaults to [7, 14, 30].

    Returns:
        DataFrame with new columns ``{column}_lag_{n}`` appended.

    Raises:
        KeyError: If *column* is missing from the DataFrame.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    if lags is None:
        lags = [7, 14, 30]

    df = df.copy()
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add rolling mean and rolling standard-deviation columns.

    Args:
        df: Input DataFrame.
        column: Column to compute rolling stats on.
        windows: Rolling window sizes in days. Defaults to [7, 14, 30].

    Returns:
        DataFrame with ``{column}_rmean_{w}`` and ``{column}_rstd_{w}``
        columns appended for each window *w*.

    Raises:
        KeyError: If *column* is missing from the DataFrame.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    if windows is None:
        windows = [7, 14, 30]

    df = df.copy()
    for w in windows:
        df[f"{column}_rmean_{w}"] = df[column].rolling(window=w, min_periods=1).mean()
        df[f"{column}_rstd_{w}"] = df[column].rolling(window=w, min_periods=1).std().fillna(0.0)
    return df


def add_momentum_features(
    df: pd.DataFrame,
    column: str,
    periods: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add percentage-change momentum features.

    Args:
        df: Input DataFrame.
        column: Column to compute momentum on.
        periods: Look-back periods in days. Defaults to [7, 14].

    Returns:
        DataFrame with ``{column}_mom_{p}`` columns appended.

    Raises:
        KeyError: If *column* is missing from the DataFrame.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    if periods is None:
        periods = [7, 14]

    df = df.copy()
    for p in periods:
        df[f"{column}_mom_{p}"] = df[column].pct_change(periods=p).fillna(0.0)
    return df


def add_calendar_features(
    df: pd.DataFrame, date_column: str = "date"
) -> pd.DataFrame:
    """Add calendar-derived features: month, day-of-week, day-of-year.

    Args:
        df: Input DataFrame with a datetime-like column.
        date_column: Name of the date column.

    Returns:
        DataFrame with ``month``, ``day_of_week``, and ``day_of_year``
        columns appended.

    Raises:
        KeyError: If *date_column* is missing from the DataFrame.
    """
    if date_column not in df.columns:
        raise KeyError(f"Column '{date_column}' not found in DataFrame")

    df = df.copy()
    dates = pd.to_datetime(df[date_column])
    df["month"] = dates.dt.month
    df["day_of_week"] = dates.dt.dayofweek
    df["day_of_year"] = dates.dt.dayofyear
    return df


# ---------------------------------------------------------------------------
# Data merging
# ---------------------------------------------------------------------------

def merge_data_sources(
    freight_rates: pd.DataFrame,
    bunker_fuel: Optional[pd.DataFrame] = None,
    crude_oil: Optional[pd.DataFrame] = None,
    sp500: Optional[pd.DataFrame] = None,
    dxy: Optional[pd.DataFrame] = None,
    gscpi: Optional[pd.DataFrame] = None,
    commodities: Optional[pd.DataFrame] = None,
    date_column: str = "date",
) -> pd.DataFrame:
    """Merge multiple data sources on date using left join + forward-fill.

    ``freight_rates`` is the anchor. Every other source is optional; if
    provided it is left-joined and forward-filled so that monthly series
    (GSCPI, commodities) propagate to daily granularity.

    Args:
        freight_rates: Must contain *date_column* and ``freight_rate``.
        bunker_fuel: Optional. Must contain *date_column* and ``ifo380_price``.
        crude_oil: Optional. Must contain *date_column*, ``brent``, ``wti``.
        sp500: Optional. Must contain *date_column* and ``sp500``.
        dxy: Optional. Must contain *date_column* and ``dxy``.
        gscpi: Optional. Must contain *date_column* and ``gscpi``.
        commodities: Optional. Must contain *date_column* and at least one
            of ``iron_ore``, ``coal``, ``grain``.
        date_column: Name of the date column common to all sources.

    Returns:
        Merged DataFrame sorted by date with NaNs forward-filled.

    Raises:
        KeyError: If *date_column* is missing from ``freight_rates``.
    """
    if date_column not in freight_rates.columns:
        raise KeyError(
            f"Column '{date_column}' not found in freight_rates DataFrame"
        )

    merged = freight_rates.copy()
    merged[date_column] = pd.to_datetime(merged[date_column])
    merged = merged.sort_values(date_column).reset_index(drop=True)

    optional_sources = [bunker_fuel, crude_oil, sp500, dxy, gscpi, commodities]
    for source in optional_sources:
        if source is not None:
            src = source.copy()
            src[date_column] = pd.to_datetime(src[date_column])
            merged = merged.merge(src, on=date_column, how="left")

    # Forward-fill monthly series (GSCPI, commodities) to daily granularity.
    # NOTE: We deliberately do NOT back-fill (bfill) — back-filling leading NaNs
    # with future values would introduce lookahead leakage into early training rows.
    # Residual leading NaNs are dropped by the valid_mask in build_feature_matrix.
    merged = merged.sort_values(date_column).reset_index(drop=True)
    merged = merged.ffill()
    return merged


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    merged_df: pd.DataFrame,
    target_column: str = "freight_rate",
    date_column: str = "date",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Transform a merged DataFrame into a feature matrix and target vector.

    Applies all feature transformations in sequence:
    1. Calendar features (month, day-of-week, day-of-year)
    2. Seasonal pressure index
    3. Lag features on target and available market columns
    4. Rolling statistics on target and available market columns
    5. Momentum features on target and available market columns

    Rows with NaN (due to lagging) are dropped from the output.

    Args:
        merged_df: Output of ``merge_data_sources``.
        target_column: Name of the target variable column.
        date_column: Name of the date column.

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the
        target Series, both with NaN rows dropped.

    Raises:
        KeyError: If *target_column* or *date_column* is missing.
    """
    if target_column not in merged_df.columns:
        raise KeyError(f"Column '{target_column}' not found in DataFrame")
    if date_column not in merged_df.columns:
        raise KeyError(f"Column '{date_column}' not found in DataFrame")

    df = merged_df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    # 1. Calendar features
    df = add_calendar_features(df, date_column)

    # 2. Seasonal pressure
    df = add_seasonal_features(df, date_column)

    # 3-5. Lag, rolling, and momentum features on target + market columns
    market_columns = [
        col for col in [
            target_column, "ifo380_price", "brent", "wti",
            "sp500", "dxy", "gscpi", "iron_ore", "coal", "grain",
        ]
        if col in df.columns
    ]

    for col in market_columns:
        df = add_lag_features(df, col, lags=[7, 14, 30])
        df = add_rolling_features(df, col, windows=[7, 14, 30])
        df = add_momentum_features(df, col, periods=[7, 14])

    # Drop non-feature columns
    drop_cols = [date_column, "seasonal_label"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Separate target
    y = df[target_column].copy()
    X = df.drop(columns=[target_column])

    # Drop rows with NaN from lagging
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    return X, y
