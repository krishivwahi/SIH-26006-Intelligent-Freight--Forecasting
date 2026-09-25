"""
Hardcoded seasonal pressure calendar for dry bulk freight markets.

Maps each month to a pressure score (0.0 to 1.0) reflecting known cyclical
demand drivers affecting vessel availability and freight rates for coking coal
imports to India's East Coast.

This is domain knowledge encoded as a feature, not learned from data.
With only ~5 years of training data, there are at most 5 observations per month.
Encoding seasonal patterns as a prior gives LightGBM a head start.
"""

from typing import Dict, Tuple

import pandas as pd

# Month -> (pressure_label, pressure_score)
# Scores normalized to 0.0-1.0 scale
SEASONAL_PRESSURE: Dict[int, Tuple[str, float]] = {
    1:  ("High",   0.80),  # S. Hemisphere grain harvest; Chinese pre-CNY restocking
    2:  ("High",   0.85),  # Grain shipments peak; Chinese New Year disruption
    3:  ("Medium", 0.50),  # Grain harvest tailing off; end of Q1 steel cycle
    4:  ("Low",    0.25),  # Shoulder season; vessel availability improves
    5:  ("Low",    0.20),  # Pre-monsoon calm; lowest seasonal pressure
    6:  ("Medium", 0.55),  # Indian monsoon onset; East Coast port throughput drops
    7:  ("High",   0.85),  # Monsoon peak; N. Hemisphere grain harvest begins
    8:  ("High",   0.80),  # Monsoon continues; US/Canada grain competes for tonnage
    9:  ("High",   0.90),  # Monsoon tail; N. grain peak; highest vessel competition
    10: ("High",   0.85),  # Peak coking coal procurement; ports recovering monsoon
    11: ("Medium", 0.60),  # Procurement continues; pre-winter N. Hemisphere demand
    12: ("Medium", 0.50),  # Procurement tails off; holiday slowdowns
}


def get_seasonal_pressure(month: int) -> float:
    """Return the seasonal pressure score for a given month (1-12).

    Args:
        month: Month number (1 = January, 12 = December).

    Returns:
        Pressure score between 0.0 and 1.0.

    Raises:
        ValueError: If month is not in range 1-12.
    """
    if not isinstance(month, int) or month < 1 or month > 12:
        raise ValueError(f"Month must be an integer between 1 and 12, got {month}")
    return SEASONAL_PRESSURE[month][1]


def get_seasonal_label(month: int) -> str:
    """Return the seasonal pressure label for a given month (1-12).

    Args:
        month: Month number (1 = January, 12 = December).

    Returns:
        One of 'High', 'Medium', or 'Low'.

    Raises:
        ValueError: If month is not in range 1-12.
    """
    if not isinstance(month, int) or month < 1 or month > 12:
        raise ValueError(f"Month must be an integer between 1 and 12, got {month}")
    return SEASONAL_PRESSURE[month][0]


def add_seasonal_features(
    df: pd.DataFrame, date_column: str = "date"
) -> pd.DataFrame:
    """Add seasonal pressure score and label columns to a DataFrame.

    Args:
        df: DataFrame with a datetime-like column.
        date_column: Name of the datetime column.

    Returns:
        Copy of the DataFrame with ``seasonal_pressure`` (float) and
        ``seasonal_label`` (str) columns added.

    Raises:
        KeyError: If *date_column* does not exist in the DataFrame.
    """
    if date_column not in df.columns:
        raise KeyError(f"Column '{date_column}' not found in DataFrame")

    df = df.copy()
    dates = pd.to_datetime(df[date_column])
    df["seasonal_pressure"] = dates.dt.month.map(
        {m: v[1] for m, v in SEASONAL_PRESSURE.items()}
    )
    df["seasonal_label"] = dates.dt.month.map(
        {m: v[0] for m, v in SEASONAL_PRESSURE.items()}
    )
    return df
