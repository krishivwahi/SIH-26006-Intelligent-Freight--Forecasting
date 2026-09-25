"""
Data ingestion and cleaning module for real-world proxy datasets.

Ingests, cleans, and standardizes data from:
1. FRED S&P 500 (SP500.csv) -> Macro equity index
2. FRED US Dollar Index (DXYUSDollar Index.csv) -> Trade-weighted USD
3. FRED Brent & WTI Crude (Crude_Oil_Combined.xlsx) -> Crude oil prices
4. USDA Bunker Fuel Prices (Daily_Bunker_Fuel_Prices_20260904.csv) -> IFO 380
5. NY Fed GSCPI (gscpi_data.xls) -> Global Supply Chain Pressure Index
6. World Bank Pink Sheet (CMO-Historical-Data-Monthly.xlsx) -> Coal, Iron Ore, Grain

Aligns all series with a calibrated freight rate target and merges them
into a single analysis-ready dataset.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from src.data.feature_engineering import merge_data_sources
from src.data.synthetic_data import generate_synthetic_freight_rates

# Default file paths
DEFAULT_RAW_DIR = os.path.join("data", "raw")
DEFAULT_PROCESSED_PATH = os.path.join("data", "processed", "merged_market_data.csv")


def load_sp500(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load and clean FRED S&P 500 daily series.

    Args:
        filepath: Path to SP500.csv. Defaults to data/raw/SP500.csv.

    Returns:
        DataFrame with columns ['date', 'sp500'].
    """
    if filepath is None:
        filepath = os.path.join(DEFAULT_RAW_DIR, "SP500.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"S&P 500 file not found at: {filepath}")

    df = pd.read_csv(filepath)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = "SP500" if "SP500" in df.columns else df.columns[1]

    clean_df = pd.DataFrame()
    clean_df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    clean_df["sp500"] = pd.to_numeric(df[val_col], errors="coerce")

    clean_df = clean_df.dropna(subset=["date", "sp500"])
    return clean_df.sort_values("date").reset_index(drop=True)


def load_dxy(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load and clean FRED Trade-Weighted US Dollar Index.

    Args:
        filepath: Path to DXYUSDollar Index.csv. Defaults to data/raw/DXYUSDollar Index.csv.

    Returns:
        DataFrame with columns ['date', 'dxy'].
    """
    if filepath is None:
        filepath = os.path.join(DEFAULT_RAW_DIR, "DXYUSDollar Index.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"DXY file not found at: {filepath}")

    df = pd.read_csv(filepath)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = "DTWEXBGS" if "DTWEXBGS" in df.columns else df.columns[1]

    clean_df = pd.DataFrame()
    clean_df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    clean_df["dxy"] = pd.to_numeric(df[val_col], errors="coerce")

    clean_df = clean_df.dropna(subset=["date", "dxy"])
    return clean_df.sort_values("date").reset_index(drop=True)


def load_crude_oil(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load and clean FRED Brent & WTI Crude oil prices.

    Args:
        filepath: Path to Crude_Oil_Combined.xlsx. Defaults to data/raw/Crude_Oil_Combined.xlsx.

    Returns:
        DataFrame with columns ['date', 'brent', 'wti'].
    """
    if filepath is None:
        filepath = os.path.join(DEFAULT_RAW_DIR, "Crude_Oil_Combined.xlsx")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Crude oil file not found at: {filepath}")

    df = pd.read_excel(filepath)
    date_col = "Date" if "Date" in df.columns else df.columns[0]

    brent_col = next((c for c in df.columns if "brent" in str(c).lower()), None)
    wti_col = next((c for c in df.columns if "wti" in str(c).lower()), None)

    clean_df = pd.DataFrame()
    clean_df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    clean_df["brent"] = pd.to_numeric(df[brent_col], errors="coerce") if brent_col else np.nan
    clean_df["wti"] = pd.to_numeric(df[wti_col], errors="coerce") if wti_col else np.nan

    clean_df = clean_df.dropna(subset=["date"])
    return clean_df.sort_values("date").reset_index(drop=True)


def load_bunker_fuel(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load and clean USDA daily bunker fuel prices.

    Extracts Intermediate Fuel Oil (IFO 380 cSt).

    Args:
        filepath: Path to Daily_Bunker_Fuel_Prices_20260904.csv.

    Returns:
        DataFrame with columns ['date', 'ifo380_price'].
    """
    if filepath is None:
        filepath = os.path.join(DEFAULT_RAW_DIR, "Daily_Bunker_Fuel_Prices_20260904.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Bunker fuel file not found at: {filepath}")

    df = pd.read_csv(filepath)
    date_col = "Day" if "Day" in df.columns else df.columns[0]

    ifo_col = next((c for c in df.columns if "380" in str(c)), None)
    if ifo_col is None:
        ifo_col = df.columns[-1]

    def _parse_currency(val) -> float:
        if pd.isna(val):
            return np.nan
        s = str(val).replace("$", "").replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return np.nan

    clean_df = pd.DataFrame()
    clean_df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    clean_df["ifo380_price"] = df[ifo_col].apply(_parse_currency)

    clean_df = clean_df.dropna(subset=["date", "ifo380_price"])
    return clean_df.sort_values("date").reset_index(drop=True)


def load_gscpi(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load and clean NY Fed Global Supply Chain Pressure Index (GSCPI).

    Args:
        filepath: Path to gscpi_data.xls. Defaults to data/raw/gscpi_data.xls.

    Returns:
        DataFrame with columns ['date', 'gscpi'].
    """
    if filepath is None:
        filepath = os.path.join(DEFAULT_RAW_DIR, "gscpi_data.xls")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GSCPI file not found at: {filepath}")

    # Read sheet "GSCPI Monthly Data" if available, else first sheet
    try:
        df = pd.read_excel(filepath, sheet_name="GSCPI Monthly Data")
    except Exception:
        df = pd.read_excel(filepath, sheet_name=0)

    date_col = "Date" if "Date" in df.columns else df.columns[0]
    val_col = "GSCPI" if "GSCPI" in df.columns else df.columns[1]

    clean_df = pd.DataFrame()
    clean_df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    clean_df["gscpi"] = pd.to_numeric(df[val_col], errors="coerce")

    clean_df = clean_df.dropna(subset=["date", "gscpi"])
    return clean_df.sort_values("date").reset_index(drop=True)


def load_commodities(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load and clean World Bank Pink Sheet monthly commodity prices.

    Extracts Australian Coal, Iron Ore (cfr spot), and Grain (Wheat/Maize).

    Args:
        filepath: Path to CMO-Historical-Data-Monthly.xlsx.

    Returns:
        DataFrame with columns ['date', 'iron_ore', 'coal', 'grain'].
    """
    if filepath is None:
        filepath = os.path.join(DEFAULT_RAW_DIR, "CMO-Historical-Data-Monthly.xlsx")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Commodities file not found at: {filepath}")

    try:
        df_raw = pd.read_excel(filepath, sheet_name="Monthly Prices")
    except Exception:
        df_raw = pd.read_excel(filepath, sheet_name=0)

    # In World Bank Pink Sheet, row 3 has product names, row 4 has units, row 5+ data
    header_idx = 3 if len(df_raw) > 5 else 0
    header_row = df_raw.iloc[header_idx]

    coal_col = None
    iron_col = None
    grain_col = None

    for col_idx in range(len(header_row)):
        name = str(header_row.iloc[col_idx]).lower()
        if "coal, australian" in name and coal_col is None:
            coal_col = col_idx
        elif "iron ore" in name and iron_col is None:
            iron_col = col_idx
        elif ("wheat" in name or "grain" in name) and grain_col is None:
            grain_col = col_idx

    # Fallbacks if exact strings not found
    if coal_col is None:
        coal_col = 5 if df_raw.shape[1] > 5 else 1
    if iron_col is None:
        iron_col = 63 if df_raw.shape[1] > 63 else min(2, df_raw.shape[1] - 1)
    if grain_col is None:
        grain_col = 37 if df_raw.shape[1] > 37 else min(3, df_raw.shape[1] - 1)

    data_start = header_idx + 2 if header_idx == 3 else 1
    df_data = df_raw.iloc[data_start:].copy()

    # Parse period string: e.g. "1960M01" -> "1960-01-01"
    period_str = df_data.iloc[:, 0].astype(str).str.replace("M", "-")
    parsed_dates = pd.to_datetime(period_str, format="%Y-%m", errors="coerce")
    # Fallback to general parser if format was different
    mask_nat = parsed_dates.isna()
    if mask_nat.any():
        parsed_dates[mask_nat] = pd.to_datetime(df_data.iloc[:, 0][mask_nat], errors="coerce")

    clean_df = pd.DataFrame({
        "date": parsed_dates,
        "coal": pd.to_numeric(df_data.iloc[:, coal_col], errors="coerce"),
        "iron_ore": pd.to_numeric(df_data.iloc[:, iron_col], errors="coerce"),
        "grain": pd.to_numeric(df_data.iloc[:, grain_col], errors="coerce"),
    })

    clean_df = clean_df.dropna(subset=["date"])
    return clean_df.sort_values("date").reset_index(drop=True)


def load_and_merge_all(
    raw_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    freight_target_df: Optional[pd.DataFrame] = None,
    freight_target_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load all raw data sources, align with freight target, and merge.

    If no freight target is passed, generates a calibrated synthetic freight
    series covering the 2024-09 to 2026-09 window per Alpha architecture.

    Args:
        raw_dir: Directory containing raw data files. Defaults to data/raw.
        output_path: Optional path to save merged CSV. Defaults to data/processed/...
        freight_target_df: Optional pre-loaded freight rate target DataFrame.
        freight_target_path: Optional CSV path to custom freight rate series.

    Returns:
        Analysis-ready merged DataFrame with all market signals.
    """
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR
    if output_path is None:
        output_path = DEFAULT_PROCESSED_PATH

    # 1. Load available external market signals
    sp500_df = None
    sp_path = os.path.join(raw_dir, "SP500.csv")
    if os.path.exists(sp_path):
        sp500_df = load_sp500(sp_path)

    dxy_df = None
    dxy_path = os.path.join(raw_dir, "DXYUSDollar Index.csv")
    if os.path.exists(dxy_path):
        dxy_df = load_dxy(dxy_path)

    crude_df = None
    crude_path = os.path.join(raw_dir, "Crude_Oil_Combined.xlsx")
    if os.path.exists(crude_path):
        crude_df = load_crude_oil(crude_path)

    bunker_df = None
    bunker_path = os.path.join(raw_dir, "Daily_Bunker_Fuel_Prices_20260904.csv")
    if os.path.exists(bunker_path):
        bunker_df = load_bunker_fuel(bunker_path)

    gscpi_df = None
    gscpi_path = os.path.join(raw_dir, "gscpi_data.xls")
    if os.path.exists(gscpi_path):
        gscpi_df = load_gscpi(gscpi_path)

    commodities_df = None
    cmo_path = os.path.join(raw_dir, "CMO-Historical-Data-Monthly.xlsx")
    if os.path.exists(cmo_path):
        commodities_df = load_commodities(cmo_path)

    # 2. Establish freight target
    if freight_target_df is not None:
        freight_rates = freight_target_df.copy()
    elif freight_target_path is not None and os.path.exists(freight_target_path):
        freight_rates = pd.read_csv(freight_target_path)
    else:
        # Determine start date and period from overlapping series
        start_date = "2024-09-01"
        periods = 735  # ~2 years to cover up to September 2026
        freight_rates = generate_synthetic_freight_rates(
            start_date=start_date, periods=periods, seed=42
        )

    # 3. Merge all series using the feature engineering module
    merged = merge_data_sources(
        freight_rates=freight_rates,
        bunker_fuel=bunker_df,
        crude_oil=crude_df,
        sp500=sp500_df,
        dxy=dxy_df,
        gscpi=gscpi_df,
        commodities=commodities_df,
    )

    # 4. Save to processed directory if specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged.to_csv(output_path, index=False)

    return merged
