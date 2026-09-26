"""Load and validate vessel metadata from an external fleet export.

The solver uses a small, stable vessel dictionary. This module is the
boundary between that contract and provider-specific CSV/JSON exports from an
AIS or fleet-management system. No provider is assumed because each source
uses different names and licensing rules for the same fields.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SHIP_DATA_PATH = Path("data/raw/vessels.csv")

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "vessel_id": ("vessel_id", "ship_id", "fleet_id"),
    "imo": ("imo", "imo_number", "imo_no"),
    "vessel_name": ("vessel_name", "ship_name", "name"),
    "vessel_type": ("vessel_type", "ship_type", "type"),
    "capacity_dwt": ("capacity_dwt", "dwt", "deadweight", "deadweight_tons"),
    "laden_speed_kn": ("laden_speed_kn", "laden_speed", "speed_kn"),
    "ballast_speed_kn": ("ballast_speed_kn", "ballast_speed"),
    "laden_fuel_mt_day": ("laden_fuel_mt_day", "laden_fuel", "fuel_mt_day"),
    "ballast_fuel_mt_day": ("ballast_fuel_mt_day", "ballast_fuel"),
    "fuel_note": ("fuel_note", "fuel_type", "fuel"),
    "active": ("active", "is_active", "status"),
}

_REQUIRED_COLUMNS = {"vessel_id", "vessel_name", "capacity_dwt"}
_NUMERIC_COLUMNS = {
    "capacity_dwt",
    "laden_speed_kn",
    "ballast_speed_kn",
    "laden_fuel_mt_day",
    "ballast_fuel_mt_day",
}


def _normalise_column_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _read_source(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Ship data file not found: {source}")

    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    if source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("vessels", payload.get("ships", payload))
        if not isinstance(payload, list):
            raise ValueError("Ship JSON must contain a list or a 'vessels'/'ships' list")
        return pd.DataFrame(payload)
    raise ValueError("Ship data must be CSV or JSON")


def _canonicalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = {_normalise_column_name(column): column for column in frame.columns}
    renamed: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        matches = [lookup[alias] for alias in aliases if alias in lookup]
        if matches:
            renamed[matches[0]] = canonical
    return frame.rename(columns=renamed)


def _parse_active(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "inactive", "laid_up"}


def load_ship_catalog(path: str | Path = DEFAULT_SHIP_DATA_PATH) -> list[dict[str, Any]]:
    """Read a provider export and return solver-compatible vessel records.

    ``vessel_id``, ``vessel_name``, and ``capacity_dwt`` are required. Missing
    operational estimates are left as ``None`` rather than fabricated. The
    current Alpha solver only requires the identity and capacity fields.
    Inactive vessels are excluded from the returned catalog.
    """
    frame = _canonicalise_columns(_read_source(path))
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Ship data is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["vessel_id"] = frame["vessel_id"].astype("string").str.strip()
    frame["vessel_name"] = frame["vessel_name"].astype("string").str.strip()
    for column in _NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["vessel_id"].isna().any() or (frame["vessel_id"] == "").any():
        raise ValueError("Ship data contains an empty vessel_id")
    if frame["vessel_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["vessel_id"].duplicated(), "vessel_id"].tolist())
        raise ValueError(f"Duplicate vessel_id values: {duplicates}")
    if frame["capacity_dwt"].isna().any() or (frame["capacity_dwt"] <= 0).any():
        raise ValueError("capacity_dwt must contain positive numeric values")

    records: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        if not _parse_active(raw.get("active")):
            continue
        record: dict[str, Any] = {
            "vessel_id": str(raw["vessel_id"]),
            "vessel_name": str(raw["vessel_name"]),
            "vessel_type": raw.get("vessel_type"),
            "capacity_dwt": int(raw["capacity_dwt"]),
            "min_cargo_dwt": int(round(raw["capacity_dwt"] * 0.85)),
            "laden_speed_kn": raw.get("laden_speed_kn"),
            "ballast_speed_kn": raw.get("ballast_speed_kn"),
            "laden_fuel_mt_day": raw.get("laden_fuel_mt_day"),
            "ballast_fuel_mt_day": raw.get("ballast_fuel_mt_day"),
            "fuel_note": raw.get("fuel_note"),
        }
        if raw.get("imo") is not None and not pd.isna(raw["imo"]):
            record["imo"] = str(raw["imo"]).split(".")[0]
        records.append(record)
    return records


def load_ship_dataframe(path: str | Path = DEFAULT_SHIP_DATA_PATH) -> pd.DataFrame:
    """Return the validated catalog as a DataFrame for reporting and UI use."""
    return pd.DataFrame(load_ship_catalog(path))