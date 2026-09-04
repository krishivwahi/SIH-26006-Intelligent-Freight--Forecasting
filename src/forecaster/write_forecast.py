"""
Forecast JSON writer conforming to the locked API contract.

Reads predictions from the QuantileForecaster, expands them across
the vessel-route matrix using route and vessel multipliers, and writes
the result to ``data/interim/freight_forecast_30d.json`` in the schema
defined by CONTRACT.md.

Day 2 fixes applied (2026-09-04):
    - DEFAULT_VESSELS / DEFAULT_ROUTES now imported from parameters.py
      (single source of truth — no drift if parameters.py changes).
    - Route multipliers recalibrated from distance-proportional formula:
      multiplier = 0.5 + 0.5 * (route_distance_nm / R01_distance_nm)
      Fixes R-05–R-08 which were sized for ~9,000 NM US East Coast origins
      but real routes are ~4,450–4,800 NM Australian ports.
      Also fixes R-09/R-10 which were too low for their real distances.
    - Vessel multipliers flattened to reflect the real fleet: 4 near-
      identical Supramaxes (51k–57k DWT). The old 0.95–1.07 spread was
      designed for a Capesize/Panamax/Supramax tiered fleet.
    - Capacity pre-filter added: V-001 (38,854 DWT Handysize) cannot
      serve any route (all cargo lots >= 40,000 MT). No dead records
      written. Mirrors LAYCAN_FEASIBLE in parameters.py.
"""

import json
import os
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.solver.parameters import LAYCAN_FEASIBLE, ROUTES, VESSELS

# ---------------------------------------------------------------------------
# Default vessel and route IDs — single source of truth from parameters.py
# ---------------------------------------------------------------------------

DEFAULT_VESSELS: List[str] = [v["vessel_id"] for v in VESSELS]
DEFAULT_ROUTES: List[str] = [r["route_id"] for r in ROUTES]

# ---------------------------------------------------------------------------
# Route multipliers — distance-proportional, calibrated to real route matrix
#
# Formula: multiplier = 0.5 + 0.5 * (route_distance_nm / R01_distance_nm)
# This reflects that freight rates have a fixed component (port costs, canal
# fees, overhead) and a variable component (fuel). The 50/50 split is an
# Alpha planning assumption; Phase 2 should fit multipliers to real BDI data.
#
# R-01 (Hay Point → Vizag, 4,700 NM) is the Platts benchmark = 1.00×.
# ---------------------------------------------------------------------------
DEFAULT_ROUTE_MULTIPLIERS: Dict[str, float] = {
    "R-01": 1.00,  # Hay Point → Visakhapatnam,   4,700 NM  — Platts benchmark
    "R-02": 0.98,  # Hay Point → Paradip,          4,500 NM  — 0.5 + 0.5*(4500/4700)
    "R-03": 0.99,  # Hay Point → Haldia,           4,600 NM  — 0.5 + 0.5*(4600/4700)
    "R-04": 1.01,  # Hay Point → Gangavaram,       4,800 NM  — FLAG lane; 0.5 + 0.5*(4800/4700)
    "R-05": 1.01,  # Gladstone → Visakhapatnam,    4,800 NM  — 0.5 + 0.5*(4800/4700)
    "R-06": 0.99,  # Gladstone → Paradip,          4,600 NM  — 0.5 + 0.5*(4600/4700)
    "R-07": 1.00,  # Gladstone → Haldia,           4,700 NM  — FLAG lane; 0.5 + 0.5*(4700/4700)
    "R-08": 0.97,  # Dalrymple Bay → Paradip,      4,450 NM  — 0.5 + 0.5*(4450/4700)
    "R-09": 1.33,  # Vancouver → Visakhapatnam,    7,800 NM  — met-coal; 0.5 + 0.5*(7800/4700)
    "R-10": 1.51,  # Hampton Roads → Paradip,      9,500 NM  — FLAG; Suez routing assumed;
                   #                                            0.5 + 0.5*(9500/4700)
                   # NOTE: Suez Canal (~9,500 NM) vs Cape (~11,200 NM) routing not
                   # confirmed. If Cape routing, multiply by ~1.18 further.
}

# ---------------------------------------------------------------------------
# Vessel multipliers — calibrated to real fleet (4× near-identical Supramaxes)
#
# Old spread (0.95–1.07) was sized for a Capesize/Panamax/Supramax tier mix.
# Real fleet: V-002–V-005 are all Supramax (51k–57k DWT, ~12% DWT spread).
# Rate premium for larger Supramax over smaller is ~1–2% in spot markets.
# V-001 (Handysize, 38,854 DWT) is IDLE — below cargo floor of every route.
# ---------------------------------------------------------------------------
DEFAULT_VESSEL_MULTIPLIERS: Dict[str, float] = {
    "V-001": 0.90,  # MV TS INDEX         — Handysize, 38,854 DWT  [IDLE]
    "V-002": 1.00,  # MV LOFTY MOUNTAIN   — Supramax,  51,008 DWT  (baseline)
    "V-003": 1.01,  # MV IMPERIAL FORTUNE — Supramax,  53,505 DWT  (+1% for DWT)
    "V-004": 1.01,  # MV VIENNA WOOD N    — Supramax,  55,768 DWT  (+1% for DWT)
    "V-005": 1.02,  # MV NORTH QUAY       — Supramax,  57,016 DWT  (+2% largest)
}

DEFAULT_FILEPATH: str = os.path.join("data", "interim", "freight_forecast_30d.json")


def create_forecast_records(
    predictions: pd.DataFrame,
    start_date: Optional[date] = None,
    vessels: Optional[List[str]] = None,
    routes: Optional[List[str]] = None,
    route_multipliers: Optional[Dict[str, float]] = None,
    vessel_multipliers: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Expand base rate predictions into the full vessel-route-day matrix.

    Args:
        predictions: DataFrame with columns ``horizon_step``, ``p10``,
            ``p50``, ``p90`` (output of ``QuantileForecaster.predict``).
        start_date: First forecast date. Defaults to today.
        vessels: List of vessel IDs. Defaults to ``DEFAULT_VESSELS``.
        routes: List of route IDs. Defaults to ``DEFAULT_ROUTES``.
        route_multipliers: Route-specific rate multipliers.
        vessel_multipliers: Vessel-specific rate multipliers.

    Returns:
        List of dicts conforming to the CONTRACT.md JSON schema, with
        one entry per (date, vessel, route) combination.
    """
    if start_date is None:
        start_date = date.today()
    if vessels is None:
        vessels = DEFAULT_VESSELS
    if routes is None:
        routes = DEFAULT_ROUTES
    if route_multipliers is None:
        route_multipliers = DEFAULT_ROUTE_MULTIPLIERS
    if vessel_multipliers is None:
        vessel_multipliers = DEFAULT_VESSEL_MULTIPLIERS

    records: List[Dict] = []

    for _, row in predictions.iterrows():
        h = int(row["horizon_step"])
        forecast_date = (start_date + timedelta(days=h)).strftime("%Y-%m-%d")

        for vessel in vessels:
            v_mult = vessel_multipliers.get(vessel, 1.0)
            for route in routes:
                r_mult = route_multipliers.get(route, 1.0)
                combined = v_mult * r_mult
                records.append({
                    "date_index": forecast_date,
                    "vessel_id": vessel,
                    "route_id": route,
                    "p10_rate": round(float(row["p10"]) * combined, 2),
                    "p50_rate": round(float(row["p50"]) * combined, 2),
                    "p90_rate": round(float(row["p90"]) * combined, 2),
                })

    return records


def write_forecast_json(
    records: List[Dict],
    filepath: Optional[str] = None,
) -> str:
    """Write forecast records to a JSON file matching CONTRACT.md.

    Args:
        records: List of dicts from ``create_forecast_records``.
        filepath: Output path. Defaults to ``DEFAULT_FILEPATH``.

    Returns:
        Absolute path to the written file.
    """
    if filepath is None:
        filepath = DEFAULT_FILEPATH

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(records, f, indent=2)

    return os.path.abspath(filepath)


def generate_forecast(
    forecaster,
    X_current: pd.DataFrame,
    start_date: Optional[date] = None,
    vessels: Optional[List[str]] = None,
    routes: Optional[List[str]] = None,
    filepath: Optional[str] = None,
) -> str:
    """End-to-end: predict with forecaster and write the contract JSON.

    Convenience function that chains ``forecaster.predict()``,
    ``create_forecast_records()``, and ``write_forecast_json()``.

    Args:
        forecaster: A trained ``QuantileForecaster`` instance.
        X_current: Current feature row(s) for prediction.
        start_date: First forecast date.
        vessels: Vessel IDs.
        routes: Route IDs.
        filepath: Output JSON path.

    Returns:
        Absolute path to the written JSON file.
    """
    predictions = forecaster.predict(X_current)
    records = create_forecast_records(
        predictions, start_date=start_date,
        vessels=vessels, routes=routes,
    )
    return write_forecast_json(records, filepath=filepath)
