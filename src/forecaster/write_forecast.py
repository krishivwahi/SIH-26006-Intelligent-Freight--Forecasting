"""
Forecast JSON writer conforming to the locked API contract.

Reads predictions from the QuantileForecaster, expands them across
the vessel-route matrix using route and vessel multipliers, and writes
the result to ``data/interim/freight_forecast_30d.json`` in the schema
defined by CONTRACT.md.
"""

import json
import os
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Default vessel and route IDs (Alpha scope: 5 vessels, 10 routes)
# ---------------------------------------------------------------------------

DEFAULT_VESSELS: List[str] = [f"V-{i:03d}" for i in range(1, 6)]
DEFAULT_ROUTES: List[str] = [f"R-{i:02d}" for i in range(1, 11)]

# Route multipliers — placeholder values pending calibration to the real matrix.
# IMPORTANT: values for R-05..R-08 were originally written assuming US East Coast
# origins (~9,000 NM). The real matrix swap moved those routes to Australian ports
# (~4,600-4,800 NM). Multiplier VALUES need recalibration before Phase 2 switch;
# COMMENTS below are updated to reflect the real port-to-port lanes.
DEFAULT_ROUTE_MULTIPLIERS: Dict[str, float] = {
    "R-01": 1.00,  # Hay Point, Australia -> Visakhapatnam       (Platts benchmark)
    "R-02": 1.05,  # Hay Point, Australia -> Paradip             (Platts benchmark)
    "R-03": 1.08,  # Hay Point, Australia -> Haldia
    "R-04": 0.98,  # Hay Point, Australia -> Gangavaram          (FLAG: Alpha extension)
    "R-05": 1.15,  # Gladstone, Australia -> Visakhapatnam       (value needs recalibration)
    "R-06": 1.18,  # Gladstone, Australia -> Paradip             (value needs recalibration)
    "R-07": 1.22,  # Gladstone, Australia -> Haldia              (FLAG: Alpha extension; value needs recalibration)
    "R-08": 1.12,  # Dalrymple Bay, Australia -> Paradip         (value needs recalibration)
    "R-09": 1.10,  # Vancouver, Canada -> Visakhapatnam          (met-coal; ~7,800 NM)
    "R-10": 1.14,  # Hampton Roads, USA -> Paradip               (FLAG: diversity lane; ~9,500 NM)
}

# Vessel multipliers — placeholder values, not yet calibrated to the real fleet.
# Real fleet is 1x Handysize + 4x Supramax (all similar size: 51k-57k DWT).
# The wide spread below (0.95-1.07) was designed for a Capesize/Panamax/Supramax
# tiered fleet that no longer matches. Recalibrate before Phase 2 switch.
DEFAULT_VESSEL_MULTIPLIERS: Dict[str, float] = {
    "V-001": 0.95,  # MV TS INDEX      -- Handysize, 38,854 DWT  [IDLE: below cargo floor of all routes]
    "V-002": 0.98,  # MV LOFTY MOUNTAIN   -- Supramax, 51,008 DWT
    "V-003": 1.00,  # MV IMPERIAL FORTUNE -- Supramax, 53,505 DWT
    "V-004": 1.03,  # MV VIENNA WOOD N    -- Supramax, 55,768 DWT
    "V-005": 1.07,  # MV NORTH QUAY       -- Supramax, 57,016 DWT (largest in fleet)
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
