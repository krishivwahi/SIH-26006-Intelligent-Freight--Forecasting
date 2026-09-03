"""
src/solver/solver.py

Deterministic MILP solver for vessel-route assignment.
Backend: PuLP 2.8.0 + CBC (bundled).

Alpha scope (AGENT_CONTEXT.md §3.4):
  - 5 vessels × 10 routes × 30 days = 1,500 binary variables
  - Objective: maximise Σ Score(v,r,t) * x[v,r,t]
  - Constraints:
      1. Each vessel used at most once across all routes and days
      2. Each (route, day) served by at most one vessel
      3. Laycan timing: x[v,r,t] = 0 outside [laycan_open, laycan_close] per route
      4. Capacity: vessel.capacity_dwt >= route.cargo_requirement_dwt (enforced by
         filtering infeasible pairings, not Big-M — Big-M comes with real data Day 2)

Day 1  → constant placeholder rates, fake laycan windows → proves mechanics work
Day 2  → real vessel/route matrix in parameters.py, real Big-M constraints
Day 3  → real forecast JSON replaces dummy JSON, no code change needed here

CONTRACT: reads from data/interim/freight_forecast_30d.json (locked schema).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pulp

from src.solver.parameters import (
    LAMBDA,
    ROUTE_MAP,
    ROUTES,
    VESSEL_MAP,
    VESSELS,
)
from src.solver.risk import compute_scores_bulk

# ── Constants ─────────────────────────────────────────────────────────────────
FORECAST_JSON_PATH = Path("data/interim/freight_forecast_30d.json")
HORIZON_DAYS = 30


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class AssignmentResult:
    """Structured solver output consumed by the Streamlit UI."""

    assignments: list[dict[str, Any]] = field(default_factory=list)
    """Each assigned triple: {vessel_id, route_id, date, score, p50_rate, p10_rate,
    origin, destination, cargo_requirement_dwt, capacity_dwt}"""

    solver_status: str = "Not Solved"
    """One of: 'Optimal', 'Infeasible', 'Unbounded', 'Not Solved'."""

    objective_value: float = 0.0
    """Sum of risk-adjusted scores for all assigned (vessel, route, day) triples."""

    solve_time_ms: float = 0.0
    """Wall-clock time for the CBC solve step (milliseconds)."""

    lambda_used: float = LAMBDA
    """The λ value used in this solve run."""

    num_vessels: int = len(VESSELS)
    num_routes: int = len(ROUTES)
    horizon_days: int = HORIZON_DAYS


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_forecast(path: Path) -> list[dict]:
    """Load and validate the contract JSON.

    Args:
        path: Path to freight_forecast_30d.json.

    Returns:
        List of forecast records.

    Raises:
        FileNotFoundError: If the JSON does not exist.
        ValueError: If required fields are missing from any record.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast JSON not found at {path}. "
            "Run dummy_generator.py or the real forecaster pipeline first."
        )
    with path.open("r", encoding="utf-8") as fh:
        records: list[dict] = json.load(fh)

    required_keys = {"date_index", "vessel_id", "route_id", "p10_rate", "p50_rate", "p90_rate"}
    for i, rec in enumerate(records):
        missing = required_keys - rec.keys()
        if missing:
            raise ValueError(
                f"Record {i} in forecast JSON is missing fields: {missing}"
            )
    return records


def _build_date_index(base_date: date, horizon: int) -> list[str]:
    """Return ISO date strings for days 0..horizon-1 from base_date."""
    return [(base_date + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(horizon)]


def _is_laycan_valid(day_offset: int, route: dict) -> bool:
    """Return True if day_offset falls within the route's laycan window."""
    return route["laycan_open"] <= day_offset <= route["laycan_close"]


def _is_capacity_feasible(vessel: dict, route: dict) -> bool:
    """Return True if the vessel can physically carry the route's cargo lot."""
    return vessel["capacity_dwt"] >= route["cargo_requirement_dwt"]


# ── Public API ─────────────────────────────────────────────────────────────────

def solve(
    lam: float = LAMBDA,
    forecast_path: Path = FORECAST_JSON_PATH,
    base_date: date | None = None,
) -> AssignmentResult:
    """Run the MILP solver and return a structured assignment result.

    Args:
        lam:           Risk aversion parameter λ ∈ [0, 1].
        forecast_path: Path to the contract JSON (default: data/interim/...).
        base_date:     Day 0 of the 30-day horizon (default: today).

    Returns:
        AssignmentResult with assignments list, status, objective, and timing.
    """
    if base_date is None:
        base_date = date.today()

    # ── 1. Load forecast and compute risk-adjusted scores ──────────────────
    records = _load_forecast(forecast_path)
    score_lookup = compute_scores_bulk(records, lam=lam)

    # Build a secondary lookup for raw rates (needed in result rows)
    rate_lookup: dict[tuple[str, str, str], dict] = {}
    for rec in records:
        key = (rec["vessel_id"], rec["route_id"], rec["date_index"])
        rate_lookup[key] = rec

    date_strings = _build_date_index(base_date, HORIZON_DAYS)

    # ── 2. Build feasible (vessel, route, day_offset) triples ─────────────
    # A triple is feasible if:
    #   a) the day_offset is within the route's laycan window
    #   b) the vessel has enough capacity for the route's cargo
    #   c) a score exists in the lookup (i.e. the forecast JSON covers it)
    feasible: list[tuple[str, str, str]] = []
    for vessel in VESSELS:
        vid = vessel["vessel_id"]
        for route in ROUTES:
            rid = route["route_id"]
            if not _is_capacity_feasible(vessel, route):
                continue
            for offset, date_str in enumerate(date_strings):
                if not _is_laycan_valid(offset, route):
                    continue
                key = (vid, rid, date_str)
                if key in score_lookup:
                    feasible.append(key)

    # ── 3. Build PuLP problem ──────────────────────────────────────────────
    prob = pulp.LpProblem("VesselRouteAssignment", pulp.LpMaximize)

    # Binary variable for each feasible triple
    x: dict[tuple[str, str, str], pulp.LpVariable] = {}
    for key in feasible:
        var_name = "x_{}_{}_{}" .format(*key).replace("-", "_")
        x[key] = pulp.LpVariable(var_name, cat="Binary")

    # Objective: maximise total risk-adjusted score
    prob += pulp.lpSum(score_lookup[key] * x[key] for key in feasible), "TotalRiskAdjustedScore"

    # Constraint 1: each vessel assigned to at most one route across all days
    for vessel in VESSELS:
        vid = vessel["vessel_id"]
        vessel_vars = [x[k] for k in feasible if k[0] == vid]
        if vessel_vars:
            prob += pulp.lpSum(vessel_vars) <= 1, f"OneAssignmentPerVessel_{vid}"

    # Constraint 2: each (route, day) served by at most one vessel
    for route in ROUTES:
        rid = route["route_id"]
        for date_str in date_strings:
            route_day_vars = [x[k] for k in feasible if k[1] == rid and k[2] == date_str]
            if route_day_vars:
                prob += pulp.lpSum(route_day_vars) <= 1, \
                    f"OneVesselPerRouteDay_{rid}_{date_str}"

    # ── 4. Solve ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    # msg=0 suppresses CBC stdout; timeLimit guards against runaway solves
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)
    solve_ms = (time.perf_counter() - t0) * 1000

    status_map = {
        pulp.LpStatusOptimal: "Optimal",
        pulp.LpStatusInfeasible: "Infeasible",
        pulp.LpStatusUnbounded: "Unbounded",
        pulp.LpStatusNotSolved: "Not Solved",
        pulp.LpStatusUndefined: "Undefined",
    }
    status_str = status_map.get(prob.status, "Unknown")

    # ── 5. Extract assignments ─────────────────────────────────────────────
    assignments: list[dict[str, Any]] = []
    if prob.status == pulp.LpStatusOptimal:
        for key, var in x.items():
            if pulp.value(var) is not None and pulp.value(var) > 0.5:
                vid, rid, date_str = key
                rec = rate_lookup.get(key, {})
                vessel_info = VESSEL_MAP[vid]
                route_info = ROUTE_MAP[rid]
                assignments.append(
                    {
                        "vessel_id": vid,
                        "route_id": rid,
                        "date": date_str,
                        "score": round(score_lookup[key], 4),
                        "p50_rate": round(float(rec.get("p50_rate", 0)), 2),
                        "p10_rate": round(float(rec.get("p10_rate", 0)), 2),
                        "p90_rate": round(float(rec.get("p90_rate", 0)), 2),
                        "origin": route_info["origin"],
                        "destination": route_info["destination"],
                        "cargo_dwt": route_info["cargo_requirement_dwt"],
                        "vessel_capacity_dwt": vessel_info["capacity_dwt"],
                        "transit_days": route_info["transit_days"],
                    }
                )
        # Sort by date then vessel for consistent display
        assignments.sort(key=lambda r: (r["date"], r["vessel_id"]))

    obj_val = pulp.value(prob.objective) or 0.0

    return AssignmentResult(
        assignments=assignments,
        solver_status=status_str,
        objective_value=round(obj_val, 4),
        solve_time_ms=round(solve_ms, 1),
        lambda_used=lam,
    )
