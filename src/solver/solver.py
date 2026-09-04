"""
src/solver/solver.py

Deterministic MILP solver for vessel-route assignment.
Backend: PuLP 2.8.0 + CBC (bundled).

Alpha scope (AGENT_CONTEXT.md §3.4):
  - 5 vessels × 10 routes × 30 days = 1,500 binary variables
  - Objective: maximise Σ Score(v,r,t) * x[v,r,t]
  - Constraints:
      1. Each vessel used at most once across all routes and days
      2. Each route served by at most one vessel across ALL days in its laycan window
         (one route = one physical cargo demand = one ship total; the MILP picks the
          optimal (vessel, loading-day) pair from all candidates in the window)
      3. Laycan timing: x[v,r,t] = 0 outside [laycan_open, laycan_close] per route
      4. Capacity: vessel.capacity_dwt >= route.cargo_requirement_dwt (enforced by
         filtering infeasible pairings, not Big-M — Big-M comes with real data Day 2)

Day 1  → constant placeholder rates, fake laycan windows → proves mechanics work
Day 2  → real vessel/route matrix in parameters.py, real Big-M constraints
Day 3  → real forecast JSON replaces dummy JSON, no code change needed here

Date-offset convention (shared with dummy_generator.py):
    Day 0 = the solver run date (base_date, today).  NOT included in the horizon.
    Days 1..30 = the 30-day planning window (t+1 … t+30).
    _build_date_index() and dummy_generator.py both produce this same t+1..t+30
    range.  The laycan_open / laycan_close offsets in parameters.py are expressed
    in this same 1-indexed space (offset 0 = today, offset 1 = tomorrow).

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
    LAYCAN_MATRIX,
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

def _load_forecast(path: Path | str) -> list[dict]:
    """Load and validate the contract JSON.

    Args:
        path: Path to freight_forecast_30d.json (Path or str).

    Returns:
        List of forecast records.

    Raises:
        FileNotFoundError: If the JSON does not exist.
        ValueError: If required fields are missing from any record.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast JSON not found at {path}. "
            "Run dummy_generator.py or the real forecaster pipeline first."
        )
    with path.open("r", encoding="utf-8") as fh:
        records: list[dict] = json.load(fh)

    required_keys = {"date_index", "vessel_id", "route_id", "p10_rate", "p50_rate", "p90_rate"}
    seen: set[tuple[str, str, str]] = set()
    for i, rec in enumerate(records):
        missing = required_keys - rec.keys()
        if missing:
            raise ValueError(
                f"Record {i} in forecast JSON is missing fields: {missing}"
            )
        # Duplicate detection (TODO): silently overwriting rate_lookup would cause
        # the solver to optimise on stale/wrong rates if the ML pipeline has a merge bug.
        key = (rec["vessel_id"], rec["route_id"], rec["date_index"])
        if key in seen:
            raise ValueError(
                f"Duplicate record at index {i}: (vessel={rec['vessel_id']}, "
                f"route={rec['route_id']}, date={rec['date_index']}) appears more than once. "
                "Check the forecast pipeline for a merge or concatenation bug."
            )
        seen.add(key)
    return records


def _build_date_index(base_date: date, horizon: int) -> list[str]:
    """Return ISO date strings for offsets 1..horizon from base_date.

    Convention: offset 0 = base_date (today, the run date) is excluded.
    Offset 1 = tomorrow = the earliest possible loading date.
    This matches dummy_generator.py (range(1, 31)) and the laycan_open /
    laycan_close offsets in parameters.py, which share the same 1-indexed space.

    Example:
        base_date = 2026-09-04, horizon = 30
        → ["2026-09-05", "2026-09-06", …, "2026-10-04"]  (30 dates, t+1..t+30)
    """
    return [(base_date + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(1, horizon + 1)]


def _is_laycan_valid(
    day_offset: int,
    route: dict,
    vessel_id: str | None = None,
    laycan_matrix: dict[tuple[str, str], dict] | None = None,
) -> bool:
    """Return True if day_offset falls within the vessel-route laycan window.

    If a pairwise laycan matrix is provided (Day 2 Laycan Matrix),
    checks the specific (vessel, route) 3-day laycan window and feasibility.
    Otherwise falls back to route-level [laycan_open, laycan_close].
    """
    if vessel_id is not None and laycan_matrix is not None:
        key = (vessel_id, route.get("route_id", ""))
        if key in laycan_matrix:
            entry = laycan_matrix[key]
            if not entry.get("feasible", True):
                return False
            return entry["laycan_open"] <= day_offset <= entry["laycan_close"]
    return route.get("laycan_open", 0) <= day_offset <= route.get("laycan_close", 30)


def _is_capacity_feasible(vessel: dict, route: dict) -> bool:
    """Return True if the vessel can physically carry the route's cargo lot."""
    return vessel["capacity_dwt"] >= route["cargo_requirement_dwt"]


def _is_min_cargo_feasible(vessel: dict, route: dict) -> bool:
    """Return True if the route's cargo lot meets the vessel's commercial loading floor.

    P-07 note: this filter is defined but NOT wired into the feasible-triple loop.
    Wiring it with the current Alpha matrix makes R-03 and R-07 (40,000 MT cargo)
    infeasible for all vessels, because every Supramax min_cargo_dwt (~85 % of DWT)
    exceeds 40,000 MT.  The team must either:
      a) Lower some cargo lots to match real vessel floors, OR
      b) Confirm that Supramaxes do accept 40,000 MT lots on these lanes.
    Once decided, replace the _is_capacity_feasible check below with both:
        if not _is_capacity_feasible(vessel, route): continue
        if not _is_min_cargo_feasible(vessel, route): continue
    """
    return route["cargo_requirement_dwt"] >= vessel.get("min_cargo_dwt", 0)


# ── Public API ─────────────────────────────────────────────────────────────────

def solve(
    lam: float = LAMBDA,
    forecast_path: Path | str = FORECAST_JSON_PATH,
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
    #   a) the day_offset is within the vessel-route's 3-day laycan window
    #   b) the vessel has enough capacity for the route's cargo
    #   c) a score exists in the lookup (i.e. the forecast JSON covers it)
    feasible: list[tuple[str, str, str]] = []
    for vessel in VESSELS:
        vid = vessel["vessel_id"]
        for route in ROUTES:
            rid = route["route_id"]
            if not _is_capacity_feasible(vessel, route):
                continue
            for offset, date_str in enumerate(date_strings, start=1):
                if not _is_laycan_valid(offset, route, vessel_id=vid, laycan_matrix=LAYCAN_MATRIX):
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

    # Constraint 2: each route (= one cargo demand) served by exactly one vessel
    # across ALL days in its laycan window.  The MILP evaluates every feasible
    # (vessel, loading-day) combination and selects the single highest-scoring one.
    # Previously this was scoped per (route, day), allowing multiple vessels to be
    # assigned to the same route on different days — incorrect domain behaviour.
    for route in ROUTES:
        rid = route["route_id"]
        route_vars = [x[k] for k in feasible if k[1] == rid]
        if route_vars:
            prob += pulp.lpSum(route_vars) <= 1, f"OneVesselPerRoute_{rid}"

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
    # TODO (CBC truncated solve): if the solver hits timeLimit it returns the best
    # feasible solution found but still sets prob.status = LpStatusOptimal.
    # Detect this by comparing prob.status vs prob.sol_status and override:
    #   if prob.status == pulp.LpStatusOptimal and prob.sol_status != 1:
    #       status_str = "Time-limited (best found)"
    # Not triggered at 1,200 variables, but wired here for Beta readiness.

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
                        # P-06: transit_days is planning metadata (informational).
                        # It is NOT enforced as a constraint in the MILP — the
                        # single-assignment-per-vessel constraint already prevents
                        # double-use within the 30-day horizon for Alpha scope.
                        "transit_days": route_info["transit_days"],
                        # P-05: review_status and cargo_type passed to UI so
                        # FLAG lanes can be visually distinguished from benchmarks.
                        "review_status": route_info.get("review_status", "KEEP"),
                        "cargo_type": route_info.get("cargo_type", "coking_coal"),
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
