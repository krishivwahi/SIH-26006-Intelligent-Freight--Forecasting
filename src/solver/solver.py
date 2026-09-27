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
Day 2  → real vessel/route matrix in parameters.py, real Big-M constraints from
          Researcher_2_Day_2_Laycan_Matrix.xlsx via LAYCAN_MATRIX in parameters.py
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
    DEMURRAGE_USD_PER_DAY,
    LAMBDA,
    LAYCAN_FEASIBLE,
    LAYCAN_MATRIX,
    PORT_WAITING_DAYS,
    ROUTE_MAP,
    ROUTES,
    VESSEL_MAP,
    VESSELS,
    VLSFO_PRICE_USD_MT,
    calculate_daily_bunker_cost,
)
from src.solver.country_risk import (
    CORRUPTION_WEIGHT,
    MAX_COUNTRY_RISK_DELAY_DAYS,
    WORKFORCE_WEIGHT,
    compute_all_route_delays,
    get_country_risk,
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
    origin, destination, cargo_requirement_dwt, capacity_dwt,
    country_risk_delay_days, country_risk_cost_usd, origin_country
    (see src/solver/country_risk.py)}"""

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
    vlsfo_price_used: float = VLSFO_PRICE_USD_MT
    freight_multiplier_used: float = 1.0


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


def _is_laycan_valid(day_offset: int, route: dict) -> bool:
    """Return True if day_offset falls within the route's laycan window."""
    return route["laycan_open"] <= day_offset <= route["laycan_close"]


def _is_capacity_feasible(vessel: dict, route: dict) -> bool:
    """Return True if the vessel can physically carry the route's cargo lot."""
    return vessel["capacity_dwt"] >= route["cargo_requirement_dwt"]


def _is_draft_feasible(vessel: dict, route: dict) -> bool:
    """Check physical port constraints like draft limit."""
    if route.get("destination") == "Haldia":
        # Official Haldia coking coal berth (Berth 4) has an 8.4m present depth limit.
        # Panamax vessels (>60,000 DWT) cannot physically berth fully laden.
        if vessel["capacity_dwt"] > 60000:
            return False
    return True


def _is_min_cargo_feasible(vessel: dict, route: dict) -> bool:
    """Return True if the route's cargo lot meets the vessel's commercial loading floor.

    SUPERSEDED FOR ALPHA by LAYCAN_MATRIX (parameters.py):
        Researcher_2_Day_2_Laycan_Matrix.xlsx explicitly marks V-002–V-005 as
        feasible=YES on R-03 and R-07 (both 40k MT cargo), even though every
        Supramax min_cargo_dwt (~85% DWT = 43k–48k MT) exceeds 40k MT.
        This is a domain confirmation (TODO option a) that Supramaxes accept
        40k MT lots on these lanes. The LAYCAN_MATRIX feasibility flag is the
        authoritative source for Alpha; this function is NOT wired into the
        feasible-triple loop.

    RETAIN FOR PHASE 2:
        Keep this function as documentation. If real commercial floors differ
        from the 85%-DWT planning assumption, wire both filters:
            if not _is_capacity_feasible(vessel, route): continue
            if not _is_min_cargo_feasible(vessel, route): continue
        and remove the relevant LAYCAN_MATRIX override.
    """
    return route["cargo_requirement_dwt"] >= vessel.get("min_cargo_dwt", 0)


def _country_risk_breakdown(origin: str) -> dict[str, float]:
    """Return the auditable corruption/workforce contributions for an origin."""
    risk = get_country_risk(origin)
    corruption_component = (100 - risk["cpi_score"]) / 100
    workforce_component = (5 - risk["lpi_score"]) / 4
    corruption_delay = CORRUPTION_WEIGHT * corruption_component * MAX_COUNTRY_RISK_DELAY_DAYS
    workforce_delay = WORKFORCE_WEIGHT * workforce_component * MAX_COUNTRY_RISK_DELAY_DAYS
    return {
        "cpi_score": float(risk["cpi_score"]),
        "lpi_score": float(risk["lpi_score"]),
        "corruption_component": round(corruption_component, 4),
        "workforce_component": round(workforce_component, 4),
        "corruption_delay_days": round(corruption_delay, 3),
        "workforce_delay_days": round(workforce_delay, 3),
        "semantic_delay_days": round(corruption_delay + workforce_delay, 3),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def solve(
    lam: float = LAMBDA,
    forecast_path: Path = FORECAST_JSON_PATH,
    base_date: date | None = None,
    vlsfo_price: float = VLSFO_PRICE_USD_MT,
    freight_multiplier: float = 1.0,
    apply_country_risk: bool = True,
) -> AssignmentResult:
    """Run the MILP solver and return a structured assignment result.

    Args:
        lam:                Risk aversion parameter λ ∈ [0, 1].
        forecast_path:      Path to the contract JSON (default: data/interim/...).
        base_date:          Day 0 of the 30-day horizon (default: today).
        vlsfo_price:        Bunker fuel price ($/MT) for VLSFO (default: parameters.py).
        freight_multiplier: What-if freight rate multiplier (default: 1.0 = baseline, 0.8 = -20% crash).
        apply_country_risk: If True (default), add each route's expected
                             origin-country corruption/workforce-efficiency
                             delay (src/solver/country_risk.py) on top of
                             PORT_WAITING_DAYS. Set False to reproduce the
                             pre-country-risk cost/objective values.

    Returns:
        AssignmentResult with assignments list, status, objective, and timing.
    """
    if base_date is None:
        base_date = date.today()

    # Expected extra loading-side delay per route, driven by the origin
    # country's corruption + workforce/logistics-efficiency composite.
    # See src/solver/country_risk.py for sources and formula.
    country_delay_lookup: dict[str, float] = (
        compute_all_route_delays(ROUTES) if apply_country_risk else {}
    )

    # ── 1. Load forecast and compute risk-adjusted scores ──────────────────
    records = _load_forecast(forecast_path)
    if abs(freight_multiplier - 1.0) > 1e-4:
        scaled_records = []
        for r in records:
            r_copy = dict(r)
            r_copy["p10_rate"] = float(r["p10_rate"]) * freight_multiplier
            r_copy["p50_rate"] = float(r["p50_rate"]) * freight_multiplier
            r_copy["p90_rate"] = float(r["p90_rate"]) * freight_multiplier
            scaled_records.append(r_copy)
        records = scaled_records

    score_lookup = compute_scores_bulk(records, lam=lam)

    # Build a secondary lookup for raw rates (needed in result rows)
    rate_lookup: dict[tuple[str, str, str], dict] = {}
    for rec in records:
        key = (rec["vessel_id"], rec["route_id"], rec["date_index"])
        rate_lookup[key] = rec

    date_strings = _build_date_index(base_date, HORIZON_DAYS)

    # ── 2. Build feasible (vessel, route, day_offset) triples ─────────────────
    # A triple is feasible if:
    #   a) LAYCAN_MATRIX marks (vessel, route) as feasible (DWT >= cargo)
    #   b) the day_offset falls within THIS vessel's specific laycan window
    #      (per-vessel windows from Day 2 xlsx, not the old route-level envelope)
    #   c) a score exists in the score_lookup (forecast JSON covers this triple)
    #
    # Infeasible (vessel, route) pairs (all V-001 rows) are NOT added to the
    # feasible list — they will be explicitly forbidden by Big-M upper_bound=0
    # constraints in step 3 so the MILP certificate is complete.
    feasible: list[tuple[str, str, str, str]] = []
    for vessel in VESSELS:
        vid = vessel["vessel_id"]
        for route in ROUTES:
            rid = route["route_id"]
            vr_entry = LAYCAN_MATRIX.get((vid, rid))
            if vr_entry is None or not vr_entry["feasible"]:
                continue  # capacity-infeasible — forbidden by Big-M below
                
            if not _is_draft_feasible(vessel, route):
                continue  # physically impossible to berth — forbidden
            lc_open  = vr_entry["laycan_open"]
            lc_close = vr_entry["laycan_close"]
            for offset, date_str in enumerate(date_strings, start=1):
                if not (lc_open <= offset <= lc_close):
                    continue  # outside THIS vessel's laycan window
                base_key = (vid, rid, date_str)
                if base_key in score_lookup:
                    feasible.append((vid, rid, date_str, "Spot"))
                    feasible.append((vid, rid, date_str, "Advance"))

    # Pre-calculate net profit for each feasible assignment
    net_profit_lookup: dict[tuple[str, str, str, str], float] = {}
    for key in feasible:
        vid, rid, date_str, ctype = key
        vessel = VESSEL_MAP[vid]
        route = ROUTE_MAP[rid]
        risk_breakdown = _country_risk_breakdown(route["origin"])
        
        # Voyage cost calculation
        semantic_delay_days = (
            country_delay_lookup.get(rid, 0.0)
            if apply_country_risk
            else 0.0
        )
        wait_days = PORT_WAITING_DAYS.get(route.get("destination", ""), 0.0)
        total_port_delay_days = wait_days + semantic_delay_days
        total_days = route["transit_days"] + total_port_delay_days
        bunker_cost = total_days * calculate_daily_bunker_cost(vessel, vlsfo_price=vlsfo_price)
        port_delay_cost = total_port_delay_days * DEMURRAGE_USD_PER_DAY
        voyage_cost = bunker_cost + port_delay_cost
        
        # Freight cost (Charterer pays freight)
        if ctype == "Spot":
            eff_score = score_lookup[(vid, rid, date_str)]
        else:
            # Advance locks in Day 1 rate + 2% forward premium. No risk penalty applies.
            day1_rate = float(rate_lookup[(vid, rid, date_strings[0])]["p50_rate"])
            eff_score = (day1_rate * 1.02)
            
        freight_cost = eff_score * route["cargo_requirement_dwt"]

        total_landed_cost = freight_cost + voyage_cost
        net_profit_lookup[key] = total_landed_cost

    # ── 3. Build PuLP problem ───────────────────────────────────────────────────
    prob = pulp.LpProblem("VesselRouteAssignment", pulp.LpMinimize)

    # Binary variable for each feasible assignment
    x: dict[tuple[str, str, str, str], pulp.LpVariable] = {}
    for key in feasible:
        var_name = "x_{}_{}_{}_{}".format(*key).replace("-", "_")
        x[key] = pulp.LpVariable(var_name, cat="Binary")

    # ── Big-M: explicitly forbid all capacity-infeasible (vessel, route) triples
    # The xlsx instructs: "FORBID (set assignment upper bound to 0)" for V-001.
    # We do this by adding a dedicated variable with ub=0 for every (V-001, route,
    # day) triple that is in the horizon but was excluded from the feasible list.
    # This makes the infeasibility certificate visible in the LP file and satisfies
    # the Day 2 handover instruction from Researcher 2 (TechLead_2_Handover sheet).
    for vessel in VESSELS:
        vid = vessel["vessel_id"]
        for route in ROUTES:
            rid = route["route_id"]
            if LAYCAN_FEASIBLE.get((vid, rid), True):
                continue  # feasible pair — already has binary vars above
                # Infeasible pair: create x=0 upper-bound variables for all horizon days
            vr_entry = LAYCAN_MATRIX.get((vid, rid), {})
            lc_open  = vr_entry.get("laycan_open",  1)
            lc_close = vr_entry.get("laycan_close", 30)
            for offset, date_str in enumerate(date_strings, start=1):
                if not (lc_open <= offset <= lc_close):
                    continue
                for ctype in ["Spot", "Advance"]:
                    key = (vid, rid, date_str, ctype)
                    var_name = "x_{}_{}_{}_{}".format(*key).replace("-", "_")
                    # upper_bound=0 hard-forces this variable to 0 — the Big-M equivalent
                    # for a binary when the right-hand-side bound is tighter than any M.
                    x[key] = pulp.LpVariable(var_name, lowBound=0, upBound=0, cat="Continuous")

    # Shortage variables for unserved cargo
    shortage_vars: dict[str, pulp.LpVariable] = {}
    for route in ROUTES:
        rid = route["route_id"]
        shortage_vars[rid] = pulp.LpVariable(f"shortage_{rid}", cat="Binary")

    # Objective: minimise Total Landed Cost (Freight + Voyage Costs) + Shortage Penalty ($10M per route)
    prob += (
        pulp.lpSum(net_profit_lookup[key] * x[key] for key in feasible)
        + pulp.lpSum(10_000_000 * shortage_vars[rid] for rid in shortage_vars)
    ), "TotalLandedCost"

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
        prob += pulp.lpSum(route_vars) + shortage_vars[rid] == 1, f"OneVesselPerRoute_{rid}"

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
                vid, rid, date_str, ctype = key
                base_key = (vid, rid, date_str)
                rec = rate_lookup.get(base_key, {})
                vessel_info = VESSEL_MAP[vid]
                route_info = ROUTE_MAP[rid]
                risk_breakdown = _country_risk_breakdown(route_info["origin"])
                # Re-calculate costs for output
                wait_days = PORT_WAITING_DAYS.get(route_info.get("destination", ""), 0.0)
                semantic_delay_days = (
                    country_delay_lookup.get(rid, 0.0)
                    if apply_country_risk
                    else 0.0
                )
                total_port_delay_days = wait_days + semantic_delay_days
                total_days = route_info["transit_days"] + total_port_delay_days
                daily_bunker = calculate_daily_bunker_cost(vessel_info, vlsfo_price=vlsfo_price)
                bunker_cost = total_days * daily_bunker
                port_delay_cost = total_port_delay_days * DEMURRAGE_USD_PER_DAY
                voyage_cost = bunker_cost + port_delay_cost
                
                if ctype == "Spot":
                    eff_score = score_lookup[base_key]
                else:
                    day1_rate = float(rate_lookup[(vid, rid, date_strings[0])]["p50_rate"])
                    eff_score = day1_rate * 1.02
                
                freight_cost = eff_score * route_info["cargo_requirement_dwt"]
                total_landed_cost = freight_cost + voyage_cost

                # Isolate the dollar cost attributable ONLY to the origin-country
                # corruption/workforce-efficiency delay (src/solver/country_risk.py),
                # separate from the destination-side PORT_WAITING_DAYS cost, so the
                # UI can state the country-semantics contribution explicitly rather
                # than burying it inside the combined voyage_cost figure.
                country_risk_cost_usd = semantic_delay_days * (daily_bunker + DEMURRAGE_USD_PER_DAY)

                assignments.append(
                    {
                        "vessel_id": vid,
                        "route_id": rid,
                        "date": date_str,
                        "contract_type": ctype,
                        "score": round(eff_score, 4),
                        "net_profit": round(total_landed_cost, 2), # UI expects net_profit field, repurposing for TCO
                        "voyage_cost": round(voyage_cost, 2),
                        "p50_rate": round(float(rec.get("p50_rate", 0)), 2),
                        "p10_rate": round(float(rec.get("p10_rate", 0)), 2),
                        "p90_rate": round(float(rec.get("p90_rate", 0)), 2),
                        "origin": route_info["origin"],
                        "destination": route_info["destination"],
                        "cargo_dwt": route_info["cargo_requirement_dwt"],
                        "vessel_capacity_dwt": vessel_info["capacity_dwt"],
                        "transit_days": route_info["transit_days"],
                        "port_waiting_days": round(total_port_delay_days, 3),
                        "total_port_delay_days": round(total_port_delay_days, 3),
                        "country_risk_delay_days": semantic_delay_days,
                        "country_risk_cost_usd": round(country_risk_cost_usd, 2),
                        "semantic_delay_days": semantic_delay_days,
                        "semantic_delay_cost": round(semantic_delay_days * DEMURRAGE_USD_PER_DAY, 2),
                        "origin_country": route_info["origin"].rsplit(",", 1)[-1].strip(),
                        **risk_breakdown,
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
        vlsfo_price_used=vlsfo_price,
        freight_multiplier_used=freight_multiplier,
    )