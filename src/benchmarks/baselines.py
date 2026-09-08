"""
src/benchmarks/baselines.py

Implementation of standard commercial maritime baseline chartering policies:
1. Naive Spot Chartering Policy:
   Charters available vessels on Day 1 of the route laycan at current spot rates.
2. Greedy Lowest-Rate Heuristic:
   Selects the day with the lowest spot freight rate in each laycan window greedily.
3. Multi-Policy Comparison Engine:
   Benchmarking against our PuLP Risk-Adjusted MILP Solver.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from src.benchmarks.metrics import calculate_cost_reduction, calculate_profit_uplift
from src.solver.parameters import (
    DEMURRAGE_USD_PER_DAY,
    HORIZON_DAYS,
    LAMBDA,
    LAYCAN_MATRIX,
    PORT_WAITING_DAYS,
    ROUTE_MAP,
    ROUTES,
    VESSEL_MAP,
    VESSELS,
    VLSFO_PRICE_USD_MT,
    calculate_daily_bunker_cost,
)
from src.solver.solver import FORECAST_JSON_PATH, AssignmentResult, _build_date_index, _load_forecast, solve


@dataclass
class PolicyResult:
    """Structured output from a benchmark chartering policy."""

    policy_name: str
    description: str
    assignments: list[dict[str, Any]] = field(default_factory=list)
    total_net_profit: float = 0.0
    total_voyage_cost: float = 0.0
    total_freight_revenue: float = 0.0
    total_cargo_dwt: int = 0
    num_assignments: int = 0


def _compute_triple_economics(
    vid: str,
    rid: str,
    date_str: str,
    p50_rate: float,
    vlsfo_price: float = VLSFO_PRICE_USD_MT,
) -> dict[str, Any]:
    """Calculate revenue, voyage cost, and net profit for an assignment triple."""
    vessel = VESSEL_MAP[vid]
    route = ROUTE_MAP[rid]

    freight_revenue = p50_rate * route["cargo_requirement_dwt"]

    wait_days = PORT_WAITING_DAYS.get(route.get("destination", ""), 0.0)
    total_days = route["transit_days"] + wait_days
    bunker_cost = total_days * calculate_daily_bunker_cost(vessel, vlsfo_price=vlsfo_price)
    port_delay_cost = wait_days * DEMURRAGE_USD_PER_DAY
    voyage_cost = bunker_cost + port_delay_cost

    net_profit = freight_revenue - voyage_cost

    return {
        "vessel_id": vid,
        "route_id": rid,
        "date": date_str,
        "p50_rate": round(p50_rate, 2),
        "cargo_dwt": route["cargo_requirement_dwt"],
        "transit_days": route["transit_days"],
        "port_waiting_days": wait_days,
        "freight_revenue": round(freight_revenue, 2),
        "voyage_cost": round(voyage_cost, 2),
        "net_profit": round(net_profit, 2),
        "origin": route["origin"],
        "destination": route["destination"],
    }


def run_naive_spot_policy(
    forecast_path: Path = FORECAST_JSON_PATH,
    base_date: date | None = None,
    vlsfo_price: float = VLSFO_PRICE_USD_MT,
    freight_multiplier: float = 1.0,
) -> PolicyResult:
    """Simulate Naive Spot Chartering: book vessels on Day 1 of laycan window.

    Real-world behavior:
        Chartering team books available vessels as early as possible (laycan open)
        at spot market rates, without optimization across forward rate cones.
    """
    if base_date is None:
        base_date = date.today()

    records = _load_forecast(forecast_path)
    date_strings = _build_date_index(base_date, HORIZON_DAYS)

    # Build rate lookup: (vid, rid, date_str) -> p50_rate
    rate_lookup: dict[tuple[str, str, str], float] = {}
    for rec in records:
        key = (rec["vessel_id"], rec["route_id"], rec["date_index"])
        rate_lookup[key] = float(rec["p50_rate"]) * freight_multiplier

    assignments: list[dict[str, Any]] = []
    used_vessels: set[str] = set()

    # Prioritize benchmark routes by earliest laycan open
    sorted_routes = sorted(ROUTES, key=lambda r: r["laycan_open"])

    for route in sorted_routes:
        rid = route["route_id"]
        # Find feasible, unassigned vessels
        candidate_vessels = [
            v["vessel_id"] for v in VESSELS
            if v["vessel_id"] not in used_vessels
            and LAYCAN_MATRIX.get((v["vessel_id"], rid), {}).get("feasible", False)
        ]
        if not candidate_vessels:
            continue

        # Naive picks first available vessel at laycan_open day
        vid = candidate_vessels[0]
        vr_entry = LAYCAN_MATRIX[(vid, rid)]
        open_offset = vr_entry["laycan_open"]

        if 1 <= open_offset <= len(date_strings):
            date_str = date_strings[open_offset - 1]
            p50 = rate_lookup.get((vid, rid, date_str), 20.0)
            trip_eco = _compute_triple_economics(
                vid=vid,
                rid=rid,
                date_str=date_str,
                p50_rate=p50,
                vlsfo_price=vlsfo_price,
            )
            assignments.append(trip_eco)
            used_vessels.add(vid)

    total_profit = sum(a["net_profit"] for a in assignments)
    total_cost = sum(a["voyage_cost"] for a in assignments)
    total_revenue = sum(a["freight_revenue"] for a in assignments)
    total_cargo = sum(a["cargo_dwt"] for a in assignments)

    return PolicyResult(
        policy_name="Naive Spot Chartering",
        description="Charters vessels on Day 1 of laycan window at spot rates without forward horizon intelligence.",
        assignments=assignments,
        total_net_profit=round(total_profit, 2),
        total_voyage_cost=round(total_cost, 2),
        total_freight_revenue=round(total_revenue, 2),
        total_cargo_dwt=total_cargo,
        num_assignments=len(assignments),
    )


def run_greedy_lowest_rate_policy(
    forecast_path: Path = FORECAST_JSON_PATH,
    base_date: date | None = None,
    vlsfo_price: float = VLSFO_PRICE_USD_MT,
    freight_multiplier: float = 1.0,
) -> PolicyResult:
    """Simulate Greedy Rate-Picking: selects lowest spot freight rate per route.

    Real-world behavior:
        Chartering manager looks at the 30-day forecast and greedily assigns the
        lowest rate day for each route, without jointly optimizing vessel availability,
        bunker burn on long routes (e.g. Canada/US vs Australia), or port delays.
    """
    if base_date is None:
        base_date = date.today()

    records = _load_forecast(forecast_path)
    date_strings = _build_date_index(base_date, HORIZON_DAYS)

    # Build rate lookup: (vid, rid, date_str) -> p50_rate
    rate_lookup: dict[tuple[str, str, str], float] = {}
    for rec in records:
        key = (rec["vessel_id"], rec["route_id"], rec["date_index"])
        rate_lookup[key] = float(rec["p50_rate"]) * freight_multiplier

    # Build all feasible candidates: (vid, rid, date_str, p50)
    all_candidates = []
    for vid in [v["vessel_id"] for v in VESSELS]:
        for route in ROUTES:
            rid = route["route_id"]
            vr_entry = LAYCAN_MATRIX.get((vid, rid))
            if not vr_entry or not vr_entry["feasible"]:
                continue
            lc_open = vr_entry["laycan_open"]
            lc_close = vr_entry["laycan_close"]
            for offset in range(lc_open, lc_close + 1):
                if 1 <= offset <= len(date_strings):
                    d_str = date_strings[offset - 1]
                    rate = rate_lookup.get((vid, rid, d_str))
                    if rate is not None:
                        all_candidates.append((vid, rid, d_str, rate))

    # Greedy sort: lowest spot rate first
    all_candidates.sort(key=lambda x: x[3])

    assignments: list[dict[str, Any]] = []
    used_vessels: set[str] = set()
    used_routes: set[str] = set()

    for vid, rid, date_str, rate in all_candidates:
        if vid in used_vessels or rid in used_routes:
            continue
        trip_eco = _compute_triple_economics(
            vid=vid,
            rid=rid,
            date_str=date_str,
            p50_rate=rate,
            vlsfo_price=vlsfo_price,
        )
        assignments.append(trip_eco)
        used_vessels.add(vid)
        used_routes.add(rid)

    total_profit = sum(a["net_profit"] for a in assignments)
    total_cost = sum(a["voyage_cost"] for a in assignments)
    total_revenue = sum(a["freight_revenue"] for a in assignments)
    total_cargo = sum(a["cargo_dwt"] for a in assignments)

    return PolicyResult(
        policy_name="Greedy Lowest-Rate Heuristic",
        description="Greedily books the lowest spot freight rate day without holistic fleet, bunker, or delay coordination.",
        assignments=assignments,
        total_net_profit=round(total_profit, 2),
        total_voyage_cost=round(total_cost, 2),
        total_freight_revenue=round(total_revenue, 2),
        total_cargo_dwt=total_cargo,
        num_assignments=len(assignments),
    )


def compare_all_policies(
    ai_result: AssignmentResult | None = None,
    lam: float = LAMBDA,
    forecast_path: Path = FORECAST_JSON_PATH,
    base_date: date | None = None,
    vlsfo_price: float = VLSFO_PRICE_USD_MT,
    freight_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Execute all policies and compute quantitative benchmark comparisons."""
    if ai_result is None:
        ai_result = solve(
            lam=lam,
            forecast_path=forecast_path,
            base_date=base_date,
            vlsfo_price=vlsfo_price,
            freight_multiplier=freight_multiplier,
        )

    naive = run_naive_spot_policy(
        forecast_path=forecast_path,
        base_date=base_date,
        vlsfo_price=vlsfo_price,
        freight_multiplier=freight_multiplier,
    )

    greedy = run_greedy_lowest_rate_policy(
        forecast_path=forecast_path,
        base_date=base_date,
        vlsfo_price=vlsfo_price,
        freight_multiplier=freight_multiplier,
    )

    ai_profit = sum(r.get("net_profit", 0.0) for r in ai_result.assignments)
    ai_cost = sum(r.get("voyage_cost", 0.0) for r in ai_result.assignments)
    ai_revenue = sum(r.get("net_profit", 0.0) + r.get("voyage_cost", 0.0) for r in ai_result.assignments)
    ai_cargo = sum(r.get("cargo_dwt", 0) for r in ai_result.assignments)

    ai_policy = PolicyResult(
        policy_name="AI Risk-Adjusted MILP Engine",
        description="Jointly optimizes downside risk (λ), 3-day laycans, bunker fuel burn, and Indian port demurrage.",
        assignments=ai_result.assignments,
        total_net_profit=round(ai_profit, 2),
        total_voyage_cost=round(ai_cost, 2),
        total_freight_revenue=round(ai_revenue, 2),
        total_cargo_dwt=ai_cargo,
        num_assignments=len(ai_result.assignments),
    )

    uplift_vs_naive = calculate_profit_uplift(ai_profit, naive.total_net_profit)
    uplift_vs_greedy = calculate_profit_uplift(ai_profit, greedy.total_net_profit)
    cost_savings_vs_naive = naive.total_voyage_cost - ai_cost
    cost_red_pct_vs_naive = calculate_cost_reduction(naive.total_voyage_cost, ai_cost)

    return {
        "ai": ai_policy,
        "naive": naive,
        "greedy": greedy,
        "uplift_vs_naive_pct": round(uplift_vs_naive, 2),
        "uplift_vs_greedy_pct": round(uplift_vs_greedy, 2),
        "cost_savings_vs_naive_usd": round(cost_savings_vs_naive, 2),
        "cost_reduction_vs_naive_pct": round(cost_red_pct_vs_naive, 2),
    }
