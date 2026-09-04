"""
src/solver/parameters.py

Real vessel and route parameter matrix — frozen Day 1 candidate.
Source: Researcher_2_Day_1_Domain_Matrix_REVIEWED.xlsx (handed over 2026-09-04).

SWAP STATUS (Day 1 → Day 2 COMPLETE):
    ✅  Vessel data:    real names, DWT, speed, fuel (from published vessel specs)
    ✅  Route data:     real ports, planning distances, corrected transit days,
                        real cargo lots (40–50 kt to fit the Alpha vessel pool)
    ✅  Laycan windows: INTEGRATED from Researcher_2_Day_2_Laycan_Matrix.xlsx
                        (50 vessel-route pairs, 3-day laycan windows, 30d horizon).

ASSUMPTIONS (see workbook Assumptions sheet for full traceability):
    A-01  Vessel names are real published specifications; NOT currently available fleet.
    A-02  Fuel figures differ in type (IFO/VLSFO/HFO) and aux-inclusion — keep notes.
          P-08: No fuel-cost normalisation is applied in Alpha. The scoring formula
          uses only freight rates (p50/p10). If fuel cost enters the objective in
          Phase 2, all five fuel types must be converted to a common basis first.
    A-03  Route distances are Alpha planning estimates, not AIS-derived.
    A-04  Sailing days = distance(NM) ÷ 13 kn ÷ 24, rounded; no port/weather buffer.
    A-05  Planning cargo (40–50 kt) is an Alpha modelling assumption, not a contract.
    A-06  10 routes prioritise Australia→India lanes + 2 North American diversity lanes.
    A-08  R-10 (Hampton Roads→Paradip) is a diversity/planning lane; less directly
          evidenced than the Hay Point benchmark routes.

All capacities in DWT (deadweight tonnage).
All speeds in knots. All fuel consumption in MT/day.
All laycan offsets are days from the solver run date (day 0 = today).
"""
from __future__ import annotations

# ── Risk aversion parameter ────────────────────────────────────────────────────
# Exposed here so the Streamlit slider can override it per-run.
# Formula: Score = p50 - LAMBDA * (p50 - p10)
LAMBDA: float = 0.5

# ── Vessel matrix (5 vessels, Phase 1 Alpha scope) ────────────────────────────
# Fields:
#   vessel_id            : matches vessel_id in the JSON contract (frozen)
#   vessel_name          : real published vessel name (see Assumption A-01)
#   vessel_type          : bulk carrier class (Handysize / Supramax)
#   capacity_dwt         : maximum deadweight tonnage (MT)
#   min_cargo_dwt        : commercial loading floor; derived as ~85 % of DWT
#                          (industry convention) — not in source, treated as assumption
#   laden_speed_kn       : sea speed fully loaded (knots)
#   ballast_speed_kn     : sea speed in ballast (knots)
#   laden_fuel_mt_day    : main-engine fuel consumption laden (MT/day)
#   ballast_fuel_mt_day  : main-engine fuel consumption ballast (MT/day)
#   fuel_note            : source wording for fuel type / auxiliary inclusion
#                          (see Assumption A-02 — do NOT silently mix fuel prices)
#
# ⚠ V-001 (38,854 DWT Handysize) is too small for every route in the real matrix
#   (all cargo lots are 40,000–50,000 MT).  It will be filtered out by
#   _is_capacity_feasible() and show as idle.  This is correct domain behaviour.
VESSELS: list[dict] = [
    {
        "vessel_id": "V-001",
        "vessel_name": "MV TS INDEX",
        "vessel_type": "Handysize",
        "capacity_dwt": 38_854,
        "min_cargo_dwt": 33_000,       # ~85 % of DWT — Alpha assumption
        "laden_speed_kn": 13.5,
        "ballast_speed_kn": 14.0,
        "laden_fuel_mt_day": 22.5,
        "ballast_fuel_mt_day": 20.5,
        "fuel_note": "IFO/RMG380 + 0.1 MT/day MDO",
    },
    {
        "vessel_id": "V-002",
        "vessel_name": "MV LOFTY MOUNTAIN",
        "vessel_type": "Supramax",
        "capacity_dwt": 51_008,
        "min_cargo_dwt": 43_000,       # ~85 % of DWT — Alpha assumption
        "laden_speed_kn": 13.5,
        "ballast_speed_kn": 14.5,
        "laden_fuel_mt_day": 30.5,
        "ballast_fuel_mt_day": 30.5,
        "fuel_note": "VLSFO + 0.1 MT/day LSMGO",
    },
    {
        "vessel_id": "V-003",
        "vessel_name": "MV IMPERIAL FORTUNE",
        "vessel_type": "Supramax",
        "capacity_dwt": 53_505,
        "min_cargo_dwt": 45_000,       # ~85 % of DWT — Alpha assumption
        "laden_speed_kn": 13.5,
        "ballast_speed_kn": 14.0,
        "laden_fuel_mt_day": 33.0,
        "ballast_fuel_mt_day": 30.0,
        "fuel_note": "IFO + 0.2 MT/day MGO (ME+GE)",
    },
    {
        "vessel_id": "V-004",
        "vessel_name": "MV VIENNA WOOD N",
        "vessel_type": "Supramax",
        "capacity_dwt": 55_768,
        "min_cargo_dwt": 47_000,       # ~85 % of DWT — Alpha assumption
        "laden_speed_kn": 14.0,
        "ballast_speed_kn": 14.5,
        "laden_fuel_mt_day": 29.0,
        "ballast_fuel_mt_day": 29.0,
        "fuel_note": "IFO + 0.1 MT/day ULSMGO (at-sea incl. auxiliaries)",
    },
    {
        "vessel_id": "V-005",
        "vessel_name": "MV NORTH QUAY",
        "vessel_type": "Supramax",
        "capacity_dwt": 57_016,
        "min_cargo_dwt": 48_000,       # ~85 % of DWT — Alpha assumption
        "laden_speed_kn": 13.0,
        "ballast_speed_kn": 13.5,
        "laden_fuel_mt_day": 31.5,
        "ballast_fuel_mt_day": 30.5,
        "fuel_note": "HFO main engine + ~2.5 MT/day HFO auxiliary (listed separately)",
    },
]

# ── Route matrix (10 routes, Phase 1 Alpha scope) ─────────────────────────────
# Fields:
#   route_id              : matches route_id in the JSON contract (frozen)
#   origin                : loading port, "Port, Country" format
#   destination           : discharge port (East Coast of India)
#   distance_nm           : approximate planning sea distance (NM) — see A-03
#   transit_days          : integer sailing days = distance_nm ÷ 13 kn ÷ 24,
#                           rounded — no port/weather buffer (see A-04)
#   cargo_requirement_dwt : planning cargo lot (MT) — see A-05
#   cargo_type            : commodity type (informational; infrastructure filter
#                           deferred to TODO — Researcher 2 has not confirmed
#                           per-vessel gear; field kept for future wiring)
#   review_status         : 'KEEP' = strongly evidenced lane;
#                           'FLAG' = Alpha planning extension, label in UI
#   laycan_open           : ⏳ PLACEHOLDER — earliest load day (offset from day 0)
#   laycan_close          : ⏳ PLACEHOLDER — latest load day (offset from day 0)
#
# Laycan windows are spread across the 30-day horizon in ~3-day bands so the
# MILP is guaranteed feasible with dummy rates.  Replace with real windows
# when Researcher 2 supplies them.
ROUTES: list[dict] = [
    {
        "route_id": "R-01",
        "origin": "Hay Point, Australia",
        "destination": "Visakhapatnam",
        "distance_nm": 4_700,
        "transit_days": 15,            # 4700 NM ÷ 13 kn ÷ 24 ≈ 15.1 d
        "cargo_requirement_dwt": 50_000,
        "cargo_type": "coking_coal",        # informational only — no vessel gear filter yet (TODO)
        "review_status": "KEEP",       # Strongly evidenced Australia→India lane
        "laycan_open": 1,              # Day 2 Matrix: 2026-09-05 (envelope start)
        "laycan_close": 6,             # Day 2 Matrix: 2026-09-10 (envelope end)
    },
    {
        "route_id": "R-02",
        "origin": "Hay Point, Australia",
        "destination": "Paradip",
        "distance_nm": 4_500,
        "transit_days": 14,            # 4500 NM ÷ 13 kn ÷ 24 ≈ 14.4 d
        "cargo_requirement_dwt": 50_000,
        "cargo_type": "coking_coal",        # informational only — no vessel gear filter yet (TODO)
        "review_status": "KEEP",       # Platts/S&P named benchmark route
        "laycan_open": 3,              # Day 2 Matrix: 2026-09-07 (envelope start)
        "laycan_close": 8,             # Day 2 Matrix: 2026-09-12 (envelope end)
    },
    {
        "route_id": "R-03",
        "origin": "Hay Point, Australia",
        "destination": "Haldia",
        "distance_nm": 4_600,
        "transit_days": 15,            # 4600 NM ÷ 13 kn ÷ 24 ≈ 14.7 d
        "cargo_requirement_dwt": 40_000,
        "cargo_type": "coking_coal",        # informational only — no vessel gear filter yet (TODO)
        "review_status": "KEEP",       # Haldia explicitly in Australia→India assessment
        "laycan_open": 5,              # Day 2 Matrix: 2026-09-09 (envelope start)
        "laycan_close": 10,            # Day 2 Matrix: 2026-09-14 (envelope end)
    },
    {
        "route_id": "R-04",
        "origin": "Hay Point, Australia",
        "destination": "Gangavaram",
        "distance_nm": 4_800,
        "transit_days": 15,            # 4800 NM ÷ 13 kn ÷ 24 ≈ 15.4 d
        "cargo_requirement_dwt": 50_000,
        "cargo_type": "coking_coal",        # informational only — no vessel gear filter yet (TODO)
        "review_status": "FLAG",       # Alpha extension — not a Platts benchmark
        "laycan_open": 7,              # Day 2 Matrix: 2026-09-11 (envelope start)
        "laycan_close": 12,            # Day 2 Matrix: 2026-09-16 (envelope end)
    },
    {
        "route_id": "R-05",
        "origin": "Gladstone, Australia",
        "destination": "Visakhapatnam",
        "distance_nm": 4_800,
        "transit_days": 15,            # 4800 NM ÷ 13 kn ÷ 24 ≈ 15.4 d
        "cargo_requirement_dwt": 50_000,
        "cargo_type": "coking_coal",        # informational only — no vessel gear filter yet (TODO)
        "review_status": "KEEP",       # Real East Coast Australia→India trade lane
        "laycan_open": 9,              # Day 2 Matrix: 2026-09-13 (envelope start)
        "laycan_close": 14,            # Day 2 Matrix: 2026-09-18 (envelope end)
    },
    {
        "route_id": "R-06",
        "origin": "Gladstone, Australia",
        "destination": "Paradip",
        "distance_nm": 4_600,
        "transit_days": 15,            # 4600 NM ÷ 13 kn ÷ 24 ≈ 14.7 d
        "cargo_requirement_dwt": 50_000,
        "cargo_type": "coking_coal",        # informational only — no vessel gear filter yet (TODO)
        "review_status": "KEEP",       # Direct documented Gladstone→Paradip voyage
        "laycan_open": 11,             # Day 2 Matrix: 2026-09-15 (envelope start)
        "laycan_close": 16,            # Day 2 Matrix: 2026-09-20 (envelope end)
    },
    {
        "route_id": "R-07",
        "origin": "Gladstone, Australia",
        "destination": "Haldia",
        "distance_nm": 4_700,
        "transit_days": 15,            # 4700 NM ÷ 13 kn ÷ 24 ≈ 15.1 d
        "cargo_requirement_dwt": 40_000,
        "cargo_type": "coking_coal",        # informational only — no vessel gear filter yet (TODO)
        "review_status": "FLAG",       # Plausible but Alpha extension, not benchmark
        "laycan_open": 13,             # Day 2 Matrix: 2026-09-17 (envelope start)
        "laycan_close": 18,            # Day 2 Matrix: 2026-09-22 (envelope end)
    },
    {
        "route_id": "R-08",
        "origin": "Dalrymple Bay, Australia",
        "destination": "Paradip",
        "distance_nm": 4_450,
        "transit_days": 14,            # 4450 NM ÷ 13 kn ÷ 24 ≈ 14.3 d
        "cargo_requirement_dwt": 50_000,
        "cargo_type": "coking_coal",        # informational only — no vessel gear filter yet (TODO)
        "review_status": "KEEP",       # Dalrymple Bay explicit in Australia→India assessment
        "laycan_open": 15,             # Day 2 Matrix: 2026-09-19 (envelope start)
        "laycan_close": 20,            # Day 2 Matrix: 2026-09-24 (envelope end)
    },
    {
        "route_id": "R-09",
        "origin": "Vancouver, Canada",
        "destination": "Visakhapatnam",
        "distance_nm": 7_800,
        "transit_days": 25,            # 7800 NM ÷ 13 kn ÷ 24 = 25.0 d
        "cargo_requirement_dwt": 50_000,
        "cargo_type": "metallurgical_coal",  # informational only — no vessel gear filter yet (TODO)
        "review_status": "KEEP",       # Platts explicitly assesses Vancouver→Vizag met-coal
        "laycan_open": 18,             # Day 2 Matrix: 2026-09-22 (envelope start)
        "laycan_close": 23,            # Day 2 Matrix: 2026-09-27 (envelope end)
    },
    {
        "route_id": "R-10",
        "origin": "Hampton Roads, USA",
        "destination": "Paradip",
        "distance_nm": 9_500,
        "transit_days": 30,            # 9500 NM ÷ 13 kn ÷ 24 ≈ 30.4 d
        "cargo_requirement_dwt": 50_000,
        "cargo_type": "coking_coal",        # informational only — no vessel gear filter yet (TODO)
        "review_status": "FLAG",       # Diversity/planning lane — see Assumption A-08
        "laycan_open": 21,             # Day 2 Matrix: 2026-09-25 (envelope start)
        "laycan_close": 26,            # Day 2 Matrix: 2026-09-30 (envelope end)
    },
]

# ── Derived lookups (computed once at import time) ────────────────────────────
VESSEL_IDS: list[str] = [v["vessel_id"] for v in VESSELS]
ROUTE_IDS: list[str] = [r["route_id"] for r in ROUTES]
ROUTE_MAP: dict[str, dict] = {r["route_id"]: r for r in ROUTES}
VESSEL_MAP: dict[str, dict] = {v["vessel_id"]: v for v in VESSELS}

# ── Day 2 Laycan Matrix (Researcher 2 Domain Scheduling) ──────────────────────
# 50 combinations: 5 vessels × 10 routes with 3-day laycan windows in 30d horizon.
# Base date convention: Day 0 = 2026-09-04; Horizon = 2026-09-05 (Day 1) .. 2026-10-04 (Day 30).
_ROUTE_STARTS: dict[str, tuple[int, str]] = {
    "R-01": (1, "2026-09-05"),
    "R-02": (3, "2026-09-07"),
    "R-03": (5, "2026-09-09"),
    "R-04": (7, "2026-09-11"),
    "R-05": (9, "2026-09-13"),
    "R-06": (11, "2026-09-15"),
    "R-07": (13, "2026-09-17"),
    "R-08": (15, "2026-09-19"),
    "R-09": (18, "2026-09-22"),
    "R-10": (21, "2026-09-25"),
}

_VESSEL_OFFSETS: dict[str, int] = {
    "V-005": 0,
    "V-002": 1,
    "V-003": 2,
    "V-004": 3,
    "V-001": 0,
}

from datetime import date as _date, timedelta as _timedelta

_BASE_PLANNING_DATE = _date(2026, 9, 4)

def _generate_static_laycan_matrix() -> dict[tuple[str, str], dict]:
    """Generate the static 50-pair laycan lookup identical to Day2_Laycan_Matrix.xlsx."""
    matrix: dict[tuple[str, str], dict] = {}
    for r in ROUTES:
        rid = r["route_id"]
        start_offset, _ = _ROUTE_STARTS[rid]
        for v in VESSELS:
            vid = v["vessel_id"]
            v_offset = _VESSEL_OFFSETS[vid]
            open_offset = start_offset + v_offset
            close_offset = open_offset + 2  # 3-day window inclusive
            start_date = (_BASE_PLANNING_DATE + _timedelta(days=open_offset)).strftime("%Y-%m-%d")
            end_date = (_BASE_PLANNING_DATE + _timedelta(days=close_offset)).strftime("%Y-%m-%d")
            feasible = (vid != "V-001")
            reason = (
                "Vessel DWT is sufficient for planning cargo."
                if feasible
                else f"Vessel DWT {v['capacity_dwt']:,} MT is below planning cargo {r['cargo_requirement_dwt']:,} MT."
            )
            matrix[(vid, rid)] = {
                "vessel_id": vid,
                "route_id": rid,
                "laycan_open": open_offset,
                "laycan_close": close_offset,
                "laycan_start": start_date,
                "laycan_end": end_date,
                "laycan_days": 3,
                "feasible": feasible,
                "feasibility_reason": reason,
            }
    return matrix

LAYCAN_MATRIX: dict[tuple[str, str], dict] = _generate_static_laycan_matrix()
