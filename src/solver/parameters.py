"""
src/solver/parameters.py

Real vessel and route parameter matrix — Day 2 complete.
Sources:
    Day 1: Researcher_2_Day_1_Domain_Matrix_REVIEWED.xlsx (2026-09-04)
    Day 2: Researcher_2_Day_2_Laycan_Matrix.xlsx          (2026-09-04)

SWAP STATUS:
    ✅  Vessel data:  real names, DWT, speed, fuel (from published vessel specs)
    ✅  Route data:   real ports, planning distances, corrected transit days,
                     real cargo lots (40–50 kt to fit the Alpha vessel pool)
    ✅  Laycan windows: REAL per-(vessel, route) windows from Day 2 xlsx.
                       Stored in LAYCAN_MATRIX for Big-M constraints in solver.py.
                       Route-level envelopes (min open / max close) also updated
                       on each ROUTES entry for display / backward compatibility.

ASSUMPTIONS (see workbook Assumptions sheet for full traceability):
    A-01  Vessel names are real published specifications; NOT currently available fleet.
    A-02  Fuel figures differ in type (IFO/VLSFO/HFO) and aux-inclusion — keep notes.
          P-08: No fuel-cost normalisation is applied in Alpha. The scoring formula
          uses only freight rates (p50/p10). If fuel cost enters the objective in
          Phase 2, all five fuel types must be converted to a common basis first.
    A-03  Route distances are Alpha planning estimates, not AIS-derived.
    A-04  Sailing days = distance(NM) ÷ 13 kn ÷ 24, rounded; no port/weather buffer.
          XLSX NOTE: Route_Matrix column header says "Approx Sailing Days" but the
          values (e.g. ~361.5, ~600.0) are actually HOURS, not days.
          transit_days in this file are correct (already divided by 24).
    A-05  Planning cargo (40–50 kt) is an Alpha modelling assumption, not a contract.
    A-06  10 routes prioritise Australia→India lanes + 2 North American diversity lanes.
    A-08  R-10 (Hampton Roads→Paradip) is a diversity/planning lane; less directly
          evidenced than the Hay Point benchmark routes.
    D2-A01 30-day horizon = 05-Sep-2026 through 04-Oct-2026 (base date 04-Sep-2026).
    D2-A02 Each feasible vessel-route pair receives a 3-day laycan window.
    D2-A03 Route windows are staggered through September.
    D2-A04 Vessel offsets vary laycan timing slightly per vessel readiness assumption.
    D2-A05 DWT >= planning cargo is the hard capacity rule (V-001 excluded everywhere).

All capacities in DWT (deadweight tonnage).
All speeds in knots. All fuel consumption in MT/day.
All laycan offsets are days from the solver run date (day 0 = 2026-09-04).
"""
from __future__ import annotations

# ── Risk aversion parameter ────────────────────────────────────────────────────
# Exposed here so the Streamlit slider can override it per-run.
# Formula: Score = p50 - LAMBDA * (p50 - p10)
LAMBDA: float = 0.5

# ── Voyage Cost Parameters (Phase 3) ───────────────────────────────────────────
DEMURRAGE_USD_PER_DAY: float = 15000.0
VLSFO_PRICE_USD_MT: float = 600.0
MGO_PRICE_USD_MT: float = 800.0

PORT_WAITING_DAYS: dict[str, float] = {
    "Paradip": 1.0,
    "Haldia": 2.0,
    "Visakhapatnam": 1.0,
    "Gangavaram": 1.0,
}

def calculate_daily_bunker_cost(
    vessel: dict,
    vlsfo_price: float = VLSFO_PRICE_USD_MT,
    mgo_price: float = MGO_PRICE_USD_MT,
) -> float:
    """Calculate daily bunker cost assuming NO SCRUBBER (forces VLSFO for main engine)."""
    # Main engine burns VLSFO (since HFO is not allowed without scrubber)
    main_engine_cost = vessel["laden_fuel_mt_day"] * vlsfo_price
    
    # Extract auxiliary fuel cost based on vessel notes
    vid = vessel["vessel_id"]
    if vid in ("V-001", "V-002", "V-004"):
        aux_cost = 0.1 * mgo_price
    elif vid == "V-003":
        aux_cost = 0.2 * mgo_price
    elif vid == "V-005":
        # 2.5 MT/day HFO auxiliary -> replace with MGO
        aux_cost = 2.5 * mgo_price
    else:
        aux_cost = 0.0
        
    return main_engine_cost + aux_cost

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
        # Laycan envelope = min(open) / max(close) across all feasible vessels — Day 2 xlsx
        "laycan_open": 1,              # V-005 earliest open (2026-09-05, offset 1)
        "laycan_close": 6,             # V-004 latest close  (2026-09-10, offset 6)
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
        "laycan_open": 3,              # V-005 earliest open (2026-09-07, offset 3)
        "laycan_close": 8,             # V-004 latest close  (2026-09-12, offset 8)
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
        "draft_check_required": True,  # Domain constraint: don't reject solely on DWT
        "laycan_open": 5,              # V-005 earliest open (2026-09-09, offset 5)
        "laycan_close": 10,            # V-004 latest close  (2026-09-14, offset 10)
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
        "laycan_open": 7,              # V-005 earliest open (2026-09-11, offset 7)
        "laycan_close": 12,            # V-004 latest close  (2026-09-16, offset 12)
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
        "laycan_open": 9,              # V-005 earliest open (2026-09-13, offset 9)
        "laycan_close": 14,            # V-004 latest close  (2026-09-18, offset 14)
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
        "laycan_open": 11,             # V-005 earliest open (2026-09-15, offset 11)
        "laycan_close": 16,            # V-004 latest close  (2026-09-20, offset 16)
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
        "draft_check_required": True,  # Domain constraint: don't reject solely on DWT
        "laycan_open": 13,             # V-005 earliest open (2026-09-17, offset 13)
        "laycan_close": 18,            # V-004 latest close  (2026-09-22, offset 18)
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
        "laycan_open": 15,             # V-005 earliest open (2026-09-19, offset 15)
        "laycan_close": 20,            # V-004 latest close  (2026-09-24, offset 20)
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
        "laycan_open": 18,             # V-005 earliest open (2026-09-22, offset 18)
        "laycan_close": 23,            # V-004 latest close  (2026-09-27, offset 23)
    },
    {
        "route_id": "R-10",
        "origin": "Hampton Roads, USA",
        "destination": "Paradip",
        "distance_nm": 9_500,
        "transit_days": 30,            # 9500 NM ÷ 13 kn ÷ 24 ≈ 30.4 d
        # ROUTING ASSUMPTION: 9,500 NM assumes Suez Canal transit.
        # Cape of Good Hope routing = ~11,200 NM (+1,700 NM, ~18% longer).
        # Suez adds ~$150k canal fees; Red Sea security risk applies.
        # Neither route is confirmed. Treat as planning estimate until
        # operator routing preference is stated.
        "cargo_requirement_dwt": 50_000,
        "cargo_type": "coking_coal",        # informational only — no vessel gear filter yet (TODO)
        "review_status": "FLAG",       # Diversity/planning lane — see Assumption A-08
        "laycan_open": 21,             # V-005 earliest open (2026-09-25, offset 21)
        "laycan_close": 26,            # V-004 latest close  (2026-09-30, offset 26)
    },
]

# ── Per-(vessel, route) laycan windows — Day 2 Big-M source ───────────────────
# Sourced from Researcher_2_Day_2_Laycan_Matrix.xlsx, sheet Day2_Laycan_Matrix.
# Key: (vessel_id, route_id)
# Value: dict with:
#   laycan_open  — earliest loading day offset from base date (day 0)
#   laycan_close — latest  loading day offset from base date (day 0)
#   feasible     — False = V-001 DWT < cargo; solver must force x[v,r,t] = 0 (Big-M)
#
# V-001 rows are kept explicitly so the Big-M constraint generator can
# FORBID those triples rather than silently skipping them.
LAYCAN_MATRIX: dict[tuple[str, str], dict] = {
    # ── R-01 Hay Point → Visakhapatnam ───────────────────────────────────────
    ("V-001", "R-01"): {"laycan_open":  1, "laycan_close":  3, "feasible": False},
    ("V-002", "R-01"): {"laycan_open":  2, "laycan_close":  4, "feasible": True},
    ("V-003", "R-01"): {"laycan_open":  3, "laycan_close":  5, "feasible": True},
    ("V-004", "R-01"): {"laycan_open":  4, "laycan_close":  6, "feasible": True},
    ("V-005", "R-01"): {"laycan_open":  1, "laycan_close":  3, "feasible": True},
    # ── R-02 Hay Point → Paradip ─────────────────────────────────────────────
    ("V-001", "R-02"): {"laycan_open":  3, "laycan_close":  5, "feasible": False},
    ("V-002", "R-02"): {"laycan_open":  4, "laycan_close":  6, "feasible": True},
    ("V-003", "R-02"): {"laycan_open":  5, "laycan_close":  7, "feasible": True},
    ("V-004", "R-02"): {"laycan_open":  6, "laycan_close":  8, "feasible": True},
    ("V-005", "R-02"): {"laycan_open":  3, "laycan_close":  5, "feasible": True},
    # ── R-03 Hay Point → Haldia ──────────────────────────────────────────────
    ("V-001", "R-03"): {"laycan_open":  5, "laycan_close":  7, "feasible": False},
    ("V-002", "R-03"): {"laycan_open":  6, "laycan_close":  8, "feasible": True},
    ("V-003", "R-03"): {"laycan_open":  7, "laycan_close":  9, "feasible": True},
    ("V-004", "R-03"): {"laycan_open":  8, "laycan_close": 10, "feasible": True},
    ("V-005", "R-03"): {"laycan_open":  5, "laycan_close":  7, "feasible": True},
    # ── R-04 Hay Point → Gangavaram ──────────────────────────────────────────
    ("V-001", "R-04"): {"laycan_open":  7, "laycan_close":  9, "feasible": False},
    ("V-002", "R-04"): {"laycan_open":  8, "laycan_close": 10, "feasible": True},
    ("V-003", "R-04"): {"laycan_open":  9, "laycan_close": 11, "feasible": True},
    ("V-004", "R-04"): {"laycan_open": 10, "laycan_close": 12, "feasible": True},
    ("V-005", "R-04"): {"laycan_open":  7, "laycan_close":  9, "feasible": True},
    # ── R-05 Gladstone → Visakhapatnam ───────────────────────────────────────
    ("V-001", "R-05"): {"laycan_open":  9, "laycan_close": 11, "feasible": False},
    ("V-002", "R-05"): {"laycan_open": 10, "laycan_close": 12, "feasible": True},
    ("V-003", "R-05"): {"laycan_open": 11, "laycan_close": 13, "feasible": True},
    ("V-004", "R-05"): {"laycan_open": 12, "laycan_close": 14, "feasible": True},
    ("V-005", "R-05"): {"laycan_open":  9, "laycan_close": 11, "feasible": True},
    # ── R-06 Gladstone → Paradip ─────────────────────────────────────────────
    ("V-001", "R-06"): {"laycan_open": 11, "laycan_close": 13, "feasible": False},
    ("V-002", "R-06"): {"laycan_open": 12, "laycan_close": 14, "feasible": True},
    ("V-003", "R-06"): {"laycan_open": 13, "laycan_close": 15, "feasible": True},
    ("V-004", "R-06"): {"laycan_open": 14, "laycan_close": 16, "feasible": True},
    ("V-005", "R-06"): {"laycan_open": 11, "laycan_close": 13, "feasible": True},
    # ── R-07 Gladstone → Haldia ──────────────────────────────────────────────
    ("V-001", "R-07"): {"laycan_open": 13, "laycan_close": 15, "feasible": False},
    ("V-002", "R-07"): {"laycan_open": 14, "laycan_close": 16, "feasible": True},
    ("V-003", "R-07"): {"laycan_open": 15, "laycan_close": 17, "feasible": True},
    ("V-004", "R-07"): {"laycan_open": 16, "laycan_close": 18, "feasible": True},
    ("V-005", "R-07"): {"laycan_open": 13, "laycan_close": 15, "feasible": True},
    # ── R-08 Dalrymple Bay → Paradip ─────────────────────────────────────────
    ("V-001", "R-08"): {"laycan_open": 15, "laycan_close": 17, "feasible": False},
    ("V-002", "R-08"): {"laycan_open": 16, "laycan_close": 18, "feasible": True},
    ("V-003", "R-08"): {"laycan_open": 17, "laycan_close": 19, "feasible": True},
    ("V-004", "R-08"): {"laycan_open": 18, "laycan_close": 20, "feasible": True},
    ("V-005", "R-08"): {"laycan_open": 15, "laycan_close": 17, "feasible": True},
    # ── R-09 Vancouver → Visakhapatnam ───────────────────────────────────────
    ("V-001", "R-09"): {"laycan_open": 18, "laycan_close": 20, "feasible": False},
    ("V-002", "R-09"): {"laycan_open": 19, "laycan_close": 21, "feasible": True},
    ("V-003", "R-09"): {"laycan_open": 20, "laycan_close": 22, "feasible": True},
    ("V-004", "R-09"): {"laycan_open": 21, "laycan_close": 23, "feasible": True},
    ("V-005", "R-09"): {"laycan_open": 18, "laycan_close": 20, "feasible": True},
    # ── R-10 Hampton Roads → Paradip ─────────────────────────────────────────
    ("V-001", "R-10"): {"laycan_open": 21, "laycan_close": 23, "feasible": False},
    ("V-002", "R-10"): {"laycan_open": 22, "laycan_close": 24, "feasible": True},
    ("V-003", "R-10"): {"laycan_open": 23, "laycan_close": 25, "feasible": True},
    ("V-004", "R-10"): {"laycan_open": 24, "laycan_close": 26, "feasible": True},
    ("V-005", "R-10"): {"laycan_open": 21, "laycan_close": 23, "feasible": True},
}

# ── Derived lookups (computed once at import time) ────────────────────────────
VESSEL_IDS: list[str] = [v["vessel_id"] for v in VESSELS]
ROUTE_IDS: list[str] = [r["route_id"] for r in ROUTES]
ROUTE_MAP: dict[str, dict] = {r["route_id"]: r for r in ROUTES}
VESSEL_MAP: dict[str, dict] = {v["vessel_id"]: v for v in VESSELS}

# Quick feasibility lookup: True if this (vessel, route) pair is physically possible.
# V-001 is False for all routes (DWT 38,854 MT < min cargo 40,000 MT).
LAYCAN_FEASIBLE: dict[tuple[str, str], bool] = {
    k: v["feasible"] for k, v in LAYCAN_MATRIX.items()
}
