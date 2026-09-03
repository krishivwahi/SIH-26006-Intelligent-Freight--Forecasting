"""
src/solver/parameters.py

Placeholder vessel and route parameter matrix for Phase 1.

SWAP PROTOCOL (Day 1 → Day 2):
    When Researcher 2 hands off the frozen real matrix, update VESSELS and
    ROUTES in this file only. solver.py, risk.py, and app.py are untouched.

All capacities in DWT (deadweight tonnage).
All laycan offsets are days from the solver run date (day 0 = today).
"""
from __future__ import annotations

# ── Risk aversion parameter ────────────────────────────────────────────────────
# Exposed here so the Streamlit slider can override it per-run.
# Formula: Score = p50 - LAMBDA * (p50 - p10)
LAMBDA: float = 0.5

# ── Vessel matrix (5 vessels, Phase 1 Alpha scope) ────────────────────────────
# Fields:
#   vessel_id       : matches vessel_id in the JSON contract
#   capacity_dwt    : maximum deadweight tonnage
#   min_cargo_dwt   : minimum cargo the vessel will accept (commercial floor)
#
# Placeholder: all vessels are Panamax-class (65,000 DWT).
# Researcher 2 will replace these with real names, classes, and fuel curves.
VESSELS: list[dict] = [
    {"vessel_id": "V-001", "capacity_dwt": 65_000, "min_cargo_dwt": 50_000},
    {"vessel_id": "V-002", "capacity_dwt": 65_000, "min_cargo_dwt": 50_000},
    {"vessel_id": "V-003", "capacity_dwt": 70_000, "min_cargo_dwt": 55_000},
    {"vessel_id": "V-004", "capacity_dwt": 70_000, "min_cargo_dwt": 55_000},
    {"vessel_id": "V-005", "capacity_dwt": 75_000, "min_cargo_dwt": 60_000},
]

# ── Route matrix (10 routes, Phase 1 Alpha scope) ─────────────────────────────
# Fields:
#   route_id            : matches route_id in the JSON contract
#   origin              : loading port (origin country)
#   destination         : discharge port (East Coast of India)
#   transit_days        : sea passage days (one-way, placeholder)
#   cargo_requirement_dwt : cargo lot size this route must carry
#   laycan_open         : earliest day (offset from day 0) vessel may load
#   laycan_close        : latest day (offset from day 0) vessel must load
#
# Laycan windows are spread across the 30-day horizon in 3-day bands so the
# MILP is guaranteed feasible with the fake rates. Real windows from
# Researcher 2 will replace laycan_open / laycan_close only.
ROUTES: list[dict] = [
    {
        "route_id": "R-01",
        "origin": "Dalrymple Bay, Australia",
        "destination": "Visakhapatnam",
        "transit_days": 14,
        "cargo_requirement_dwt": 60_000,
        "laycan_open": 0,
        "laycan_close": 4,
    },
    {
        "route_id": "R-02",
        "origin": "Hay Point, Australia",
        "destination": "Paradip",
        "transit_days": 14,
        "cargo_requirement_dwt": 62_000,
        "laycan_open": 3,
        "laycan_close": 7,
    },
    {
        "route_id": "R-03",
        "origin": "Newcastle, Australia",
        "destination": "Haldia",
        "transit_days": 16,
        "cargo_requirement_dwt": 58_000,
        "laycan_open": 6,
        "laycan_close": 10,
    },
    {
        "route_id": "R-04",
        "origin": "Gladstone, Australia",
        "destination": "Gangavaram",
        "transit_days": 15,
        "cargo_requirement_dwt": 63_000,
        "laycan_open": 9,
        "laycan_close": 13,
    },
    {
        "route_id": "R-05",
        "origin": "Hampton Roads, USA",
        "destination": "Visakhapatnam",
        "transit_days": 22,
        "cargo_requirement_dwt": 68_000,
        "laycan_open": 12,
        "laycan_close": 16,
    },
    {
        "route_id": "R-06",
        "origin": "Norfolk, USA",
        "destination": "Paradip",
        "transit_days": 22,
        "cargo_requirement_dwt": 65_000,
        "laycan_open": 15,
        "laycan_close": 19,
    },
    {
        "route_id": "R-07",
        "origin": "Baltimore, USA",
        "destination": "Haldia",
        "transit_days": 23,
        "cargo_requirement_dwt": 64_000,
        "laycan_open": 18,
        "laycan_close": 22,
    },
    {
        "route_id": "R-08",
        "origin": "Vancouver, Canada",
        "destination": "Visakhapatnam",
        "transit_days": 20,
        "cargo_requirement_dwt": 70_000,
        "laycan_open": 21,
        "laycan_close": 25,
    },
    {
        "route_id": "R-09",
        "origin": "Prince Rupert, Canada",
        "destination": "Gangavaram",
        "transit_days": 19,
        "cargo_requirement_dwt": 67_000,
        "laycan_open": 24,
        "laycan_close": 28,
    },
    {
        "route_id": "R-10",
        "origin": "Roberts Bank, Canada",
        "destination": "Paradip",
        "transit_days": 20,
        "cargo_requirement_dwt": 66_000,
        "laycan_open": 27,
        "laycan_close": 29,
    },
]

# ── Derived lookups (computed once at import time) ────────────────────────────
VESSEL_IDS: list[str] = [v["vessel_id"] for v in VESSELS]
ROUTE_IDS: list[str] = [r["route_id"] for r in ROUTES]
ROUTE_MAP: dict[str, dict] = {r["route_id"]: r for r in ROUTES}
VESSEL_MAP: dict[str, dict] = {v["vessel_id"]: v for v in VESSELS}
