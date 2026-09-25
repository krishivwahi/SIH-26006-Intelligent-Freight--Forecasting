"""
dummy_generator.py

Generates a placeholder freight_forecast_30d.json for Phase 1 smoke-testing.

Intentional constraints applied so the JSON mirrors realistic feasibility:
  - Only (vessel, route) pairs where vessel.capacity_dwt >= route.cargo_requirement_dwt
    receive forecast records.  Infeasible pairs (e.g. V-001 Handysize on any
    50,000 MT route) are silently skipped — matching the solver's own pre-filter
    in _is_capacity_feasible().
  - A summary is printed so the operator knows exactly how many records were
    generated and which vessels were excluded.

Date-offset convention (matches solver.py _build_date_index exactly):
    Offset 0 = today (base_date) — excluded.
    Offsets 1..30 = the 30-day planning window (t+1 .. t+30).
    range(1, 31) here mirrors range(1, horizon + 1) in _build_date_index,
    so the date sets produced by generator and solver are always identical.

Rates: uniform random $15-25/t with P10 = 80 %, P90 = 130 % of P50.
       These are SYNTHETIC PROXY values, NOT real Baltic Exchange data.
"""
import json
import random
import warnings
from datetime import date, timedelta
from pathlib import Path

from src.solver.parameters import ROUTES, VESSELS


def generate_dummy_json(
    filepath: str | Path = Path(__file__).parent / "data" / "interim" / "freight_forecast_30d.json",
) -> None:
    """Generate placeholder forecast JSON, skipping capacity-infeasible pairs.

    Args:
        filepath: Destination path for the JSON file.  Defaults to the repo-root
                  relative path so it is safe to call from any working directory.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    base_date = date.today()
    records: list[dict] = []

    # Track which vessels have at least one feasible route for the summary.
    vessel_feasible_routes: dict[str, list[str]] = {v["vessel_id"]: [] for v in VESSELS}

    for day_offset in range(1, 31):  # offsets 1..30 = t+1..t+30; mirrors solver range(1, horizon+1)
        current_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        for route in ROUTES:
            rid = route["route_id"]
            for vessel in VESSELS:
                vid = vessel["vessel_id"]

                # ── Capacity pre-filter ───────────────────────────────────────
                # Mirror solver._is_capacity_feasible() so the JSON never
                # contains rates for (vessel, route) pairs the solver will
                # immediately discard.  This prevents silent dead-weight records.
                if vessel["capacity_dwt"] < route["cargo_requirement_dwt"]:
                    continue

                if rid not in vessel_feasible_routes[vid]:
                    vessel_feasible_routes[vid].append(rid)

                base_rate = round(random.uniform(15, 25), 2)
                records.append(
                    {
                        "date_index": current_date,
                        "vessel_id": vid,
                        "route_id": rid,
                        "p10_rate": round(base_rate * 0.8, 2),
                        "p50_rate": base_rate,
                        "p90_rate": round(base_rate * 1.3, 2),
                    }
                )

    # ── Warn about completely idle vessels ────────────────────────────────────
    idle_vessels = [
        vid for vid, routes in vessel_feasible_routes.items() if not routes
    ]
    if idle_vessels:
        warnings.warn(
            f"The following vessels have NO feasible routes (capacity too small "
            f"for every cargo lot in the route matrix): {idle_vessels}. "
            f"No forecast records were generated for them.  "
            f"Consider adjusting cargo_requirement_dwt for at least one route, "
            f"or replacing the vessel.",
            stacklevel=2,
        )

    filepath.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Generated {len(records)} records -> {filepath}")
    print(f"Date range : {base_date + timedelta(days=1)} to {base_date + timedelta(days=30)}")
    for vessel in VESSELS:
        vid = vessel["vessel_id"]
        n_routes = len(vessel_feasible_routes[vid])
        name = vessel["vessel_name"]
        cap = vessel["capacity_dwt"]
        status = f"{n_routes}/10 routes feasible" if n_routes > 0 else "[IDLE] no feasible routes"
        print(f"  {vid}  {name:<22}  {cap:>6} DWT  ->  {status}")


if __name__ == "__main__":
    generate_dummy_json()