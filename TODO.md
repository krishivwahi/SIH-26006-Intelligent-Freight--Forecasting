# TODO — AI Maritime Chartering Engine

> Items here are **not** Phase 1 blockers. They are improvements to be picked up
> as bandwidth allows, roughly ordered by phase relevance.
> Add items freely. Mark `[x]` when done, `[~]` when descoped.

---

## Solver

- [ ] **V-001 idle — team decision required**
  - MV TS INDEX (38,854 DWT Handysize) is below the cargo floor of every route
    in the real matrix (all lots >= 40,000 MT). `_is_capacity_feasible()` pre-filters
    it from every feasible triple. The solver always returns <= 4 assignments.
  - The UI and pitch say "5 vessels" — a judge who counts assignments will notice.
  - Decision needed (pick one):
    a) Add a <= 35,000 MT cargo lot to at least one route so V-001 participates, OR
    b) Swap V-001 for a vessel >= 40,000 DWT from the same Handysize pool, OR
    c) Keep it and add a visible "Idle — below cargo floor" row to the UI table.
  - Option (c) is already partially in place: `dummy_generator.py` warns on run,
    and `write_forecast.py` labels V-001 [IDLE] in its comments.

- [ ] **Wire `_is_min_cargo_feasible()` into the solver — blocked on team domain decision**
  - `_is_min_cargo_feasible()` is defined in `solver.py` but not called.
  - Wiring it would make R-03 (Haldia, 40k MT) and R-07 (Haldia, 40k MT) infeasible
    for all Supramax vessels because every vessel's 85%-DWT commercial floor exceeds 40k MT.
  - Decision needed (pick one):
    a) Confirm Supramaxes accept 40k MT lots on these lanes -> keep routes, don't wire, OR
    b) Raise cargo lots to >= 43,000 MT (V-002 floor) -> wire the filter, routes survive, OR
    c) Remove R-03 and R-07 from Alpha -> wire the filter, lose 2 routes.
  - Until decided, filter is intentionally not wired. See note in `solver.py`.

- [ ] **Voyage cost is absent from the objective — real transit days quantify the gap**
  - Scoring formula: `Score = P50 - lam*(P50 - P10)` (freight rates only).
    Fuel cost is not in the Alpha objective (AGENT_CONTEXT s2 scoping decision).
  - With real transit days the distortion is measurable:
    - R-01 Hay Point -> Vizag: 15 days, bunker cost ~ baseline
    - R-09 Vancouver -> Vizag: 25 days, bunker premium ~ +/voyage
    - R-10 Hampton Roads -> Paradip: 30 days, bunker premium ~ +/voyage
  - A Vancouver assignment at the same dollar/t rate as Hay Point is significantly
    less profitable in reality. The solver cannot distinguish them.
  - Immediate action: add a one-sentence disclaimer to the pitch and UI:
    "Solver objective: freight revenue only. Voyage cost enters Phase 3."
  - Fix in Phase 3: add voyage_cost = transit_days x fuel_consumption_mtd x fuel_price
    to the objective as a penalty term.

- [ ] **Cargo type as a solver input field**
  - `cargo_type` is now present on all routes in `parameters.py` (informational only).
  - Fix: add `supported_infrastructure: list[str]` to each vessel; add
    `_is_infrastructure_feasible(vessel, route)` pre-filter in `solver.py`.
  - Blocked on: Researcher 2 confirming gear per vessel.

- [ ] **`date.today()` as the solve base — demo midnight risk**
  - Found in: `solver.py` — `base_date = date.today()`.
  - Risk: demo running across midnight causes date index shift -> Infeasible result.
  - Fix: pin `base_date` once per UI session in `st.session_state`.

- [ ] **CBC `timeLimit` returns "Optimal" on a truncated solve**
  - Fix blueprint is commented in `solver.py` (`prob.sol_status` check).
  - Uncomment and activate when problem scale exceeds ~10,000 variables (Beta).

- [ ] **No infeasibility diagnostics**
  - Fix: add `diagnose_infeasibility(vessels, routes, score_lookup)` helper that
    reports which routes have zero feasible vessels and which vessels have zero
    feasible routes, shown in an `st.expander` below the warning.

---

## Data Integrity

- [ ] **No rate range validation on forecast JSON**
  - `_load_forecast()` validates field presence but not values.
  - Risk: negative/zero rates or rates outside expected range (~5-80 dollar/t real BDI)
    will silently produce a nonsense assignment.
  - Fix: add `_validate_rates(records)` that warns if any p10/p50/p90 falls outside
    a configurable sanity band.

- [ ] **No contract JSON schema version field**
  - Risk: schema drift between Tech Lead 1 pipeline and solver is silent.
  - Fix: add optional `schema_version: "1.0"` to the JSON; solver warns on mismatch.

- [ ] **`HORIZON_DAYS` defined in two places**
  - `solver.py` and `dummy_generator.py` both hardcode 30 independently.
  - Fix: move `HORIZON_DAYS = 30` to `parameters.py` and import everywhere.

---

## Phase 2 Readiness (write_forecast.py)

> These items are deferred until the dummy generator is retired. Must be resolved
> before `write_forecast.generate_forecast()` replaces `dummy_generator.py`.

- [ ] **`write_forecast.py` will re-introduce the V-001 dead-record problem (W-01)**
  - `generate_forecast()` has no capacity pre-filter. 300 dead V-001 records will
    be silently written and discarded — the P-01/P-02 problem reappears.
  - Fix: import `VESSELS`/`ROUTES` from `parameters.py` and apply
    `_is_capacity_feasible()` before appending records in `create_forecast_records()`.

- [ ] **Route multiplier VALUES need recalibration for R-05-R-08 (W-02)**
  - Comments are now correct (fixed 2026-09-04). Values remain wrong:
    R-05-R-08 multipliers (1.12-1.22x) were sized for ~9,000 NM US East Coast
    origins. Real routes are Australian (~4,600-4,800 NM). Will over-inflate
    Gladstone routes relative to Hay Point benchmark (R-01 = 1.00x).
  - Fix: recalibrate multipliers proportional to real distance ratios vs R-01.

- [ ] **Vessel multiplier VALUES need recalibration for the real fleet (W-03)**
  - Comments are now correct (fixed 2026-09-04). Values remain wrong:
    0.95-1.07x spread designed for Capesize/Panamax/Supramax tiers. Real fleet is
    4x Supramax (51k-57k DWT, narrow spread). Flat 1.00x for V-002-V-005 is more
    defensible until Researcher 2 provides real per-vessel rate data.

- [ ] **`DEFAULT_VESSELS`/`DEFAULT_ROUTES` hardcoded as string lists (W-05)**
  - Two sources of truth for IDs. If parameters.py changes, forecaster and solver
    silently diverge.
  - Fix (do together with dummy generator removal): replace with list comprehensions
    from the imported VESSELS/ROUTES dicts.

---

## UI

- [ ] **Session state lost on server restart during demo**
  - Fix: pickle last `AssignmentResult` to `data/interim/last_result.pkl` after
    each successful solve; reload on startup if session state is empty.

- [ ] **CSS uses Streamlit internal `data-testid` selectors (fragile)**
  - Add `# FRAGILE` comment above each testid selector so upgraders know to recheck.

- [ ] **Colour-code normalised score column** (green -> high, red -> low)

- [ ] **Add route map panel**
  - Plotly `scatter_geo` arcs from origin -> destination for each assigned pair.

---

## Infrastructure



---

## Testing

- [ ] **No test for malformed forecast JSON**
  - Add `test_malformed_rates_raises` to `tests/test_solver.py`.

- [ ] **No test for duplicate record detection**
  - `_load_forecast()` raises on duplicates but no test covers this path.
  - Add `test_duplicate_records_raises` alongside the malformed-rates test.

- [ ] **Zero UI test coverage**
  - Add `tests/test_ui.py` with at least one smoke test using
    `streamlit.testing.v1.AppTest` (available in Streamlit >= 1.28).

---

## Phase 2+ (do not touch until forecasts are real)

- [ ] Wire SHAP waterfall chart per selected assignment row (Phase 3).
- [ ] Add freight rate crash slider and fuel price shock slider (Phase 3).
- [ ] Add voyage cost penalty term to solver objective (Phase 3 — see Voyage cost item above).
- [ ] Backtest profit-uplift metric vs. naive baseline (Phase 4).

