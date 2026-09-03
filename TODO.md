# TODO — AI Maritime Chartering Engine

> Items here are **not** Phase 1 blockers. They are improvements to be picked up
> as bandwidth allows, roughly ordered by phase relevance.
> Add items freely. Mark `[x]` when done, `[~]` when descoped.

---

## 🧮 Solver

- [ ] **Score normalisation across vessels**
  - Problem: all 5 placeholder vessels score ~22.3–22.5 (constant dummy rates).
    The bar chart and table look identical — assignment ranking is invisible.
  - Fix: add `normalise_scores(scores: dict) -> dict` to `src/solver/risk.py`.
    Formula: `norm = (score - min_score) / (max_score - min_score) * 100`.
    Add "Normalised Score (0–100)" as a display-only column in `app.py`.
    Raw score stays unchanged in the PuLP objective — normalisation is UI-only.

- [ ] **Cargo type as a solver input field**
  - Problem: coking coal, thermal coal, and iron ore need different vessel
    infrastructure (grab cranes, conveyor belts, self-unloaders). A vessel
    without the right gear is infeasible regardless of rate.
  - Fix: add `cargo_type` + `required_infrastructure: list[str]` to each route
    in `parameters.py`; add `supported_infrastructure: list[str]` to each vessel.
    Add `_is_infrastructure_feasible(vessel, route)` pre-filter in `solver.py`.
    Add a **Cargo Type** dropdown to the Streamlit sidebar.
  - Cargo types: `coking_coal`, `thermal_coal`, `iron_ore`.
  - Tags: `grab_crane`, `conveyor_belt`, `self_unloader`, `pneumatic_system`.
  - **Blocked on:** Researcher 2 confirming gear per vessel in the real matrix.

- [ ] **Duplicate record detection in forecast JSON loader**
  - Found in: `solver.py` L147–150 — `rate_lookup[key] = rec` silently overwrites
    if the same (vessel, route, date) triple appears twice in the JSON.
  - Risk: if Tech Lead 1's pipeline has a merge bug, the solver optimises on the
    wrong rate with no warning.
  - Fix: raise `ValueError` if a duplicate key is detected before building the lookup.

- [ ] **`date.today()` as the solve base — demo midnight risk**
  - Found in: `solver.py` L140 — `base_date = date.today()`.
  - Risk: if the demo runs across midnight, the date index shifts mid-session.
    The forecast JSON was generated earlier in the day, so dates will not match
    the solver's rebuilt date index → feasible triple set becomes empty → Infeasible.
  - Fix: pin `base_date` once per UI session in `st.session_state` and pass it
    into `solve()`. Do not re-derive from `date.today()` on each re-run.

- [ ] **CBC `timeLimit` returns "Optimal" on a truncated solve**
  - Found in: `solver.py` L204 — `timeLimit=30`.
  - Risk: if CBC hits 30 s (won't happen at 1,500 vars, but could at Beta scale),
    it returns the best feasible solution found so far. PuLP still reports status
    `LpStatusOptimal` even though it is not provably optimal. The UI shows "✔ Optimal"
    misleadingly.
  - Fix: check `prob.sol_status` (not just `prob.status`) and emit a "Time-limited
    (best found)" badge if the two disagree.

- [ ] **No infeasibility diagnostics**
  - Found in: `app.py` L179 — "Solver returned no assignments. Check laycan windows
    or capacity constraints."
  - Risk: during integration testing with real data, infeasibility will be common
    (tight real laycan windows + real capacity mismatches). The current message gives
    the user no actionable information.
  - Fix: add a `diagnose_infeasibility(vessels, routes, score_lookup)` helper that
    reports which routes have zero feasible vessels and which vessels have zero
    feasible routes, shown in an `st.expander` below the warning.

---

## 📦 Data Integrity

- [ ] **No rate range validation on forecast JSON**
  - Found in: `_load_forecast()` validates field presence but not values.
  - Risk: negative rates, zero rates, or rates 10× outside expected range
    (15–25 $/t placeholder, ~5–80 $/t real BDI range) will silently produce a
    nonsense assignment.
  - Fix: add a `_validate_rates(records)` step that warns (not raises) if any
    p10/p50/p90 falls outside a configurable sanity band, and logs the offending
    records.

- [ ] **No contract JSON schema version field**
  - Risk: if Tech Lead 1 adds a field or renames one, the solver reads stale/wrong
    data silently. Both sides build against CONTRACT.md independently — schema drift
    is a real integration risk.
  - Fix: add an optional `schema_version: "1.0"` field to the JSON. Solver checks
    it on load and warns if absent or mismatched.

- [ ] **`HORIZON_DAYS` defined in two places**
  - Found in: `solver.py` L45 (`HORIZON_DAYS = 30`) and `dummy_generator.py` L15
    (`range(30)`), with no shared constant between them.
  - Risk: if dummy_generator is updated to 28 days for testing, the solver silently
    treats missing dates as infeasible without raising.
  - Fix: move `HORIZON_DAYS = 30` to `parameters.py` as the single source of truth
    and import it in both files.

---

## 🖥️ UI

- [ ] **Session state lost on server restart during demo**
  - Found in: `app.py` L125–128 — result stored only in `st.session_state`.
  - Risk: if Streamlit crashes or is restarted during the judged demo, the previous
    solve result is gone and the presenter must re-click Run Solver.
  - Fix: pickle the last `AssignmentResult` to `data/interim/last_result.pkl` after
    each successful solve and reload it on startup if session state is empty.

- [ ] **CSS uses Streamlit internal `data-testid` selectors**
  - Found in: `app.py` L47, L51, L56 — `[data-testid="stAppViewContainer"]` etc.
  - Risk: Streamlit can rename these testids between minor versions and silently
    break the dark theme. Already flagged: we are on 1.35.0 and the note says a
    newer version is available.
  - Fix: pin `streamlit==1.35.0` (already done) and add a `# FRAGILE` comment above
    each testid selector so any upgrader knows to recheck styling.

- [ ] **Colour-code normalised score column** (green → high, red → low)
  - Deferred from score normalisation item above.

- [ ] **Add route map panel** — Plotly `scatter_geo` arcs from origin → destination
  for each assigned vessel-route pair. Visual proof the routes are geographically
  sensible.

---

## 🐳 Infrastructure

- [ ] **No `.gitignore` — `.venv` would be committed** ✅ FIXED (this push)
- [ ] **No `.dockerignore` — `COPY . .` would copy 300 MB `.venv` into image** ✅ FIXED (this push)

- [ ] **`dummy_generator.py` uses a relative path**
  - Found in: `dummy_generator.py` L6 — `filepath="data/interim/freight_forecast_30d.json"`.
  - Risk: if called from any directory other than the repo root (e.g., from inside
    `src/`), it silently writes the JSON to the wrong path.
  - Fix: resolve the path relative to `__file__` using `Path(__file__).parent / filepath`.

---

## 🧪 Testing

- [ ] **No test for malformed forecast JSON**
  - Current tests cover missing file and missing fields. No test covers valid JSON
    with wrong types (e.g., `p50_rate: "N/A"`), which would cause a silent
    `float()` cast failure deep in the solver.
  - Fix: add `test_malformed_rates_raises` to `tests/test_solver.py`.

- [ ] **Zero UI test coverage**
  - `app.py` is completely untested. A signature change to `AssignmentResult` or
    `solve()` would silently break the UI until someone opens a browser.
  - Fix: add a `tests/test_ui.py` with at least one smoke test using
    `streamlit.testing.v1.AppTest` (available in Streamlit ≥ 1.28).

---

## Phase 2+ (do not touch until forecasts are real)

- [ ] Wire SHAP waterfall chart per selected assignment row (Phase 3).
- [ ] Add freight rate crash slider and fuel price shock slider (Phase 3).
- [ ] Backtest profit-uplift metric vs. naive baseline (Phase 4).

