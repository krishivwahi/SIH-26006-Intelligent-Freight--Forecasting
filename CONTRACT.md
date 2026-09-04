**Communication Protocol:**
The ML pipeline will write a single JSON artifact to disk. The MILP solver will read from this exact path. No REST APIs.

**File Path:** `data/interim/freight_forecast_30d.json`

**Forecast Horizon (Option A Locked):**
- Forward-looking 30 days: `t+1` to `t+30` (where `t` is base run date / today).
- In maritime chartering, operations require tender / notice of readiness, so loading laycans start at `t+1` (tomorrow) through `t+30`.
- All `date_index` strings correspond to `base_date + timedelta(days=d)` for `d ∈ [1, 30]`.

**Required JSON Structure:**
[
  {
    "date_index": "YYYY-MM-DD",
    "vessel_id": "String (e.g., V-001)",
    "route_id": "String (e.g., R-01)",
    "p10_rate": "Float (10th percentile rate)",
    "p50_rate": "Float (50th percentile rate)",
    "p90_rate": "Float (90th percentile rate)"
  }
]