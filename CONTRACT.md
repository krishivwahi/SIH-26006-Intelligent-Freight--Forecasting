**Communication Protocol:**
The ML pipeline will write a single JSON artifact to disk. The MILP solver will read from this exact path. No REST APIs.

**File Path:** `data/interim/freight_forecast_30d.json`

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