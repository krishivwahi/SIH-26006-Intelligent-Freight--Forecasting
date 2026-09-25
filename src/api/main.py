"""FastAPI backend separating ML inference from UI."""
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.forecaster.serve_forecast import ForecastServer
from src.solver.solver import solve

app = FastAPI(title="SIH Freight Forecasting API", version="1.0")

# Load models ONCE at startup — not per-request
server = ForecastServer()

@app.on_event("startup")
async def startup():
    server.load_models()

# --- Request/Response schemas ---
class SolveRequest(BaseModel):
    lam: float = 0.5
    vlsfo_price: float = 600.0
    freight_multiplier: float = 1.0

class ForecastRequest(BaseModel):
    base_date: Optional[str] = None  # ISO format

# --- Endpoints ---
@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": server._loaded,
        "model_version": server.model_version,
    }

@app.post("/api/v1/forecast/refresh")
async def refresh_forecast(req: ForecastRequest):
    """Re-generate forecast from cached models (no retraining)."""
    bd = date.fromisoformat(req.base_date) if req.base_date else date.today()
    try:
        result = server.generate_forecast(base_date=bd)
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/api/v1/solve")
async def run_solver(req: SolveRequest):
    """Run MILP solver with given parameters."""
    result = solve(
        lam=req.lam,
        vlsfo_price=req.vlsfo_price,
        freight_multiplier=req.freight_multiplier,
    )
    return {
        "status": result.solver_status,
        "objective_value": result.objective_value,
        "solve_time_ms": result.solve_time_ms,
        "assignments": result.assignments,
        "model_version": server.model_version,
    }

@app.get("/api/v1/model/info")
async def model_info():
    """Return current model metadata."""
    import json, os
    meta_path = os.path.join("models", server.model_version or "", "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {"version": server.model_version, "metadata": "not found"}
