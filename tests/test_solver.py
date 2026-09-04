"""
tests/test_solver.py

Unit tests for the Phase 1 placeholder solver.

These tests verify solver mechanics BEFORE any real forecast data exists.
They must remain green throughout Phase 1 and must not be broken by
the Day 2 real-parameter swap or Day 3 real-forecast swap.
"""
from __future__ import annotations

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.solver.risk import compute_score, compute_scores_bulk
from src.solver.solver import AssignmentResult, solve


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def dummy_forecast_path(tmp_path: Path) -> Path:
    """Generate a minimal but complete dummy forecast JSON in a temp dir.

    Mirrors dummy_generator.py but is self-contained for test isolation.
    5 vessels × 10 routes × 30 days = 1,500 records.
    """
    vessels = [f"V-00{i}" for i in range(1, 6)]
    routes = [f"R-{i:02d}" for i in range(1, 11)]
    base_date = date.today()
    records = []
    for day_offset in range(1, 31):
        date_str = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        for rid in routes:
            for vid in vessels:
                records.append(
                    {
                        "date_index": date_str,
                        "vessel_id": vid,
                        "route_id": rid,
                        "p10_rate": 16.0,
                        "p50_rate": 20.0,
                        "p90_rate": 26.0,
                    }
                )
    filepath = tmp_path / "freight_forecast_30d.json"
    filepath.write_text(json.dumps(records), encoding="utf-8")
    return filepath


# ── Risk score unit tests ──────────────────────────────────────────────────────

class TestComputeScore:
    def test_formula_correctness(self) -> None:
        """Score = P50 - λ*(P50 - P10). With p50=20, p10=16, λ=0.5 → 18.0"""
        assert compute_score(p50=20.0, p10=16.0, lam=0.5) == pytest.approx(18.0)

    def test_lambda_zero_returns_p50(self) -> None:
        """λ=0 means no risk penalty — score equals the median rate."""
        assert compute_score(p50=20.0, p10=16.0, lam=0.0) == pytest.approx(20.0)

    def test_lambda_one_returns_p10(self) -> None:
        """λ=1 means maximise worst-case floor — score equals P10."""
        assert compute_score(p50=20.0, p10=16.0, lam=1.0) == pytest.approx(16.0)

    def test_equal_quantiles_returns_rate(self) -> None:
        """When P10 == P50 (zero uncertainty), score equals that rate."""
        assert compute_score(p50=20.0, p10=20.0, lam=0.5) == pytest.approx(20.0)

    def test_invalid_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="λ must be in"):
            compute_score(p50=20.0, p10=16.0, lam=1.5)

    def test_quantile_inversion_raises(self) -> None:
        """p10 > p50 violates quantile ordering and must raise."""
        with pytest.raises(ValueError, match="Quantile ordering violated"):
            compute_score(p50=15.0, p10=20.0, lam=0.5)

    def test_bulk_scores_count(self) -> None:
        """compute_scores_bulk should return one score per record."""
        records = [
            {"vessel_id": "V-001", "route_id": "R-01", "date_index": "2026-09-03",
             "p10_rate": 16.0, "p50_rate": 20.0, "p90_rate": 26.0},
            {"vessel_id": "V-002", "route_id": "R-02", "date_index": "2026-09-03",
             "p10_rate": 14.0, "p50_rate": 18.0, "p90_rate": 24.0},
        ]
        scores = compute_scores_bulk(records, lam=0.5)
        assert len(scores) == 2
        assert scores[("V-001", "R-01", "2026-09-03")] == pytest.approx(18.0)


# ── Solver integration tests ───────────────────────────────────────────────────

class TestSolver:
    def test_solver_returns_optimal_status(self, dummy_forecast_path: Path) -> None:
        """Solver must find an optimal solution on well-formed dummy data."""
        result = solve(lam=0.5, forecast_path=dummy_forecast_path)
        assert result.solver_status == "Optimal", (
            f"Expected 'Optimal', got '{result.solver_status}'. "
            "Check that laycan windows in parameters.py are within the 30-day horizon."
        )

    def test_solver_returns_assignment_result_type(self, dummy_forecast_path: Path) -> None:
        result = solve(lam=0.5, forecast_path=dummy_forecast_path)
        assert isinstance(result, AssignmentResult)

    def test_positive_objective_value(self, dummy_forecast_path: Path) -> None:
        """With positive rates, objective must be positive."""
        result = solve(lam=0.5, forecast_path=dummy_forecast_path)
        assert result.objective_value > 0.0

    def test_no_vessel_double_assigned(self, dummy_forecast_path: Path) -> None:
        """Each vessel may appear in at most one assignment (Constraint 1)."""
        result = solve(lam=0.5, forecast_path=dummy_forecast_path)
        vessel_ids = [a["vessel_id"] for a in result.assignments]
        assert len(vessel_ids) == len(set(vessel_ids)), (
            f"Vessel(s) appear more than once: "
            f"{[v for v in vessel_ids if vessel_ids.count(v) > 1]}"
        )

    def test_no_route_day_double_served(self, dummy_forecast_path: Path) -> None:
        """Each (route, day) pair must be served by at most one vessel (Constraint 2)."""
        result = solve(lam=0.5, forecast_path=dummy_forecast_path)
        route_days = [(a["route_id"], a["date"]) for a in result.assignments]
        assert len(route_days) == len(set(route_days)), (
            "A (route, day) pair was served by more than one vessel."
        )

    def test_laycan_respected(self, dummy_forecast_path: Path) -> None:
        """All assigned loading dates must fall within each route/vessel laycan window."""
        from src.solver.parameters import LAYCAN_MATRIX, ROUTE_MAP
        result = solve(lam=0.5, forecast_path=dummy_forecast_path)
        base_date = date.today()
        for assignment in result.assignments:
            rid = assignment["route_id"]
            vid = assignment["vessel_id"]
            route = ROUTE_MAP[rid]
            loading_date = date.fromisoformat(assignment["date"])
            day_offset = (loading_date - base_date).days
            if (vid, rid) in LAYCAN_MATRIX:
                laycan = LAYCAN_MATRIX[(vid, rid)]
                assert laycan["laycan_open"] <= day_offset <= laycan["laycan_close"], (
                    f"Assignment ({vid}, {rid}) on day {day_offset}, "
                    f"outside laycan [{laycan['laycan_open']}, {laycan['laycan_close']}]"
                )
            else:
                assert route["laycan_open"] <= day_offset <= route["laycan_close"], (
                    f"Route {rid} assigned on day {day_offset}, "
                    f"outside laycan [{route['laycan_open']}, {route['laycan_close']}]"
                )
            assert 1 <= day_offset <= 30, f"Loading date outside 30-day horizon (offset {day_offset})"

    def test_day2_laycan_matrix_structure(self) -> None:
        """Verify Researcher 2 Day 2 Laycan Matrix coverage and feasibility."""
        from src.solver.parameters import LAYCAN_MATRIX, ROUTES, VESSELS

        assert len(LAYCAN_MATRIX) == len(VESSELS) * len(ROUTES)  # 50 pairs
        for (vid, rid), entry in LAYCAN_MATRIX.items():
            assert entry["laycan_days"] == 3
            assert entry["laycan_close"] - entry["laycan_open"] + 1 == 3
            assert 1 <= entry["laycan_open"] <= 30
            assert 1 <= entry["laycan_close"] <= 30
            if vid == "V-001":
                assert not entry["feasible"], "V-001 Handysize must be marked infeasible"
            else:
                assert entry["feasible"], f"Vessel {vid} on {rid} should be feasible"

    def test_date_index_forward_looking(self) -> None:
        """_build_date_index must generate t+1..t+30 forward-looking dates."""
        from src.solver.solver import _build_date_index
        base_date = date(2026, 9, 4)
        dates = _build_date_index(base_date, 30)
        assert len(dates) == 30
        assert dates[0] == "2026-09-05"  # t+1
        assert dates[-1] == "2026-10-04"  # t+30

    def test_forecaster_writer_to_solver_compatibility(self, tmp_path: Path) -> None:
        """Forecasts written by create_forecast_records must seamlessly solve in solver."""
        import pandas as pd
        from src.forecaster.write_forecast import create_forecast_records, write_forecast_json

        # 30-day horizon predictions (horizon_step 1..30)
        preds = pd.DataFrame({
            "horizon_step": list(range(1, 31)),
            "p10": [15.0] * 30,
            "p50": [20.0] * 30,
            "p90": [25.0] * 30,
        })
        base_date = date.today()
        records = create_forecast_records(preds, start_date=base_date)
        filepath = tmp_path / "forecaster_output.json"
        write_forecast_json(records, filepath=str(filepath))

        # Solve with the forecaster's written output
        result = solve(lam=0.5, forecast_path=filepath, base_date=base_date)
        assert result.solver_status == "Optimal"
        assert len(result.assignments) > 0
        for a in result.assignments:
            loading_date = date.fromisoformat(a["date"])
            offset = (loading_date - base_date).days
            assert 1 <= offset <= 30

    def test_assignments_have_required_fields(self, dummy_forecast_path: Path) -> None:
        """Every assignment row must carry all fields the UI expects."""
        required = {
            "vessel_id", "route_id", "date", "score",
            "p50_rate", "p10_rate", "p90_rate",
            "origin", "destination", "cargo_dwt",
            "vessel_capacity_dwt", "transit_days",
            "review_status", "cargo_type",  # P-05: needed for FLAG lane badge in UI
        }
        result = solve(lam=0.5, forecast_path=dummy_forecast_path)
        for i, assignment in enumerate(result.assignments):
            missing = required - assignment.keys()
            assert not missing, f"Assignment {i} missing fields: {missing}"

    def test_solver_missing_json_raises(self, tmp_path: Path) -> None:
        """Solver must raise FileNotFoundError if the contract JSON is absent."""
        with pytest.raises(FileNotFoundError):
            solve(forecast_path=tmp_path / "nonexistent.json")

    def test_solve_time_recorded(self, dummy_forecast_path: Path) -> None:
        """solve_time_ms must be positive after a solve."""
        result = solve(lam=0.5, forecast_path=dummy_forecast_path)
        assert result.solve_time_ms > 0.0

    def test_lambda_affects_scores(self, dummy_forecast_path: Path) -> None:
        """Higher λ must yield lower or equal objective (more conservative)."""
        result_low = solve(lam=0.0, forecast_path=dummy_forecast_path)
        result_high = solve(lam=1.0, forecast_path=dummy_forecast_path)
        assert result_high.objective_value <= result_low.objective_value, (
            "Higher λ should penalise downside more, yielding lower or equal objective."
        )
