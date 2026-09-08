"""
tests/test_benchmarks.py

Unit tests for Phase 4 benchmarking suite:
- Directional Accuracy & error metrics
- Profit Uplift and Cost Reduction calculations
- Naive Spot Chartering & Greedy Lowest-Rate policies
- Multi-policy comparative evaluation
- Historical regime backtesting summary
"""

from __future__ import annotations

import pytest
from src.benchmarks.baselines import (
    compare_all_policies,
    run_greedy_lowest_rate_policy,
    run_naive_spot_policy,
)
from src.benchmarks.backtest import run_historical_regime_backtest
from src.benchmarks.metrics import (
    calculate_cost_reduction,
    calculate_directional_accuracy,
    calculate_forecast_errors,
    calculate_profit_uplift,
)


class TestBenchmarkMetrics:
    """Test suite for statistical and business evaluation metrics."""

    def test_directional_accuracy_perfect(self) -> None:
        """Identical directional shifts should yield 100% DA."""
        y_true = [20.0, 21.0, 23.0, 22.0, 25.0]
        y_pred = [19.0, 20.5, 24.0, 21.5, 27.0]
        da = calculate_directional_accuracy(y_true, y_pred)
        assert da == 100.0

    def test_directional_accuracy_inverted(self) -> None:
        """Exactly opposite directional shifts should yield 0% DA."""
        y_true = [20.0, 22.0, 24.0]
        y_pred = [20.0, 18.0, 16.0]
        da = calculate_directional_accuracy(y_true, y_pred)
        assert da == 0.0

    def test_directional_accuracy_mixed(self) -> None:
        """Half matching directional shifts should yield 50% DA."""
        y_true = [20.0, 22.0, 21.0]  # up, down
        y_pred = [20.0, 23.0, 24.0]  # up, up
        da = calculate_directional_accuracy(y_true, y_pred)
        assert da == 50.0

    def test_directional_accuracy_validation(self) -> None:
        """Mismatched lengths or too few points must raise ValueError."""
        with pytest.raises(ValueError):
            calculate_directional_accuracy([20.0], [20.0])
        with pytest.raises(ValueError):
            calculate_directional_accuracy([20.0, 21.0], [20.0, 21.0, 22.0])

    def test_profit_uplift(self) -> None:
        """Verify profit uplift percentage formula."""
        assert calculate_profit_uplift(120.0, 100.0) == 20.0
        assert calculate_profit_uplift(90.0, 100.0) == -10.0
        assert calculate_profit_uplift(100.0, 0.0) == 0.0

    def test_cost_reduction(self) -> None:
        """Verify voyage cost reduction percentage formula."""
        assert calculate_cost_reduction(100.0, 85.0) == 15.0
        assert calculate_cost_reduction(100.0, 110.0) == -10.0
        assert calculate_cost_reduction(0.0, 50.0) == 0.0

    def test_forecast_errors(self) -> None:
        """Verify statistical error dictionary values."""
        y_true = [20.0, 22.0, 24.0]
        y_pred = [20.0, 22.0, 24.0]
        err = calculate_forecast_errors(y_true, y_pred)
        assert err["mae"] == 0.0
        assert err["rmse"] == 0.0
        assert err["r2"] == 1.0
        assert err["mape"] == 0.0


class TestBaselinePolicies:
    """Test suite for commercial chartering baseline policies."""

    def test_naive_spot_policy_execution(self) -> None:
        """Naive spot policy must assign vessels without exceeding fleet constraints."""
        naive = run_naive_spot_policy()
        assert naive.policy_name == "Naive Spot Chartering"
        assert 1 <= naive.num_assignments <= 4  # V-001 idle
        assert naive.total_net_profit > 0
        assert naive.total_voyage_cost > 0

        # Unique vessels and routes
        assigned_vids = [a["vessel_id"] for a in naive.assignments]
        assigned_rids = [a["route_id"] for a in naive.assignments]
        assert len(assigned_vids) == len(set(assigned_vids)), "Duplicate vessel assigned in Naive policy"
        assert len(assigned_rids) == len(set(assigned_rids)), "Duplicate route assigned in Naive policy"
        assert "V-001" not in assigned_vids, "V-001 Handysize should not be assigned"

    def test_greedy_lowest_rate_policy_execution(self) -> None:
        """Greedy lowest-rate policy must book vessels without duplicate allocations."""
        greedy = run_greedy_lowest_rate_policy()
        assert greedy.policy_name == "Greedy Lowest-Rate Heuristic"
        assert 1 <= greedy.num_assignments <= 4
        assert greedy.total_net_profit > 0

        assigned_vids = [a["vessel_id"] for a in greedy.assignments]
        assigned_rids = [a["route_id"] for a in greedy.assignments]
        assert len(assigned_vids) == len(set(assigned_vids)), "Duplicate vessel in Greedy policy"
        assert len(assigned_rids) == len(set(assigned_rids)), "Duplicate route in Greedy policy"
        assert "V-001" not in assigned_vids

    def test_multi_policy_comparison(self) -> None:
        """AI MILP engine should outperform Naive Spot under baseline rates."""
        comparison = compare_all_policies()
        assert "ai" in comparison
        assert "naive" in comparison
        assert "greedy" in comparison
        assert "uplift_vs_naive_pct" in comparison
        assert comparison["ai"].total_net_profit >= comparison["naive"].total_net_profit


class TestHistoricalBacktestRegimes:
    """Test suite for domain historical backtest scenarios."""

    def test_historical_regimes_summary(self) -> None:
        """Verify backtest returns 3 historical regimes meeting paper thresholds."""
        res = run_historical_regime_backtest()
        assert len(res["regimes"]) == 3
        agg = res["aggregate"]
        assert agg["mean_directional_accuracy_pct"] >= 65.0
        assert agg["mean_profit_uplift_pct"] >= 10.0
        assert agg["mean_cost_reduction_usd"] > 100000.0
