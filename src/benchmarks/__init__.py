"""
src/benchmarks

Phase 4: Backtesting and Profit Uplift Benchmarking module.
Provides baseline chartering policies (Naive Spot, Greedy Lowest-Rate),
evaluation metrics (Directional Accuracy, Profit Uplift %, Cost Reduction %),
and historical rolling-window backtesting engines.
"""

from src.benchmarks.metrics import (
    calculate_directional_accuracy,
    calculate_profit_uplift,
    calculate_cost_reduction,
    calculate_forecast_errors,
)
from src.benchmarks.baselines import (
    PolicyResult,
    run_naive_spot_policy,
    run_greedy_lowest_rate_policy,
    compare_all_policies,
)
from src.benchmarks.backtest import (
    RegimeBacktestResult,
    run_historical_regime_backtest,
)

__all__ = [
    "calculate_directional_accuracy",
    "calculate_profit_uplift",
    "calculate_cost_reduction",
    "calculate_forecast_errors",
    "PolicyResult",
    "run_naive_spot_policy",
    "run_greedy_lowest_rate_policy",
    "compare_all_policies",
    "RegimeBacktestResult",
    "run_historical_regime_backtest",
]
