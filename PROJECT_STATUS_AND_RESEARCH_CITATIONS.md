# Comprehensive Project Status, Architecture, and Research Citations

> **Repository:** `SIH-26006-Intelligent-Freight--Forecasting`  
> **Problem Statement:** SIH 2026 Problem Statement 26006 — *Development of an Intelligent Freight Forecasting Model for Optimized Vessel Chartering and Bulk Cargo Procurement from Overseas to East Coast of India*  
> **Sponsoring Agency:** Ministry of Steel, Government of India  
> **Target Ports:** Visakhapatnam, Paradip, Haldia, Gangavaram (East Coast of India)  
> **Target Cargo:** Coking Coal imports from Australia (Dalrymple Bay, Hay Point, Newcastle, Gladstone), USA (Hampton Roads, Norfolk, Baltimore), and Canada (Vancouver, Prince Rupert, Roberts Bank)  
> **Last Updated:** 2026-09-04  

---

## 1. Executive Summary & Current Phase Status

### 1.1 Phase Completion Overview
The project follows an 18-day structured sprint plan designed for the Smart India Hackathon (SIH) prototype delivery:

| Phase | Description | Status | Scope Delivered / Remaining |
|---|---|---|---|
| **Phase 1 (Days 1–3)** | **Foundation & Floor Deliverable** | **100% COMPLETE** | • API JSON contract locked (`CONTRACT.md`)<br>• Standalone dummy generator (`dummy_generator.py`)<br>• PuLP CBC MILP optimizer engine (`src/solver/solver.py`)<br>• Risk-adjusted scoring (`src/solver/risk.py`)<br>• Interactive Streamlit UI (`src/ui/app.py`)<br>• Offline Dockerfile (`docker/Dockerfile`)<br>• Synchronized $t+1 \dots t+30$ horizon |
| **Phase 2 (Days 4–7)** | **Machine Learning & Real Data Ingestion** | **100% COMPLETE** | • LightGBM Multi-Step Direct Quantile Regressor (`src/forecaster/quantile_model.py`)<br>• AutoARIMA statistical baseline signal (`src/forecaster/baseline_arima.py`)<br>• Contract Forecast Writer (`src/forecaster/write_forecast.py`)<br>• Temporal 80/20 train/test holdout split with out-of-sample directional accuracy & error metrics (`src/forecaster/train_pipeline.py`)<br>• Feature Engineering Pipeline without lookahead leakage (`src/data/feature_engineering.py`)<br>• Seasonal Domain Calendar (`src/data/seasonal_calendar.py`)<br>• **6 Real Proxy Datasets Ingested** in `data/raw/`<br>• SHAP TreeExplainer local and global attributions cached (`models/shap_summary.json`) |
| **Phase 3 (Days 8–10)** | **Risk Scoring & Counterfactual UI** | **100% COMPLETE** | • Downside-penalty risk scoring ($\lambda$ tuning) fully operational<br>• Real-time Streamlit sliders for Freight Rate Shock ($\pm 30\%$) and Fuel Price Shock ($400–$1,000/MT)<br>• Real-time 3-card `Scenario Impact vs Baseline` delta grid<br>• **Decision Advisor Panel** with plain-English chartering recommendations and per-vessel guidance cards<br>• Voyage Gantt schedule and 30-day probabilistic Forecast Cones ($P_{10}/P_{50}/P_{90}$)<br>• Full SHAP Waterfall and Global Macro Driver UI tabs |
| **Phase 4 (Days 11–13)** | **Backtesting & Commercial Benchmarking** | **100% COMPLETE** | • Commercial baselines: Naive Spot Policy (Day 1 booking) & Greedy Heuristic (global spot minimum)<br>• 3 Historical Regime Stress Tests: Normal (2019), 2021 Super-Spike, 2020 COVID Crash (`src/benchmarks/regimes.py`)<br>• Continuous 12-Cycle Walk-Forward Rolling Simulation (`src/benchmarks/walk_forward.py`) showing **$584K honest cost savings** and 100% win-rate without artificial floors |
| **Phase 5 (Days 14–16)** | **Hardening & Verification** | **100% COMPLETE** | • All 7 SIH Judge audit loopholes resolved (charterer cost accounting, no artificial floors, zero lookahead leakage, out-of-sample holdout validation, V-001 feasibility, centralized INR rate)<br>• Dockerfile and `.dockerignore` configured for zero-internet execution<br>• **Full test suite passes: 205 passed in 12.86s** |
| **Phase 6 (Days 17–18)** | **Freeze, Rehearsal & Pitch Polish** | **IN PROGRESS** | • Code frozen on `main`<br>• Pitch & Demo Playbook created (`PITCH_AND_DEMO_PLAYBOOK.md`)<br>• 2-minute live demo choreography & Q&A defense rehearsal |

---

## 2. Implemented Codebase Architecture & Function Index

The codebase is organized into a modular three-layer architecture:
1. **Layer 1: Forecaster & Data Pipeline (`src/data/`, `src/forecaster/`)**
2. **Layer 2: Optimization Solver & Commercial Benchmarking (`src/solver/`, `src/benchmarks/`)**
3. **Layer 3: Interactive Decision UI (`src/ui/`)**

```
SIH-26006-Intelligent-Freight--Forecasting/
├── AGENT_CONTEXT.md              # Living master project blueprint
├── BETA_ROADMAP.md               # 20-day roadmap for Beta phase
├── CONTRACT.md                   # Locked JSON API contract
├── PITCH_AND_DEMO_PLAYBOOK.md    # SIH Judge pitch, demo & defense playbook
├── docker/
│   └── Dockerfile                # Offline container specification
├── dummy_generator.py            # Calibrated contract JSON generator
├── requirements.txt              # Production Python dependencies
├── data/
│   ├── raw/                      # 6 real proxy datasets
│   ├── processed/                # Merged feature matrices
│   └── interim/                  # freight_forecast_30d.json
├── src/
│   ├── data/
│   │   ├── feature_engineering.py# Lags, rolling stats, momentum, calendar
│   │   ├── load_real_data.py     # Real proxy data loaders & cleaners
│   │   ├── seasonal_calendar.py  # Monsoon, harvest, CNY domain pressure
│   │   └── synthetic_data.py     # Calibrated synthetic market generators
│   ├── forecaster/
│   │   ├── baseline_arima.py     # Statistical AutoARIMA baseline signal
│   │   ├── quantile_model.py     # LightGBM multi-step direct quantile engine
│   │   ├── explainability.py     # SHAP TreeExplainer feature attributions
│   │   ├── train_pipeline.py     # 80/20 train/test split & training pipeline
│   │   └── write_forecast.py     # Multiplier expansion & contract writer
│   ├── solver/
│   │   ├── parameters.py         # 5 vessels, 10 routes, laycans, capacities
│   │   ├── risk.py               # Downside-penalty risk scoring formula
│   │   └── solver.py             # PuLP CBC MILP assignment optimizer
│   ├── benchmarks/
│   │   ├── baselines.py          # Naive Spot & Greedy heuristic policies
│   │   ├── regimes.py            # Historical regime stress-testing (3 regimes)
│   │   └── walk_forward.py       # 12-cycle continuous rolling simulation
│   └── ui/
│       └── app.py                # VarunSetu decision dashboard & simulator
└── tests/                        # 11 test suites (205 passed tests)
```

### 2.1 Detailed Module & Function Reference

#### A. Data Engineering (`src/data/feature_engineering.py`)
- `add_lag_features(df, target_col="freight_rate", lags=[1, 7, 14, 30])`: Constructs historical autoregressive lags to capture autocorrelation.
- `add_rolling_features(df, target_col="freight_rate", windows=[7, 14, 30])`: Computes rolling moving averages, standard deviations, min, and max for regime and volatility tracking.
- `add_momentum_features(df, target_col="freight_rate", windows=[7, 14, 30])`: Computes rate of change (ROC) and temporal differences to detect price acceleration/deceleration.
- `add_calendar_features(df, date_col="date")`: Extracts calendar attributes (month, day of week, day of year) and cyclical trigonometric transformations ($\sin, \cos$).
- `merge_data_sources(freight_rates, bunker_fuel, crude_oil, sp500, dxy, gscpi, commodities, ...)`: Performs date-aligned merging of disparate market data using forward filling to prevent lookahead leakage.
- `build_feature_matrix(df, target_col="freight_rate", ...)`: End-to-end transformation pipeline yielding $(X, y)$ ready for machine learning.

#### B. Seasonal Domain Logic (`src/data/seasonal_calendar.py`)
- `SEASONAL_PRESSURE`: Quantified monthly index ($0.0 \dots 1.0$) capturing maritime shipping seasonality:
  - *Q1 (Jan–Mar):* Chinese New Year industrial slowdown followed by Southern Hemisphere grain export surge.
  - *Q2 (Apr–Jun):* Post-monsoon pre-stocking and steady industrial coal flows.
  - *Q3 (Jul–Sep):* Southwest Monsoon dip causing Indian port congestion and lower discharge rates.
  - *Q4 (Oct–Dec):* Peak winter thermal stocking and Northern Hemisphere grain harvest demand.
- `get_seasonal_pressure(date_val)`: Evaluates numerical seasonal pressure score for any calendar date.
- `get_seasonal_label(date_val)`: Provides domain explanation for the seasonal regime.
- `add_seasonal_features(df, date_col="date")`: Appends continuous scores and one-hot seasonal indicators.

#### C. Real Data Ingestion & Cleaning (`src/data/load_real_data.py`)
- `load_sp500(filepath)`: Ingests and cleans daily FRED S&P 500 equity index.
- `load_dxy(filepath)`: Ingests and cleans FRED Trade-Weighted US Dollar Index.
- `load_crude_oil(filepath)`: Parses FRED Brent and WTI crude oil price series from Excel.
- `load_bunker_fuel(filepath)`: Ingests USDA daily bunker fuel (IFO 380 cSt) quotes and normalizes currency strings.
- `load_gscpi(filepath)`: Ingests NY Fed Global Supply Chain Pressure Index monthly time series.
- `load_commodities(filepath)`: Parses World Bank Pink Sheet multi-level headers to extract Australian Coking Coal, Iron Ore (CFR spot), and Grain prices.
- `load_and_merge_all(raw_dir, output_path)`: Orchestrates cleaning, timestamp alignment, and dataset export.

#### D. Quantile Forecaster Engine (`src/forecaster/quantile_model.py`)
- `class QuantileForecaster`:
  - `__init__(quantiles=[0.1, 0.5, 0.9], horizon=30, n_estimators=200, learning_rate=0.05, num_leaves=31)`: Configures LightGBM regressors with pinball loss.
  - `_build_multi_step_target(y)`: Constructs direct multi-step target matrix for horizons $h \in [1, 30]$ to eliminate error compounding inherent to recursive forecasting.
  - `train(X, y)`: Fits $3 \times 30 = 90$ distinct quantile models or multi-step regressors.
  - `predict(X_current)`: Generates tabular output with columns `[horizon_step, p10, p50, p90]`.
  - `save(dirpath)` / `load(dirpath)`: Serialization via `joblib`.

#### E. Contract Serialization (`src/forecaster/write_forecast.py`)
- `create_forecast_records(predictions, start_date, vessels, routes, route_multipliers, vessel_multipliers)`: Expands base rate predictions across the 5 vessels $\times$ 10 routes $\times$ 30 days matrix ($1,500$ records) conforming to `CONTRACT.md` on the forward-looking $t+1 \dots t+30$ window.
- `write_forecast_json(records, filepath)`: Exports validated JSON artifact to `data/interim/freight_forecast_30d.json`.
- `generate_forecast(forecaster, X_current, ...)`: End-to-end chaining wrapper.

#### F. Solver Risk Scoring (`src/solver/risk.py`)
- `compute_score(p50, p10, lam=0.5)`: Computes the downside-penalty score:
  $$\text{Score} = P50 - \lambda \times (P50 - P10)$$
  Enforces structural checks ($\lambda \in [0, 1]$, $P10 \le P50$).
- `compute_scores_bulk(records, lam=0.5)`: Precomputes dictionary lookup `(vessel_id, route_id, date_index) -> score`.

#### G. MILP Optimization Engine (`src/solver/solver.py`)
- `class AssignmentResult`: Dataclass containing assignment list, status (`Optimal`), objective value, solve time (ms), and parameters.
- `_build_date_index(base_date, horizon)`: Produces ISO date strings for days $1 \dots 30$ ($t+1 \dots t+30$).
- `_is_laycan_valid(day_offset, route)`: Filters candidate departure days against route laycan windows.
- `_is_capacity_feasible(vessel, route)`: Verifies vessel DWT meets or exceeds route cargo lot size.
- `solve(lam, forecast_path, base_date)`: Formulates binary decision variables $x[v, r, t]$ and optimizes fleet profit using PuLP CBC under operational constraints.

#### H. Interactive UI (`src/ui/app.py`)
- Streamlit application displaying:
  - Top KPI cards: Total Objective Value, Number of Assignments, CBC Solve Time, $\lambda$ used.
  - Interactive DataFrame of optimal vessel assignments with routes, loading dates, rates, and transit days.
  - Plotly visualization showing risk-adjusted scores across assignments.
  - Real-time what-if sliders:
    1. *Risk aversion parameter ($\lambda$)*
    2. *Freight rate crash shock (% shock to P50 rates)*
    3. *Bunker fuel price shock (% change to voyage expenses)*

---

## 3. Data Inventory & Specifications

| Dataset Name | Source / Provider | File Location | Frequency | Description & Relevance to Dry Bulk |
|---|---|---|---|---|
| **S&P 500 Index** | Federal Reserve Bank of St. Louis (FRED) | `data/raw/SP500.csv` | Daily | Benchmark equity index representing broader macroeconomic demand, global liquidity, and industrial sentiment. |
| **US Dollar Index (DXY)** | Federal Reserve Bank of St. Louis (FRED) | `data/raw/DXYUSDollar Index.csv` | Daily | Trade-weighted US Dollar Index. Dry bulk shipping freight rates and bunker fuel are denominated in USD; dollar movements inversely impact commodity purchasing power. |
| **Brent & WTI Crude Oil** | Federal Reserve Bank of St. Louis (FRED) | `data/raw/Crude_Oil_Combined.xlsx` | Daily | Global energy benchmarks. Primary leading indicator for refined maritime bunker fuel prices. |
| **IFO 380 Bunker Fuel Prices** | USDA Agricultural Marketing Service | `data/raw/Daily_Bunker_Fuel_Prices_20260904.csv` | Daily | Intermediate Fuel Oil (IFO 380 cSt) spot prices across primary bunkering ports. Fuel constitutes 30–50% of total voyage expenses. |
| **Global Supply Chain Pressure Index (GSCPI)** | Federal Reserve Bank of New York | `data/raw/gscpi_data.xls` | Monthly | Composite index integrating container freight rates, air freight, port backlogs, delivery times, and inventory accumulation. |
| **Commodity Pink Sheet** | World Bank Development Prospects Group | `data/raw/CMO-Historical-Data-Monthly.xlsx` | Monthly | Global commodity benchmark prices for Australian Coking Coal, Iron Ore (62% CFR spot), and Grain. Direct proxy for steel mill raw material demand. |
| **Synthetic Freight Target** | Generated internally via GBM | `src/data/synthetic_data.py` | Daily | Calibrated proxy series matching historical BDI bounds ($15–$25/ton) with mean-reverting stochastic drift and seasonal volatility. |
| **Contract Interim JSON** | ML pipeline / Dummy generator | `data/interim/freight_forecast_30d.json` | 30-Day Forward | Standardized handoff artifact containing 1,500 records ($5 \text{ vessels} \times 10 \text{ routes} \times 30 \text{ days}$). |

---

## 4. Research Papers: Citations, Findings, and Project Implementation

The design and mathematical formulation of this project are grounded in **four academic research papers** spanning time-series econometrics, machine learning, maritime logistics optimization, and signal decomposition.

---

### Research Paper 1: Traditional Time-Series Econometrics of the BDI
- **Focus Area:** Time-series analysis, long-memory behavior, and econometric forecasting of the Baltic Dry Index.
- **Core Findings of the Paper:**
  1. Dry bulk freight rate indices exhibit non-stationary dynamics, autoregressive clustering, and long-memory persistence (fractional integration).
  2. Pure machine learning models do not uniformly outperform classical statistical baselines across all market regimes.
  3. Forecast combination (hybridizing linear statistical models with nonlinear estimators) frequently produces lower generalized forecast error than any standalone model.
- **What We Implemented in the Project:**
  - **Statistical Baseline Requirement:** Defined AutoARIMA (`pmdarima>=2.0`) as a mandatory benchmark to validate LightGBM performance against classical econometrics.
  - **Autoregressive Temporal Feature Extraction:** Built multi-scale lag features ($t-1, t-7, t-14, t-30$) in `src/data/feature_engineering.py` mirroring the autoregressive memory identified in the paper.
  - **Volatilty & Regime Tracking:** Extracted multi-window rolling standard deviations ($7d, 14d, 30d$) to detect market regime shifts.
  - **Decision-Centric Evaluation:** Upgraded the paper's objective from pure statistical error minimization (RMSE/MAE) to evaluating downstream chartering business profit.
- **Deferred to Beta Roadmap:**
  - Hurst exponent computation to mathematically test for fractional differencing.
  - Full FARIMA (Fractionally Integrated ARIMA) long-memory implementation.

---

### Research Paper 2: Machine Learning & Financial Feature Integration for BDI Prediction
- **Focus Area:** Multi-source freight rate forecasting using tree-based machine learning ensembles and macroeconomic/financial exogenous variables.
- **Core Findings of the Paper:**
  1. Incorporating exogenous macroeconomic indicators (equity markets, commodity prices, exchange rates) significantly improves out-of-sample directional and level accuracy compared to univariate models.
  2. Tree-based gradient boosted models (LightGBM, XGBoost, Extra Trees) outperform neural networks and linear regressions on tabular macroeconomic series due to robust handling of irregular volatility and multicollinearity.
  3. Post-hoc feature attribution using SHAP (SHapley Additive exPlanations) demonstrates that energy prices, currency strength, and steel inputs are primary drivers of freight rate trends.
- **What We Implemented in the Project:**
  - **Six-Source Exogenous Feature Matrix:** Ingested S&P 500, DXY, Brent/WTI Crude Oil, IFO 380 Bunker Fuel, GSCPI, and World Bank Pink Sheet commodities into `data/raw/` and built the alignment engine in `src/data/load_real_data.py`.
  - **LightGBM as Primary Forecaster:** Implemented LightGBM (`src/forecaster/quantile_model.py`) as the core forecasting engine.
  - **SHAP Explainability Protocol:** Architected SHAP TreeExplainer integration into `AGENT_CONTEXT.md` to compute per-prediction feature contributions.
  - **Multi-Metric Evaluation Framework:** Standardized MAE, RMSE, and $R^2$ as core quantitative metrics across high- and low-volatility regimes.
- **Deferred to Beta Roadmap:**
  - Model zoo comparison benchmarking LightGBM against CatBoost and Extra Trees.
  - Global SHAP summary and dependence dashboards in the UI.

---

### Research Paper 3: Wang et al. — Stochastic Vessel Chartering and Fleet Optimization
- **Focus Area:** Stochastic optimization and decision-making for maritime shipping logistics, advance chartering contracts, and spot market scheduling under rate uncertainty.
- **Core Findings of the Paper:**
  1. Naive spot-market chartering policies incur high volatility penalties, whereas advance chartering mitigates price spikes.
  2. Two-stage stochastic optimization incorporating rate uncertainty achieves an average **12.7% cost reduction** compared to deterministic static rules.
  3. Maritime operational constraints (vessel deadweight capacity, cargo requirements, port laycan windows) must constrain the decision space to avoid infeasible schedules.
- **What We Implemented in the Project:**
  - **Downside-Penalty Risk Formulation:** Derived the risk-adjusted scoring objective in `src/solver/risk.py`:
    $$\text{Score}(v, r, t) = P50 - \lambda \times (P50 - P10)$$
    penalizing routes with high downside rate volatility.
  - **Combinatorial MILP Solver:** Implemented binary integer programming ($x[v, r, t]$) in `src/solver/solver.py` ensuring fleet uniqueness and route-day uniqueness.
  - **Operational Domain Constraints:** Modeled real maritime constraints: vessel deadweight tonnage floors ($65,000 \dots 75,000$ DWT), route cargo requirements ($58,000 \dots 70,000$ DWT), and laycan departure windows.
  - **Synchronized Forward Horizon ($t+1 \dots t+30$):** Aligned solver and forecaster contracts (Option A) to respect advance tender notice requirements.
  - **Uplift Benchmarking Protocol:** Defined profit uplift benchmarking against naive spot chartering and greedy heuristics.
- **Deferred to Beta Roadmap:**
  - Two-Stage Adaptive Rolling Re-Optimization with switching cost penalties.
  - Explicit Advance vs. Spot charter rate premium modeling ($1–7$ days spot vs $8–30$ days advance).
  - Formal Rockafellar-Uryasev CVaR formulation with Monte Carlo scenario sampling.

---

### Research Paper 4: Wu et al. (2026) — "Dry Bulk Freight Index: Hybrid + Probabilistic Forecasting" (CEEMDAN-GF-BO-GPR)
- **Focus Area:** Probabilistic BDI forecasting using Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN), Gaussian Filter (GF) denoising, Bayesian Optimization (BO), and Gaussian Process Regression (GPR).
- **Core Findings of the Paper:**
  1. Freight rate data contains complex multi-scale temporal patterns: high-frequency short-term volatility and low-frequency macroeconomic trends. Direct modeling of raw series leads to error accumulation.
  2. Decomposing series into Intrinsic Mode Functions (IMFs), filtering high-frequency noise with Gaussian filters, and optimizing GPR via Bayesian Optimization achieved **99.82% fitting accuracy** and **84.98% directional accuracy**.
  3. Probabilistic forecasts providing calibrated uncertainty intervals are significantly more valuable for chartering decisions than deterministic point forecasts.
- **What We Implemented in the Project:**
  - **Directional Accuracy (DA) Metric:** Formally adopted Directional Accuracy as a core evaluation metric in `AGENT_CONTEXT.md`:
    $$\text{DA} = \frac{1}{N} \sum_{t=1}^N \mathbf{1}\left(\operatorname{sign}(\hat{y}_{t+1} - y_t) = \operatorname{sign}(y_{t+1} - y_t)\right)$$
    reflecting chartering managers' priority on predicting market direction.
  - **Uncertainty-Aware Quantile Architecture:** Replaced point forecasting with three quantile models ($\alpha=0.1, 0.5, 0.9$) to provide honest uncertainty bounds ($P10, P50, P90$).
  - **Implicit Time-Scale Decomposition:** Implemented multi-window rolling features ($7d, 14d, 30d$) in `src/data/feature_engineering.py` as an Alpha approximation of multi-scale decomposition.
  - **Derived Direction & Volatility Signals:** Formulated market direction indicators ($\Delta P50$) and volatility spreads ($P90 - P10$) to guide downstream risk scoring.
- **Deferred to Beta Roadmap:**
  - Explicit CEEMDAN decomposition using `PyEMD` into Intrinsic Mode Functions.
  - Gaussian Filter denoising of high-frequency components.
  - Gaussian Process Regression (GPR) with RBF + WhiteKernel as an alternative probabilistic forecaster.
  - Bayesian Optimization via Optuna to tune hyperparameters and $\lambda$ jointly.

---

## 5. Verification & Testing Status

The repository maintains an automated unit test suite executed via `pytest`:
- **Total Test Cases:** **181 passing tests** across 8 test suites.
- **Execution Time:** ~6.8 seconds.
- **Test Suites Breakdown:**
  1. `tests/test_feature_engineering.py`: 39 tests covering lags, rolling statistics, momentum, calendar cyclical features, and dataset merging.
  2. `tests/test_seasonal_calendar.py`: 48 tests verifying monthly domain pressures, boundary cases, and feature encoding.
  3. `tests/test_synthetic_data.py`: 26 tests verifying statistical bounds, reproducibility, and synthetic schema conformance.
  4. `tests/test_quantile_model.py`: 24 tests validating multi-step target alignment, pinball loss quantile ordering ($P10 \le P50 \le P90$), and model serialization.
  5. `tests/test_write_forecast.py`: 17 tests validating $t+1 \dots t+30$ horizon calculations, multiplier scaling, and JSON contract validation.
  6. `tests/test_solver.py`: 19 tests validating PuLP CBC optimization, laycan enforcement, fleet uniqueness, lambda sensitivity, and forecaster-solver end-to-end compatibility.
  7. `tests/test_load_real_data.py`: 8 tests verifying ingestion, column parsing, date monotonicity, and missing-file handling for all 6 raw data sources.
