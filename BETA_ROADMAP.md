# Beta Roadmap: AI Maritime Chartering Decision Engine

> **Status:** Planning. No Beta work begins until Alpha Phase 6 (Freeze and Rehearse) is complete.
> **Last updated:** 2026-09-04

This document consolidates all Beta-deferred features from AGENT_CONTEXT.md into a structured workplan. Each item includes the research paper or design decision that motivated it, the implementation approach, and estimated effort.

---

## 1. Forecasting Model Upgrades

### 1.1 CEEMDAN Signal Decomposition
**Source:** Wu et al. (2026) — CEEMDAN-GF-BO-GPR for BDI forecasting.

**What:** Decompose the freight rate series into Intrinsic Mode Functions (IMFs) using Complete Ensemble Empirical Mode Decomposition with Adaptive Noise. High-frequency IMFs capture short-term volatility; low-frequency IMFs capture market trends. Denoise high-frequency components with a Gaussian filter before feeding into the forecaster.

**Why:** Alpha uses rolling statistics at multiple windows (7d, 14d, 30d) as an implicit decomposition. CEEMDAN is a principled, adaptive decomposition that extracts data-driven modes rather than fixed-window averages. The paper reports 99.82% fitting accuracy using decomposed components.

**Implementation:**
- Add `PyEMD` library for CEEMDAN decomposition
- Create `src/data/signal_decomposition.py` with `decompose_freight_series(series) -> dict[str, pd.Series]`
- Feed IMF-derived features (trend slope, volatility amplitude, dominant cycle period) into LightGBM alongside existing features
- Benchmark: decomposed features vs current rolling features via backtest

**Effort:** 3-4 days. Medium complexity.

---

### 1.2 Gaussian Process Regression (GPR)
**Source:** Wu et al. (2026).

**What:** Add GPR as an alternative forecasting model. GPR natively produces a mean prediction with a confidence interval (uncertainty estimate), unlike LightGBM which requires separate quantile models.

**Why:** GPR uncertainty is calibrated by construction (Bayesian posterior), whereas LightGBM quantiles are distribution-free approximations. For small-sample regimes (new routes, limited data), GPR may outperform tree-based methods.

**Implementation:**
- Use `scikit-learn.gaussian_process.GaussianProcessRegressor` with RBF + WhiteKernel
- Create `src/forecaster/gpr_model.py` with the same `train/predict/save/load` interface as `QuantileForecaster`
- Ensemble: compare LightGBM quantiles vs GPR intervals vs a weighted combination
- Benchmark on same backtest windows

**Effort:** 2-3 days. Medium complexity.

---

### 1.3 Bayesian Optimization for Hyperparameters
**Source:** Wu et al. (2026) used BO to tune GPR kernel parameters.

**What:** Replace manual hyperparameter selection with Bayesian Optimization using `optuna` or `scikit-optimize`. Tune LightGBM `n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples`, and the risk aversion parameter lambda jointly.

**Why:** Alpha uses LightGBM defaults (200 estimators, 0.05 lr, 31 leaves) and hand-tuned lambda=0.5. Systematic optimization could improve forecast accuracy and risk calibration.

**Implementation:**
- Add `optuna>=3.5` to dependencies
- Create `src/tuning/hyperparameter_search.py`
- Objective function: minimize pinball loss on time-series cross-validation
- Store best hyperparameters in `models/best_params.json`

**Effort:** 2 days. Low complexity.

---

### 1.4 Temporal Fusion Transformer (TFT)
**Source:** Architecture decision from early planning.

**What:** Add TFT as a deep learning forecasting model that handles multiple time series, static covariates, and known future inputs natively.

**Why:** TFT captures temporal patterns that tree models miss (long-range dependencies, attention over past inputs). It also provides built-in interpretability via attention weights.

**Implementation:**
- Use `pytorch-forecasting` library
- Requires GPU for training (not available in Alpha Docker constraint)
- Pre-train on cloud, export ONNX model for offline inference

**Effort:** 5-7 days. High complexity. Requires GPU infrastructure.

---

### 1.5 FARIMA (Fractionally Integrated ARIMA)
**Source:** Research paper 1 — long-memory analysis in freight rate data.

**What:** If freight rate data shows long-term dependence (measured by Hurst exponent > 0.5), FARIMA captures this better than standard ARIMA.

**Why:** Standard ARIMA/AutoARIMA assumes short memory. Freight rates may exhibit long-memory behavior (persistent trends lasting months). FARIMA explicitly models fractional differencing.

**Implementation:**
- Compute Hurst exponent on freight rate series
- If H > 0.5, fit FARIMA via `statsmodels` or custom fractional differencing
- Feed FARIMA forecast as an additional feature to LightGBM (same pattern as AutoARIMA)

**Effort:** 2 days. Low-medium complexity. Only implement if Hurst test justifies it.

---

## 2. Solver and Optimization Upgrades

### 2.1 Two-Stage Adaptive Re-Optimization
**Source:** Wang et al. — stochastic chartering optimization.

**What:** Instead of a single-stage deterministic MILP run, implement a rolling re-optimization framework. Stage 1: charter now based on current forecasts. Stage 2: when new data arrives (weekly), re-run the optimizer and recommend changes only when the profit uplift exceeds a switching cost threshold.

**Why:** The paper demonstrated 12.7% cost reduction using stochastic vs deterministic approaches. Our Alpha runs a single deterministic solve. Beta adds the adaptive feedback loop.

**Implementation:**
- Create `src/solver/adaptive_solver.py`
- Define switching cost threshold (penalty for changing a charter decision)
- Implement `compare_assignments(old, new) -> list[recommended_changes]`
- UI: show "Recommended Changes" panel with uplift estimates

**Effort:** 4-5 days. High complexity.

---

### 2.2 Advance Charter vs Spot Market Modeling
**Source:** Wang et al.

**What:** Add explicit modeling of the advance-vs-spot charter decision. Routes chartered within the next 7 days use spot-market pricing (higher, uncertain). Routes chartered 8-30 days out get a forward discount (lower, locked-in). The solver chooses the optimal mix.

**Implementation:**
- Add `charter_type` variable to the MILP: `advance` (days 8-30) or `spot` (days 1-7)
- Spot premium parameter: `spot_rate = base_rate * (1 + spot_premium)`
- Expose spot premium as a UI slider

**Effort:** 2-3 days. Medium complexity.

---

### 2.3 Per-Constraint Big-M Tightening
**Source:** Architecture decision.

**What:** Replace the single global Big-M constant with per-constraint tightened Big-M values. This improves solver performance at larger problem sizes.

**Effort:** 1-2 days. Low complexity. Mechanical refactor.

---

### 2.4 Commercial Solver Evaluation
**Source:** Scaling requirement.

**What:** Benchmark CBC against Gurobi and CPLEX on realistic fleet-scale problems (50+ vessels, 100+ routes). Determine if a commercial solver license is needed for production.

**Effort:** 2-3 days. Requires solver licenses.

---

### 2.5 Formal CVaR with Scenario Generation
**Source:** Architecture decision, deferred from Alpha risk treatment.

**What:** Replace the downside-penalty risk score with formal Conditional Value-at-Risk. Generate scenarios from the quantile forecast distribution and optimize over the worst-α% outcomes.

**Implementation:**
- Generate 100-500 scenarios per (vessel, route, day) from P10/P50/P90 using interpolation
- Formulate CVaR as a linear program (Rockafellar-Uryasev)
- Add `src/risk/cvar.py`

**Effort:** 3-4 days. High complexity.

---

## 3. Data and Features

### 3.1 Live Meteorological API
**Source:** User requirement, deferred from Alpha to maintain offline Docker capability.

**What:** Integrate a weather API (e.g., OpenWeatherMap, NOAA) for operational-layer decisions: port closures due to cyclones, monsoon-driven congestion, sea state affecting transit times.

**Implementation:**
- API integration with caching layer
- Weather features: wind speed at port, sea state index, cyclone proximity
- Fallback to cached data when offline

**Effort:** 2-3 days. Medium complexity. Requires API key management.

---

### 3.2 Geopolitical Shock Detection
**Source:** User requirement.

**What:** Detect and model sanctions, canal disruptions (Suez, Panama), trade policy changes, and geopolitical events that cause freight rate spikes or route closures.

**Implementation:**
- News sentiment API or curated event feed
- Event type classification: sanctions, canal disruption, trade war, port strike
- Scenario modeling: "What if Suez Canal closes?" slider in the UI
- Route feasibility constraints updated based on active events

**Effort:** 5-7 days. High complexity. Requires NLP or curated data source.

---

### 3.3 Licensed Baltic Exchange Feed
**Source:** Data quality requirement.

**What:** Replace synthetic freight rate proxy with real Baltic Exchange data (BDI, BCI, BSI, BPI sub-indices).

**Cost:** Thousands of GBP per year for data licensing.

**Effort:** 1-2 days integration (once license is acquired).

---

### 3.4 Forecasted Bunker Cost
**Source:** Architecture decision, deferred from Alpha fuel cost.

**What:** Replace static historical average bunker cost with a forecasted bunker cost feeding the solver objective function. Use crude oil price features to predict IFO380 prices 30 days out.

**Implementation:**
- Train a separate LightGBM model on bunker fuel prices using crude oil, DXY, and GSCPI features
- Feed predicted bunker cost into the solver profit calculation

**Effort:** 2 days. Low-medium complexity.

---

## 4. UI and Presentation

### 4.1 Full Model Audit Dashboard
**Source:** Explainability scope expansion.

**What:** Expand beyond per-assignment SHAP waterfalls to include global feature importance, partial dependence plots, model comparison charts, and forecast error distribution.

**Effort:** 3-4 days. Medium complexity.

---

### 4.2 Three-Variable What-If Simulator
**Source:** Alpha has two sliders (freight crash, fuel shock). Beta adds port congestion.

**What:** Add a third slider for port congestion delay (days added to transit time). This affects voyage cost and vessel availability.

**Effort:** 1-2 days. Low complexity.

---

### 4.3 Route Map Visualization
**Source:** Tech Lead 2 TODO.md.

**What:** Plotly `scatter_geo` arcs from origin to destination for each assigned vessel-route pair. Visual proof the routes are geographically sensible.

**Effort:** 1 day. Low complexity.

---

### 4.4 Cargo Type Constraints
**Source:** Tech Lead 2 TODO.md.

**What:** Add cargo type (coking coal, thermal coal, iron ore) and vessel infrastructure compatibility (grab cranes, conveyors, self-unloaders) as solver constraints.

**Blocked on:** Researcher 2 confirming gear per vessel in the real parameter matrix.

**Effort:** 2 days. Medium complexity.

---

## 5. Infrastructure

### 5.1 Production Docker Image
**What:** Non-root user, slim base, model pre-baking, health checks, multi-stage build.

**Effort:** 1-2 days.

---

### 5.2 CI/CD Pipeline
**What:** GitHub Actions for automated testing, linting, and Docker image building on push.

**Effort:** 1 day.

---

## 6. Prioritized Beta Sprint Plan

If Beta gets 10 working days, this is the recommended order:

| Priority | Item | Days | Cumulative |
|----------|------|------|------------|
| 1 | Bayesian Optimization (1.3) | 2 | 2 |
| 2 | Forecasted Bunker Cost (3.4) | 2 | 4 |
| 3 | Advance vs Spot Modeling (2.2) | 2 | 6 |
| 4 | CEEMDAN Signal Decomposition (1.1) | 3 | 9 |
| 5 | Route Map Visualization (4.3) | 1 | 10 |

If Beta gets 20 working days, add:

| Priority | Item | Days | Cumulative |
|----------|------|------|------------|
| 6 | Two-Stage Adaptive Re-Optimization (2.1) | 4 | 14 |
| 7 | GPR Model (1.2) | 2 | 16 |
| 8 | Formal CVaR (2.5) | 3 | 19 |
| 9 | Cargo Type Constraints (4.4) | 1 | 20 |

Items 3.1 (Weather API), 3.2 (Geopolitical Shocks), and 1.4 (TFT) are deferred to a production roadmap beyond Beta due to infrastructure requirements.

---

## 7. Research Paper References

| Paper | Key Contribution to Our System |
|-------|-------------------------------|
| Paper 1: BDI time-series analysis | AutoARIMA baseline, forecast combination, long-memory analysis (FARIMA for Beta) |
| Paper 2: BDI prediction with ML and financial features | External market features (S&P 500, DXY), SHAP explainability, feature importance, model evaluation framework |
| Paper 3: Wang et al. stochastic chartering | Two-stage adaptive optimization, advance-vs-spot chartering, business impact evaluation |
| Paper 4: Wu et al. CEEMDAN-GF-BO-GPR | Signal decomposition, GPR uncertainty, Bayesian hyperparameter optimization, directional accuracy metric |
