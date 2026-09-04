# AGENT CONTEXT — AI Maritime Chartering Decision Engine

> **Purpose of this file:** Any AI agent, copilot, or human contributor scanning this repository should read this file first. It contains the full architectural context, domain constraints, mathematical formulations, team conventions, and current progress state. This is a living document updated as phases complete.

---

## 1. Problem Statement

**SIH 2026 — Problem ID 26006**
*Development of an Intelligent Freight Forecasting Model for Optimized Vessel Chartering and Bulk Cargo Procurement from overseas to East Coast of India.*

**Sponsor:** Ministry of Steel, Government of India.

**Business case:** In 2013, the government exempted PSUs (SAIL, RINL) from mandatory chartering through Transchart, allowing direct vessel chartering. This created a need for sophisticated in-house decision support to replace the intermediary's expertise. Our system is that decision support engine.

**Cargo:** Coking coal imports from Australia, the United States, and Canada.

**Destination ports:** Visakhapatnam, Paradip, Haldia, Gangavaram (East Coast of India).

---

## 2. Alpha Scope (Locked)

This is an Alpha prototype for the SIH internal selection round. Everything below is a deliberate scope cut, not a missing feature. Beta scope is documented separately.

| Component               | Alpha Scope                                                       | Deferred to Beta                                              |
|--------------------------|-------------------------------------------------------------------|---------------------------------------------------------------|
| Forecasting model        | LightGBM quantile (P10, P50, P90) + AutoARIMA statistical baseline; AutoARIMA point forecast fed as feature into LightGBM | Temporal Fusion Transformer, FARIMA, GPR, full multi-model bake-off |
| Signal decomposition     | Not in Alpha (rolling stats at multiple windows serve as implicit decomposition) | CEEMDAN decomposition of freight series into trend + volatility IMFs before forecasting |
| Hyperparameter tuning    | LightGBM defaults with manual lambda tuning                       | Bayesian Optimization for LightGBM and model hyperparameters  |
| Problem size             | Hardcoded 5 vessels, 10 routes, 30-day horizon                    | Realistic fleet-scale problem sizing, re-benchmarked          |
| Solver                   | Deterministic MILP, one global Big-M, CBC via PuLP or OR-Tools   | Per-constraint tightened Big-M, commercial solver evaluation  |
| Risk treatment           | Downside penalty on P50-to-P10 gap                                | Formal CVaR with scenario generation                          |
| Fuel cost                | Static historical average, labeled as such on every slide         | Forecasted bunker cost feeding the objective function          |
| Feasibility constraints  | Laycan timing window only                                         | Demurrage, sanctions, flag-state, IMO CII, EEXI screening    |
| Counterfactual simulator | Two sliders: freight rate crash, fuel price shock                 | Full three-variable simulator including port congestion       |
| Data                     | Free public proxies + calibrated synthetic freight series          | Licensed Baltic Exchange feed                                 |
| Seasonal awareness       | Hardcoded seasonal pressure calendar (harvest cycles, monsoon, procurement peaks) | Live meteorological API for operational-layer weather decisions |
| Explainability           | SHAP waterfall charts per assignment showing feature contributions | Full model audit dashboard with global and local explanations  |
| Geopolitical shocks      | Not in Alpha                                                       | Sanctions/canal disruption detection and scenario modeling     |
| Chartering strategy      | Single-stage deterministic assignment                              | Two-stage adaptive re-optimization; explicit advance-vs-spot charter decision modeling |

---

## 3. Technical Architecture

### 3.1 Data Pipeline

**Free confirmed data sources:**

| Purpose                   | Source                                      | Access         | Notes                                                             |
|---------------------------|---------------------------------------------|----------------|-------------------------------------------------------------------|
| Fuel cost baseline        | USDA Daily Bunker Fuel Prices (agtransport.usda.gov) | Free, no signup | IFO380, IFO180, Marine Gas Oil across 20+ ports                 |
| Crude oil trend feature   | FRED series DCOILBRENTEU and DCOILWTICO     | Free           | Daily Brent and WTI. Leading indicator for bunker cost feature   |
| Global economic activity  | FRED series SP500                           | Free           | Daily S&P 500. Proxy for global economic health and trade demand |
| USD strength              | FRED series DTWEXBGS                        | Free           | Trade-weighted US Dollar Index. Freight is USD-denominated; dollar strength affects rate dynamics |
| Global freight stress     | NY Fed Global Supply Chain Pressure Index   | Free           | Monthly. Genuine freight-market stress signal                     |
| Commodity demand proxies  | World Bank Commodity Price Data (Pink Sheet) | Free, monthly  | Iron ore, coal, grain prices. Demand-side features               |

**Target variable (freight rates):** No confirmed free, clean, five-year daily Baltic Dry Index series exists. The Alpha uses a calibrated synthetic series shaped to match publicly known BDI ranges (500-3,000 post-2010, up to 11,000 at historical extremes). This is labeled explicitly as a synthetic proxy everywhere. Never present it as real BDI data. A judge who catches misrepresented data discounts everything; a judge who hears the pipeline was validated on a transparent proxy while licensing is already open reads that as maturity.

### 3.2 Forecaster (Tech Lead 1 owns this)

**Dual-model architecture:**

1. **AutoARIMA (statistical baseline).** Trained via `pmdarima` on the freight rate series alone. Produces a point forecast for each horizon step. Serves two purposes: (a) a standalone baseline to benchmark LightGBM against in Phase 4, and (b) an input feature to LightGBM so the ML model can learn when to trust or override the statistical forecast.

2. **LightGBM (primary ML model, sole quantile producer).** Three separate regressors with `objective='quantile'` and `alpha` set to 0.1, 0.5, and 0.9. These are native LightGBM objectives. No custom loss function. LightGBM receives the AutoARIMA point forecast as one of its input features, creating a principled model combination without the methodological problems of blending parametric (ARIMA Gaussian intervals) and non-parametric (LightGBM quantile) uncertainty estimates.

**Why LightGBM remains the sole quantile producer:** AutoARIMA prediction intervals assume Gaussian residuals. LightGBM quantiles are distribution-free. Naively averaging their P10s would blend incompatible uncertainty paradigms. Instead, AutoARIMA's point forecast flows into LightGBM as a feature, and LightGBM learns the combination internally through its tree structure.

**Horizon:** 30-day forecast. Direct multi-step forecasting (train a model per horizon step to predict from current features directly), not recursive forecasting. Recursive compounds error across 30 steps and is hard to tune on a compressed timeline.

**Feature engineering targets:**
- AutoARIMA point forecast for the same horizon step (the ensemble bridge)
- Seasonal pressure index (hardcoded domain calendar, see section 3.2.1 below)
- Lagged bunker fuel prices (USDA)
- Crude oil price trends and momentum (FRED Brent/WTI)
- S&P 500 level and momentum (FRED, global economic activity proxy)
- US Dollar Index and momentum (FRED DTWEXBGS, currency strength affects USD-denominated freight rates)
- Supply chain pressure index (NY Fed GSCPI)
- Commodity price proxies for demand signals (World Bank Pink Sheet: iron ore, coal, grain)
- Calendar features (month, day-of-week, seasonal dummies)
- Rolling statistics (7d, 14d, 30d rolling means and volatilities)
- Rolling volatility (7d, 14d, 30d standard deviations) for implicit regime awareness

#### 3.2.1 Seasonal Pressure Calendar (Hardcoded)

A domain-informed lookup that maps each month to a seasonal pressure score reflecting known dry bulk demand drivers. This is not a learned feature; it encodes expert knowledge about cyclical patterns that affect vessel availability and freight rates for coking coal imports to India's East Coast.

| Month   | Pressure | Primary Driver                                                                 |
|---------|----------|--------------------------------------------------------------------------------|
| Jan     | High     | Southern Hemisphere grain harvest begins (Australia, Argentina). Chinese pre-New Year steel restocking. |
| Feb     | High     | Grain shipments peak. Chinese New Year disruption (demand lull then surge).    |
| Mar     | Medium   | Grain harvest tailing off. End of Q1 steel production cycle.                   |
| Apr     | Low      | Shoulder season. Vessel availability improves.                                 |
| May     | Low      | Pre-monsoon calm. Lowest seasonal freight pressure.                            |
| Jun     | Medium   | Indian monsoon onset. East Coast port throughput begins to drop (Paradip, Vizag, Haldia). |
| Jul     | High     | Monsoon peak. Northern Hemisphere grain harvest begins (US, Canada). Dual pressure on vessel supply. |
| Aug     | High     | Monsoon continues. US/Canada grain shipments compete for Panamax/Supramax tonnage. |
| Sep     | High     | Monsoon tail. Northern grain harvest peak. Highest seasonal competition for vessels from our origin ports. |
| Oct     | High     | Peak coking coal procurement season. Steel mills ramp for Q1 demand. Indian ports recovering from monsoon backlog. |
| Nov     | Medium   | Procurement continues. Pre-winter demand in Northern Hemisphere.               |
| Dec     | Medium   | Procurement tails off. Holiday slowdowns in Western markets.                   |

**Implementation:** A simple dictionary mapping `month -> float` (0.0 to 1.0 scale, normalized). Fed directly as a feature column to LightGBM. The model learns the interaction between seasonal pressure and the other market signals.

**Why hardcoded, not learned:** With only 5 years of training data, there are at most 5 observations per month. LightGBM cannot reliably learn seasonal patterns from that alone. Encoding domain knowledge as a prior gives the model a head start. If the pattern does not hold, the model will assign it low feature importance, which is itself a useful finding for the pitch.

### 3.3 Forecaster-to-Solver API Contract

**This is the single integration interface.** The ML pipeline writes JSON to disk. The solver reads from the same path. No REST APIs.

**File path:** `data/interim/freight_forecast_30d.json`

**Schema (locked Day 1):**
```json
[
  {
    "date_index": "YYYY-MM-DD",
    "vessel_id": "String (e.g., V-001)",
    "route_id": "String (e.g., R-01)",
    "p10_rate": "Float (10th percentile rate, $/ton)",
    "p50_rate": "Float (50th percentile rate, $/ton)",
    "p90_rate": "Float (90th percentile rate, $/ton)"
  }
]
```

**Dimensions:** 30 days x 5 vessels x 10 routes = 1,500 records per forecast run.

**Forecast Horizon Alignment (Option A):**
- Forward-looking 30 days: $t+1 \dots t+30$ (where $t$ is base run date / today).
- Operational justification: In dry-bulk chartering, vessels cannot load on day 0 without advance notice / tender window (24–48h notice of readiness); operational loading laycans start at $t+1$ (tomorrow) through $t+30$.
- Both `write_forecast.py`, `dummy_generator.py`, and `solver.py` are strictly synchronized to generate and evaluate dates for horizon steps $1 \dots 30$.

### 3.4 Solver (Tech Lead 2 owns this)

**Type:** Deterministic Mixed-Integer Linear Programming (MILP).

**Solver backend:** CBC (open source) via PuLP or OR-Tools.

**Decision variables:** Binary assignment `x[v, r, t]` indicating whether vessel `v` is assigned to route `r` departing on day `t`.

**Risk-adjusted scoring formula:**
```
Score(v, r, t) = P50_Profit(v, r, t) - λ * (P50_Profit(v, r, t) - P10_Profit(v, r, t))
```
This penalizes assignments where the gap between expected and worst-plausible-case profit is large, without requiring a full scenario distribution. Lambda starts at 0.5. Tune it against one stable route and one volatile route by hand. The number in the pitch must be defensible as a considered choice.

**Do not call this CVaR.** Call it the "risk-adjusted score" or "downside-adjusted expected profit." If a judge asks whether this is CVaR, the honest answer is: it is a simplified downside penalty; full CVaR with scenario generation is scoped and specified for Beta.

**Constraints (Alpha):**
- Each vessel assigned to at most one route per time window
- Each route has minimum/maximum cargo capacity requirements
- Laycan timing windows (loading date constraints)
- Total fleet utilization bounds

**Big-M:** One global Big-M constant. Per-constraint tightening deferred to Beta.

### 3.5 UI (Tech Lead 2 + Researcher 3)

**Framework:** Streamlit.

**Core views:**
1. Optimal vessel-route assignment table (solver output)
2. 30-day forecast visualization with P10/P50/P90 bands
3. Risk score breakdown per assignment
4. SHAP waterfall chart per selected assignment ("why did the model predict this rate?")
5. Forecast direction and volatility indicators per route (derived from P10/P50/P90, no model change needed)

**Counterfactual simulator (two sliders):**
- Freight rate crash slider (% shock to all P50 rates)
- Fuel price shock slider (% change to bunker cost)
- Each slider triggers a solver re-run and updates the assignment table live

**Deployment:** Offline Docker container. Zero internet dependency at demo time.

---

## 4. Repository Structure (Current and Target)

```
SIH-26006-Intelligent-Freight--Forecasting/
├── AGENT_CONTEXT.md          # This file (living project context)
├── CONTRACT.md               # Forecaster-to-solver JSON schema contract
├── LICENSE
├── requirements.txt          # Python dependencies
├── dummy_generator.py        # Generates placeholder forecast JSON
├── data/
│   ├── raw/                  # [Phase 1] Raw downloaded proxy data
│   ├── processed/            # [Phase 1] Cleaned, feature-engineered datasets
│   └── interim/
│       └── freight_forecast_30d.json  # The contract artifact (currently dummy)
├── src/                      # [Phase 1+] Source code
│   ├── data/                 # Data ingestion and feature engineering
│   ├── forecaster/           # AutoARIMA baseline + LightGBM quantile models
│   ├── solver/               # MILP solver
│   ├── risk/                 # Risk scoring module
│   └── ui/                   # Streamlit app
├── models/                   # [Phase 2] Serialized trained models
├── notebooks/                # [Optional] Exploration notebooks
├── tests/                    # [Phase 2+] Unit and integration tests
├── docker/                   # [Phase 5] Dockerfile and compose
│   └── Dockerfile
└── docs/                     # [Phase 6] Pitch deck assets
```

---

## 5. The 18-Day De-risked Schedule

Every phase after Phase 1 assumes continuous integration: the moment a real number exists anywhere in the pipeline, it replaces the placeholder the same day.

| Phase | Days   | Focus                              | End-of-Phase Checkpoint                                                                                     |
|-------|--------|------------------------------------|-------------------------------------------------------------------------------------------------------------|
| 1     | 1-3    | Contracts, data, floor deliverable | Button click returns a vessel-route assignment on screen, using placeholder constant rate. Ugly but real and running end to end. |
| 2     | 4-7    | Real forecasts, integrated as they land | AutoARIMA baseline trained. LightGBM quantile models trained with AutoARIMA feature. Pipeline runs on real output. |
| 3     | 8-10   | Risk scoring and interactive UI    | A judge can move a slider and watch the assignment change live, with real or transparently labeled proxy numbers. |
| 4     | 11-13  | Backtest                           | Profit-uplift number against baselines. Model comparison: AutoARIMA vs LightGBM vs LightGBM+AutoARIMA. Segmented by volatility regime. |
| 5     | 14-16  | Hardening and real buffer          | Offline container runs clean, or the descope ladder has been applied and the core pipeline still works.      |
| 6     | 17-18  | Freeze and rehearse                | Code frozen morning of Day 17. Full run-throughs completed.                                                  |

### Phase 1 Day-by-Day Breakdown

**Day 1:**
- Lock the forecaster-to-solver JSON contract in writing (DONE — see CONTRACT.md)
- Researcher 2 freezes the 5-vessel, 10-route parameter matrix with real names, capacities, speeds, fuel curves
- Researcher 3 scaffolds the repo and a minimal Dockerfile
- Researcher 1 begins pulling all four confirmed free data sources
- Researcher 1 decides synthetic-vs-partial-real target variable approach by end of day
- Tech Lead 1 begins feature engineering against whatever data has landed

**Day 2:**
- Tech Lead 2 builds a trivial solver that reads the contract JSON and returns a feasible assignment
- Tech Lead 2 wires it to a bare-bones Streamlit UI shell
- Tech Lead 1 continues feature engineering
- Researcher 1 delivers cleaned proxy datasets

**Day 3:**
- End-to-end pipeline runs: dummy data flows through forecaster slot, solver picks an assignment, UI displays it
- This is the floor deliverable. Nobody leaves without this working.

### Descope Ladder (if Phase 5 opens behind schedule)

| Tier | Cut These First                                                      | Why It Is Safe to Cut                                              |
|------|----------------------------------------------------------------------|--------------------------------------------------------------------|
| 1    | Second what-if slider (keep freight shock only), visual polish, trim backtest baselines from three to two | None of these change what the system proves. They change presentation. |
| 1.5  | Drop SHAP panel; drop AutoARIMA baseline and feature; LightGBM runs standalone | SHAP is a presentation add-on. AutoARIMA is a rigour add-on. Core pipeline works without either. |
| 2    | Drop P90 quantile model, keep P10 and P50 only                       | The downside-penalty formula only needs P10 and P50. P90 is presentation depth. |
| 3    | **Never cut:** The forecaster-to-solver-to-UI pipeline itself         | Real numbers, one risk adjustment, one working slider. That pipeline is the entire thesis. |

---

## 6. Team Roles and Ownership

| Role                     | Primary Responsibility                                        | Key Checkpoint Owned                                 |
|--------------------------|---------------------------------------------------------------|------------------------------------------------------|
| Tech Lead 1 (Backend/ML) | AutoARIMA baseline, LightGBM quantile models, feature engineering, forecast API | Phase 2: real forecasts replace the placeholder      |
| Tech Lead 2 (Solver/UI)  | MILP solver, downside-penalty scoring, interactive frontend   | Phase 1 and Phase 3: floor deliverable, then live sliders |
| Researcher 1 (Data)      | Secures proxy and target-variable data, feature eng support   | Phase 1: data plan executed by Day 3                 |
| Researcher 2 (Domain)    | Fixed vessel and route parameters, laycan windows             | Phase 1 and Phase 2: parameter matrix frozen, laycans finalized |
| Researcher 3 (Integration) | Docker scaffold from Day 1, what-if slider wiring, offline QA | Phase 5: clean offline container run                 |
| Orator (Pitch)           | Narrative, deck, judge Q&A; frames Alpha as proof-of-concept  | Phase 6: rehearsed answers to anticipated questions   |

---

## 7. Anticipated Judge Questions (and Honest Answers)

| Question                          | Answer                                                                                                         |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------|
| Is this real CVaR?                | No. It is a downside-penalty simplification for Alpha. Full CVaR with scenario generation is scoped for Beta.  |
| Is this live Baltic Exchange data? | No. We use free public proxies plus a calibrated synthetic series matching real BDI ranges. Direct licensing runs thousands of pounds per year. We validated the pipeline on a transparent proxy while the licensing conversation is already open. |
| Why only 5 vessels and 10 routes?  | Deliberate Alpha scope decision to guarantee a live, responsive demo without needing a commercial solver. The solver scales; the demo is deliberately small. |
| What happens with a real fleet?    | Point to the Beta roadmap: per-constraint Big-M tightening, re-benchmarked solve time, commercial-solver evaluation if needed. |
| Why both ARIMA and LightGBM?       | AutoARIMA captures linear time-series structure; LightGBM captures nonlinear feature interactions. We feed AutoARIMA's forecast as a feature into LightGBM so the ML model learns when to trust or override the statistical baseline. Backtest proves whether the combination adds value. |
| How do you handle different market regimes? | Rolling volatility features let LightGBM split on regime implicitly. We segment backtest results by high- and low-volatility windows to validate performance is not regime-dependent. |
| How do you explain the model's decisions? | SHAP (SHapley Additive exPlanations) on LightGBM shows exactly which features pushed each prediction up or down. The UI displays a waterfall chart per assignment so a chartering manager can see, for example, "iron ore prices rising +$1.20, monsoon season +$0.80, USD weakening +$0.40" rather than a black-box number. |
| Why not re-optimize when new data arrives? | Alpha runs a single-stage deterministic optimization. Beta adds two-stage adaptive re-optimization: run the solver weekly, compare the new assignment against the previous one, and recommend changes only when uplift exceeds a switching cost threshold. |

---

## 8. Key Mathematical Formulations

### 8.1 Quantile Regression Loss (LightGBM native)

For quantile `α`, the pinball loss on a single observation:

```
L_α(y, ŷ) = α * max(y - ŷ, 0) + (1 - α) * max(ŷ - y, 0)
```

Three models trained with α = 0.1, α = 0.5, α = 0.9.

### 8.2 Risk-Adjusted Score (Solver Objective)

```
Score(v, r, t) = P50_Profit(v, r, t) - λ * (P50_Profit(v, r, t) - P10_Profit(v, r, t))
```

Where:
- `P50_Profit` = expected profit at the median freight rate forecast
- `P10_Profit` = profit at the pessimistic (10th percentile) freight rate forecast
- `λ` = risk aversion parameter (start at 0.5, tune against one stable and one volatile route)
- Higher λ = more conservative (heavier penalty for downside exposure)

### 8.3 MILP Formulation (Simplified)

```
Maximize:  Σ_v Σ_r Σ_t  Score(v, r, t) * x[v, r, t]

Subject to:
  Σ_r Σ_t  x[v, r, t] <= 1          ∀ v        (each vessel used at most once)
  Σ_v      x[v, r, t] <= 1          ∀ r, t     (each route-day served by at most one vessel)
  x[v, r, t] ∈ {0, 1}                ∀ v, r, t  (binary assignment)
  Laycan constraints on t per route
  Capacity constraints linking vessel size to route cargo requirements
```

---

## 9. Conventions and Standards

### Code Style
- Python 3.10+
- Type hints on all public function signatures
- Docstrings on all modules and public functions
- `black` formatting, `isort` import ordering

### Data Conventions
- All dates in ISO 8601 format (`YYYY-MM-DD`)
- All freight rates in USD per metric ton
- All fuel costs in USD per metric ton
- Vessel capacities in deadweight tonnage (DWT)

### Git Conventions
- Feature branches off `main`
- Branch naming: `phase{N}/{role}/{description}` (e.g., `phase1/tl1/feature-engineering`)
- Commit messages: imperative mood, reference phase number

### File Naming
- Snake_case for Python files
- Data files: `{source}_{frequency}_{description}.csv`
- Model artifacts: `lgb_q{quantile}_{version}.pkl`, `arima_{version}.pkl`

---

## 10. Current Progress Tracker

> **Last updated:** 2026-09-02 (Day 0, pre-kickoff)

### Phase 1: Foundation and Floor Deliverable
- [x] Forecaster-to-solver JSON contract locked (CONTRACT.md)
- [x] Dummy data generator written (dummy_generator.py)
- [x] Dummy forecast JSON artifact generated (data/interim/freight_forecast_30d.json)
- [x] Core ML dependencies declared (requirements.txt)
- [ ] Vessel-route parameter matrix frozen (Researcher 2)
- [ ] Raw proxy data pulled for all six sources (Researcher 1: USDA bunker, FRED crude, FRED S&P 500, FRED DXY, NY Fed GSCPI, World Bank Pink Sheet)
- [ ] Target variable approach decided: synthetic vs partial-real (Researcher 1)
- [ ] Repo scaffolded with full directory structure (Researcher 3)
- [ ] Minimal Dockerfile created (Researcher 3)
- [ ] Feature engineering started (Tech Lead 1)
- [ ] Trivial placeholder solver built (Tech Lead 2)
- [ ] Bare-bones Streamlit UI shell wired (Tech Lead 2)
- [ ] End-to-end pipeline runs on dummy data (all)

### Phase 2: Real Forecasts — COMPLETE
- [x] AutoARIMA trained on freight rate series (Tech Lead 1)
- [x] LightGBM quantile models trained with all features including AutoARIMA forecast (Tech Lead 1)
- [x] SHAP integration: generate feature contribution values per prediction (Tech Lead 1)
- [x] Real forecasts replace dummy JSON in pipeline (Tech Lead 1 + Tech Lead 2)

### Phase 3: Risk Scoring and Interactive UI — NOT STARTED
- [ ] SHAP waterfall chart panel added to Streamlit UI (Tech Lead 2 + Researcher 3)

### Phase 4: Backtest — NOT STARTED
- [ ] Evaluate with MAE, RMSE, R², and directional accuracy (Tech Lead 1)
- [ ] Compare: AutoARIMA alone vs LightGBM alone vs LightGBM+AutoARIMA (Tech Lead 1)
- [ ] Segment results by high- and low-volatility regimes (Tech Lead 1)
- [ ] Business metrics: profit uplift and downside risk reduction vs baselines (Tech Lead 1 + Tech Lead 2)
### Phase 5: Hardening — NOT STARTED
### Phase 6: Freeze and Rehearse — NOT STARTED

---

## 11. Dependencies (Full Alpha Stack)

```
# ML and Data
lightgbm==4.3.0
pandas==2.2.1
scikit-learn==1.4.1.post1
numpy==1.26.4
pmdarima>=2.0              # AutoARIMA statistical baseline
shap>=0.45                 # SHAP explainability for LightGBM

# Solver (to be added Phase 1)
pulp>=2.7
# OR: ortools>=9.9

# UI (to be added Phase 1)
streamlit>=1.32

# Containerization (Phase 5)
# Docker (external)

# Backtesting utilities (Phase 4)
matplotlib>=3.8
```

---

## 12. Critical Reminders for Any Agent

1. **Never present the synthetic freight series as real BDI data.** It is a calibrated proxy. Label it everywhere.
2. **The JSON contract in CONTRACT.md is frozen.** Both Tech Lead 1 and Tech Lead 2 build against it independently. Do not change the schema without explicit team consensus.
3. **The risk score is NOT CVaR.** Do not use that term in code, comments, UI labels, or documentation. Call it "downside-penalty risk score" or "risk-adjusted score."
4. **Direct multi-step forecasting, not recursive.** Train separate models or a single model that takes horizon step as a feature. Do not feed predictions back as inputs.
5. **The floor deliverable (Phase 1, Day 3) is non-negotiable.** An ugly working pipeline beats a beautiful broken one. The team walks into the judged round with something that runs.
6. **Lambda (λ) in the risk formula starts at 0.5.** It is a tunable parameter exposed in the UI, not a hardcoded constant.
7. **5 vessels, 10 routes is deliberate.** Do not expand the problem size in Alpha. The solver must respond instantly during the live demo.
8. **LightGBM is the sole quantile producer.** AutoARIMA provides a point forecast feature, not quantile forecasts. Do not average or blend AutoARIMA prediction intervals with LightGBM quantiles.
9. **AutoARIMA is on the descope ladder.** If Phase 2 runs long, cut AutoARIMA first. The core pipeline works with LightGBM alone.
10. **Measure both forecast metrics and business metrics.** RMSE/MAE/R²/directional accuracy for forecast accuracy. Profit uplift and downside risk reduction for business value. Lead with business metrics in the pitch.
11. **SHAP runs on LightGBM only.** Use `shap.TreeExplainer` for speed. Generate per-prediction waterfall data during forecast, store alongside the JSON contract for the UI to read.
12. **Six data sources, not four.** S&P 500 (FRED SP500) and US Dollar Index (FRED DTWEXBGS) are now part of the feature set. Researcher 1 must pull these alongside the original four.
13. **Directional accuracy is a first-class metric.** Chartering managers care about "will rates go up or down?" as much as exact rate values. Measure it in backtest, display it in the pitch. Formula: `DA = count(sign(predicted_change) == sign(actual_change)) / total`.
14. **Direction and volatility labels are derived, not modeled.** Direction = P50(t+1) > P50(t). Volatility = P90 - P10 spread. Risk label = High/Medium/Low based on spread thresholds. These are post-processing on existing quantile outputs, not new models.
