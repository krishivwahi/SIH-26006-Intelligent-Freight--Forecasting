"""
src/ui/app.py

Streamlit UI shell — Phase 1 floor deliverable.

Researcher 3 integration:
    - λ risk-aversion slider
    - Freight-rate what-if shock slider
    - VLSFO fuel-price what-if slider
    - Automatic solver re-run when any control changes

LANGUAGE NOTE:
    This file is 100% Python. Streamlit compiles it to a React frontend
    internally. There is no JavaScript, TypeScript, or HTML in this codebase.

Layout:
  Sidebar  → λ slider + Freight Shock slider + Fuel Price slider + Run Solver button
  Main     → status badge, objective metric, assignment table, data label footer

Day 1: table renders from solver output (live PuLP call)
Day 3: same code, same button — only the forecast JSON changes
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# ── Path bootstrap (run from repo root: streamlit run src/ui/app.py) ──────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.solver.parameters import LAMBDA, ROUTES, VESSELS, VLSFO_PRICE_USD_MT
from src.solver.risk import normalise_scores
from src.solver.solver import AssignmentResult, solve


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Maritime Chartering Engine",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark premium background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #e6edf3;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #1c2128 100%);
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 18px;
    }

    /* Status pill */
    .status-optimal {
        background: #1a4731;
        color: #3fb950;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .status-bad {
        background: #4d1c1c;
        color: #f85149;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .data-label {
        background: #21262d;
        border: 1px solid #f0883e44;
        border-radius: 6px;
        padding: 8px 14px;
        color: #f0883e;
        font-size: 0.82rem;
        margin-top: 18px;
    }

    h1 {
        color: #58a6ff !important;
    }

    h3 {
        color: #79c0ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Freight shock integration helper ──────────────────────────────────────────
def _create_shocked_forecast(freight_shock_pct: float) -> Path:
    """
    Create a temporary forecast JSON with the freight-rate what-if shock applied.

    Example:
        +10% → all P10/P50/P90 freight rates increase by 10%
        -10% → all P10/P50/P90 freight rates decrease by 10%

    The original forecast JSON is never modified.
    """
    source_path = ROOT / "data" / "interim" / "freight_forecast_30d.json"

    if not source_path.exists():
        raise FileNotFoundError(
            f"Forecast JSON not found at {source_path}. "
            "Run dummy_generator.py or the real forecaster pipeline first."
        )

    with source_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    multiplier = 1.0 + (freight_shock_pct / 100.0)

    shocked_records = []

    for record in records:
        shocked_record = dict(record)

        shocked_record["p10_rate"] = (
            float(record["p10_rate"]) * multiplier
        )

        shocked_record["p50_rate"] = (
            float(record["p50_rate"]) * multiplier
        )

        shocked_record["p90_rate"] = (
            float(record["p90_rate"]) * multiplier
        )

        shocked_records.append(shocked_record)

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="freight_shock_",
        delete=False,
        encoding="utf-8",
    )

    with temp_file:
        json.dump(shocked_records, temp_file, indent=2)

    return Path(temp_file.name)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Solver Controls")
    st.markdown("---")

    # ── λ Risk Aversion ────────────────────────────────────────────────────────
    lam = st.slider(
        label="λ — Risk Aversion",
        min_value=0.0,
        max_value=1.0,
        value=LAMBDA,
        step=0.05,
        help=(
            "Controls the downside penalty in the risk-adjusted score.\n\n"
            "Score = P50 − λ × (P50 − P10)\n\n"
            "λ = 0 → pure expected value\n"
            "λ = 1 → maximise P10 (worst-case floor)"
        ),
    )

    st.markdown(
        f"**Formula:**  `Score = P50 − {lam:.2f} × (P50 − P10)`"
    )

    st.markdown("---")

    # ── Freight-rate What-If ───────────────────────────────────────────────────
    freight_shock = st.slider(
        label="📉 Freight Rate Shock",
        min_value=-30,
        max_value=30,
        value=0,
        step=5,
        format="%d%%",
        help=(
            "What-if scenario applied to the forecast freight rates.\n\n"
            "Negative values simulate a freight-rate crash.\n"
            "Positive values simulate a freight-rate increase.\n\n"
            "The original forecast JSON is not modified."
        ),
    )

    if freight_shock == 0:
        st.caption("Freight scenario: baseline")
    elif freight_shock < 0:
        st.caption(
            f"Freight scenario: {abs(freight_shock)}% rate reduction"
        )
    else:
        st.caption(
            f"Freight scenario: {freight_shock}% rate increase"
        )

    st.markdown("---")

    # ── Fuel Price What-If ────────────────────────────────────────────────────
    st.markdown("**⛽ Fuel Price Shock**")

    vlsfo_price = st.slider(
        label="VLSFO Bunker Fuel Price ($/MT)",
        min_value=400.0,
        max_value=1000.0,
        value=float(VLSFO_PRICE_USD_MT),
        step=25.0,
        help=(
            "What-if scenario applied to the VLSFO bunker fuel price.\n\n"
            "Higher values increase vessel bunker costs.\n"
            "Lower values decrease vessel bunker costs.\n\n"
            "Default VLSFO price: USD 600/MT."
        ),
    )

    st.caption(
        f"Fuel scenario: USD {vlsfo_price:,.0}/MT VLSFO"
    )

    st.markdown("---")

    run_clicked = st.button(
        "▶ Run Solver",
        type="primary",
        use_container_width=True,
        help="Invoke PuLP CBC solver and update the assignment table.",
    )

    st.markdown("---")

    st.markdown("**Problem size**")
    st.markdown(f"- Vessels: `{len(VESSELS)}`")
    st.markdown(f"- Routes: `{len(ROUTES)}`")
    st.markdown("- Horizon: `30 days`")
    st.markdown(
        f"- Variables: `{len(VESSELS) * len(ROUTES) * 30:,}` binary"
    )

    st.markdown("---")
    st.markdown("**Solver:** PuLP 2.8.0 + CBC")
    st.markdown("**Beta:** OR-Tools (planned)")


# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("# 🚢 AI Maritime Chartering Decision Engine")

st.markdown(
    "**SIH 2026 · Problem 26006** — Ministry of Steel · East Coast of India · "
    "Coking coal imports (Australia / USA / Canada)"
)

st.markdown("---")


# ── Session state ──────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result: AssignmentResult | None = None

if "last_lam" not in st.session_state:
    st.session_state.last_lam: float = lam

if "last_freight_shock" not in st.session_state:
    st.session_state.last_freight_shock: int = freight_shock

if "last_vlsfo_price" not in st.session_state:
    st.session_state.last_vlsfo_price: float = vlsfo_price


# ── Detect control changes ─────────────────────────────────────────────────────
lam_changed = (
    st.session_state.result is not None
    and abs(lam - st.session_state.last_lam) > 1e-6
)

freight_shock_changed = (
    st.session_state.result is not None
    and freight_shock != st.session_state.last_freight_shock
)

vlsfo_changed = (
    st.session_state.result is not None
    and abs(vlsfo_price - st.session_state.last_vlsfo_price) > 1e-6
)


# ── Run solver ─────────────────────────────────────────────────────────────────
if (
    run_clicked
    or lam_changed
    or freight_shock_changed
    or vlsfo_changed
):
    temp_forecast_path: Path | None = None

    with st.spinner("🔧 Running CBC solver…"):
        try:
            # Baseline scenario: use the normal forecast JSON.
            # What-if scenario: create a temporary shocked forecast.
            if freight_shock == 0:
                forecast_path = (
                    ROOT
                    / "data"
                    / "interim"
                    / "freight_forecast_30d.json"
                )
            else:
                temp_forecast_path = _create_shocked_forecast(
                    freight_shock
                )
                forecast_path = temp_forecast_path

            result = solve(
                lam=lam,
                forecast_path=forecast_path,
                vlsfo_price=vlsfo_price,
            )

            st.session_state.result = result
            st.session_state.last_lam = lam
            st.session_state.last_freight_shock = freight_shock
            st.session_state.last_vlsfo_price = vlsfo_price

        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

        except Exception as exc:
            st.error(f"Solver error: {exc}")
            st.stop()

        finally:
            # Remove temporary scenario file after the solver has finished.
            if temp_forecast_path is not None:
                try:
                    temp_forecast_path.unlink(missing_ok=True)
                except OSError:
                    pass


# ── Results display ────────────────────────────────────────────────────────────
result: AssignmentResult | None = st.session_state.result


if result is None:
    st.info(
        "👈 Click **▶ Run Solver** in the sidebar to compute the optimal assignment."
    )

    st.markdown(
        "<div class='data-label'>⚠️ <strong>Phase 1 — placeholder data.</strong> "
        "Freight rates are constant synthetic proxies (not real BDI data). "
        "Real forecast JSON will replace this on Day 3.</div>",
        unsafe_allow_html=True,
    )

else:
    # ── Scenario summary ──────────────────────────────────────────────────────
    if freight_shock == 0:
        scenario_text = "Baseline freight scenario"
    elif freight_shock < 0:
        scenario_text = (
            f"Freight shock: {freight_shock}% "
            f"(simulated freight-rate reduction)"
        )
    else:
        scenario_text = (
            f"Freight shock: +{freight_shock}% "
            f"(simulated freight-rate increase)"
        )

    st.info(f"🎯 **Active scenario:** {scenario_text}")

    # ── Status pill ────────────────────────────────────────────────────────────
    if result.solver_status == "Optimal":
        pill_html = (
            f"<span class='status-optimal'>✔ "
            f"{result.solver_status}</span>"
        )
    else:
        pill_html = (
            f"<span class='status-bad'>✘ "
            f"{result.solver_status}</span>"
        )

    st.markdown(
        f"**Solver status:** {pill_html}",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── KPI metrics ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Objective Value",
        f"{result.objective_value:.2f}",
        help="Sum of risk-adjusted scores",
    )

    col2.metric(
        "Assignments",
        len(result.assignments),
        help="Vessel-route pairs selected",
    )

    col3.metric(
        "Solve Time",
        f"{result.solve_time_ms:.0f} ms",
    )

    col4.metric(
        "λ Used",
        f"{result.lambda_used:.2f}",
        help="Risk aversion parameter",
    )

    st.markdown("---")

    # ── Assignment table ──────────────────────────────────────────────────────
    if not result.assignments:
        st.warning(
            "Solver returned no assignments. "
            "Check laycan windows or capacity constraints."
        )

    else:
        st.markdown("### 📋 Optimal Vessel–Route Assignment")

        df = pd.DataFrame(result.assignments)

        # ── P-04: Normalised score ────────────────────────────────────────────
        # Display-only; raw score drives the MILP.
        raw_scores = {
            i: row["score"]
            for i, row in enumerate(result.assignments)
        }

        norm_map = normalise_scores(raw_scores)

        df["norm_score"] = [
            norm_map[i]
            for i in range(len(df))
        ]

        # ── P-05: Lane type badge ─────────────────────────────────────────────
        df["lane_type"] = df["review_status"].map(
            {
                "KEEP": "Benchmark",
                "FLAG": "Alpha lane",
            }
        ).fillna("Benchmark")

        # ── Rename columns for display ────────────────────────────────────────
        df_display = df.rename(
            columns={
                "vessel_id": "Vessel",
                "route_id": "Route",
                "date": "Loading Date",
                "score": "Risk-Adj. Score",
                "norm_score": "Norm. Score (0-100)",
                "lane_type": "Lane Type",
                "p50_rate": "P50 Rate ($/t)",
                "p10_rate": "P10 Rate ($/t)",
                "p90_rate": "P90 Rate ($/t)",
                "origin": "Origin",
                "destination": "Destination",
                "cargo_dwt": "Cargo (DWT)",
                "vessel_capacity_dwt": "Vessel Cap. (DWT)",
                "transit_days": "Transit (days)",
            }
        )

        # ── Assignment table ──────────────────────────────────────────────────
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk-Adj. Score": st.column_config.NumberColumn(
                    format="%.4f"
                ),
                "Norm. Score (0-100)": st.column_config.ProgressColumn(
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                    help=(
                        "Score normalised 0-100 for visual comparison "
                        "(display only — raw score drives the MILP)"
                    ),
                ),
                "Lane Type": st.column_config.TextColumn(
                    help=(
                        "Benchmark = directly evidenced Platts route. "
                        "Alpha lane = planning extension (FLAG)."
                    )
                ),
                "P50 Rate ($/t)": st.column_config.NumberColumn(
                    format="$%.2f"
                ),
                "P10 Rate ($/t)": st.column_config.NumberColumn(
                    format="$%.2f"
                ),
                "P90 Rate ($/t)": st.column_config.NumberColumn(
                    format="$%.2f"
                ),
                "Cargo (DWT)": st.column_config.NumberColumn(
                    format="%d"
                ),
                "Vessel Cap. (DWT)": st.column_config.NumberColumn(
                    format="%d"
                ),
            },
        )

        # ── Score bar chart ────────────────────────────────────────────────────
        st.markdown("### Score Breakdown per Assignment")

        bar_labels = [
            f"{r['vessel_id']} -> {r['route_id']}"
            for r in result.assignments
        ]

        bar_colors = [
            "#f0883e"
            if r.get("review_status") == "FLAG"
            else "#58a6ff"
            for r in result.assignments
        ]

        bar_scores = [
            norm_map[i]
            for i in range(len(result.assignments))
        ]

        fig = go.Figure(
            go.Bar(
                x=bar_labels,
                y=bar_scores,
                marker_color=bar_colors,
                text=[f"{s:.1f}" for s in bar_scores],
                textposition="outside",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Assignment (Vessel -> Route)",
            yaxis_title="Normalised Score (0-100)",
            yaxis_range=[0, 115],
            margin=dict(t=20, b=60),
            height=350,
            annotations=[
                dict(
                    x=0.99,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=(
                        "<span style='color:#58a6ff'>"
                        "blue = Benchmark</span>  "
                        "<span style='color:#f0883e'>"
                        "orange = Alpha lane (FLAG)</span>"
                    ),
                    showarrow=False,
                    align="right",
                    font=dict(size=11),
                )
            ],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ── Data provenance label ─────────────────────────────────────────────────
    st.markdown(
        "<div class='data-label'>⚠️ <strong>Phase 1 — placeholder data.</strong> "
        "Freight rates are constant synthetic proxies calibrated to real BDI ranges "
        "(15–25 $/t). This is <em>not</em> real Baltic Exchange data. "
        "Real LightGBM quantile forecast will replace this on Day 3.</div>",
        unsafe_allow_html=True,
    )