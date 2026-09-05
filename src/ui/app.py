"""
src/ui/app.py

Streamlit UI shell — Phase 1 floor deliverable.

LANGUAGE NOTE:
    This file is 100% Python. Streamlit compiles it to a React frontend
    internally. There is no JavaScript, TypeScript, or HTML in this codebase.

Layout:
  Sidebar  → λ slider + Run Solver button
  Main     → status badge, objective metric, assignment table, data label footer

Day 1: table renders from solver output (live PuLP call)
Day 3: same code, same button — only the forecast JSON changes
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Path bootstrap (run from repo root: streamlit run src/ui/app.py) ──────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.solver.parameters import LAMBDA, ROUTES, VESSELS
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
    .status-optimal  { background:#1a4731; color:#3fb950; padding:4px 14px;
                       border-radius:20px; font-weight:600; font-size:0.9rem; }
    .status-bad      { background:#4d1c1c; color:#f85149; padding:4px 14px;
                       border-radius:20px; font-weight:600; font-size:0.9rem; }
    .data-label      { background:#21262d; border:1px solid #f0883e44;
                       border-radius:6px; padding:8px 14px; color:#f0883e;
                       font-size:0.82rem; margin-top:18px; }
    h1 { color:#58a6ff !important; }
    h3 { color:#79c0ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Solver Controls")
    st.markdown("---")

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

    st.markdown(f"**Formula:**  `Score = P50 − {lam:.2f} × (P50 − P10)`")
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
    st.markdown(f"- Horizon: `30 days`")
    st.markdown(f"- Variables: `{len(VESSELS) * len(ROUTES) * 30:,}` binary")
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

# Re-run automatically if λ changed after a previous solve
lam_changed = (
    st.session_state.result is not None
    and abs(lam - st.session_state.last_lam) > 1e-6
)

if run_clicked or lam_changed:
    with st.spinner("🔧 Running CBC solver…"):
        try:
            result = solve(lam=lam)
            st.session_state.result = result
            st.session_state.last_lam = lam
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Solver error: {exc}")
            st.stop()

# ── Results display ────────────────────────────────────────────────────────────
result: AssignmentResult | None = st.session_state.result

if result is None:
    st.info("👈 Click **▶ Run Solver** in the sidebar to compute the optimal assignment.")
    st.markdown(
        "<div class='data-label'>⚠️ <strong>Phase 1 — placeholder data.</strong> "
        "Freight rates are constant synthetic proxies (not real BDI data). "
        "Real forecast JSON will replace this on Day 3.</div>",
        unsafe_allow_html=True,
    )
else:
    # Status pill
    if result.solver_status == "Optimal":
        pill_html = f"<span class='status-optimal'>✔ {result.solver_status}</span>"
    else:
        pill_html = f"<span class='status-bad'>✘ {result.solver_status}</span>"
    st.markdown(f"**Solver status:** {pill_html}", unsafe_allow_html=True)
    st.markdown("")

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Objective Value", f"{result.objective_value:.2f}", help="Sum of risk-adjusted scores")
    col2.metric("Assignments", len(result.assignments), help="Vessel-route pairs selected")
    col3.metric("Solve Time", f"{result.solve_time_ms:.0f} ms")
    col4.metric("λ Used", f"{result.lambda_used:.2f}", help="Risk aversion parameter")

    st.markdown("---")

    if not result.assignments:
        st.warning("Solver returned no assignments. Check laycan windows or capacity constraints.")
    else:
        st.markdown("### 📋 Optimal Vessel–Route Assignment")

        df = pd.DataFrame(result.assignments)

        # ── P-04: Normalised score (display-only; raw score drives the MILP) ──
        raw_scores = {i: row["score"] for i, row in enumerate(result.assignments)}
        norm_map = normalise_scores(raw_scores)
        df["norm_score"] = [norm_map[i] for i in range(len(df))]

        # ── P-05: Lane type badge — FLAG = Alpha planning lane ──────────────
        df["lane_type"] = df["review_status"].map(
            {"KEEP": "Benchmark", "FLAG": "Alpha lane"}
        ).fillna("Benchmark")

        # Rename columns for display
        df_display = df.rename(columns={
            "vessel_id":            "Vessel",
            "route_id":             "Route",
            "date":                 "Loading Date",
            "score":                "Risk-Adj. Score",
            "norm_score":           "Norm. Score (0-100)",
            "net_profit":           "Net Profit ($)",
            "voyage_cost":          "Voyage Cost ($)",
            "lane_type":            "Lane Type",
            "p50_rate":             "P50 Rate ($/t)",
            "p10_rate":             "P10 Rate ($/t)",
            "p90_rate":             "P90 Rate ($/t)",
            "origin":               "Origin",
            "destination":          "Destination",
            "cargo_dwt":            "Cargo (DWT)",
            "vessel_capacity_dwt":  "Vessel Cap. (DWT)",
            "transit_days":         "Transit (days)",
            "port_waiting_days":    "Port Wait (days)",
        })

        # Colour-code score column
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk-Adj. Score":    st.column_config.NumberColumn(format="%.4f"),
                "Norm. Score (0-100)": st.column_config.ProgressColumn(
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                    help="Score normalised 0-100 for visual comparison (display only — raw score drives the MILP)",
                ),
                "Lane Type":          st.column_config.TextColumn(
                    help="Benchmark = directly evidenced Platts route. Alpha lane = planning extension (FLAG)."
                ),
                "Net Profit ($)":     st.column_config.NumberColumn(format="$%.2f"),
                "Voyage Cost ($)":    st.column_config.NumberColumn(format="$%.2f"),
                "P50 Rate ($/t)":     st.column_config.NumberColumn(format="$%.2f"),
                "P10 Rate ($/t)":     st.column_config.NumberColumn(format="$%.2f"),
                "P90 Rate ($/t)":     st.column_config.NumberColumn(format="$%.2f"),
                "Cargo (DWT)":        st.column_config.NumberColumn(format="%d"),
                "Vessel Cap. (DWT)":  st.column_config.NumberColumn(format="%d"),
                "Port Wait (days)":   st.column_config.NumberColumn(format="%.1f"),
            },
        )

        # Score bar chart — bars coloured by lane type (P-05)
        st.markdown("### Score Breakdown per Assignment")
        bar_labels = [f"{r['vessel_id']} -> {r['route_id']}" for r in result.assignments]
        bar_colors = [
            "#f0883e" if r.get("review_status") == "FLAG" else "#58a6ff"
            for r in result.assignments
        ]
        bar_scores = [norm_map[i] for i in range(len(result.assignments))]
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
                    x=0.99, y=0.98, xref="paper", yref="paper",
                    text="<span style='color:#58a6ff'>blue = Benchmark</span>  "
                         "<span style='color:#f0883e'>orange = Alpha lane (FLAG)</span>",
                    showarrow=False, align="right",
                    font=dict(size=11),
                )
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Data provenance label (always shown after a solve)
    st.markdown(
        "<div class='data-label'>⚠️ <strong>Phase 1 — placeholder data.</strong> "
        "Freight rates are constant synthetic proxies calibrated to real BDI ranges "
        "(15–25 $/t). This is <em>not</em> real Baltic Exchange data. "
        "Real LightGBM quantile forecast will replace this on Day 3.</div>",
        unsafe_allow_html=True,
    )
