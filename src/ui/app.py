"""
src/ui/app.py

Streamlit UI shell — Phase 1 floor deliverable.

Researcher 3 integration:
    - λ risk-aversion slider
    - Freight-rate what-if shock slider
    - VLSFO fuel-price what-if slider
    - Automatic solver re-run when any control changes.

LANGUAGE NOTE:
    This file is 100% Python. Streamlit compiles it to a React frontend
    internally. There is no JavaScript, TypeScript, or HTML in this codebase.

Layout:
  Sidebar  → λ slider + Freight Shock slider + VLSFO slider + Run Solver button
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
    page_title="VarunSetu",
    page_icon="VS",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --navy-950: #04111a;
        --navy-900: #071b28;
        --navy-850: #0a2636;
        --navy-800: #0d3143;
        --navy-700: #12485d;
        --teal: #31c4b6;
        --teal-bright: #67ddd2;
        --steel: #b4c6cd;
        --text: #f0f7f8;
        --muted: #8faab5;
        --line: rgba(116, 188, 205, 0.16);
        --line-strong: rgba(103, 221, 210, 0.28);
        --shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
    }

    html { scroll-behavior: smooth; }
    *, *::before, *::after { box-sizing: border-box; }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 84% 8%, rgba(49, 196, 182, 0.075), transparent 25%),
            radial-gradient(circle at 10% 18%, rgba(30, 104, 137, 0.12), transparent 28%),
            linear-gradient(145deg, var(--navy-950) 0%, #061b28 52%, #04111a 100%);
        color: var(--text);
        font-family: Helvetica, Arial, sans-serif;
    }

    [data-testid="stAppViewContainer"] * {
        font-family: Helvetica, Arial, sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061722 0%, #092432 100%);
        border-right: 1px solid var(--line);
        box-shadow: 16px 0 40px rgba(0,0,0,0.12);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }

    .main .block-container {
        max-width: 1420px;
        padding-top: 5.35rem !important;
        padding-bottom: 5rem !important;
    }

    /* Scroll / entrance motion: subtle, never distracting. */
    @keyframes vs-rise {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes vs-glow {
        0%, 100% { box-shadow: 0 0 0 rgba(49,196,182,0); }
        50% { box-shadow: 0 0 22px rgba(49,196,182,0.08); }
    }

    /* Result content reveals only as it enters the viewport. */
    .stDataFrame, [data-testid="stPlotlyChart"] {
        opacity: 0;
        transform: translateY(26px);
    }

    @supports (animation-timeline: view()) {
        .stDataFrame, [data-testid="stPlotlyChart"] {
            animation: vs-rise linear both;
            animation-timeline: view(block);
            animation-range: entry 8% cover 28%;
        }
    }

    @supports not (animation-timeline: view()) {
        .stDataFrame, [data-testid="stPlotlyChart"] {
            opacity: 1;
            transform: none;
            animation: vs-rise 0.55s ease both;
        }
    }

    .vs-hero {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        margin-top: 24px;
        padding: 34px 34px 30px;
        transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
        border: 1px solid var(--line-strong);
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(13, 54, 70, 0.82), rgba(6, 25, 37, 0.91)),
            radial-gradient(circle at 88% 14%, rgba(49,196,182,0.15), transparent 28%);
        box-shadow: var(--shadow);
    }

    .vs-hero::before {
        content: "";
        position: absolute;
        inset: auto -8% -60% auto;
        width: 430px;
        height: 430px;
        border-radius: 50%;
        border: 1px solid rgba(103,221,210,0.08);
        box-shadow:
            0 0 0 40px rgba(103,221,210,0.02),
            0 0 0 80px rgba(103,221,210,0.015);
        z-index: -1;
    }

    .vs-hero:hover {
        transform: translateY(-2px) scale(1.012);
        border-color: rgba(103,221,210,0.34);
        box-shadow: 0 24px 52px rgba(0,0,0,0.28), 0 0 28px rgba(49,196,182,0.07);
    }

    .vs-hero::after {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 38%;
        height: 2px;
        background: linear-gradient(90deg, var(--teal), transparent);
        opacity: 0.85;
    }

    .vs-kicker {
        color: var(--teal-bright);
        font-size: 0.69rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .vs-title {
        color: var(--text);
        font-size: clamp(2.35rem, 4.5vw, 4rem);
        font-weight: 800;
        line-height: 0.98;
        letter-spacing: -0.055em;
        margin: 0;
    }

    .vs-subtitle {
        color: #b8cbd2;
        margin-top: 13px;
        max-width: 960px;
        font-size: 1rem;
        line-height: 1.6;
    }

    .vs-live {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        color: #9fc4cc;
        font-size: 0.77rem;
        letter-spacing: 0.02em;
    }

    .live-dot, .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--teal);
        box-shadow: 0 0 0 4px rgba(49,196,182,0.08);
        display: inline-block;
    }

    .vs-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 22px;
    }

    .vs-chip {
        display: inline-flex;
        align-items: center;
        padding: 7px 11px;
        border-radius: 999px;
        border: 1px solid rgba(118,225,215,0.14);
        background: rgba(3,16,24,0.42);
        color: #cadce1;
        font-size: 0.72rem;
        transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
    }

    .vs-chip:hover {
        transform: translateY(-2px);
        border-color: rgba(103,221,210,0.34);
        background: rgba(14,53,66,0.56);
    }

    .vs-section {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #dcebee;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-top: 4px;
        margin-bottom: 10px;
    }

    .section-kicker {
        color: var(--teal-bright);
        font-size: 0.66rem;
        letter-spacing: 0.12em;
        font-weight: 800;
        opacity: 0.85;
    }

    /* KPI cards */
    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        background: linear-gradient(160deg, rgba(13,52,67,0.92), rgba(7,29,41,0.98));
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 18px 18px 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        min-height: 118px;
        transition: transform 220ms ease, border-color 220ms ease, box-shadow 220ms ease, background 220ms ease;
    }

    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 28%;
        height: 2px;
        background: linear-gradient(90deg, var(--teal), transparent);
        opacity: 0.75;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(103,221,210,0.34);
        box-shadow: 0 18px 42px rgba(0,0,0,0.24), 0 0 24px rgba(49,196,182,0.06);
        background: linear-gradient(160deg, rgba(16,62,78,0.98), rgba(7,29,41,1));
    }

    [data-testid="stMetricLabel"] {
        color: #8faab5 !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        color: #f2f9fa !important;
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.045em;
    }


    /* Decision summary */
    .decision-card {
        position: relative;
        overflow: hidden;
        margin: 14px 0 14px;
        padding: 20px 22px;
        border-radius: 18px;
        border: 1px solid rgba(103,221,210,0.24);
        background: linear-gradient(135deg, rgba(15,61,78,0.94), rgba(6,29,42,0.98));
        box-shadow: 0 16px 38px rgba(0,0,0,0.16);
        transition: transform 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
    }
    .decision-card:hover {
        transform: translateY(-3px) scale(1.006);
        border-color: rgba(103,221,210,0.40);
        box-shadow: 0 22px 46px rgba(0,0,0,0.24), 0 0 28px rgba(43,193,178,0.06);
    }
    .decision-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: linear-gradient(180deg, var(--teal-bright), transparent);
    }
    .decision-eyebrow { color:#80c9c3; font-size:0.67rem; font-weight:800; letter-spacing:0.14em; text-transform:uppercase; }
    .decision-head { display:flex; align-items:flex-end; justify-content:space-between; gap:20px; margin-top:6px; }
    .decision-title { color:#f3fbfc; font-size:1.30rem; font-weight:800; letter-spacing:-0.02em; }
    .decision-route { color:#b8cfd5; font-size:0.84rem; margin-top:4px; }
    .decision-value { color:#f4fbfc; font-size:2rem; font-weight:800; letter-spacing:-0.045em; text-align:right; white-space:nowrap; }
    .decision-meta { margin-top:14px; display:flex; flex-wrap:wrap; gap:8px; }
    .decision-chip { padding:6px 9px; border-radius:999px; border:1px solid rgba(126,220,214,0.13); background:rgba(2,18,27,0.36); color:#b8d0d7; font-size:0.72rem; }

    .impact-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:12px 0 20px; }
    .impact-card {
        padding:14px 16px;
        border:1px solid var(--line);
        border-radius:15px;
        background:rgba(8,33,46,0.80);
        transition:transform 200ms ease, border-color 200ms ease, box-shadow 200ms ease;
    }
    .impact-card:hover { transform:translateY(-3px) scale(1.008); border-color:rgba(103,221,210,0.26); box-shadow:0 16px 32px rgba(0,0,0,0.16); }
    .impact-label { color:#86a5af; font-size:0.68rem; font-weight:800; letter-spacing:0.10em; text-transform:uppercase; }
    .impact-value { color:#eff8fa; font-size:1.10rem; font-weight:800; margin-top:5px; }
    .impact-note { color:#8faab5; font-size:0.72rem; margin-top:3px; }

    /* Slider polish */
    div[data-baseweb="slider"] {
        padding: 8px 3px 4px;
    }

    div[data-baseweb="slider"] [role="slider"] {
        background: #071a24 !important;
        border: 2px solid var(--teal) !important;
        width: 18px !important;
        height: 18px !important;
        box-shadow: 0 0 0 4px rgba(49,196,182,0.08), 0 5px 16px rgba(0,0,0,0.25);
        transition: transform 160ms ease, box-shadow 160ms ease;
    }

    div[data-baseweb="slider"] [role="slider"]:hover {
        transform: scale(1.12);
        box-shadow: 0 0 0 6px rgba(49,196,182,0.10), 0 8px 20px rgba(0,0,0,0.32);
    }

    div[data-baseweb="slider"] [data-testid="stTickBar"] {
        opacity: 0.35;
    }

    div[data-baseweb="slider"] > div > div > div {
        border-radius: 999px !important;
    }

    .scenario-card {
        position: relative;
        padding: 13px 16px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(13,51,65,0.82), rgba(8,31,43,0.92));
        border: 1px solid rgba(118,225,215,0.15);
        color: #cbdde2;
        font-size: 0.83rem;
        margin: 8px 0 18px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.10);
        transition: transform 200ms ease, border-color 200ms ease, box-shadow 200ms ease;
    }

    .scenario-card:hover {
        transform: translateY(-3px);
        border-color: rgba(103,221,210,0.30);
        box-shadow: 0 15px 32px rgba(0,0,0,0.18);
    }

    .status-optimal, .status-bad {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.76rem;
        letter-spacing: 0.04em;
    }

    .status-optimal {
        background: rgba(49,196,182,0.09);
        border: 1px solid rgba(49,196,182,0.24);
        color: #73e5da;
    }

    .status-bad {
        background: rgba(218,87,87,0.10);
        border: 1px solid rgba(218,87,87,0.25);
        color: #ff9f9f;
    }

    .bad-dot { background: #ff7f7f; box-shadow: 0 0 0 4px rgba(255,127,127,0.08); }

    .data-label {
        background: rgba(12,40,51,0.78);
        border: 1px solid rgba(204,167,97,0.16);
        border-radius: 12px;
        padding: 11px 15px;
        color: #cbbd9d;
        font-size: 0.75rem;
        line-height: 1.55;
        margin-top: 20px;
        transition: transform 200ms ease, border-color 200ms ease;
    }

    .data-label:hover {
        transform: translateY(-2px);
        border-color: rgba(204,167,97,0.25);
    }

    .sidebar-brand {
        padding: 4px 4px 18px;
    }

    .sidebar-brand-title {
        color: var(--text);
        font-size: 1.18rem;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 9px;
        letter-spacing: -0.02em;
    }

    .brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 29px;
        height: 29px;
        border-radius: 9px;
        background: linear-gradient(135deg, #2cbfaf, #163f50);
        color: #04131b;
        font-size: 0.63rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        box-shadow: 0 6px 18px rgba(49,196,182,0.18);
    }

    .sidebar-brand-sub { color: var(--muted); font-size: 0.72rem; margin-top: 5px; }
    .sidebar-label { color: #9cb5be; font-size: 0.71rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }
    .mini-stat { color: #c8dce2; font-size: 0.82rem; margin: 4px 0; }
    .mini-stat strong { color: #eef8fa; }

    [data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #2ac1b1, #36d3c1);
        border: 1px solid rgba(103,221,210,0.28);
        color: #04141c;
        font-weight: 800;
        border-radius: 11px;
        min-height: 46px;
        box-shadow: 0 10px 25px rgba(39,194,179,0.16);
        transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
    }

    [data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-2px);
        filter: brightness(1.04);
        box-shadow: 0 15px 30px rgba(39,194,179,0.22);
    }

    .stDataFrame {
        border: 1px solid var(--line);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 14px 34px rgba(0,0,0,0.14);
        transition: transform 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
    }

    .stDataFrame:hover {
        transform: translateY(-3px) scale(1.006);
        border-color: rgba(103,221,210,0.24);
        box-shadow: 0 20px 40px rgba(0,0,0,0.20);
    }

    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--line);
        border-radius: 15px;
        overflow: hidden;
        padding: 4px;
        background: rgba(5,19,28,0.34);
        transition: transform 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
    }

    [data-testid="stPlotlyChart"]:hover {
        transform: translateY(-3px) scale(1.006);
        border-color: rgba(103,221,210,0.22);
        box-shadow: 0 18px 40px rgba(0,0,0,0.18);
    }

    h1, h2, h3 { color: var(--text) !important; }
    h3 { margin-top: 4px; }
    hr { border-color: var(--line) !important; }

    /* Reduce default Streamlit visual noise */
    [data-testid="stCaptionContainer"] { color: #7898a4 !important; }
    [data-testid="stMarkdownContainer"] p { line-height: 1.55; }
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
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title"><span class="brand-mark">VS</span> VarunSetu</div>
            <div class="sidebar-brand-sub">Maritime Chartering Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-label">Scenario Controls</div>', unsafe_allow_html=True)
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
        label="Freight Rate Shock",
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

    # ── VLSFO Fuel Price ──────────────────────────────────────────────────────
    vlsfo_price = st.slider(
        label="VLSFO Bunker Fuel Price ($/MT)",
        min_value=400.0,
        max_value=1000.0,
        value=float(VLSFO_PRICE_USD_MT),
        step=25.0,
        help=(
            "Simulate vessel voyage cost sensitivity to bunker fuel price fluctuations.\n\n"
            "Default: $600/MT."
        ),
    )

    st.caption(f"Fuel scenario: ${vlsfo_price:,.0f}/MT VLSFO")

    st.markdown("---")

    run_clicked = st.button(
        "Run Solver",
        type="primary",
        use_container_width=True,
        help="Invoke PuLP CBC solver and update the assignment table.",
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Problem Snapshot</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mini-stat">Vessels <strong>{len(VESSELS)}</strong></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mini-stat">Routes <strong>{len(ROUTES)}</strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="mini-stat">Planning horizon <strong>30 days</strong></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="mini-stat">Decision variables <strong>{len(VESSELS) * len(ROUTES) * 30:,}</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Optimization engine: PuLP + CBC")
    st.caption("OR-Tools integration planned")


# ── Main header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="vs-hero">
        <div class="vs-kicker">SIH 2026 · Problem 26006 · Ministry of Steel</div>
        <div class="vs-title">VarunSetu</div>
        <div class="vs-subtitle">Intelligent Maritime Freight Forecasting &amp; Vessel Chartering Decision Engine</div>
        <div class="vs-live"><span class="live-dot"></span> Decision workspace · Scenario-ready</div>
        <div class="vs-chip-row">
            <span class="vs-chip">East Coast of India</span>
            <span class="vs-chip">Coking Coal Imports</span>
            <span class="vs-chip">Australia</span>
            <span class="vs-chip">USA</span>
            <span class="vs-chip">Canada</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")


# ── Session state ──────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result: AssignmentResult | None = None

if "last_lam" not in st.session_state:
    st.session_state.last_lam: float = lam

if "last_freight_shock" not in st.session_state:
    st.session_state.last_freight_shock: int = freight_shock

if "last_vlsfo_price" not in st.session_state:
    st.session_state.last_vlsfo_price: float = vlsfo_price

if "baseline_snapshot" not in st.session_state:
    st.session_state.baseline_snapshot: dict | None = None


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
if run_clicked or lam_changed or freight_shock_changed or vlsfo_changed:
    temp_forecast_path: Path | None = None

    with st.spinner("Running CBC solver…"):
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

            if freight_shock == 0:
                st.session_state.baseline_snapshot = {
                    "objective": float(result.objective_value),
                    "net_profit": float(sum(r.get("net_profit", 0.0) for r in result.assignments)),
                    "assignments": len(result.assignments),
                    "vlsfo_price": float(vlsfo_price),
                    "lam": float(lam),
                }

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
    st.info("Open the Scenario Controls and run the solver to compute the optimal assignment.")
    st.markdown(
        "<div class='data-label'><strong>Phase 1 · Placeholder data.</strong> "
        "Freight rates are constant synthetic proxies. Real forecast JSON will replace this on Day 3.</div>",
        unsafe_allow_html=True,
    )
else:
    if freight_shock == 0:
        scenario_text = "Baseline freight scenario"
    elif freight_shock < 0:
        scenario_text = f"Freight shock: {freight_shock}% reduction"
    else:
        scenario_text = f"Freight shock: +{freight_shock}% increase"

    st.markdown(
        f'<div class="scenario-card"><strong>Active scenario</strong> · {scenario_text} · VLSFO ${vlsfo_price:,.0f}/MT · λ {lam:.2f}</div>',
        unsafe_allow_html=True,
    )

    assignments = list(result.assignments)
    total_net_profit = float(sum(r.get("net_profit", 0.0) for r in assignments))

    if assignments:
        lead = max(assignments, key=lambda r: float(r.get("score", 0.0)))
        lead_vessel = lead.get("vessel_id", "—")
        lead_route = lead.get("route_id", "—")
        lead_origin = lead.get("origin", "—")
        lead_destination = lead.get("destination", "—")
        lead_date = lead.get("date", "—")
        lead_score = float(lead.get("score", 0.0))
    else:
        lead_vessel = lead_route = lead_origin = lead_destination = lead_date = "—"
        lead_score = 0.0

    decision_html = (
        "<div class='decision-eyebrow'>Recommended plan</div>"
        "<div class='decision-head'>"
        "<div><div class='decision-title'>" + str(lead_vessel) + " → " + str(lead_destination) + "</div>"
        "<div class='decision-route'>" + str(lead_origin) + " · Route " + str(lead_route) + " · Loading " + str(lead_date) + "</div></div>"
        "<div><div class='decision-eyebrow'>Selected net profit</div>"
        f"<div class='decision-value'>${total_net_profit:,.0f}</div></div>"
        "</div>"
        "<div class='decision-meta'>"
        f"<span class='decision-chip'>{len(assignments)} vessel assignments</span>"
        f"<span class='decision-chip'>Lead risk-adjusted score {lead_score:.2f}</span>"
        f"<span class='decision-chip'>VLSFO ${vlsfo_price:,.0f}/MT</span>"
        "</div>"
    )
    st.markdown(f'<div class="decision-card">{decision_html}</div>', unsafe_allow_html=True)

    if result.solver_status == "Optimal":
        pill_html = f"<span class='status-optimal'><span class='status-dot'></span>{result.solver_status}</span>"
    else:
        pill_html = f"<span class='status-bad'><span class='status-dot bad-dot'></span>{result.solver_status}</span>"
    st.markdown(f"**Solver status:** {pill_html}", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Objective Value", f"{result.objective_value:,.0f}", help="Optimization objective returned by the solver.")
    col2.metric("Assignments", len(assignments), help="Selected vessel-route assignments.")
    col3.metric("Solve Time", f"{result.solve_time_ms:.0f} ms")
    col4.metric("VLSFO Price", f"${result.vlsfo_price_used:,.0f}/MT", help="Bunker fuel price used by the solver.")

    baseline = st.session_state.baseline_snapshot
    if baseline is not None:
        objective_delta = float(result.objective_value) - baseline["objective"]
        profit_delta = total_net_profit - baseline["net_profit"]
        objective_pct = (objective_delta / baseline["objective"] * 100.0) if baseline["objective"] else 0.0
        profit_pct = (profit_delta / baseline["net_profit"] * 100.0) if baseline["net_profit"] else 0.0
        impact_html = (
            '<div class="vs-section"><span class="section-kicker">00</span>Scenario Impact vs Baseline</div>'
            '<div class="impact-grid">'
            f'<div class="impact-card"><div class="impact-label">Objective change</div><div class="impact-value">{objective_delta:+,.0f} ({objective_pct:+.1f}%)</div><div class="impact-note">Baseline: {baseline["objective"]:,.0f}</div></div>'
            f'<div class="impact-card"><div class="impact-label">Selected net profit</div><div class="impact-value">{profit_delta:+,.0f} ({profit_pct:+.1f}%)</div><div class="impact-note">Baseline: ${baseline["net_profit"]:,.0f}</div></div>'
            f'<div class="impact-card"><div class="impact-label">Scenario inputs</div><div class="impact-value">Freight {freight_shock:+d}% · Fuel ${vlsfo_price:,.0f}</div><div class="impact-note">Baseline fuel: ${baseline["vlsfo_price"]:,.0f}/MT · λ {baseline["lam"]:.2f}</div></div>'
            '</div>'
        )
        st.markdown(impact_html, unsafe_allow_html=True)

    st.markdown("---")

    if not assignments:
        st.warning("Solver returned no assignments. Check laycan windows or capacity constraints.")
    else:
        st.markdown('<div class="vs-section"><span class="section-kicker">01</span>Recommended Vessel–Route Assignments</div>', unsafe_allow_html=True)

        df = pd.DataFrame(assignments)
        raw_scores = {i: row["score"] for i, row in enumerate(assignments)}
        norm_map = normalise_scores(raw_scores)
        df["norm_score"] = [norm_map[i] for i in range(len(df))]
        df["lane_type"] = df["review_status"].map({"KEEP": "Benchmark", "FLAG": "Alpha lane"}).fillna("Benchmark")

        df_display = df.rename(columns={
            "vessel_id": "Vessel", "route_id": "Route", "date": "Loading Date",
            "score": "Risk-Adj. Score", "norm_score": "Norm. Score (0-100)",
            "lane_type": "Lane Type", "p50_rate": "P50 Rate ($/t)", "p10_rate": "P10 Rate ($/t)",
            "p90_rate": "P90 Rate ($/t)", "origin": "Origin", "destination": "Destination",
            "cargo_dwt": "Cargo (DWT)", "vessel_capacity_dwt": "Vessel Cap. (DWT)",
            "transit_days": "Transit (days)", "net_profit": "Net Profit ($)",
            "voyage_cost": "Voyage Cost ($)", "port_waiting_days": "Port Waiting (days)",
        })

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk-Adj. Score": st.column_config.NumberColumn(format="%.4f"),
                "Norm. Score (0-100)": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "Lane Type": st.column_config.TextColumn(help="Benchmark = directly evidenced route; Alpha lane = planning extension."),
                "P50 Rate ($/t)": st.column_config.NumberColumn(format="$%.2f"),
                "P10 Rate ($/t)": st.column_config.NumberColumn(format="$%.2f"),
                "P90 Rate ($/t)": st.column_config.NumberColumn(format="$%.2f"),
                "Cargo (DWT)": st.column_config.NumberColumn(format="%d"),
                "Vessel Cap. (DWT)": st.column_config.NumberColumn(format="%d"),
                "Net Profit ($)": st.column_config.NumberColumn(format="$%.0f"),
                "Voyage Cost ($)": st.column_config.NumberColumn(format="$%.0f"),
                "Port Waiting (days)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        st.markdown('<div class="vs-section"><span class="section-kicker">02</span>Assignment Score Comparison</div>', unsafe_allow_html=True)
        bar_labels = [f"{r['vessel_id']} → {r['route_id']}" for r in assignments]
        bar_scores = [norm_map[i] for i in range(len(assignments))]
        bar_colors = ["#f1a15a" if r.get("review_status") == "FLAG" else "#49c8bc" for r in assignments]

        fig = go.Figure(go.Bar(
            x=bar_labels,
            y=bar_scores,
            marker_color=bar_colors,
            text=[f"{s:.1f}" for s in bar_scores],
            textposition="outside",
            hovertemplate="%{x}<br>Normalised score: %{y:.1f}<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Assignment",
            yaxis_title="Normalised Score (0–100)",
            yaxis_range=[0, 115],
            margin=dict(t=24, b=56, l=42, r=24),
            height=360,
            font=dict(family="Helvetica, Arial, sans-serif", color="#dcebee"),
            xaxis=dict(gridcolor="rgba(120,201,215,0.08)", zerolinecolor="rgba(120,201,215,0.10)"),
            yaxis=dict(gridcolor="rgba(120,201,215,0.08)", zerolinecolor="rgba(120,201,215,0.10)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Data provenance label ─────────────────────────────────────────────────
    st.markdown(
        "<div class='data-label'><strong>Phase 1 · Placeholder data.</strong> "
        "Freight rates are constant synthetic proxies calibrated to real BDI ranges "
        "(15–25 $/t). This is <em>not</em> real Baltic Exchange data. "
        "Real LightGBM quantile forecast will replace this on Day 3.</div>",
        unsafe_allow_html=True,
    )