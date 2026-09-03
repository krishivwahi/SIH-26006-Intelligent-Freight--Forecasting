"""
src/solver/risk.py

Isolated risk-adjusted score computation.

Formula (from AGENT_CONTEXT.md §8.2):
    Score(v, r, t) = P50_Profit - λ * (P50_Profit - P10_Profit)

This module is intentionally framework-free so it can be unit-tested
independently of PuLP and Streamlit.

NOTE: Do NOT call this CVaR. It is a downside-penalty risk score.
Full CVaR with scenario generation is scoped for Beta.
"""
from __future__ import annotations


def compute_score(p50: float, p10: float, lam: float = 0.5) -> float:
    """Compute the risk-adjusted score for a single (vessel, route, day) triple.

    Args:
        p50: Median (50th percentile) freight rate forecast in $/ton.
        p10: Pessimistic (10th percentile) freight rate forecast in $/ton.
        lam: Risk aversion parameter λ ∈ [0, 1]. Higher = more conservative.
             Default 0.5 per AGENT_CONTEXT.md.

    Returns:
        Risk-adjusted score. Higher is better.

    Raises:
        ValueError: If p10 > p50 (violates quantile ordering) or lam not in [0, 1].
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"λ must be in [0, 1], got {lam}")
    if p10 > p50:
        raise ValueError(
            f"Quantile ordering violated: p10={p10} > p50={p50}. "
            "Check forecast JSON for this (vessel, route, date) triple."
        )
    return p50 - lam * (p50 - p10)


def compute_scores_bulk(
    records: list[dict],
    lam: float = 0.5,
) -> dict[tuple[str, str, str], float]:
    """Compute risk-adjusted scores for all forecast records in one pass.

    Args:
        records: List of dicts matching the JSON contract schema:
                 {date_index, vessel_id, route_id, p10_rate, p50_rate, p90_rate}
        lam: Risk aversion parameter λ.

    Returns:
        Dict keyed by (vessel_id, route_id, date_index) → score.
    """
    scores: dict[tuple[str, str, str], float] = {}
    for rec in records:
        key = (rec["vessel_id"], rec["route_id"], rec["date_index"])
        scores[key] = compute_score(
            p50=float(rec["p50_rate"]),
            p10=float(rec["p10_rate"]),
            lam=lam,
        )
    return scores
