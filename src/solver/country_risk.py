"""
src/solver/country_risk.py

Semantic country-risk variable: public-sector corruption + workforce /
logistics efficiency at the ORIGIN country, converted into an expected extra
port-side delay (days) that feeds the voyage-cost model in solver.py the same
way PORT_WAITING_DAYS already does for destination ports.

WHY THIS EXISTS
    AGENT_CONTEXT.md §2 lists "Geopolitical shocks" as "Not in Alpha" and
    BETA_ROADMAP.md §3.2 scopes full sanctions/canal-disruption detection as a
    5-7 day Beta item requiring an NLP/news pipeline. This module is a much
    narrower, Alpha-sized slice of that idea: a static, per-origin-country
    corruption + logistics-efficiency composite, applied as a deterministic
    day-count adjustment. It is intentionally simple enough to defend by hand,
    the same way LAMBDA=0.5 is defended in risk.py — NOT a fitted model.

DATA SOURCES (public, free, re-checkable — see COUNTRY_RISK_INDEX below)
    - Corruption Perceptions Index (CPI), Transparency International.
      Scale 0 (highly corrupt) .. 100 (very clean).
      https://www.transparency.org/en/cpi
    - Logistics Performance Index (LPI), World Bank.
      Scale 1 (low) .. 5 (high); overall score aggregates six sub-components,
      including "competence and quality of logistics services" and "efficiency
      of customs clearance" — the closest public, citable proxy for
      "workforce efficiency" at ports. https://lpi.worldbank.org

    NOTE ON "UN data" / "Bloomberg reports" (as originally requested):
    The UN does not itself publish a standalone corruption or workforce-
    efficiency index, and UNCTAD's own maritime-trade analysis (Review of
    Maritime Transport) leans on the World Bank LPI as the standard logistics
    benchmark — so LPI is used here as the UN-adjacent proxy. A Bloomberg
    terminal subscription was not available in this environment, so no
    Bloomberg-sourced figures are included; CPI + CPI/LPI are the standard
    public substitutes analysts cite for this exact purpose. If your team has
    a Bloomberg/UNCTAD port-congestion feed, drop the real numbers into
    COUNTRY_RISK_INDEX below — no other function in this file needs to change.

SWAP STATUS:
     COUNTRY_RISK_INDEX currently covers exactly the three origin countries
       present in parameters.ROUTES today (Australia, Canada, USA). Add any
       new origin country here BEFORE it appears in ROUTES — get_country_risk()
       fails loudly (KeyError) on an unmapped country rather than silently
       treating it as zero risk, matching the fail-loud convention already
       used for duplicate/missing keys in solver.py's _load_forecast().

ASSUMPTIONS (traceable, same convention as parameters.py):
    CR-A01  CPI figures are Transparency International's 2025 release
            (published Feb 2026). LPI figures are the World Bank's 2023
            release (the most recent edition at the time of writing; the next
            LPI edition is not yet published).
    CR-A02  Composite weighting is 50% corruption / 50% workforce-efficiency,
            an even split chosen for defensibility, not fitted to historical
            demurrage data (no such dataset was available in this repo).
    CR-A03  MAX_COUNTRY_RISK_DELAY_DAYS=3.0 is a planning ceiling: the extra
            delay a fully-corrupt (CPI=0), least-efficient (LPI=1) origin
            country would contribute, on the same order of magnitude as the
            existing PORT_WAITING_DAYS entries (1-2 days) in parameters.py.
    CR-A04  Applied per ORIGIN country only (loading side). Destination-side
            delay is already modelled by PORT_WAITING_DAYS; the two are
            additive, not double-counted against the same leg.
"""
from __future__ import annotations

# CPI: 0 (highly corrupt) .. 100 (very clean) — Transparency International 2025.
# LPI: 1 (low) .. 5 (high) overall score — World Bank Logistics Performance Index 2023.
COUNTRY_RISK_INDEX: dict[str, dict] = {
    "Australia": {
        "cpi_score": 76,
        "lpi_score": 3.7,
        "source_cpi": "Transparency International CPI 2025",
        "source_lpi": "World Bank LPI 2023",
    },
    "Canada": {
        "cpi_score": 75,
        "lpi_score": 4.0,
        "source_cpi": "Transparency International CPI 2025",
        "source_lpi": "World Bank LPI 2023",
    },
    "USA": {
        "cpi_score": 64,
        "lpi_score": 3.8,
        "source_cpi": "Transparency International CPI 2025",
        "source_lpi": "World Bank LPI 2023",
    },
    "Indonesia": {
        "cpi_score": 34,
        "lpi_score": 3.0,
        "source_cpi": "Transparency International CPI 2025",
        "source_lpi": "World Bank LPI 2023",
    },
    "Mozambique": {
        "cpi_score": 25,
        "lpi_score": 2.4,
        "source_cpi": "Transparency International CPI 2025",
        "source_lpi": "World Bank LPI 2023",
    },
}

# Tunable knob, exposed here so a future Streamlit slider can override it per
# run — same pattern as LAMBDA in parameters.py. See ASSUMPTION CR-A03.
MAX_COUNTRY_RISK_DELAY_DAYS: float = 3.0

# Composite weighting between the two components. See ASSUMPTION CR-A02.
CORRUPTION_WEIGHT: float = 0.5
WORKFORCE_WEIGHT: float = 0.5


def _origin_country(origin: str) -> str:
    """Extract the country name from a "Port, Country" origin string.

    Example: "Hay Point, Australia" -> "Australia"

    Raises:
        ValueError: if origin does not follow the "Port, Country" convention
        used throughout parameters.ROUTES.
    """
    if "," not in origin:
        raise ValueError(
            f"origin '{origin}' does not match the 'Port, Country' convention "
            "used in parameters.ROUTES — cannot determine origin country."
        )
    return origin.rsplit(",", 1)[-1].strip()


def get_country_risk(origin: str) -> dict:
    """Look up the corruption/logistics-efficiency entry for a route's origin.

    Args:
        origin: Route origin string, e.g. "Hay Point, Australia".

    Returns:
        The COUNTRY_RISK_INDEX entry for that country.

    Raises:
        KeyError: if the country has no entry — fail loud, not silent 0 risk.
    """
    country = _origin_country(origin)
    if country not in COUNTRY_RISK_INDEX:
        raise KeyError(
            f"No corruption/workforce-efficiency data for '{country}' "
            f"(origin='{origin}'). Add it to COUNTRY_RISK_INDEX in "
            "src/solver/country_risk.py before using this origin in ROUTES."
        )
    return COUNTRY_RISK_INDEX[country]


def compute_expected_delay_days(origin: str) -> float:
    """Composite corruption + workforce-efficiency score -> expected extra port days.

    Formula (Alpha modelling assumption — see CR-A02):
        corruption_component = (100 - cpi_score) / 100   # 0 (clean) .. 1 (corrupt)
        workforce_component  = (5 - lpi_score) / 4        # 0 (efficient) .. 1 (inefficient)
        composite            = CORRUPTION_WEIGHT * corruption_component
                              + WORKFORCE_WEIGHT  * workforce_component
        delay_days           = composite * MAX_COUNTRY_RISK_DELAY_DAYS

    Higher corruption and lower logistics/workforce efficiency at the origin
    country both push the expected loading delay up.

    Args:
        origin: Route origin string, e.g. "Hay Point, Australia".

    Returns:
        Expected extra port-side delay in days (>= 0), rounded to 3 dp.
    """
    risk = get_country_risk(origin)
    corruption_component = (100 - risk["cpi_score"]) / 100
    workforce_component = (5 - risk["lpi_score"]) / 4
    composite = (
        CORRUPTION_WEIGHT * corruption_component
        + WORKFORCE_WEIGHT * workforce_component
    )
    return round(composite * MAX_COUNTRY_RISK_DELAY_DAYS, 3)


def compute_all_route_delays(routes: list[dict]) -> dict[str, float]:
    """Bulk helper: {route_id: expected_country_risk_delay_days} for every route.

    Args:
        routes: parameters.ROUTES (or any list of dicts with route_id + origin).

    Returns:
        Dict keyed by route_id.
    """
    return {r["route_id"]: compute_expected_delay_days(r["origin"]) for r in routes}
