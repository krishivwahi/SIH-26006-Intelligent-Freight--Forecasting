"""
tests/test_country_risk.py

Unit tests for src/solver/country_risk.py — the corruption + workforce/
logistics-efficiency delay variable.
"""
from __future__ import annotations

import pytest

from src.solver.country_risk import (
    COUNTRY_RISK_INDEX,
    MAX_COUNTRY_RISK_DELAY_DAYS,
    compute_all_route_delays,
    compute_expected_delay_days,
    get_country_risk,
)
from src.solver.parameters import ROUTES


class TestOriginParsing:
    def test_known_country_resolves(self) -> None:
        assert get_country_risk("Hay Point, Australia")["cpi_score"] == 76

    def test_extra_whitespace_is_stripped(self) -> None:
        assert get_country_risk("Hampton Roads,   USA") == COUNTRY_RISK_INDEX["USA"]

    def test_malformed_origin_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            get_country_risk("Nowhere")

    def test_unmapped_country_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_country_risk("Some Port, Atlantis")


class TestComputeExpectedDelayDays:
    def test_delay_is_non_negative_and_bounded(self) -> None:
        for country in COUNTRY_RISK_INDEX:
            delay = compute_expected_delay_days(f"Some Port, {country}")
            assert 0.0 <= delay <= MAX_COUNTRY_RISK_DELAY_DAYS

    def test_more_corrupt_or_less_efficient_means_more_delay(self) -> None:
        """USA (CPI 64) should show a larger delay than Canada (CPI 75, higher LPI)."""
        usa_delay = compute_expected_delay_days("Hampton Roads, USA")
        canada_delay = compute_expected_delay_days("Vancouver, Canada")
        assert usa_delay > canada_delay

    def test_formula_matches_hand_calculation(self) -> None:
        """Australia: cpi=76, lpi=3.7 -> composite=0.5*0.24 + 0.5*0.325=0.2825 -> *3.0"""
        expected = round((0.5 * (100 - 76) / 100 + 0.5 * (5 - 3.7) / 4) * MAX_COUNTRY_RISK_DELAY_DAYS, 3)
        assert compute_expected_delay_days("Hay Point, Australia") == pytest.approx(expected)


class TestComputeAllRouteDelays:
    def test_covers_every_route_in_parameters(self) -> None:
        """Every real ROUTES origin must already have a COUNTRY_RISK_INDEX entry —
        this is the 'fail loud, not silent' contract with parameters.py."""
        delays = compute_all_route_delays(ROUTES)
        assert set(delays.keys()) == {r["route_id"] for r in ROUTES}
        assert all(d >= 0.0 for d in delays.values())
