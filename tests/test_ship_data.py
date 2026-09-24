from __future__ import annotations

import json

import pytest

from src.data.ship_data import load_ship_catalog


def test_loads_aliases_and_excludes_inactive_vessels(tmp_path):
    source = tmp_path / "fleet.csv"
    source.write_text(
        "ship_id,imo_number,name,ship_type,dwt,speed_kn,active\n"
        "V-101,1234567,SEA STAR,Supramax,52000,13.5,yes\n"
        "V-102,7654321,OLD STAR,Supramax,50000,13,no\n",
        encoding="utf-8",
    )

    vessels = load_ship_catalog(source)

    assert len(vessels) == 1
    assert vessels[0]["vessel_id"] == "V-101"
    assert vessels[0]["imo"] == "1234567"
    assert vessels[0]["capacity_dwt"] == 52000
    assert vessels[0]["min_cargo_dwt"] == 44200


def test_loads_json_vessels(tmp_path):
    source = tmp_path / "fleet.json"
    source.write_text(
        json.dumps({"vessels": [{"vessel_id": "V-201", "vessel_name": "OCEAN", "capacity_dwt": 51000}]}),
        encoding="utf-8",
    )

    assert load_ship_catalog(source)[0]["vessel_name"] == "OCEAN"


def test_rejects_duplicate_ids(tmp_path):
    source = tmp_path / "fleet.csv"
    source.write_text(
        "vessel_id,vessel_name,capacity_dwt\nV-1,A,50000\nV-1,B,51000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate vessel_id"):
        load_ship_catalog(source)


def test_rejects_missing_required_columns(tmp_path):
    source = tmp_path / "fleet.csv"
    source.write_text("vessel_id,vessel_name\nV-1,A\n", encoding="utf-8")

    with pytest.raises(ValueError, match="capacity_dwt"):
        load_ship_catalog(source)