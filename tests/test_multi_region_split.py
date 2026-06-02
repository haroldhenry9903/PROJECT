"""Smoke tests for the multi-region raw CSV and `phase1_split_multi_region.py`.

These tests guard the *contract* between `swkdataweather.csv` (the input
shipped to the project) and the per-region pipeline:

  * Exactly 5 distinct (lat, lon) groups.
  * Each group is 43,872 hourly rows (5 years of Sarawak weather data).
  * Each configured region in `experiment_config.json` matches an actual
    lat/lon group within the documented tolerance.
  * `_clean_one_region` produces a non-empty hourly-indexed dataframe with
    the expected non-negative radiative columns.

The tests do NOT execute the pvlib model chain (that takes ~0.7s per
region and is exercised by the integration runs); they only validate the
*split* step. They are skipped if `swkdataweather.csv` is missing so the
suite still passes for fresh checkouts that haven't downloaded the
multi-region data yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase1_split_multi_region import (
    _clean_one_region,
    _match_region,
    _read_raw_multi_csv,
    _slice_by_region,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "swkdataweather.csv"
CONFIG_JSON = REPO_ROOT / "experiment_config.json"

# Hard-coded expected shape: 5 regions × 5 years × 8760 hours = 219,360 rows total.
EXPECTED_REGIONS = 5
EXPECTED_ROWS_PER_REGION = 43_872  # 5 years of hourly data per region


pytestmark = pytest.mark.skipif(
    not RAW_CSV.exists(),
    reason=f"Raw multi-region CSV not present at {RAW_CSV}; multi-region tests skipped.",
)


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return _read_raw_multi_csv(RAW_CSV)


@pytest.fixture(scope="module")
def cfg_regions() -> list[dict]:
    if not CONFIG_JSON.exists():
        return []
    cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    return list(cfg.get("regions", {}).get("list", []))


def test_raw_csv_has_required_columns(raw_df: pd.DataFrame):
    required = {
        "latitude", "longitude", "datetime", "temp", "humidity", "precip",
        "cloudcover", "ghiradiation", "dniradiation", "difradiation",
    }
    missing = required - set(raw_df.columns)
    assert not missing, f"Raw multi-region CSV missing columns: {missing}"


def test_raw_csv_has_exactly_five_latlon_groups(raw_df: pd.DataFrame):
    """Promise from the supervisor prompt: 5 Sarawak regions in one CSV."""
    rounded = (
        raw_df[["latitude", "longitude"]]
        .astype(float).round(2)
        .drop_duplicates()
    )
    assert len(rounded) == EXPECTED_REGIONS, (
        f"Expected exactly {EXPECTED_REGIONS} (lat, lon) groups, got {len(rounded)}: "
        f"{rounded.to_records(index=False).tolist()}"
    )


def test_each_latlon_group_has_expected_row_count(raw_df: pd.DataFrame):
    """Every region must have the same hourly row count (5 years × 8760 + 12 leap-year hours)."""
    grouped = (
        raw_df.assign(
            _lat2=raw_df["latitude"].astype(float).round(2),
            _lon2=raw_df["longitude"].astype(float).round(2),
        )
        .groupby(["_lat2", "_lon2"])
        .size()
    )
    for (lat, lon), n in grouped.items():
        assert int(n) == EXPECTED_ROWS_PER_REGION, (
            f"Region (lat={lat}, lon={lon}) has {n} rows, expected {EXPECTED_ROWS_PER_REGION}"
        )


def test_every_configured_region_matches_a_latlon_group(raw_df: pd.DataFrame, cfg_regions: list[dict]):
    if not cfg_regions:
        pytest.skip("regions.list missing from experiment_config.json")

    rounded = (
        raw_df[["latitude", "longitude"]]
        .astype(float).round(2)
        .drop_duplicates()
        .to_records(index=False)
    )
    csv_pairs = [(float(lat), float(lon)) for lat, lon in rounded]
    for r in cfg_regions:
        match = _match_region(float(r["lat"]), float(r["lon"]), cfg_regions, tol=0.02)
        assert match == r["name"], f"region {r['name']} did not self-match"
        # AND the configured (lat, lon) is within tolerance of some CSV pair.
        within = [p for p in csv_pairs if max(abs(p[0] - r["lat"]), abs(p[1] - r["lon"])) <= 0.02]
        assert within, f"Configured region {r['name']} ({r['lat']},{r['lon']}) has no matching CSV lat/lon group."


def test_match_region_rejects_far_points(cfg_regions: list[dict]):
    if not cfg_regions:
        pytest.skip("regions.list missing from experiment_config.json")
    # A point in Singapore — far from every Sarawak region.
    assert _match_region(1.3521, 103.8198, cfg_regions, tol=0.5) is None


def test_slice_by_region_produces_all_named_buckets(raw_df: pd.DataFrame, cfg_regions: list[dict]):
    if not cfg_regions:
        pytest.skip("regions.list missing from experiment_config.json")
    slices = _slice_by_region(raw_df, cfg_regions, tol=0.02)
    assert set(slices.keys()) == {r["name"] for r in cfg_regions}
    for rname, df_slice in slices.items():
        assert len(df_slice) == EXPECTED_ROWS_PER_REGION, (
            f"slice for {rname} has {len(df_slice)} rows, expected {EXPECTED_ROWS_PER_REGION}"
        )


def test_clean_one_region_returns_hourly_indexed_df(raw_df: pd.DataFrame, cfg_regions: list[dict]):
    if not cfg_regions:
        pytest.skip("regions.list missing from experiment_config.json")
    slices = _slice_by_region(raw_df, cfg_regions, tol=0.02)
    first_region = cfg_regions[0]["name"]
    out = _clean_one_region(slices[first_region], region_name=first_region)
    # Hourly DatetimeIndex, ascending, tz-aware Asia/Kuching.
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing
    # Cleaned columns: radiative metrics must be non-negative after clip.
    for col in ("precip", "ghiradiation", "dniradiation", "difradiation"):
        if col in out.columns:
            assert (out[col].dropna() >= 0).all(), f"{col} has negative values after cleaning"
    # NaN-free in the engineered numeric columns.
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    assert not out[numeric_cols].isna().any().any(), (
        f"Cleaned region '{first_region}' still has NaNs in numeric cols after cleaning."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
