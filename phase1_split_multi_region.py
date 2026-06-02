"""Phase 1 multi-region: split a single multi-site weather CSV into per-region
cleaned + processed (Target_Power_kW) CSVs.

Input contract (`paths.raw_multi_region_csv`):
  A single CSV containing rows for several sites, with columns:
    latitude, longitude, datetime, temp, humidity, precip, cloudcover,
    solarradiation, ghiradiation, dniradiation, difradiation, conditions, source

The lat/lon values are matched (rounded to 2 decimal places) against the
`regions.list` in `experiment_config.json`. Each region's slice is then
cleaned with the same rules as the legacy `archive/legacy/phase1a_data_cleaning.py` (tz-localize to
Asia/Kuching, resample 1H, interpolate limit=3, bfill/ffill, clip non-neg
radiative columns) and run through `phase1b_physics_synthesis.simulate_pv_for_location`
using that region's coordinates.

Outputs (per region, paths from config templates):
  data/processed/{region}_weather_cleaned.csv
  data/processed/{region}_weather_processed.csv   (includes Target_Power_kW)

The Kuching processed CSV is also copied (if needed) to the legacy
`paths.phase1_processed_csv` location so existing single-region scripts keep
working without modification.

Idempotency:
  Skip a region whose processed output already exists AND is newer than
  the raw multi-region CSV, unless `--force` is passed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from phase1b_physics_synthesis import simulate_pv_for_location
from repo_paths import repo_root, resolve_repo_path


_RADIATIVE_COLS_NONNEG = ("precip", "solarradiation", "ghiradiation", "dniradiation", "difradiation")


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_raw_multi_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in ("latitude", "longitude", "datetime") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw multi-region CSV missing required columns {missing} (got {list(df.columns)})"
        )
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    if df["datetime"].isna().any():
        bad = int(df["datetime"].isna().sum())
        raise ValueError(f"{bad} rows have unparseable `datetime` in {csv_path.name}.")
    return df


def _match_region(row_lat: float, row_lon: float, region_list: Iterable[dict], tol: float) -> str | None:
    """Return the region.name whose (lat,lon) is within `tol` degrees of the row, else None."""
    best_name: str | None = None
    best_dist = float("inf")
    for r in region_list:
        d = max(abs(float(r["lat"]) - row_lat), abs(float(r["lon"]) - row_lon))
        if d <= tol and d < best_dist:
            best_dist = d
            best_name = str(r["name"])
    return best_name


def _slice_by_region(df_raw: pd.DataFrame, regions: list[dict], tol: float) -> dict[str, pd.DataFrame]:
    """Group `df_raw` rows into per-region dataframes using rounded (lat,lon)."""
    rounded_lat = df_raw["latitude"].astype(float).round(2)
    rounded_lon = df_raw["longitude"].astype(float).round(2)
    key_df = pd.DataFrame({"lat": rounded_lat, "lon": rounded_lon})
    keys = key_df.drop_duplicates().to_records(index=False)

    region_name_by_key: dict[tuple[float, float], str | None] = {}
    for lat, lon in keys:
        region_name_by_key[(float(lat), float(lon))] = _match_region(float(lat), float(lon), regions, tol)

    unmatched = sorted({k for k, v in region_name_by_key.items() if v is None})
    if unmatched:
        print(
            f"WARNING: {len(unmatched)} (lat,lon) pairs did not match any configured region "
            f"within tol={tol} deg; their rows will be dropped. Examples: {unmatched[:3]}"
        )

    region_to_rows: dict[str, list[pd.Series]] = {r["name"]: [] for r in regions}
    df_raw = df_raw.copy()
    df_raw["_lat2"] = rounded_lat
    df_raw["_lon2"] = rounded_lon

    out: dict[str, pd.DataFrame] = {}
    for r in regions:
        rname = r["name"]
        mask = pd.Series(False, index=df_raw.index)
        for (lat, lon), matched in region_name_by_key.items():
            if matched == rname:
                mask |= (df_raw["_lat2"] == lat) & (df_raw["_lon2"] == lon)
        slice_df = df_raw.loc[mask].drop(columns=["_lat2", "_lon2"]).copy()
        out[rname] = slice_df
    return out


def _clean_one_region(df_region: pd.DataFrame, *, region_name: str) -> pd.DataFrame:
    """Apply the Phase-1a cleaning rules to a single region's slice."""
    if df_region.empty:
        return df_region.copy()

    df = df_region.copy()
    df = df.set_index("datetime").sort_index()
    # Localize tz only if naive (the multi-region CSV ships tz-naive)
    if df.index.tz is None:
        df = df.tz_localize("Asia/Kuching", ambiguous="infer", nonexistent="shift_forward")
    else:
        df = df.tz_convert("Asia/Kuching")

    df = df.resample("1h").asfreq()
    # Only numeric columns participate in linear interpolation; categorical
    # columns like `conditions`/`source` get forward/back-filled instead. This
    # also silences the pandas FutureWarning about object-dtype interpolation.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3)
    df = df.bfill().ffill()

    for col in _RADIATIVE_COLS_NONNEG:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    missing_after = int(df.isna().sum().sum())
    if missing_after > 0:
        print(
            f"WARNING [{region_name}]: {missing_after} NaN cells remain after interpolation/bfill/ffill."
        )
    return df


def _newer_than(target: Path, source: Path) -> bool:
    """True iff target exists and was modified after source."""
    return target.exists() and target.stat().st_mtime >= source.stat().st_mtime


def split_and_process_regions(cfg: dict, *, force: bool = False, verbose: bool = True) -> dict[str, Path]:
    """Top-level entry: produces per-region cleaned + processed CSVs.

    Returns a mapping `region_name -> processed_csv_path` for the regions that
    were either rebuilt or already up to date.
    """
    repo = repo_root()
    paths_cfg = cfg.get("paths", {}) or {}
    regions_cfg = cfg.get("regions", {}) or {}
    region_list = list(regions_cfg.get("list", []))
    tol = float(regions_cfg.get("match_tolerance_deg", 0.02))

    raw_csv_rel = regions_cfg.get("raw_multi_csv") or paths_cfg.get("raw_multi_region_csv") or "swkdataweather.csv"
    raw_csv = resolve_repo_path(repo, str(raw_csv_rel))
    if not raw_csv.exists():
        raise FileNotFoundError(f"Multi-region raw CSV not found at {raw_csv}.")

    cleaned_template = paths_cfg.get("cleaned_csv_template", "data/processed/{region}_weather_cleaned.csv")
    processed_template = paths_cfg.get("processed_csv_template", "data/processed/{region}_weather_processed.csv")
    legacy_processed_rel = paths_cfg.get("phase1_processed_csv")

    processed_out = resolve_repo_path(repo, processed_template.format(region=region_list[0]["name"]))
    processed_out.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[multi-region] Reading {raw_csv} ...")
    t0 = time.perf_counter()
    df_raw = _read_raw_multi_csv(raw_csv)
    if verbose:
        print(f"[multi-region] Loaded {len(df_raw):,} rows in {time.perf_counter()-t0:.1f}s")

    slices = _slice_by_region(df_raw, region_list, tol)

    produced: dict[str, Path] = {}
    for region in region_list:
        rname = region["name"]
        lat = float(region["lat"])
        lon = float(region["lon"])
        cleaned_path = resolve_repo_path(repo, cleaned_template.format(region=rname))
        processed_path = resolve_repo_path(repo, processed_template.format(region=rname))
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        if not force and _newer_than(processed_path, raw_csv):
            if verbose:
                print(f"[{rname}] up-to-date: {processed_path.name} (skip; use --force to rebuild)")
            produced[rname] = processed_path
            continue

        df_slice = slices.get(rname, pd.DataFrame())
        if df_slice.empty:
            print(f"[{rname}] WARNING: no rows matched in raw CSV; skipping.")
            continue

        if verbose:
            print(f"[{rname}] cleaning {len(df_slice):,} rows (lat={lat}, lon={lon})...")
        t_clean = time.perf_counter()
        df_clean = _clean_one_region(df_slice, region_name=rname)
        df_clean.to_csv(cleaned_path)
        if verbose:
            print(f"[{rname}] cleaned -> {cleaned_path.name} ({time.perf_counter()-t_clean:.1f}s)")

        if verbose:
            print(f"[{rname}] running pvlib model chain ...")
        t_pv = time.perf_counter()
        df_processed = simulate_pv_for_location(
            df_clean,
            latitude=lat,
            longitude=lon,
            region_name=rname,
            tz="Asia/Kuching",
        )
        df_processed.index = df_processed.index.tz_localize(None) if getattr(df_processed.index, "tz", None) is not None else df_processed.index
        df_processed.to_csv(processed_path)
        if verbose:
            print(
                f"[{rname}] processed -> {processed_path.name} "
                f"(rows={len(df_processed):,}; pv-energy-yr={float(df_processed['Target_Power_kW'].sum()):.0f} kWh; "
                f"{time.perf_counter()-t_pv:.1f}s)"
            )
        produced[rname] = processed_path

    # Back-compat: mirror the active region (default: kuching) into the legacy single-region path.
    if legacy_processed_rel:
        active = str(regions_cfg.get("active") or region_list[0]["name"])
        active_processed = produced.get(active)
        if active_processed and active_processed.exists():
            legacy_path = resolve_repo_path(repo, legacy_processed_rel)
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            if not _newer_than(legacy_path, active_processed):
                shutil.copy2(active_processed, legacy_path)
                if verbose:
                    print(f"[legacy] copied {active_processed.name} -> {legacy_path}")

    return produced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment_config.json", help="path to experiment_config.json")
    parser.add_argument("--force", action="store_true", help="rebuild outputs even if up-to-date")
    args = parser.parse_args(argv)

    repo = repo_root()
    cfg = _load_config(resolve_repo_path(repo, args.config))
    produced = split_and_process_regions(cfg, force=args.force, verbose=True)
    print(f"Done. Produced {len(produced)} region datasets: {sorted(produced)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
