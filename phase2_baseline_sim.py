import json
import sys

import numpy as np
import pandas as pd

from core_battery_model import Battery
from repo_paths import repo_root, resolve_repo_path


def generate_synthetic_load(index: pd.DatetimeIndex, *, rng: np.random.Generator) -> pd.Series:
    """
    Generates a synthetic residential load profile.
    - Lower at night
    - Morning peak (06:00 - 08:00)
    - Evening peak (18:00 - 22:00) is the highest
    - Weekend slightly higher than weekday
    - Daily consumption targeted around 15-20 kWh
    - Added stochastic noise
    """
    hours = index.hour
    
    # Base load profile mapping per hour (approximate relative shape)
    # We will scale this base shape and add noise.
    base_shape = np.zeros(len(index))
    
    # Simple predefined shape rules (relative shape, will be scaled per-day to a target kWh)
    # Night time (00:00 - 05:00): Base/essential loads
    mask_night = (hours >= 0) & (hours < 6)
    base_shape[mask_night] = 0.45
    
    # Morning peak (06:00 - 08:00): Ramping up
    mask_morning = (hours >= 6) & (hours <= 8)
    base_shape[mask_morning] = 0.8
    
    # Mid-day (09:00 - 17:00): Moderate consumption
    mask_day = (hours >= 9) & (hours <= 17)
    base_shape[mask_day] = 0.6
    
    # Evening peak (18:00 - 22:00): High consumption
    mask_evening = (hours >= 18) & (hours <= 22)
    base_shape[mask_evening] = 1.8
    
    # Late night (23:00): Winding down
    mask_late = (hours == 23)
    base_shape[mask_late] = 0.5
    
    # Add day-by-day stochastic noise (normal distribution)
    # Adding a random scaling factor per day to keep the overall shape but vary intensity
    days = index.normalize()
    unique_days = pd.Index(days.unique()).sort_values()
    
    load = pd.Series(base_shape, index=index)
    
    for day in unique_days:
        day_mask = days == day
        # Weekend effect (Sat/Sun)
        is_weekend = pd.Timestamp(day).dayofweek >= 5
        weekend_mult = 1.10 if is_weekend else 1.00

        # Target daily energy (kWh) ~ 15-20 kWh
        target_kwh = float(rng.uniform(15.0, 20.0)) * weekend_mult

        # Day-level stochastic variation
        daily_variation = float(rng.normal(loc=1.0, scale=0.08))
        # Add hourly noise
        hourly_noise = rng.normal(loc=1.0, scale=0.1, size=day_mask.sum())
        
        profile = load.loc[day_mask].astype(float) * daily_variation * hourly_noise
        profile = profile.clip(lower=0.1)

        # Scale to match target daily kWh (since 1-hour steps, sum(kW) ≈ kWh/day)
        day_sum = float(profile.sum())
        scale = 1.0 if day_sum <= 0 else (target_kwh / day_sum)
        load.loc[day_mask] = (profile * scale).clip(lower=0.1)
        
    # Ensure no negative loads
    load = load.clip(lower=0.1)
    
    return load

def get_tariff_rate(hour: int) -> float:
    """
    Returns Sarawak Energy Tariff C3 rate based on the hour.
    Peak (07:00-22:00): 0.245 RM/kWh
    Off-Peak (22:00-07:00): 0.139 RM/kWh
    """
    if 7 <= hour < 22:
        return 0.245
    else:
        return 0.139


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_datetime_series_to_naive(s: pd.Series) -> pd.Series:
    """Parse datetimes and ensure tz-naive timestamps (legacy tz-aware CSVs are normalized)."""
    dt = pd.to_datetime(s, errors="coerce")
    if getattr(dt.dt, "tz", None) is not None:
        # Drop timezone while keeping local clock time (pipeline standard is tz-naive)
        dt = dt.dt.tz_localize(None)
    return dt


def tariff_rate_from_config(hour: int, *, tariff_cfg: dict) -> float:
    """Tariff from config (single source of truth). Falls back to C3 defaults if keys missing."""
    peak_start = int(tariff_cfg.get("peak_start_hour", 7))
    peak_end = int(tariff_cfg.get("peak_end_hour", 22))
    peak_rate = float(tariff_cfg.get("peak_rate_rm_per_kwh", 0.245))
    offpeak_rate = float(tariff_cfg.get("offpeak_rate_rm_per_kwh", 0.139))
    return peak_rate if peak_start <= hour < peak_end else offpeak_rate

def main():
    repo = repo_root()
    argv = sys.argv[1:]
    config_path = "experiment_config.json"
    region: str | None = None
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]; i += 2; continue
        if argv[i] == "--region" and i + 1 < len(argv):
            region = argv[i + 1]; i += 2; continue
        i += 1

    cfg_path = resolve_repo_path(repo, config_path)
    cfg = _load_config(str(cfg_path))
    paths_cfg = cfg.get("paths", {}) or {}

    # Region-aware path resolution. When --region is supplied AND the config
    # defines templates, use them; otherwise fall back to the flat single-region
    # paths so legacy callers (`python phase2_baseline_sim.py`) keep working.
    processed_tpl = paths_cfg.get("processed_csv_template")
    baseline_tpl = paths_cfg.get("baseline_results_template")
    if region and processed_tpl and baseline_tpl:
        input_filepath = str(resolve_repo_path(repo, processed_tpl.format(region=region)))
        output_filepath = str(resolve_repo_path(repo, baseline_tpl.format(region=region)))
        print(f"[phase2] region={region}  in={input_filepath}  out={output_filepath}")
    else:
        input_filepath = str(
            resolve_repo_path(repo, paths_cfg.get("phase1_processed_csv", "data/processed/sarawak_weather_processed.csv"))
        )
        output_filepath = str(
            resolve_repo_path(repo, paths_cfg.get("baseline_results_csv", "baseline_simulation_results.csv"))
        )
    seed = int(cfg.get("seed", {}).get("numpy_seed", 42))
    rng = np.random.default_rng(seed)
    tariff_cfg = cfg.get("tariff", {}) if isinstance(cfg, dict) else {}
    
    print(f"Loading Phase 1 synthesized data from {input_filepath}...")
    df = pd.read_csv(input_filepath)
    df['datetime'] = normalize_datetime_series_to_naive(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    print("Generating synthetic residential load profile...")
    df['Load_kW'] = generate_synthetic_load(df.index, rng=rng)
    
    print("Initializing Battery and storing simulation results...")
    batt_cfg = cfg.get("battery", {})
    capacity_kwh = float(batt_cfg.get("capacity_kwh", 10.0))
    inverter_power_kw = float(batt_cfg.get("inverter_power_kw", 5.0))
    soc_min = float(batt_cfg.get("soc_min", 0.15))
    efficiency_rt = float(batt_cfg.get("efficiency_rt", 0.95))
    battery = Battery(
        capacity_kwh=capacity_kwh,
        max_charge_kw=inverter_power_kw,
        max_discharge_kw=inverter_power_kw,
        soc_min=soc_min,
        efficiency_rt=efficiency_rt,
    )
    
    # Prepare lists to store our hourly results
    soc_history = []
    grid_import_history = []
    hourly_cost_history = []
    battery_action_history = [] # For debugging/viz tracking
    
    print("Running baseline simulation hour-by-hour...")
    for dt, row in df.iterrows():
        p_load = row['Load_kW']
        p_pv = row['Target_Power_kW']
        
        # Calculate net power
        # Positive P_net means load is higher than PV (Deficit) -> Battery should discharge
        # Negative P_net means PV is higher than load (Surplus) -> Battery should charge
        p_net = p_load - p_pv
        
        # P_batt is the actual power supplied (+) or absorbed (-) by the battery
        p_batt = battery.update_soc(p_request_kw=p_net)
        
        # The remaining power must come from the grid.
        # Grid import is positive. Grid export (if any) is negative.
        # P_grid = P_net - P_batt
        # Example 1 (Deficit): P_load = 2, P_pv = 0 -> P_net = 2. Battery supplies P_batt = 1. Grid = 2 - 1 = 1 kW import.
        # Example 2 (Surplus): P_load = 1, P_pv = 3 -> P_net = -2. Battery absorbs P_batt = -2. Grid = -2 - (-2) = 0 kW.
        p_grid = p_net - p_batt
        
        # If the battery cannot absorb all surplus, P_grid might be negative (exporting).
        # We will assume zero compensation for exporting back to the grid for this baseline,
        # so grid costs are only associated with importing (P_grid > 0).
        grid_import_kw = max(0.0, p_grid)
        
        # Calculate cost
        tariff_rate = tariff_rate_from_config(dt.hour, tariff_cfg=tariff_cfg) if tariff_cfg else get_tariff_rate(dt.hour)
        hourly_cost_rm = grid_import_kw * tariff_rate
        
        # Store tracking info
        soc_history.append(battery.soc)
        battery_action_history.append(p_batt)
        grid_import_history.append(grid_import_kw)
        hourly_cost_history.append(hourly_cost_rm)
        
    print("Appending results to dataframe...")
    df['SoC'] = soc_history
    df['Batt_Action_kW'] = battery_action_history
    df['Grid_Import_kW'] = grid_import_history
    df['Hourly_Cost_RM'] = hourly_cost_history
    
    print(f"Saving baseline simulation dataframe to {output_filepath}...")
    from pathlib import Path as _Path  # local import keeps module headers untouched
    _Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_filepath)

    print("Phase 2: Baseline Environment Simulation completed successfully!")

if __name__ == "__main__":
    main()
