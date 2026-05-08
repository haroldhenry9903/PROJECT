import json
import sys

import joblib
import numpy as np
import pandas as pd

from core_battery_model import Battery
from repo_paths import repo_root, resolve_repo_path

def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_datetime_series_to_naive(s: pd.Series) -> pd.Series:
    """Parse datetimes and ensure tz-naive timestamps (legacy tz-aware CSVs are normalized)."""
    dt = pd.to_datetime(s, errors="coerce")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt


def get_tariff_rate(hour: int, *, peak_start_hour: int, peak_end_hour: int, peak_rate: float, offpeak_rate: float) -> float:
    """Time-of-use tariff rate given hour."""
    if peak_start_hour <= hour < peak_end_hour:
        return peak_rate
    return offpeak_rate

def main():
    repo = repo_root()
    config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "experiment_config.json"
    cfg_path = resolve_repo_path(repo, config_path)
    cfg = _load_config(str(cfg_path))

    # Load dataset from Phase 2 which already contains the generated Load_kW
    data_path = str(
        resolve_repo_path(repo, cfg.get("paths", {}).get("baseline_results_csv", "baseline_simulation_results.csv"))
    )
    model_path = str(resolve_repo_path(repo, cfg.get("paths", {}).get("rf_model_path", "models/rf_forecast_model.joblib")))
    output_path = str(
        resolve_repo_path(repo, cfg.get("paths", {}).get("smart_results_csv", "smart_simulation_results.csv"))
    )
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['datetime'] = normalize_datetime_series_to_naive(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    print(f"Loading trained AI model from {model_path}...")
    rf_model = joblib.load(model_path)
    
    # Ensure temporal features exist for the model
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    if 'month' not in df.columns:
        df['month'] = df.index.month
        
    features = ['ghiradiation', 'temp', 'humidity', 'precip', 'cloudcover', 'hour', 'month']
    
    print("Initializing Smart Battery Controller...")
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
    max_discharge_kw_nominal = float(battery.max_discharge_kw)
    
    conservation_mode = False
    controller_cfg = cfg.get("controller", {})
    tariff_cfg = cfg.get("tariff", {})
    peak_start_hour = int(tariff_cfg.get("peak_start_hour", 7))
    peak_end_hour = int(tariff_cfg.get("peak_end_hour", 22))
    peak_rate = float(tariff_cfg.get("peak_rate_rm_per_kwh", 0.245))
    offpeak_rate = float(tariff_cfg.get("offpeak_rate_rm_per_kwh", 0.139))

    e_crit = float(controller_cfg.get("e_crit_kwh", 18.0))  # Critical energy threshold in kWh
    essential_discharge_kw = float(controller_cfg.get("essential_discharge_kw", 0.0))
    soc_reserve = float(controller_cfg.get("soc_reserve", 0.50))
    
    # Lists to store results
    soc_history = []
    grid_import_history = []
    hourly_cost_history = []
    conservation_mode_history = []
    battery_action_history = []
    
    print("Running Smart Optimizer Simulation...")
    for i, (dt, row) in enumerate(df.iterrows()):
        
        # ============================================================
        # ALGORITHM STEP 1: Midnight Daily Forecast (runs once per day at 00:00)
        # ============================================================
        # For Non-Solar Readers:
        # Every midnight, the AI "looks ahead" at tomorrow's weather forecast.
        # It asks: "Will the sun shine enough tomorrow to generate meaningful power?"
        # If the answer is NO (cloudy/rainy day expected), it activates Conservation Mode.
        if dt.hour == 0:
            next_24_rows = df.iloc[i : i+24]
            X_24h = next_24_rows[features]
            
            # Safety check: avoid indexing errors on the last day of the dataset
            if len(X_24h) < 24:
                conservation_mode = False
            else:
                # AI predicts hourly solar power for the next 24 hours
                predictions = rf_model.predict(X_24h)
                # Sum up the TOTAL expected solar energy for the day (kWh)
                e_pred = predictions.sum()
                
                # DECISION THRESHOLD (e_crit = 18 kWh):
                # If the AI expects less than 18 kWh of solar generation,
                # it means tomorrow will be too cloudy to rely on solar alone.
                # We MUST save battery energy now for the expensive peak tariff window (07:00-22:00).
                if e_pred < e_crit:
                    conservation_mode = True   # Activate hoarding!
                else:
                    conservation_mode = False  # Sunny day expected, operate normally.
                
        # ============================================================
        # ALGORITHM STEP 2: Peak-Reserve Policy (runs every hour)
        # ============================================================
        # On Conservation days, preserve a minimum SoC reserve for the peak-tariff window.
        # - During OFF-PEAK (outside 07:00–22:00):
        #   If SoC is at/below reserve, restrict discharge to essential loads only.
        # - During PEAK:
        #   Allow normal discharge to reduce expensive grid imports.
        if conservation_mode:
            is_peak = peak_start_hour <= dt.hour < peak_end_hour
            if (not is_peak) and (battery.soc <= soc_reserve):
                battery.max_discharge_kw = essential_discharge_kw
            else:
                battery.max_discharge_kw = max_discharge_kw_nominal
        else:
            battery.max_discharge_kw = max_discharge_kw_nominal
            
        # 3. Energy Flow Calculation
        p_load = row['Load_kW']
        p_pv = row['Target_Power_kW']
        
        p_net = p_load - p_pv
        p_batt = battery.update_soc(p_request_kw=p_net)
        
        p_grid = p_net - p_batt
        grid_import_kw = max(0.0, p_grid)
        
        # 4. Hourly Cost Calculation using Target Tariff C3
        tariff_rate = get_tariff_rate(
            dt.hour,
            peak_start_hour=peak_start_hour,
            peak_end_hour=peak_end_hour,
            peak_rate=peak_rate,
            offpeak_rate=offpeak_rate,
        )
        hourly_cost_rm = grid_import_kw * tariff_rate
        
        # 5. Store Data
        soc_history.append(battery.soc)
        battery_action_history.append(p_batt)
        grid_import_history.append(grid_import_kw)
        hourly_cost_history.append(hourly_cost_rm)
        conservation_mode_history.append(conservation_mode)
        
    print("Appending new results to dataframe...")
    df['SoC'] = soc_history
    df['Batt_Action_kW'] = battery_action_history
    df['Grid_Import_kW'] = grid_import_history
    df['Hourly_Cost_RM'] = hourly_cost_history
    df['Conservation_Mode_Active'] = conservation_mode_history
    
    print(f"Saving Smart Simulation Results to {output_path}...")
    df.to_csv(output_path)
    print("Phase 4: Smart Optimizer Simulation completed successfully!")

if __name__ == '__main__':
    main()
