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


def _resolve_forecast_seed(cfg: dict) -> int:
    """Resolve the dedicated forecast RNG seed.

    Precedence:
      1. cfg["forecast"]["seed"]
      2. cfg["seed"]["forecast_seed"]
      3. cfg["seed"]["numpy_seed"] + 1  (final fallback so it never collides with the load RNG)
    """
    fc = cfg.get("forecast", {}) or {}
    if "seed" in fc and fc["seed"] is not None:
        return int(fc["seed"])
    seed_block = cfg.get("seed", {}) or {}
    if "forecast_seed" in seed_block and seed_block["forecast_seed"] is not None:
        return int(seed_block["forecast_seed"])
    return int(seed_block.get("numpy_seed", 42)) + 1


def apply_forecast_noise(
    X_24h: pd.DataFrame,
    *,
    forecast_cfg: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a noisy copy of a 24-row weather feature window.

    Closes the perfect-foresight loophole: the smart controller no longer reads
    ground-truth next-day weather. Instead a configurable, seeded error model is
    applied to the features that would normally come from a day-ahead forecast.

    Noise model (per column, only if the column is present):
      - ghiradiation : multiplicative log-normal with bias ->
          x' = max(0, x * (1 + ghi_bias) * exp(N(0, ghi_noise_sigma)))
      - precip       : multiplicative log-normal ->
          x' = max(0, x * exp(N(0, precip_noise_sigma)))
      - temp         : additive gaussian (degC) ->
          x' = x + N(0, temp_noise_sigma)
      - humidity     : additive gaussian, clipped to [0, 100] ->
          x' = clip(x + N(0, humidity_noise_sigma), 0, 100)
      - cloudcover   : additive gaussian, clipped to [0, 100] ->
          x' = clip(x + N(0, cloudcover_noise_sigma), 0, 100)
      - hour, month  : never perturbed (calendar features are known exactly).

    The input dataframe is not mutated; a copy is returned.
    """
    out = X_24h.copy()
    if not forecast_cfg or not bool(forecast_cfg.get("enabled", False)):
        return out

    n = len(out)
    if n == 0:
        return out

    ghi_sigma = float(forecast_cfg.get("ghi_noise_sigma", 0.0))
    ghi_bias = float(forecast_cfg.get("ghi_bias", 0.0))
    cloud_sigma = float(forecast_cfg.get("cloudcover_noise_sigma", 0.0))
    temp_sigma = float(forecast_cfg.get("temp_noise_sigma", 0.0))
    hum_sigma = float(forecast_cfg.get("humidity_noise_sigma", 0.0))
    precip_sigma = float(forecast_cfg.get("precip_noise_sigma", 0.0))
    clip_nonneg = bool(forecast_cfg.get("clip_nonneg", True))

    if "ghiradiation" in out.columns:
        ghi_mult = (1.0 + ghi_bias) * np.exp(rng.normal(loc=0.0, scale=ghi_sigma, size=n))
        v = out["ghiradiation"].astype(float).to_numpy() * ghi_mult
        if clip_nonneg:
            v = np.maximum(0.0, v)
        out["ghiradiation"] = v

    if "precip" in out.columns:
        precip_mult = np.exp(rng.normal(loc=0.0, scale=precip_sigma, size=n))
        v = out["precip"].astype(float).to_numpy() * precip_mult
        if clip_nonneg:
            v = np.maximum(0.0, v)
        out["precip"] = v

    if "temp" in out.columns:
        v = out["temp"].astype(float).to_numpy() + rng.normal(loc=0.0, scale=temp_sigma, size=n)
        out["temp"] = v

    if "humidity" in out.columns:
        v = out["humidity"].astype(float).to_numpy() + rng.normal(loc=0.0, scale=hum_sigma, size=n)
        v = np.clip(v, 0.0, 100.0)
        out["humidity"] = v

    if "cloudcover" in out.columns:
        v = out["cloudcover"].astype(float).to_numpy() + rng.normal(loc=0.0, scale=cloud_sigma, size=n)
        v = np.clip(v, 0.0, 100.0)
        out["cloudcover"] = v

    return out

FEATURE_COLUMNS = ['ghiradiation', 'temp', 'humidity', 'precip', 'cloudcover', 'hour', 'month']


def simulate(df_input: pd.DataFrame, model, cfg: dict, *, verbose: bool = False) -> pd.DataFrame:
    """Run the smart-controller simulation in-process and return a NEW dataframe.

    This is the importable, zero-IO core of Phase 4. `run_all.py`'s sensitivity
    sweep calls this 50+ times in-process, so it must be fast and side-effect
    free.

    The forecaster is treated as a duck-typed sklearn estimator: anything with
    a `.predict(X)` method that accepts a 24-row dataframe of FEATURE_COLUMNS
    works. The project ships two: a Random Forest (`models/rf_forecast_model_{region}.joblib`)
    and an SVR pipeline (`models/svm_forecast_model_{region}.joblib`); both
    were trained on the same chronological split, so the smart controller is
    indifferent to which one is plugged in.

    Contract:
      - Does NOT mutate `df_input`.
      - Returns a new dataframe with the original columns plus
        `SoC`, `Batt_Action_kW`, `Grid_Import_kW`, `Hourly_Cost_RM`,
        `Conservation_Mode_Active`, `Forecast_Energy_kWh_Predicted`.
      - Honors `cfg["forecast"]` exactly as the CLI path does (same noise,
        same RNG, same seed precedence). With `forecast.enabled=false` (or
        section absent) the simulation is byte-for-byte identical to the
        legacy perfect-foresight code path.
    """
    df = df_input.copy()
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    if 'month' not in df.columns:
        df['month'] = df.index.month

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

    controller_cfg = cfg.get("controller", {})
    tariff_cfg = cfg.get("tariff", {})
    peak_start_hour = int(tariff_cfg.get("peak_start_hour", 7))
    peak_end_hour = int(tariff_cfg.get("peak_end_hour", 22))
    peak_rate = float(tariff_cfg.get("peak_rate_rm_per_kwh", 0.245))
    offpeak_rate = float(tariff_cfg.get("offpeak_rate_rm_per_kwh", 0.139))

    e_crit = float(controller_cfg.get("e_crit_kwh", 18.0))
    essential_discharge_kw = float(controller_cfg.get("essential_discharge_kw", 0.0))
    soc_reserve = float(controller_cfg.get("soc_reserve", 0.50))

    forecast_cfg = cfg.get("forecast", {}) or {}
    forecast_enabled = bool(forecast_cfg.get("enabled", False))
    forecast_rng = np.random.default_rng(_resolve_forecast_seed(cfg)) if forecast_enabled else None
    if verbose:
        if forecast_enabled:
            print(
                f"Forecast noise ENABLED (seed={_resolve_forecast_seed(cfg)}, "
                f"ghi_sigma={float(forecast_cfg.get('ghi_noise_sigma', 0.0))}, "
                f"ghi_bias={float(forecast_cfg.get('ghi_bias', 0.0))})"
            )
        else:
            print("Forecast noise DISABLED -> perfect-foresight upper bound (legacy behaviour).")

    n = len(df)
    load_arr = df['Load_kW'].astype(float).to_numpy()
    pv_arr = df['Target_Power_kW'].astype(float).to_numpy()
    hour_arr = df.index.hour.to_numpy()
    feat_df = df[FEATURE_COLUMNS]

    # ===========================================================
    # PRE-COMPUTE: forecast decisions for every midnight in one shot.
    #
    # Why: the previous iter-and-predict loop called rf_model.predict 1830
    # times (once per midnight) on 24-row slices. Each call has fixed
    # overhead (tree dispatch, Python<->C marshalling), and at five years
    # of hourly data this dominated wall time (~36 s of a 42 s run).
    # Concatenating all noisy 24h windows into one matrix and calling
    # rf_model.predict ONCE drops that to ~2 s without changing semantics:
    # the noise RNG advances in exactly the same order it did in the per-
    # midnight loop (one apply_forecast_noise call per midnight, in
    # chronological order), so the noise draws and therefore the
    # predictions are bit-identical.
    # ===========================================================
    all_midnight_indices = np.where(hour_arr == 0)[0]
    n_midnights = int(len(all_midnight_indices))
    e_pred_per_midnight = np.full(n_midnights, np.nan, dtype=float)
    conservation_per_midnight = np.zeros(n_midnights, dtype=bool)

    full_window_positions: list[int] = []
    full_window_arrays: list[np.ndarray] = []
    for k, mi in enumerate(all_midnight_indices):
        if mi + 24 > n:
            continue  # truncated last-day midnight: leave defaults (NaN, False)
        X_24h_truth = feat_df.iloc[mi : mi + 24]
        if forecast_enabled and forecast_rng is not None:
            X_24h = apply_forecast_noise(
                X_24h_truth,
                forecast_cfg=forecast_cfg,
                rng=forecast_rng,
            )
        else:
            X_24h = X_24h_truth
        full_window_positions.append(k)
        full_window_arrays.append(X_24h.to_numpy(dtype=float))

    if full_window_arrays:
        big_X = np.vstack(full_window_arrays)
        # Wrap in a DataFrame with the original feature names so sklearn does
        # not emit "X does not have valid feature names" warnings (the model
        # was fitted on a DataFrame in Phase 3).
        big_X_df = pd.DataFrame(big_X, columns=FEATURE_COLUMNS)
        big_pred = np.asarray(model.predict(big_X_df), dtype=float).reshape(-1, 24)
        e_preds = big_pred.sum(axis=1)
        for j, k in enumerate(full_window_positions):
            e_val = float(e_preds[j])
            e_pred_per_midnight[k] = e_val
            conservation_per_midnight[k] = bool(e_val < e_crit)

    # Forward-fill each midnight's decision until the next midnight.
    e_pred_by_hour = np.full(n, np.nan, dtype=float)
    conservation_by_hour = np.zeros(n, dtype=bool)
    for k, mi in enumerate(all_midnight_indices):
        next_mi = int(all_midnight_indices[k + 1]) if k + 1 < n_midnights else n
        e_pred_by_hour[mi:next_mi] = e_pred_per_midnight[k]
        conservation_by_hour[mi:next_mi] = conservation_per_midnight[k]

    # ===========================================================
    # FAST DISPATCH LOOP (no RF predicts inside; pure scalar Python on numpy).
    # ===========================================================
    soc_history = np.empty(n, dtype=float)
    battery_action_history = np.empty(n, dtype=float)
    grid_import_history = np.empty(n, dtype=float)
    hourly_cost_history = np.empty(n, dtype=float)

    for i in range(n):
        h = int(hour_arr[i])
        cons_mode = bool(conservation_by_hour[i])
        is_peak = peak_start_hour <= h < peak_end_hour

        if cons_mode and (not is_peak) and (battery.soc <= soc_reserve):
            battery.max_discharge_kw = essential_discharge_kw
        else:
            battery.max_discharge_kw = max_discharge_kw_nominal

        p_load = load_arr[i]
        p_pv = pv_arr[i]
        p_net = p_load - p_pv
        p_batt = battery.update_soc(p_request_kw=p_net)
        p_grid = p_net - p_batt
        grid_import_kw = p_grid if p_grid > 0.0 else 0.0
        tariff_rate = peak_rate if is_peak else offpeak_rate

        soc_history[i] = battery.soc
        battery_action_history[i] = p_batt
        grid_import_history[i] = grid_import_kw
        hourly_cost_history[i] = grid_import_kw * tariff_rate

    conservation_mode_history = conservation_by_hour
    predicted_daily_energy_history = e_pred_by_hour

    df['SoC'] = soc_history
    df['Batt_Action_kW'] = battery_action_history
    df['Grid_Import_kW'] = grid_import_history
    df['Hourly_Cost_RM'] = hourly_cost_history
    df['Conservation_Mode_Active'] = conservation_mode_history
    # Forecast audit column: the daily-total PV the controller "thought" it would
    # see, repeated for every hour of that day. Compare against the actual daily
    # sum of `Target_Power_kW` to evaluate forecast skill / MAPE.
    df['Forecast_Energy_kWh_Predicted'] = predicted_daily_energy_history

    return df


def _resolve_path_template(repo, template: str, **kwargs: str) -> str:
    return str(resolve_repo_path(repo, template.format(**kwargs)))


def main():
    """CLI entry point.

    Usage examples::

        # Legacy single-region path (uses paths.baseline_results_csv etc.)
        python phase4_smart_optimizer.py

        # Per-region + per-algo run (preferred for the multi-region project)
        python phase4_smart_optimizer.py --region miri --algo svm

    Resolution order for inputs:
      1. If both --region and --algo are passed AND the config defines path
         templates (`baseline_results_template`, `model_template`,
         `smart_results_template`), use those.
      2. Otherwise fall back to the flat single-region paths
         (`baseline_results_csv`, `rf_model_path`, `smart_results_csv`).
    """
    repo = repo_root()
    # Parse arguments manually to avoid changing the legacy `--config` behaviour.
    argv = sys.argv[1:]
    config_path = "experiment_config.json"
    region: str | None = None
    algo: str | None = None
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]; i += 2; continue
        if argv[i] == "--region" and i + 1 < len(argv):
            region = argv[i + 1]; i += 2; continue
        if argv[i] == "--algo" and i + 1 < len(argv):
            algo = argv[i + 1]; i += 2; continue
        i += 1

    cfg_path = resolve_repo_path(repo, config_path)
    cfg = _load_config(str(cfg_path))
    paths_cfg = cfg.get("paths", {}) or {}

    regions_cfg = cfg.get("regions", {}) or {}
    algos_cfg = (cfg.get("models", {}) or {}).get("algorithms", ["rf"])
    if region is None:
        region = str(regions_cfg.get("active", "kuching"))
    if algo is None:
        algo = str(algos_cfg[0] if algos_cfg else "rf")

    # Per-region path resolution if templates exist; else fall back to legacy flat paths.
    baseline_tpl = paths_cfg.get("baseline_results_template")
    model_tpl = paths_cfg.get("model_template")
    smart_tpl = paths_cfg.get("smart_results_template")
    if baseline_tpl and model_tpl and smart_tpl:
        data_path = _resolve_path_template(repo, baseline_tpl, region=region, algo=algo)
        model_path = _resolve_path_template(repo, model_tpl, region=region, algo=algo)
        output_path = _resolve_path_template(repo, smart_tpl, region=region, algo=algo)
    else:
        data_path = str(resolve_repo_path(repo, paths_cfg.get("baseline_results_csv", "baseline_simulation_results.csv")))
        model_path = str(resolve_repo_path(repo, paths_cfg.get("rf_model_path", "models/rf_forecast_model_kuching.joblib")))
        output_path = str(resolve_repo_path(repo, paths_cfg.get("smart_results_csv", "smart_simulation_results.csv")))

    print(f"[phase4] region={region}  algo={algo}")
    print(f"[phase4] baseline csv : {data_path}")
    print(f"[phase4] forecaster   : {model_path}")
    print(f"[phase4] smart output : {output_path}")

    df = pd.read_csv(data_path)
    df['datetime'] = normalize_datetime_series_to_naive(df['datetime'])
    df.set_index('datetime', inplace=True)

    print(f"Loading trained AI model from {model_path}...")
    model = joblib.load(model_path)

    print("Running Smart Optimizer Simulation...")
    out = simulate(df, model, cfg, verbose=True)

    print(f"Saving Smart Simulation Results to {output_path}...")
    from pathlib import Path as _Path  # local import to avoid touching module headers
    _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path)
    print("Phase 4: Smart Optimizer Simulation completed successfully!")

if __name__ == '__main__':
    main()
