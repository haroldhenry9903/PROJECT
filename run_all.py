from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel, wilcoxon

from core_battery_model import Battery
from phase4_smart_optimizer import simulate as phase4_simulate


@dataclass(frozen=True)
class Config:
    raw: dict

    @property
    def paths(self) -> dict:
        return self.raw.get("paths", {})

    @property
    def seed(self) -> dict:
        return self.raw.get("seed", {})

    @property
    def battery(self) -> dict:
        return self.raw.get("battery", {})

    @property
    def tariff(self) -> dict:
        return self.raw.get("tariff", {})

    @property
    def controller(self) -> dict:
        return self.raw.get("controller", {})

    @property
    def scenario(self) -> dict:
        return self.raw.get("scenario", {})

    @property
    def pipeline(self) -> dict:
        return self.raw.get("pipeline", {})

    @property
    def evaluation(self) -> dict:
        return self.raw.get("evaluation", {})

    @property
    def forecast(self) -> dict:
        return self.raw.get("forecast", {})


def load_config(config_path: Path) -> Config:
    with config_path.open("r", encoding="utf-8") as f:
        return Config(json.load(f))

def normalize_datetime_index_to_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ensure tz-naive index; legacy tz-aware indices are normalized to tz-naive."""
    if idx.tz is not None:
        # Drop timezone while keeping local clock time (pipeline standard is tz-naive)
        return idx.tz_localize(None)
    return idx


def _tariff_rate(hour: int, *, peak_start: int, peak_end: int, peak_rate: float, offpeak_rate: float) -> float:
    return peak_rate if peak_start <= hour < peak_end else offpeak_rate


def _run(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  exit_code: {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def validate_timeseries(df: pd.DataFrame, *, name: str, soc_min: float) -> None:
    required = {"SoC", "Grid_Import_kW", "Hourly_Cost_RM"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: index must be a DatetimeIndex")

    # Pipeline standard: tz-naive. If tz-aware slipped through, normalize earlier during load.
    if df.index.tz is not None:
        raise ValueError(f"{name}: datetime index must be tz-naive after normalization (got {df.index.tz})")

    soc = df["SoC"].astype(float)
    if float(soc.min()) < soc_min - 1e-9 or float(soc.max()) > 1.0 + 1e-9:
        raise ValueError(f"{name}: SoC out of bounds (min={soc.min()}, max={soc.max()}, expected [{soc_min}, 1.0])")

    grid = df["Grid_Import_kW"].astype(float)
    if float(grid.min()) < -1e-9:
        raise ValueError(f"{name}: Grid_Import_kW has negative values (min={grid.min()})")

    cost = df["Hourly_Cost_RM"].astype(float)
    if float(cost.min()) < -1e-9:
        raise ValueError(f"{name}: Hourly_Cost_RM has negative values (min={cost.min()})")

    # Basic continuity checks
    if len(df.index) >= 2:
        diffs = df.index.to_series().diff().dropna()
        # Allow only hourly steps for this study
        bad = diffs[diffs != pd.Timedelta(hours=1)]
        if len(bad) > 0:
            raise ValueError(f"{name}: non-hourly time steps detected (examples: {bad.head(3).to_dict()})")


def validate_energy_balance(df: pd.DataFrame, *, name: str, pv_col: str = "Target_Power_kW") -> None:
    """
    Check P_grid ~= P_net - P_batt where:
      P_net = P_load - P_pv
      P_grid_import = max(0, P_grid)

    Since the CSV stores only import (clipped), we check the unclipped identity by reconstructing
    P_grid_unclipped from P_net and Batt_Action_kW, then confirming import is max(0, unclipped).
    """
    required = {"Load_kW", pv_col, "Batt_Action_kW", "Grid_Import_kW"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name}: cannot validate energy balance; missing: {missing}")

    load = df["Load_kW"].astype(float).to_numpy()
    pv = df[pv_col].astype(float).to_numpy()
    p_batt = df["Batt_Action_kW"].astype(float).to_numpy()
    grid_import = df["Grid_Import_kW"].astype(float).to_numpy()

    p_net = load - pv
    p_grid_unclipped = p_net - p_batt
    grid_import_expected = np.maximum(0.0, p_grid_unclipped)

    max_abs_err = float(np.max(np.abs(grid_import_expected - grid_import)))
    if max_abs_err > 1e-6:
        raise ValueError(f"{name}: energy balance/import mismatch; max_abs_err={max_abs_err}")


def validate_tariff_cost_identity(
    df_base: pd.DataFrame,
    df_smart: pd.DataFrame,
    *,
    tariff: dict,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    """
    Spot-check Hourly_Cost_RM ≈ Grid_Import_kW * tariff(hour) on a few representative hours
    for both baseline and smart CSVs.
    """
    peak_start = int(tariff.get("peak_start_hour", 7))
    peak_end = int(tariff.get("peak_end_hour", 22))
    peak_rate = float(tariff.get("peak_rate_rm_per_kwh", 0.245))
    offpeak_rate = float(tariff.get("offpeak_rate_rm_per_kwh", 0.139))

    hours_to_sample: List[int] = [
        (peak_start - 1) % 24,
        peak_start,
        peak_end - 1,
        peak_end % 24,
    ]

    max_err = 0.0
    samples: List[Dict[str, Any]] = []

    for ds_name, df in (("baseline", df_base), ("smart", df_smart)):
        dfc = df.copy()
        if "hour" not in dfc.columns:
            dfc["hour"] = dfc.index.hour
        hrs = dfc["hour"].astype(int)
        for target_h in hours_to_sample:
            sel = dfc.loc[hrs == int(target_h)]
            if sel.empty:
                continue
            row = sel.iloc[0]
            hr = int(row["hour"])
            gi = float(row["Grid_Import_kW"])
            cost = float(row["Hourly_Cost_RM"])
            rate = _tariff_rate(hr, peak_start=peak_start, peak_end=peak_end, peak_rate=peak_rate, offpeak_rate=offpeak_rate)
            expected = gi * rate
            err = abs(cost - expected)
            max_err = max(max_err, err)
            samples.append({"dataset": ds_name, "hour": hr, "abs_err": err})

    passed = bool(len(samples) > 0 and max_err <= tol)
    return {
        "tariff_cost_identity_passed": passed,
        "tariff_cost_identity_max_abs_err": float(max_err),
        "tariff_cost_identity_tol": float(tol),
        "samples_checked": int(len(samples)),
    }


def compute_daily_results(
    df_base: pd.DataFrame,
    df_smart: pd.DataFrame,
    *,
    soc_init: float,
    batt: dict,
    tariff: dict,
    policy: dict,
    scenario: dict,
) -> pd.DataFrame:
    peak_start = int(tariff.get("peak_start_hour", 7))
    peak_end = int(tariff.get("peak_end_hour", 22))
    peak_rate = float(tariff.get("peak_rate_rm_per_kwh", 0.245))
    offpeak_rate = float(tariff.get("offpeak_rate_rm_per_kwh", 0.139))

    capacity_kwh = float(batt.get("capacity_kwh", 10.0))
    inverter_power_kw = float(batt.get("inverter_power_kw", 5.0))
    soc_min = float(batt.get("soc_min", 0.15))
    efficiency_rt = float(batt.get("efficiency_rt", 0.95))

    pv_scale = float(scenario.get("pv_scale", 1.0))
    load_scale = float(scenario.get("load_scale", 1.0))

    soc_reserve = float(policy.get("soc_reserve", 0.50))
    essential_discharge_kw = float(policy.get("essential_discharge_kw", 0.0))

    df = df_base.copy()
    if "hour" not in df.columns:
        df["hour"] = df.index.hour

    df_s = df_smart.copy()
    if "Conservation_Mode_Active" not in df_s.columns:
        df_s["Conservation_Mode_Active"] = False

    days = df.index.normalize()
    unique_days = pd.Index(days.unique()).sort_values()

    rows = []
    for day in unique_days:
        day_mask = days == day
        df_day = df.loc[day_mask]
        df_s_day = df_s.loc[df_s.index.normalize() == day]

        # Smart flag is set at midnight and should remain constant for that day; take first value.
        cm = False
        if len(df_s_day) > 0:
            try:
                cm = bool(df_s_day["Conservation_Mode_Active"].iloc[0])
            except Exception:
                cm = False

        def simulate_day(controller: str) -> Tuple[float, float, float]:
            b = Battery(
                capacity_kwh=capacity_kwh,
                max_charge_kw=inverter_power_kw,
                max_discharge_kw=inverter_power_kw,
                soc_min=soc_min,
                efficiency_rt=efficiency_rt,
            )
            b.soc = float(np.clip(soc_init, soc_min, 1.0))

            total_cost = 0.0
            peak_import = 0.0
            min_soc = float(b.soc)

            for _, r in df_day.iterrows():
                hr = int(r["hour"])
                if controller == "smart" and cm:
                    is_peak = peak_start <= hr < peak_end
                    if (not is_peak) and (b.soc <= soc_reserve):
                        b.max_discharge_kw = essential_discharge_kw
                    else:
                        b.max_discharge_kw = inverter_power_kw
                else:
                    b.max_discharge_kw = inverter_power_kw

                p_load = float(r["Load_kW"]) * load_scale
                p_pv = float(r["Target_Power_kW"]) * pv_scale
                p_net = p_load - p_pv

                p_batt = b.update_soc(p_request_kw=p_net)
                p_grid = p_net - p_batt
                grid_import_kw = max(0.0, p_grid)

                rate = _tariff_rate(hr, peak_start=peak_start, peak_end=peak_end, peak_rate=peak_rate, offpeak_rate=offpeak_rate)
                total_cost += grid_import_kw * rate
                if peak_start <= hr < peak_end:
                    peak_import += grid_import_kw
                min_soc = min(min_soc, float(b.soc))

            return float(total_cost), float(peak_import), float(min_soc)

        base_cost, base_peak, base_min_soc = simulate_day("baseline")
        smart_cost, smart_peak, smart_min_soc = simulate_day("smart")

        rows.append(
            {
                "date": pd.Timestamp(day).date().isoformat(),
                "conservation_mode": bool(cm),
                "baseline_cost_rm": base_cost,
                "smart_cost_rm": smart_cost,
                "savings_rm": base_cost - smart_cost,
                "baseline_peak_import_kwh": base_peak,
                "smart_peak_import_kwh": smart_peak,
                "peak_import_reduction_kwh": base_peak - smart_peak,
                "baseline_min_soc": base_min_soc,
                "smart_min_soc": smart_min_soc,
            }
        )

    return pd.DataFrame(rows)


def _paired_significance_tests(daily: pd.DataFrame) -> Dict[str, Any]:
    """Run Wilcoxon signed-rank + paired t-test on per-day savings.

    Treats each calendar day as a paired observation: `baseline_cost_rm[d]`
    vs `smart_cost_rm[d]`. With ~365 paired days and a stable savings
    distribution we expect p << 0.001 when the AI is genuinely beating the
    baseline; we report both stat and p-value so the thesis can cite either.

    Edge cases:
      - if `daily` is empty or all paired differences are zero, return
        sentinel NaN payloads so downstream consumers (dashboard, summary
        CSV) render gracefully instead of crashing.
    """
    if len(daily) == 0:
        return {
            "wilcoxon_signed_rank": {"stat": float("nan"), "p_value": float("nan"), "n_pairs": 0},
            "paired_t_test": {"stat": float("nan"), "p_value": float("nan"), "n_pairs": 0},
        }

    base = daily["baseline_cost_rm"].astype(float).to_numpy()
    smart = daily["smart_cost_rm"].astype(float).to_numpy()
    diff = base - smart
    n_pairs = int(len(diff))

    try:
        w_stat, w_p = wilcoxon(base, smart)
        w_payload = {"stat": float(w_stat), "p_value": float(w_p), "n_pairs": n_pairs}
    except ValueError:
        # All zero-differences -> wilcoxon raises; treat as p=1.0 (no signal).
        w_payload = {"stat": float("nan"), "p_value": 1.0, "n_pairs": n_pairs}

    try:
        t_stat, t_p = ttest_rel(base, smart)
        t_payload = {"stat": float(t_stat), "p_value": float(t_p), "n_pairs": n_pairs}
    except Exception:
        t_payload = {"stat": float("nan"), "p_value": float("nan"), "n_pairs": n_pairs}

    return {
        "wilcoxon_signed_rank": w_payload,
        "paired_t_test": t_payload,
        "mean_daily_saving_rm": float(np.mean(diff)),
        "std_daily_saving_rm": float(np.std(diff, ddof=1)) if n_pairs > 1 else 0.0,
    }


def _perfect_foresight_pct(results_dir: Path, current_total_savings_rm: float) -> Dict[str, Any]:
    """Return `pct_of_perfect_foresight_ceiling` once the σ-sweep has written its CSV.

    The σ=0 row of `forecast_sensitivity.csv` is the oracle upper bound
    (`run_forecast_sensitivity_sweep` already produces it for free). We
    compare the current run's headline `total_savings_rm` against that row's
    savings:

        pct = 100 * current / oracle_ceiling

    During the first `compute_metrics` call inside `run_all.py:main` the
    sweep CSV may not exist yet (the sweep runs *after* `compute_metrics`),
    in which case we return NaN and the post-sweep code patches the value
    back in. Callers that have the ceiling in hand can pass it directly via
    `oracle_override_rm`.
    """
    sweep_csv = results_dir / "forecast_sensitivity.csv"
    if not sweep_csv.exists():
        return {"pct_of_perfect_foresight_ceiling": float("nan"), "oracle_ceiling_rm": float("nan")}
    try:
        df = pd.read_csv(sweep_csv)
        row = df.loc[df["ghi_sigma"].astype(float).abs().idxmin()]
        ceiling = float(row["total_savings_rm"])
    except Exception:
        return {"pct_of_perfect_foresight_ceiling": float("nan"), "oracle_ceiling_rm": float("nan")}
    pct = (100.0 * current_total_savings_rm / ceiling) if ceiling > 0 else float("nan")
    return {"pct_of_perfect_foresight_ceiling": float(pct), "oracle_ceiling_rm": ceiling}


def compute_metrics(
    df_base: pd.DataFrame,
    df_smart: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    tariff: dict,
    daily_reset_soc_init: float,
    evaluation_cfg: Optional[dict] = None,
    tariff_validation: Optional[Dict[str, Any]] = None,
    forecast_cfg: Optional[dict] = None,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    peak_start = int(tariff.get("peak_start_hour", 7))
    peak_end = int(tariff.get("peak_end_hour", 22))

    def _kpis(df: pd.DataFrame) -> Dict[str, float]:
        total_cost = float(df["Hourly_Cost_RM"].astype(float).sum())
        total_load = float(df["Load_kW"].astype(float).sum())
        total_import = float(df["Grid_Import_kW"].astype(float).sum())
        ssr = 0.0 if total_load <= 0 else (1.0 - (total_import / total_load)) * 100.0
        peak_mask = df.index.hour.astype(int)
        peak_import = float(df.loc[(peak_mask >= peak_start) & (peak_mask < peak_end), "Grid_Import_kW"].astype(float).sum())
        return {
            "total_cost_rm": total_cost,
            "total_load_kwh": total_load,
            "total_import_kwh": total_import,
            "ssr_pct": ssr,
            "peak_import_kwh": peak_import,
        }

    kb = _kpis(df_base)
    ks = _kpis(df_smart)

    savings = kb["total_cost_rm"] - ks["total_cost_rm"]
    peak_reduction = kb["peak_import_kwh"] - ks["peak_import_kwh"]

    daily_savings = daily["savings_rm"].astype(float)
    daily_peak_red = daily["peak_import_reduction_kwh"].astype(float)

    sum_daily_baseline = float(daily["baseline_cost_rm"].astype(float).sum()) if len(daily) else float("nan")
    sum_daily_smart = float(daily["smart_cost_rm"].astype(float).sum()) if len(daily) else float("nan")
    sum_daily_savings = float(daily_savings.sum()) if len(daily) else float("nan")
    gap_savings = float("nan") if math.isnan(sum_daily_savings) else float(savings) - float(sum_daily_savings)

    ev = evaluation_cfg or {}
    headline = str(ev.get("headline_primary", "continuous_csv_cost_and_peak"))

    by_continuous = {
        "baseline_total_cost_rm": kb["total_cost_rm"],
        "smart_total_cost_rm": ks["total_cost_rm"],
        "total_savings_rm": float(savings),
        "baseline_total_import_kwh": kb["total_import_kwh"],
        "smart_total_import_kwh": ks["total_import_kwh"],
        "baseline_ssr_pct": kb["ssr_pct"],
        "smart_ssr_pct": ks["ssr_pct"],
        "baseline_peak_import_kwh": kb["peak_import_kwh"],
        "smart_peak_import_kwh": ks["peak_import_kwh"],
        "peak_import_reduction_kwh": float(peak_reduction),
    }

    by_daily_reset = {
        "conservation_days_count": int(daily["conservation_mode"].astype(bool).sum()),
        "daily_savings_mean_rm": float(daily_savings.mean()) if len(daily_savings) else float("nan"),
        "daily_savings_median_rm": float(daily_savings.median()) if len(daily_savings) else float("nan"),
        "daily_savings_min_rm": float(daily_savings.min()) if len(daily_savings) else float("nan"),
        "daily_savings_max_rm": float(daily_savings.max()) if len(daily_savings) else float("nan"),
        "daily_peak_import_reduction_mean_kwh": float(daily_peak_red.mean()) if len(daily_peak_red) else float("nan"),
        "sum_daily_baseline_cost_rm": sum_daily_baseline,
        "sum_daily_smart_cost_rm": sum_daily_smart,
        "sum_daily_savings_rm": sum_daily_savings,
    }

    meta: Dict[str, Any] = {
        "schema_version": "1.1",
        "headline_primary": headline,
        "definitions": {
            "continuous_csv": (
                "Sums stored columns on the full continuous timelines copied from "
                "`baseline_simulation_results.csv` / `smart_simulation_results.csv`: "
                "`Hourly_Cost_RM`, `Grid_Import_kW`, `Load_kW`; peak import uses hours "
                f"[{peak_start}, {peak_end}) on the datetime index."
            ),
            "daily_reset": (
                "Per-calendar-day re-simulation with SoC reset each day to "
                f"{daily_reset_soc_init:.4f} (from config `scenario.initial_soc_daily_reset`); "
                "uses saved `Conservation_Mode_Active` flag per day for the smart branch. "
                "Not the same counterfactual as the continuous CSV year run."
            ),
        },
        "columns_used": {
            "continuous_total_cost": "Hourly_Cost_RM",
            "continuous_total_import": "Grid_Import_kW",
            "continuous_total_load": "Load_kW",
        },
        "daily_reset_soc_init": float(daily_reset_soc_init),
        "gap_continuous_total_savings_rm_minus_sum_daily_savings_rm": gap_savings,
        "gap_explainer": (
            "`total_savings_rm` uses continuous CSV cost sums. `sum_daily_savings_rm` sums per-day "
            "(baseline_cost - smart_cost) from independent daily-reset simulations. A non-zero gap is expected."
        ),
        "validation": tariff_validation or {},
        "forecast": dict(forecast_cfg) if isinstance(forecast_cfg, dict) else {},
        "note_perfect_foresight_disabled": (
            "When `forecast.enabled` is true the smart controller queries the RF "
            "with NOISY day-ahead weather features (seeded). Setting "
            "`forecast.ghi_noise_sigma=0`, `*_noise_sigma=0` and "
            "`ghi_bias=0` reproduces the perfect-foresight upper bound. See "
            "`forecast_sensitivity.csv` and `figures/forecast_sensitivity.png` "
            "for the savings curve as a function of forecast error."
        ),
    }

    legacy: Dict[str, Any] = {
        "baseline_total_cost_rm": kb["total_cost_rm"],
        "smart_total_cost_rm": ks["total_cost_rm"],
        "total_savings_rm": float(savings),
        "baseline_total_import_kwh": kb["total_import_kwh"],
        "smart_total_import_kwh": ks["total_import_kwh"],
        "baseline_ssr_pct": kb["ssr_pct"],
        "smart_ssr_pct": ks["ssr_pct"],
        "baseline_peak_import_kwh": kb["peak_import_kwh"],
        "smart_peak_import_kwh": ks["peak_import_kwh"],
        "peak_import_reduction_kwh": float(peak_reduction),
        "conservation_days_count": int(daily["conservation_mode"].astype(bool).sum()),
        "daily_savings_mean_rm": float(daily_savings.mean()) if len(daily_savings) else float("nan"),
        "daily_savings_median_rm": float(daily_savings.median()) if len(daily_savings) else float("nan"),
        "daily_savings_min_rm": float(daily_savings.min()) if len(daily_savings) else float("nan"),
        "daily_savings_max_rm": float(daily_savings.max()) if len(daily_savings) else float("nan"),
        "daily_peak_import_reduction_mean_kwh": float(daily_peak_red.mean()) if len(daily_peak_red) else float("nan"),
    }

    legacy["by_evaluation_mode"] = {"continuous_csv": by_continuous, "daily_reset": by_daily_reset}
    legacy["_meta"] = meta

    # ---- Statistical evaluation of the simulation result (Wilcoxon + paired-t)
    stats_block = _paired_significance_tests(daily)
    if results_dir is not None:
        stats_block.update(_perfect_foresight_pct(results_dir, float(savings)))
    else:
        stats_block.setdefault("pct_of_perfect_foresight_ceiling", float("nan"))
        stats_block.setdefault("oracle_ceiling_rm", float("nan"))
    legacy["statistical_tests"] = stats_block

    return legacy


def _daily_pv_actual_kwh(df: pd.DataFrame) -> pd.Series:
    """Daily-total actual PV energy from the simulation CSV (kW with 1 h step ≈ kWh)."""
    by_day = df.groupby(df.index.normalize())["Target_Power_kW"].sum().astype(float)
    by_day.index = pd.to_datetime(by_day.index)
    return by_day


def _daily_pv_predicted_kwh(df_smart: pd.DataFrame) -> pd.Series:
    """Daily-total predicted PV energy from the smart CSV (one value per day, taken at hour 0)."""
    if "Forecast_Energy_kWh_Predicted" not in df_smart.columns:
        return pd.Series(dtype=float)
    s = df_smart["Forecast_Energy_kWh_Predicted"].astype(float)
    by_day = s.groupby(df_smart.index.normalize()).first()
    by_day.index = pd.to_datetime(by_day.index)
    return by_day


def _forecast_mape_daily(df_smart: pd.DataFrame) -> float:
    """MAPE between predicted-daily-PV and actual-daily-PV, ignoring days where actual <= 0."""
    pred = _daily_pv_predicted_kwh(df_smart)
    actual = _daily_pv_actual_kwh(df_smart)
    common = pred.index.intersection(actual.index)
    if len(common) == 0:
        return float("nan")
    p = pred.loc[common].dropna()
    a = actual.loc[p.index]
    mask = a > 1e-6
    if mask.sum() == 0:
        return float("nan")
    return float(((p[mask] - a[mask]).abs() / a[mask]).mean() * 100.0)


def _continuous_kpis_for_smart(df_smart: pd.DataFrame, *, tariff: dict) -> Dict[str, float]:
    peak_start = int(tariff.get("peak_start_hour", 7))
    peak_end = int(tariff.get("peak_end_hour", 22))
    cost = float(df_smart["Hourly_Cost_RM"].astype(float).sum())
    imp = float(df_smart["Grid_Import_kW"].astype(float).sum())
    hrs = df_smart.index.hour.astype(int)
    peak_imp = float(df_smart.loc[(hrs >= peak_start) & (hrs < peak_end), "Grid_Import_kW"].astype(float).sum())
    cm_days = int(df_smart.groupby(df_smart.index.normalize())["Conservation_Mode_Active"].first().astype(bool).sum()) if "Conservation_Mode_Active" in df_smart.columns else 0
    return {
        "smart_total_cost_rm": cost,
        "smart_total_import_kwh": imp,
        "smart_peak_import_kwh": peak_imp,
        "conservation_days_count": cm_days,
    }


def _kpis_from_smart_df(df_s: pd.DataFrame, *, tariff: dict, base_cost: float, base_peak: float) -> Dict[str, float]:
    smart = _continuous_kpis_for_smart(df_s, tariff=tariff)
    savings = base_cost - smart["smart_total_cost_rm"]
    peak_red = base_peak - smart["smart_peak_import_kwh"]
    return {
        "smart_total_cost_rm": smart["smart_total_cost_rm"],
        "smart_total_import_kwh": smart["smart_total_import_kwh"],
        "smart_peak_import_kwh": smart["smart_peak_import_kwh"],
        "conservation_days_count": int(smart["conservation_days_count"]),
        "total_savings_rm": float(savings),
        "savings_pct_of_baseline": float(100.0 * savings / base_cost) if base_cost > 0 else float("nan"),
        "peak_import_reduction_kwh": float(peak_red),
        "rf_forecast_mape_daily": _forecast_mape_daily(df_s),
    }


def run_forecast_sensitivity_sweep(
    cfg: Config,
    repo: Path,
    results_dir: Path,
    config_path: Path,
    df_base: pd.DataFrame,
    *,
    df_phase4_input: Optional[pd.DataFrame] = None,
    rf_model: Optional[Any] = None,
    sigmas: Optional[List[float]] = None,
    seeds: Optional[int] = None,
) -> Path:
    """In-process, multi-seed σ-sensitivity sweep.

    For each (σ, seed) trial the smart simulation is invoked via
    `phase4_simulate(...)` directly; no sub-process spawn, no extra Python /
    sklearn / RF-load overhead per trial.

    Outputs (under `results_dir`):
      - `forecast_sensitivity_raw.csv` — one row per (σ, seed) with the
        per-trial KPIs and a `seed` column.
      - `forecast_sensitivity.csv` — one row per σ with the mean as the
        primary value plus `*_std`, `*_min`, `*_max` siblings on the four
        bottom-line KPIs (savings, peak reduction, conservation days, daily
        MAPE).
      - `figures/forecast_sensitivity.png` — `plt.errorbar` plot of mean
        savings ±1 std vs σ, with a vertical dashed line at the active
        `forecast.ghi_noise_sigma`.

    With `forecast.sweep_seeds=1` the summary CSV reproduces the previous
    single-seed sweep exactly (mean = the single trial value, std = 0). This
    is the regression guarantee for legacy users.
    """
    fc_cfg = cfg.forecast or {}
    if sigmas is None:
        sigmas = list(fc_cfg.get("sweep_sigmas", [0.00, 0.10, 0.20, 0.30, 0.40]))
    if seeds is None:
        seeds = int(fc_cfg.get("sweep_seeds", 10))
    seeds = max(1, int(seeds))

    base_seed = int(fc_cfg.get("seed", 1337))

    # Baseline KPIs (constant across the sweep — only the smart side varies).
    peak_start = int(cfg.tariff.get("peak_start_hour", 7))
    peak_end = int(cfg.tariff.get("peak_end_hour", 22))
    base_hrs = df_base.index.hour.astype(int)
    base_cost = float(df_base["Hourly_Cost_RM"].astype(float).sum())
    base_peak = float(
        df_base.loc[(base_hrs >= peak_start) & (base_hrs < peak_end), "Grid_Import_kW"].astype(float).sum()
    )
    base_imp = float(df_base["Grid_Import_kW"].astype(float).sum())

    # Phase 4 input + RF model. When orchestrated from `main()` we pass these
    # in to avoid loading them twice; if called standalone, fall back to disk.
    if df_phase4_input is None:
        baseline_csv_path = repo / cfg.paths.get("baseline_results_csv", "baseline_simulation_results.csv")
        df_phase4_input = pd.read_csv(baseline_csv_path)
        df_phase4_input["datetime"] = pd.to_datetime(df_phase4_input["datetime"], errors="coerce")
        df_phase4_input = df_phase4_input.set_index("datetime").sort_index()
        df_phase4_input.index = normalize_datetime_index_to_naive(df_phase4_input.index)
    if rf_model is None:
        rf_path = repo / cfg.paths.get("rf_model_path", "models/rf_forecast_model_kuching.joblib")
        rf_model = joblib.load(rf_path)

    raw_rows: List[Dict[str, Any]] = []
    sweep_t0 = time.perf_counter()
    for sigma in sigmas:
        for seed_offset in range(seeds):
            cfg_trial: Dict[str, Any] = deepcopy(cfg.raw)
            fc = cfg_trial.setdefault("forecast", {})
            fc["enabled"] = True
            fc["ghi_noise_sigma"] = float(sigma)
            # When σ=0 we also pin every other noise-knob to zero so this row
            # is the *true* perfect-foresight reference (matches the unit-test
            # invariant `test_apply_forecast_noise_then_predict_round_trip`).
            if float(sigma) == 0.0:
                fc["ghi_bias"] = 0.0
                fc["cloudcover_noise_sigma"] = 0.0
                fc["temp_noise_sigma"] = 0.0
                fc["humidity_noise_sigma"] = 0.0
                fc["precip_noise_sigma"] = 0.0
            fc["seed"] = int(base_seed + seed_offset)

            df_trial = phase4_simulate(df_phase4_input, rf_model, cfg_trial)
            kpis = _kpis_from_smart_df(df_trial, tariff=cfg.tariff, base_cost=base_cost, base_peak=base_peak)
            raw_rows.append(
                {
                    "ghi_sigma": float(sigma),
                    "seed": int(base_seed + seed_offset),
                    "baseline_total_cost_rm": base_cost,
                    "baseline_total_import_kwh": base_imp,
                    "baseline_peak_import_kwh": base_peak,
                    **kpis,
                }
            )

    sweep_seconds = time.perf_counter() - sweep_t0
    print(
        f"Forecast sweep: {len(sigmas)} sigmas x {seeds} seeds "
        f"= {len(raw_rows)} trials in {sweep_seconds:.1f}s "
        f"({sweep_seconds/max(1,len(raw_rows)):.2f}s/trial)"
    )

    # Persist raw per-trial rows.
    raw_df = pd.DataFrame(raw_rows)
    raw_csv = results_dir / "forecast_sensitivity_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    # Aggregate per-σ mean/std/min/max for the four bottom-line KPIs.
    agg_kpis = ["total_savings_rm", "peak_import_reduction_kwh", "conservation_days_count", "rf_forecast_mape_daily"]
    summary_rows: List[Dict[str, Any]] = []
    for sigma in sigmas:
        sub = raw_df.loc[raw_df["ghi_sigma"] == float(sigma)]
        if len(sub) == 0:
            continue
        row: Dict[str, Any] = {
            "ghi_sigma": float(sigma),
            "n_seeds": int(len(sub)),
            "baseline_total_cost_rm": float(base_cost),
            # The dashboard expects these to be the headline (mean) numbers.
            "smart_total_cost_rm": float(sub["smart_total_cost_rm"].mean()),
            "savings_pct_of_baseline": float(sub["savings_pct_of_baseline"].mean()),
        }
        for kpi in agg_kpis:
            vals = sub[kpi].astype(float)
            row[kpi] = float(vals.mean())  # primary value = mean
            # std: ddof=0 when only one trial (so we report 0.0 deterministically)
            ddof = 1 if len(vals) > 1 else 0
            row[f"{kpi}_std"] = float(vals.std(ddof=ddof)) if len(vals) > 0 else 0.0
            row[f"{kpi}_min"] = float(vals.min())
            row[f"{kpi}_max"] = float(vals.max())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "forecast_sensitivity.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Figure: errorbar mean ± 1 std vs sigma, with active-σ vertical line.
    fig_dir = results_dir / "figures"
    _safe_mkdir(fig_dir)
    plt.figure(figsize=(7, 4))
    x = summary_df["ghi_sigma"].astype(float).to_numpy()
    y = summary_df["total_savings_rm"].astype(float).to_numpy()
    yerr = summary_df["total_savings_rm_std"].astype(float).to_numpy()
    plt.errorbar(
        x, y, yerr=yerr,
        fmt="o-",
        color="#2E86AB",
        ecolor="#888888",
        elinewidth=1.2,
        capsize=4,
        linewidth=2,
    )
    plt.axhline(0.0, color="#888888", linestyle=":", linewidth=1)
    active_sigma = float(fc_cfg.get("ghi_noise_sigma", 0.20))
    if any(abs(active_sigma - float(s)) < 1e-9 for s in sigmas):
        plt.axvline(active_sigma, color="#F18F01", linestyle="--", linewidth=1.2, alpha=0.8)
        ymin, ymax = plt.ylim()
        plt.text(
            active_sigma,
            ymin + 0.05 * (ymax - ymin),
            f" headline σ={active_sigma:.2f}",
            fontsize=9,
            color="#F18F01",
            va="bottom",
            ha="left",
        )
    for xi, yi in zip(x, y):
        plt.text(xi, yi, f"  {yi:.1f}", fontsize=9, va="center")
    plt.xlabel("GHI forecast noise σ (multiplicative log-normal)")
    plt.ylabel(f"Annual savings vs baseline (RM, mean ± 1σ over {seeds} seed{'s' if seeds != 1 else ''})")
    plt.title("Smart-controller savings vs forecast error")
    plt.tight_layout()
    plt.savefig(fig_dir / "forecast_sensitivity.png", dpi=180)
    plt.close()

    return summary_csv


def run_e_crit_sensitivity_sweep(
    cfg: Config,
    repo: Path,
    results_dir: Path,
    df_base: pd.DataFrame,
    *,
    df_phase4_input: Optional[pd.DataFrame] = None,
    rf_model: Optional[Any] = None,
    e_crit_values: Optional[List[float]] = None,
    seeds: Optional[int] = None,
) -> Tuple[Path, float]:
    """In-process grid search over `controller.e_crit_kwh`.

    For each candidate `e_crit_kwh` value, the smart simulation is re-run
    (with the *configured*, non-zero forecast noise — we are tuning the
    rule for the realistic noisy-forecast regime, not the oracle). Each
    e_crit is replicated across `seeds` forecast-noise seeds so the
    reported curve has error bars rather than relying on a single noisy
    draw.

    Outputs (under `results_dir`):
      - `e_crit_sensitivity_raw.csv` — one row per (e_crit, seed) trial.
      - `e_crit_sensitivity.csv` — per-e_crit summary (mean ± std savings,
        peak reduction, conservation days, MAPE).
      - `figures/e_crit_sensitivity.png` — `plt.errorbar` plot of mean
        savings ±1 std vs e_crit, with vertical lines at (a) the
        currently-configured `controller.e_crit_kwh` and (b) the argmax
        e_crit by mean total savings.

    Returns a tuple of `(summary_csv_path, best_e_crit_kwh)` so the
    caller can log the chosen value alongside the headline metrics.
    """
    ctrl_cfg = cfg.controller or {}
    fc_cfg = cfg.forecast or {}

    if e_crit_values is None:
        e_crit_values = list(
            ctrl_cfg.get(
                "e_crit_sweep_values",
                [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
            )
        )
    if seeds is None:
        seeds = int(ctrl_cfg.get("e_crit_sweep_seeds", 3))
    seeds = max(1, int(seeds))

    base_seed = int(fc_cfg.get("seed", 1337))

    # Baseline KPIs (constant across the sweep — only the smart side varies).
    peak_start = int(cfg.tariff.get("peak_start_hour", 7))
    peak_end = int(cfg.tariff.get("peak_end_hour", 22))
    base_hrs = df_base.index.hour.astype(int)
    base_cost = float(df_base["Hourly_Cost_RM"].astype(float).sum())
    base_peak = float(
        df_base.loc[(base_hrs >= peak_start) & (base_hrs < peak_end), "Grid_Import_kW"].astype(float).sum()
    )
    base_imp = float(df_base["Grid_Import_kW"].astype(float).sum())

    if df_phase4_input is None:
        baseline_csv_path = repo / cfg.paths.get("baseline_results_csv", "baseline_simulation_results.csv")
        df_phase4_input = pd.read_csv(baseline_csv_path)
        df_phase4_input["datetime"] = pd.to_datetime(df_phase4_input["datetime"], errors="coerce")
        df_phase4_input = df_phase4_input.set_index("datetime").sort_index()
        df_phase4_input.index = normalize_datetime_index_to_naive(df_phase4_input.index)
    if rf_model is None:
        rf_path = repo / cfg.paths.get("rf_model_path", "models/rf_forecast_model_kuching.joblib")
        rf_model = joblib.load(rf_path)

    raw_rows: List[Dict[str, Any]] = []
    sweep_t0 = time.perf_counter()
    for e_crit in e_crit_values:
        for seed_offset in range(seeds):
            cfg_trial: Dict[str, Any] = deepcopy(cfg.raw)
            ctrl = cfg_trial.setdefault("controller", {})
            ctrl["e_crit_kwh"] = float(e_crit)
            # Use the configured (non-zero) forecast noise so the sweep
            # picks the e_crit that's best under realistic forecast error.
            fc = cfg_trial.setdefault("forecast", {})
            fc["enabled"] = True
            fc["seed"] = int(base_seed + seed_offset)

            df_trial = phase4_simulate(df_phase4_input, rf_model, cfg_trial)
            kpis = _kpis_from_smart_df(df_trial, tariff=cfg.tariff, base_cost=base_cost, base_peak=base_peak)
            raw_rows.append(
                {
                    "e_crit_kwh": float(e_crit),
                    "seed": int(base_seed + seed_offset),
                    "baseline_total_cost_rm": base_cost,
                    "baseline_total_import_kwh": base_imp,
                    "baseline_peak_import_kwh": base_peak,
                    **kpis,
                }
            )

    sweep_seconds = time.perf_counter() - sweep_t0
    print(
        f"E_crit sweep: {len(e_crit_values)} values x {seeds} seeds "
        f"= {len(raw_rows)} trials in {sweep_seconds:.1f}s "
        f"({sweep_seconds/max(1,len(raw_rows)):.2f}s/trial)"
    )

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = results_dir / "e_crit_sensitivity_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    agg_kpis = ["total_savings_rm", "peak_import_reduction_kwh", "conservation_days_count", "rf_forecast_mape_daily"]
    summary_rows: List[Dict[str, Any]] = []
    for e_crit in e_crit_values:
        sub = raw_df.loc[raw_df["e_crit_kwh"] == float(e_crit)]
        if len(sub) == 0:
            continue
        row: Dict[str, Any] = {
            "e_crit_kwh": float(e_crit),
            "n_seeds": int(len(sub)),
            "baseline_total_cost_rm": float(base_cost),
            "smart_total_cost_rm": float(sub["smart_total_cost_rm"].mean()),
            "savings_pct_of_baseline": float(sub["savings_pct_of_baseline"].mean()),
        }
        for kpi in agg_kpis:
            vals = sub[kpi].astype(float)
            row[kpi] = float(vals.mean())
            ddof = 1 if len(vals) > 1 else 0
            row[f"{kpi}_std"] = float(vals.std(ddof=ddof)) if len(vals) > 0 else 0.0
            row[f"{kpi}_min"] = float(vals.min())
            row[f"{kpi}_max"] = float(vals.max())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "e_crit_sensitivity.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Pick the argmax of mean total savings as the "optimised" e_crit.
    best_idx = int(summary_df["total_savings_rm"].astype(float).idxmax())
    best_e_crit = float(summary_df.loc[best_idx, "e_crit_kwh"])
    best_savings = float(summary_df.loc[best_idx, "total_savings_rm"])
    active_e_crit = float(ctrl_cfg.get("e_crit_kwh", 18.0))

    # Figure: errorbar mean ± 1 std vs e_crit, with active + best vlines.
    fig_dir = results_dir / "figures"
    _safe_mkdir(fig_dir)
    plt.figure(figsize=(7.5, 4.2))
    x = summary_df["e_crit_kwh"].astype(float).to_numpy()
    y = summary_df["total_savings_rm"].astype(float).to_numpy()
    yerr = summary_df["total_savings_rm_std"].astype(float).to_numpy()
    plt.errorbar(
        x, y, yerr=yerr,
        fmt="o-",
        color="#2E86AB",
        ecolor="#888888",
        elinewidth=1.2,
        capsize=4,
        linewidth=2,
    )
    plt.axhline(0.0, color="#888888", linestyle=":", linewidth=1)
    plt.axvline(active_e_crit, color="#F18F01", linestyle="--", linewidth=1.2, alpha=0.8,
                label=f"configured E_crit = {active_e_crit:.1f} kWh")
    plt.axvline(best_e_crit, color="#2E8B57", linestyle="-.", linewidth=1.4, alpha=0.9,
                label=f"argmax E_crit = {best_e_crit:.1f} kWh (RM {best_savings:.1f})")
    for xi, yi in zip(x, y):
        plt.text(xi, yi, f"  {yi:.1f}", fontsize=8, va="center")
    plt.xlabel("Conservation-mode threshold E_crit (kWh/day)")
    plt.ylabel(f"Annual savings vs baseline (RM, mean ± 1σ over {seeds} seed{'s' if seeds != 1 else ''})")
    plt.title("Smart-controller savings vs conservation-mode threshold E_crit")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "e_crit_sensitivity.png", dpi=180)
    plt.close()

    print(
        f"E_crit sweep: argmax E_crit = {best_e_crit:.1f} kWh "
        f"(annual savings = RM {best_savings:.2f}); "
        f"configured E_crit = {active_e_crit:.1f} kWh."
    )

    return summary_csv, best_e_crit


def save_figures(df_base: pd.DataFrame, df_smart: pd.DataFrame, daily: pd.DataFrame, *, out_dir: Path, metrics: dict) -> None:
    _safe_mkdir(out_dir)

    # 1) Total cost bars
    plt.figure(figsize=(6, 4))
    names = ["Baseline", "AI"]
    vals = [float(metrics["baseline_total_cost_rm"]), float(metrics["smart_total_cost_rm"])]
    plt.bar(names, vals, color=["#888888", "#2E86AB"])
    plt.ylabel("Total Cost (RM)")
    plt.title("Total electricity cost (continuous CSV sums; headline KPI)")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "total_cost_comparison.png", dpi=180)
    plt.close()

    # 2) Daily savings histogram
    plt.figure(figsize=(7, 4))
    s = daily["savings_rm"].astype(float)
    plt.hist(s, bins=30, color="#2E86AB", alpha=0.85)
    plt.axvline(float(s.mean()), color="black", linestyle="--", linewidth=1, label=f"mean={s.mean():.2f}")
    plt.xlabel("Daily savings (RM)  [baseline - AI]")
    plt.ylabel("Days")
    plt.title("Distribution of daily savings (daily-reset evaluation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "daily_savings_hist.png", dpi=180)
    plt.close()

    # 3) Daily peak import reduction histogram
    plt.figure(figsize=(7, 4))
    p = daily["peak_import_reduction_kwh"].astype(float)
    plt.hist(p, bins=30, color="#F18F01", alpha=0.85)
    plt.axvline(float(p.mean()), color="black", linestyle="--", linewidth=1, label=f"mean={p.mean():.2f}")
    plt.xlabel("Daily peak import reduction (kWh)  [baseline - AI]")
    plt.ylabel("Days")
    plt.title("Distribution of daily peak import reduction (daily-reset evaluation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "daily_peak_import_reduction_hist.png", dpi=180)
    plt.close()

    # 4) SoC distribution (from whole-run CSVs)
    plt.figure(figsize=(7, 4))
    plt.hist(df_base["SoC"].astype(float), bins=40, alpha=0.5, label="Baseline", color="#888888")
    plt.hist(df_smart["SoC"].astype(float), bins=40, alpha=0.5, label="AI", color="#2E86AB")
    plt.xlabel("SoC")
    plt.ylabel("Hours")
    plt.title("SoC distribution (whole dataset)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "soc_distribution.png", dpi=180)
    plt.close()


def _resolve_per_region_baseline_csv(repo: Path, cfg: Config, region: str) -> Path:
    """Path where `phase2_baseline_sim.py --region <region>` writes its CSV."""
    tpl = cfg.paths.get("baseline_results_template")
    if tpl:
        return repo / tpl.format(region=region)
    return repo / cfg.paths.get("baseline_results_csv", "baseline_simulation_results.csv")


def _resolve_per_region_smart_csv(repo: Path, cfg: Config, region: str, algo: str) -> Path:
    tpl = cfg.paths.get("smart_results_template")
    if tpl:
        return repo / tpl.format(region=region, algo=algo)
    return repo / cfg.paths.get("smart_results_csv", "smart_simulation_results.csv")


def _resolve_per_region_model(repo: Path, cfg: Config, region: str, algo: str) -> Path:
    tpl = cfg.paths.get("model_template")
    if tpl:
        return repo / tpl.format(region=region, algo=algo)
    return repo / cfg.paths.get("rf_model_path", "models/rf_forecast_model_kuching.joblib")


def run_one_region_algo(
    *,
    region: str,
    algo: str,
    cfg: Config,
    repo: Path,
    config_path: Path,
    run_root: Path,
    df_phase4_input: pd.DataFrame,
    skip_sweeps: bool = False,
) -> Dict[str, Any]:
    """End-to-end Phase 4 + validations + sweeps + metrics for one (region, algo).

    Pre-conditions:
      - Phase 2 has already produced the per-region baseline CSV (it is shared
        across algos: same load profile, same baseline → same `df_phase4_input`).
      - The forecaster `models/{algo}_forecast_model_{region}.joblib` exists.

    Side effects (under `run_root/{region}/{algo}/`):
      - `smart_simulation_results.csv`
      - `daily_results.csv`
      - `metrics.json`
      - `forecast_sensitivity.csv` / `..._raw.csv` / `figures/forecast_sensitivity.png`
      - `e_crit_sensitivity.csv` / `..._raw.csv` / `figures/e_crit_sensitivity.png`
      - `figures/*.png` (cost bars, daily savings histogram, etc.)
    """
    cfg_for_phase4 = deepcopy(cfg.raw)  # keep the master cfg pristine

    region_dir = run_root / region / algo
    figures_dir = region_dir / "figures"
    _safe_mkdir(figures_dir)

    # --- forecaster
    model_path = _resolve_per_region_model(repo, cfg, region, algo)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Forecaster for ({region}, {algo}) missing: {model_path}. "
            f"Set `pipeline.run_phase3_train_model=true` or run "
            f"`python phase3_ai_training.py --all-regions` first."
        )
    print(f"[{region}/{algo}] loading model: {model_path}")
    model = joblib.load(model_path)

    # --- Phase 4 headline (in-process)
    smart_csv = _resolve_per_region_smart_csv(repo, cfg, region, algo)
    smart_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{region}/{algo}] running smart-controller simulation (in-process)...")
    t_headline = time.perf_counter()
    df_smart_full = phase4_simulate(df_phase4_input, model, cfg_for_phase4, verbose=False)
    print(f"[{region}/{algo}] phase4 simulate done in {time.perf_counter()-t_headline:.1f}s")
    df_smart_full.to_csv(smart_csv)
    shutil.copy2(smart_csv, region_dir / "smart_simulation_results.csv")

    # Copy the shared baseline CSV next to it so the dashboard can read one folder.
    base_csv = _resolve_per_region_baseline_csv(repo, cfg, region)
    shutil.copy2(base_csv, region_dir / "baseline_simulation_results.csv")

    # --- reload + normalize indices (same dance the old single-region main() did)
    df_base = pd.read_csv(region_dir / "baseline_simulation_results.csv")
    df_smart = pd.read_csv(region_dir / "smart_simulation_results.csv")
    df_base["datetime"] = pd.to_datetime(df_base["datetime"], errors="coerce")
    df_smart["datetime"] = pd.to_datetime(df_smart["datetime"], errors="coerce")
    df_base = df_base.set_index("datetime").sort_index()
    df_smart = df_smart.set_index("datetime").sort_index()
    df_base.index = normalize_datetime_index_to_naive(df_base.index)
    df_smart.index = normalize_datetime_index_to_naive(df_smart.index)
    if df_base.index.tz != df_smart.index.tz:
        raise ValueError(
            f"[{region}/{algo}] Timezone mismatch: baseline tz={df_base.index.tz}, smart tz={df_smart.index.tz}"
        )

    # --- validations (4 formulas from docs/VALIDATION.md surface here)
    soc_min = float(cfg.battery.get("soc_min", 0.15))
    validate_timeseries(df_base, name=f"{region}/{algo}/baseline", soc_min=soc_min)
    validate_timeseries(df_smart, name=f"{region}/{algo}/smart", soc_min=soc_min)
    validate_energy_balance(df_base, name=f"{region}/{algo}/baseline")
    validate_energy_balance(df_smart, name=f"{region}/{algo}/smart")

    tariff_check = validate_tariff_cost_identity(df_base, df_smart, tariff=cfg.tariff)
    if int(tariff_check.get("samples_checked", 0)) <= 0:
        raise RuntimeError(f"[{region}/{algo}] tariff spot-check found no rows: {tariff_check}")
    if not bool(tariff_check.get("tariff_cost_identity_passed", False)):
        raise RuntimeError(f"[{region}/{algo}] tariff identity check failed: {tariff_check}")

    # --- daily-reset evaluation table + headline metrics
    soc_init_daily = float(cfg.scenario.get("initial_soc_daily_reset", 0.50))
    daily = compute_daily_results(
        df_base, df_smart,
        soc_init=soc_init_daily,
        batt=cfg.battery,
        tariff=cfg.tariff,
        policy=cfg.controller,
        scenario=cfg.scenario,
    )
    daily.to_csv(region_dir / "daily_results.csv", index=False)

    metrics = compute_metrics(
        df_base, df_smart, daily,
        tariff=cfg.tariff,
        daily_reset_soc_init=soc_init_daily,
        evaluation_cfg=cfg.evaluation,
        tariff_validation=tariff_check,
        forecast_cfg=cfg.forecast,
        results_dir=region_dir,  # for perfect-foresight % once the sweep CSV is in place
    )
    metrics["region"] = region
    metrics["algo"] = algo
    (region_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_figures(df_base, df_smart, daily, out_dir=figures_dir, metrics=metrics)

    if skip_sweeps:
        print(f"[{region}/{algo}] --no-sweeps: skipping forecast sigma-sweep and e_crit sweep")
        best_e_crit = float(cfg.controller.get("e_crit_kwh", 18.0))
        e_crit_summary_csv = region_dir / "e_crit_sensitivity.csv"  # may not exist; OK
    else:
        # --- forecast σ-sensitivity sweep (oracle ceiling lives at σ=0)
        sweep_csv = run_forecast_sensitivity_sweep(
            cfg=cfg, repo=repo,
            results_dir=region_dir,
            config_path=config_path,
            df_base=df_base,
            df_phase4_input=df_phase4_input,
            rf_model=model,
        )
        print(f"[{region}/{algo}] forecast sensitivity -> {sweep_csv}")

        # --- E_crit sweep (rule-based optimiser for the conservation-mode threshold)
        e_crit_summary_csv, best_e_crit = run_e_crit_sensitivity_sweep(
            cfg=cfg, repo=repo,
            results_dir=region_dir,
            df_base=df_base,
            df_phase4_input=df_phase4_input,
            rf_model=model,
        )
        print(f"[{region}/{algo}] e_crit sweep -> {e_crit_summary_csv}; argmax E_crit = {best_e_crit:.1f} kWh")

    # --- now that the σ-sweep CSV exists, re-patch the metrics with the
    # perfect-foresight ceiling % (which compute_metrics couldn't compute
    # the first time because the CSV didn't exist yet).
    try:
        metrics_disk = json.loads((region_dir / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        metrics_disk = metrics
    ppc = _perfect_foresight_pct(region_dir, float(metrics_disk.get("total_savings_rm", 0.0)))
    metrics_disk.setdefault("statistical_tests", {}).update(ppc)
    metrics_disk.setdefault("_meta", {})
    metrics_disk["_meta"]["e_crit_optimisation"] = {
        "configured_e_crit_kwh": float(cfg.controller.get("e_crit_kwh", 18.0)),
        "argmax_e_crit_kwh": float(best_e_crit),
        "sweep_summary_csv": Path(e_crit_summary_csv).name,
    }
    (region_dir / "metrics.json").write_text(json.dumps(metrics_disk, indent=2), encoding="utf-8")

    # --- one row for the cross-region/cross-algo summary CSV
    cv_report_path = repo / cfg.paths.get("cv_report_template", "models/cv_report_{region}.json").format(region=region)
    cv_rmse_mean = cv_rmse_std = test_rmse = test_r2 = float("nan")
    if cv_report_path.exists():
        try:
            rep = json.loads(cv_report_path.read_text(encoding="utf-8"))
            algo_block = rep.get("algorithms", {}).get(algo, {})
            cv_rmse_mean = float(algo_block.get("cv_rmse_mean", float("nan")))
            cv_rmse_std = float(algo_block.get("cv_rmse_std", float("nan")))
            test_rmse = float(algo_block.get("test_rmse", float("nan")))
            test_r2 = float(algo_block.get("test_r2", float("nan")))
        except Exception:
            pass

    stats = metrics_disk.get("statistical_tests", {})
    return {
        "region": region,
        "algo": algo,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "baseline_total_cost_rm": float(metrics_disk.get("baseline_total_cost_rm", float("nan"))),
        "smart_total_cost_rm": float(metrics_disk.get("smart_total_cost_rm", float("nan"))),
        "annual_savings_rm": float(metrics_disk.get("total_savings_rm", float("nan"))),
        "peak_reduction_kwh": float(metrics_disk.get("peak_import_reduction_kwh", float("nan"))),
        "ssr_smart_pct": float(metrics_disk.get("smart_ssr_pct", float("nan"))),
        "ssr_baseline_pct": float(metrics_disk.get("baseline_ssr_pct", float("nan"))),
        "wilcoxon_p": float(stats.get("wilcoxon_signed_rank", {}).get("p_value", float("nan"))),
        "paired_t_p": float(stats.get("paired_t_test", {}).get("p_value", float("nan"))),
        "pct_of_perfect_foresight_ceiling": float(stats.get("pct_of_perfect_foresight_ceiling", float("nan"))),
        "best_e_crit_kwh": float(best_e_crit),
        "metrics_json_path": str(region_dir / "metrics.json"),
    }


@dataclass(frozen=True)
class RunAllArgs:
    config_path: Path
    regions_filter: Optional[List[str]]
    algos_filter: Optional[List[str]]
    skip_sweeps: bool


def _parse_run_all_args(argv: List[str]) -> RunAllArgs:
    """Parse `run_all.py [config_path] [--region a] [--region b] [--algo rf] [--no-sweeps]`.

    Multiple `--region`/`--algo` flags are allowed for ad-hoc subsets.
    Positional `argv[1]` (if not starting with `-`) is the config file path,
    matching the legacy invocation `python run_all.py experiment_config.json`.

    `--no-sweeps` disables the σ-sensitivity and e_crit sweeps (which iterate
    Phase 4 50+ times each and are the main wall-clock cost for SVM, where a
    single predict on 40k rows takes ~45s instead of ~2s for RF). The flag is
    intended for fast iteration; the thesis-grade nightly run should NOT
    pass it.
    """
    config_path = "experiment_config.json"
    regions_filter: List[str] = []
    algos_filter: List[str] = []
    skip_sweeps = False
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--region" and i + 1 < len(argv):
            regions_filter.append(argv[i + 1]); i += 2; continue
        if tok == "--algo" and i + 1 < len(argv):
            algos_filter.append(argv[i + 1]); i += 2; continue
        if tok == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]; i += 2; continue
        if tok == "--no-sweeps":
            skip_sweeps = True; i += 1; continue
        if not tok.startswith("-"):
            config_path = tok
            i += 1; continue
        i += 1
    return RunAllArgs(
        config_path=Path(config_path).resolve(),
        regions_filter=(regions_filter or None),
        algos_filter=(algos_filter or None),
        skip_sweeps=skip_sweeps,
    )


def main() -> None:
    pipeline_t0 = time.perf_counter()
    repo = Path(__file__).resolve().parent
    args = _parse_run_all_args(sys.argv[1:])
    config_path = args.config_path
    regions_filter = args.regions_filter
    algos_filter = args.algos_filter
    skip_sweeps = args.skip_sweeps
    if not config_path.is_absolute():
        config_path = (repo / config_path).resolve()
    cfg = load_config(config_path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = repo / "results" / stamp
    _safe_mkdir(results_dir)

    # Snapshot config used
    (results_dir / "config_snapshot.json").write_text(json.dumps(cfg.raw, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Resolve the list of (region, algorithm) pairs to process
    # ------------------------------------------------------------------
    regions_cfg = cfg.raw.get("regions", {}) or {}
    region_list = list(regions_cfg.get("list", []))
    if not region_list:
        # Fully backward-compat path: no regions block -> single-region legacy run
        region_list = [{"name": "default", "lat": None, "lon": None}]
    algos = list((cfg.raw.get("models", {}) or {}).get("algorithms", ["rf"]))

    # CLI filters: `--region NAME` and/or `--algo NAME` narrow the loop
    # without editing the config (useful for smoke-testing on a single region).
    if regions_filter:
        names_in_cfg = {r["name"] for r in region_list}
        unknown = [r for r in regions_filter if r not in names_in_cfg]
        if unknown:
            raise SystemExit(
                f"--region values not found in config.regions.list: {unknown} "
                f"(available: {sorted(names_in_cfg)})"
            )
        region_list = [r for r in region_list if r["name"] in regions_filter]
    if algos_filter:
        unknown = [a for a in algos_filter if a not in algos]
        if unknown:
            raise SystemExit(
                f"--algo values not found in config.models.algorithms: {unknown} (available: {algos})"
            )
        algos = list(algos_filter)

    print(
        f"[run_all] regions={[r['name'] for r in region_list]}  algos={algos}  "
        f"-> {len(region_list) * len(algos)} (region, algo) simulations"
    )

    # ------------------------------------------------------------------
    # Optional: multi-region Phase 1 (split + clean + pvlib for every region)
    # ------------------------------------------------------------------
    if bool(cfg.pipeline.get("run_phase1", False)):
        splitter = repo / "phase1_split_multi_region.py"
        if splitter.exists():
            print("[run_all] running multi-region Phase 1 split...")
            _run([sys.executable, str(splitter), "--config", str(config_path)], cwd=repo)

    # ------------------------------------------------------------------
    # Phase 2 baseline -- ONE per region (load profile depends only on the
    # shared seed + the hourly index; PV differs per region via Target_Power_kW)
    # ------------------------------------------------------------------
    for region in region_list:
        rname = region["name"]
        print(f"[run_all] Phase 2 baseline for region={rname}...")
        _run([
            sys.executable, str(repo / "phase2_baseline_sim.py"),
            "--config", str(config_path),
            "--region", rname,
        ], cwd=repo)

    # ------------------------------------------------------------------
    # Optional: retrain models (run once at the start; produces all
    # `models/{algo}_forecast_model_{region}.joblib` artefacts).
    # ------------------------------------------------------------------
    if bool(cfg.pipeline.get("run_phase3_train_model", False)):
        phase3 = repo / "phase3_ai_training.py"
        if phase3.exists():
            print("[run_all] training all (region, algo) forecasters...")
            _run([
                sys.executable, str(phase3),
                "--config", str(config_path),
                "--all-regions",
            ], cwd=repo)

    # ------------------------------------------------------------------
    # Phase 4 + sweeps + metrics PER (region, algo)
    # ------------------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    for region in region_list:
        rname = region["name"]
        # Load this region's baseline ONCE; share the dataframe across both algos.
        baseline_csv = _resolve_per_region_baseline_csv(repo, cfg, rname)
        print(f"[run_all] [{rname}] loading baseline: {baseline_csv}")
        df_phase4_input = pd.read_csv(baseline_csv)
        df_phase4_input["datetime"] = pd.to_datetime(df_phase4_input["datetime"], errors="coerce")
        df_phase4_input = df_phase4_input.set_index("datetime").sort_index()
        df_phase4_input.index = normalize_datetime_index_to_naive(df_phase4_input.index)

        for algo in algos:
            row = run_one_region_algo(
                region=rname,
                algo=algo,
                cfg=cfg,
                repo=repo,
                config_path=config_path,
                run_root=results_dir,
                df_phase4_input=df_phase4_input,
                skip_sweeps=skip_sweeps,
            )
            summary_rows.append(row)

    # ------------------------------------------------------------------
    # Cross-(region, algo) summary CSV + winner-per-region verdict
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[run_all] summary -> {summary_csv}")
    print(summary_df.to_string(index=False))

    # Best algo per region: the one with the lowest test_rmse (forecast accuracy).
    # If test_rmse is NaN everywhere (no CV report present), fall back to the algo
    # with the highest annual_savings_rm.
    verdict_lines: List[str] = []
    for rname in [r["name"] for r in region_list]:
        sub = summary_df.loc[summary_df["region"] == rname]
        if sub.empty:
            continue
        if sub["test_rmse"].notna().any():
            best_row = sub.loc[sub["test_rmse"].astype(float).idxmin()]
            criterion = "lowest_test_rmse"
        else:
            best_row = sub.loc[sub["annual_savings_rm"].astype(float).idxmax()]
            criterion = "highest_savings"
        verdict_lines.append(f"{rname}={best_row['algo']} ({criterion})")
    verdict = "Best model per region: " + ", ".join(verdict_lines)
    print(f"\n[run_all] {verdict}")

    (results_dir / "verdict.txt").write_text(verdict + "\n", encoding="utf-8")

    pipeline_seconds = time.perf_counter() - pipeline_t0
    print(f"\n[run_all] Total wall clock: {pipeline_seconds:.1f}s ({pipeline_seconds/60.0:.2f} min)")


if __name__ == "__main__":
    main()

