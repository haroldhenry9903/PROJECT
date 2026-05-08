from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from core_battery_model import Battery


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


def compute_metrics(
    df_base: pd.DataFrame,
    df_smart: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    tariff: dict,
    daily_reset_soc_init: float,
    evaluation_cfg: Optional[dict] = None,
    tariff_validation: Optional[Dict[str, Any]] = None,
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
    return legacy


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


def main() -> None:
    repo = Path(__file__).resolve().parent
    config_path = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else (repo / "experiment_config.json")
    cfg = load_config(config_path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = repo / "results" / stamp
    figures_dir = results_dir / "figures"
    _safe_mkdir(figures_dir)

    # Snapshot config used
    (results_dir / "config_snapshot.json").write_text(json.dumps(cfg.raw, indent=2), encoding="utf-8")

    # Optional: Phase 1 / Phase 3 hooks (kept minimal; scripts may or may not exist)
    if bool(cfg.pipeline.get("run_phase1", False)):
        phase1a = repo / "phase1a_data_cleaning.py"
        phase1b = repo / "phase1b_physics_synthesis.py"
        if phase1a.exists():
            _run([sys.executable, str(phase1a)], cwd=repo)
        if phase1b.exists():
            _run([sys.executable, str(phase1b)], cwd=repo)

    # Phase 2 baseline
    _run([sys.executable, str(repo / "phase2_baseline_sim.py"), "--config", str(config_path)], cwd=repo)

    # Optional: retrain model
    if bool(cfg.pipeline.get("run_phase3_train_model", False)):
        phase3 = repo / "phase3_ai_training.py"
        if phase3.exists():
            _run([sys.executable, str(phase3), "--config", str(config_path)], cwd=repo)

    # Phase 4 smart
    _run([sys.executable, str(repo / "phase4_smart_optimizer.py"), "--config", str(config_path)], cwd=repo)

    # Load outputs
    base_csv = repo / cfg.paths.get("baseline_results_csv", "baseline_simulation_results.csv")
    smart_csv = repo / cfg.paths.get("smart_results_csv", "smart_simulation_results.csv")

    if not base_csv.exists():
        raise FileNotFoundError(f"Missing baseline output: {base_csv}")
    if not smart_csv.exists():
        raise FileNotFoundError(f"Missing smart output: {smart_csv}")

    # Copy raw CSVs into results folder (reproducible artifact)
    shutil.copy2(base_csv, results_dir / "baseline_simulation_results.csv")
    shutil.copy2(smart_csv, results_dir / "smart_simulation_results.csv")

    df_base = pd.read_csv(results_dir / "baseline_simulation_results.csv")
    df_smart = pd.read_csv(results_dir / "smart_simulation_results.csv")

    df_base["datetime"] = pd.to_datetime(df_base["datetime"], errors="coerce")
    df_smart["datetime"] = pd.to_datetime(df_smart["datetime"], errors="coerce")
    df_base = df_base.set_index("datetime").sort_index()
    df_smart = df_smart.set_index("datetime").sort_index()

    df_base.index = normalize_datetime_index_to_naive(df_base.index)
    df_smart.index = normalize_datetime_index_to_naive(df_smart.index)

    # Timezone consistency guard (both must match)
    if df_base.index.tz != df_smart.index.tz:
        raise ValueError(f"Timezone mismatch: baseline tz={df_base.index.tz}, smart tz={df_smart.index.tz}")

    soc_min = float(cfg.battery.get("soc_min", 0.15))

    validate_timeseries(df_base, name="baseline", soc_min=soc_min)
    validate_timeseries(df_smart, name="smart", soc_min=soc_min)

    # Stronger self-checks (requires Batt_Action_kW in both CSVs)
    validate_energy_balance(df_base, name="baseline")
    validate_energy_balance(df_smart, name="smart")

    # Daily-reset evaluation (fair day-isolated comparisons)
    soc_init_daily = float(cfg.scenario.get("initial_soc_daily_reset", 0.50))
    daily = compute_daily_results(
        df_base,
        df_smart,
        soc_init=soc_init_daily,
        batt=cfg.battery,
        tariff=cfg.tariff,
        policy=cfg.controller,
        scenario=cfg.scenario,
    )
    daily.to_csv(results_dir / "daily_results.csv", index=False)

    tariff_check = validate_tariff_cost_identity(df_base, df_smart, tariff=cfg.tariff)
    if int(tariff_check.get("samples_checked", 0)) <= 0:
        raise RuntimeError(f"Tariff spot-check found no hourly rows to validate: {tariff_check}")
    if not bool(tariff_check.get("tariff_cost_identity_passed", False)):
        raise RuntimeError(f"Tariff vs Hourly_Cost_RM identity check failed: {tariff_check}")

    metrics = compute_metrics(
        df_base,
        df_smart,
        daily,
        tariff=cfg.tariff,
        daily_reset_soc_init=soc_init_daily,
        evaluation_cfg=cfg.evaluation,
        tariff_validation=tariff_check,
    )
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_figures(df_base, df_smart, daily, out_dir=figures_dir, metrics=metrics)


if __name__ == "__main__":
    main()

