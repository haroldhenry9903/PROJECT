from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from core_battery_model import Battery
from phase2_baseline_sim import get_tariff_rate
from repo_paths import repo_root, resolve_repo_path


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AI-Predicted Solar Generation for Optimised Rule-Based Scheduling of Hybrid Solar Battery Systems: A Sarawak Simulation",
    layout="wide",
)


# -----------------------------
# Guided help + Streamlit capability detection
# -----------------------------
_HAS_ST_DIALOG = hasattr(st, "dialog")
_HAS_ST_POPOVER = hasattr(st, "popover")

HELP = {
    "guided_help_toggle": (
        "Turns on extra explanations (expanders) across the dashboard. Hover help still works when this is off."
    ),
    "experiment_settings_expander": (
        "Shows the exact JSON parameters from `experiment_config.json` (your reproducible “lab notebook settings”). "
        "Use this to compare a saved `results/...` run against what the dashboard sliders currently say."
    ),
    "results_run_toggle": (
        "Loads CSV outputs from a saved `results/YYYYMMDD_HHMM/` folder (created by `run_all.py`). "
        "Use this when you want the dashboard to match saved `metrics.json` / `daily_results.csv`."
    ),
    "results_run_select": (
        "Pick which saved run folder to load. Newest is selected by default."
    ),
    "date_picker": (
        "Pick one calendar day from the loaded dataset. Tab 2 simulates that day; Tab 3 uses the full year."
    ),
    "battery_capacity": (
        "Battery size in kWh (how much energy it can store). Bigger storage can carry more solar into later hours."
    ),
    "inverter_power": (
        "Max charge/discharge power in kW (how fast the battery can move energy). Higher kW can cover bigger spikes."
    ),
    "soc_min": (
        "Minimum allowed battery fullness (%). Higher floors protect the battery but reduce usable energy."
    ),
    "pv_scale": (
        "What-if multiplier for solar output. 1.0 means “as modeled”. Use >1 or <1 to explore bigger/smaller PV systems."
    ),
    "load_scale": (
        "What-if multiplier for household demand. 1.0 means “as modeled”. Use >1 or <1 to explore heavier/lighter usage."
    ),
    "custom_tariff_toggle": (
        "Switch between the default tariff mode and your own peak/off-peak hours and prices."
    ),
    "peak_start": ("First hour counted as expensive (peak) pricing."),
    "peak_end": ("Hour after the last peak-priced hour (peak is [start, end) )."),
    "peak_rate": ("Price per kWh during peak hours (RM/kWh)."),
    "offpeak_rate": ("Price per kWh during off-peak hours (RM/kWh)."),
    "e_crit": (
        "If predicted tomorrow solar energy is below this threshold, the day is treated as a low-solar day "
        "(Conservation Mode). Lower E_crit means fewer conservation days."
    ),
    "hoard_start": (
        "Used for the shaded ‘hoarding window’ in Tab 2’s SoC chart. It does not change the smart controller math; "
        "the policy uses off-peak vs peak hours + reserve SoC."
    ),
    "hoard_end": (
        "End hour for the shaded hoarding window in Tab 2 (visual aid only)."
    ),
    "soc_reserve": (
        "Target battery fullness to protect for expensive peak hours on conservation days. Higher values hoard more."
    ),
    "essential_discharge": (
        "How much battery discharge is still allowed during off-peak reserve protection. 0 kW is strict hoarding."
    ),
    "tab2_initial_soc": (
        "Starting battery fullness for the one-day interactive simulation in Tab 2. "
        "This only affects Tab 2 unless you intentionally match Tab 3’s daily-reset slider."
    ),
    "tab2_run_button": (
        "Re-simulates the selected calendar day using your current sidebar hardware/tariff/AI settings. "
        "This does not rewrite files in `results/`."
    ),
    "tab3_year": ("Pick which calendar year to summarize (requires overlapping baseline + smart data)."),
    "tab3_agg": (
        "Resamples the hourly saved timelines before plotting. Hourly shows raw series; Daily/Monthly sums energy-like quantities."
    ),
    "tab3_conservation_only": (
        "Filters the view to days where the saved smart run marked Conservation Mode active (based on saved `Conservation_Mode_Active`)."
    ),
    "tab3_daily_reset": (
        "When ON, the headline KPIs recompute each day with the same starting SoC (independent days). "
        "The time-series plots still come from the saved hourly CSV unless you regenerate CSVs."
    ),
    "tab3_daily_reset_soc": (
        "Starting SoC used for Tab 3’s daily-reset KPI recomputation (each day starts here again)."
    ),
}


def _guided_help() -> bool:
    return bool(st.session_state.get("guided_help", False))


def section_help(title: str, body_md: str) -> None:
    if not _guided_help():
        return
    with st.expander(title, expanded=False):
        st.markdown(body_md)


def chart_help(caption: str, title: str, body_md: str) -> None:
    st.caption(caption)
    if not _guided_help():
        return
    with st.expander(title, expanded=False):
        st.markdown(body_md)


def explain_tab_header(tab_key: str, title: str, body_md: str) -> None:
    """
    Per-tab explainer:
    - Prefer Streamlit dialog when available
    - Else fall back to expander
    """
    open_key = f"{tab_key}_explain_open"
    all_open_keys = [f"{k}_explain_open" for k in ("tab1", "tab2", "tab3", "tab4")]

    def _clear_open() -> None:
        st.session_state.pop(open_key, None)

    def _clear_other_explain_tabs() -> None:
        for k in all_open_keys:
            if k != open_key:
                st.session_state.pop(k, None)

    c1, _c2 = st.columns([1, 6])
    with c1:
        if st.button("Explain tab", key=f"{tab_key}_explain_btn"):
            _clear_other_explain_tabs()
            st.session_state[open_key] = True

    if st.session_state.get(open_key, False):
        if _HAS_ST_DIALOG:

            @st.dialog(title, on_dismiss=_clear_open)
            def _dlg() -> None:
                st.markdown(body_md)
                if st.button("Close", key=f"{tab_key}_explain_close"):
                    _clear_open()
                    st.rerun()

            _dlg()
        elif _HAS_ST_POPOVER:
            with st.popover("Explain tab (popover)", use_container_width=True):
                st.markdown(body_md)
                if st.button("Close", key=f"{tab_key}_explain_close_pop"):
                    _clear_open()
                    st.rerun()
        else:
            with st.expander("Explain tab", expanded=True):
                st.markdown(body_md)
                if st.button("Close", key=f"{tab_key}_explain_close_exp"):
                    _clear_open()
                    st.rerun()


def _fmt_metric(x: Any, *, nd: int = 2) -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return "—"
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def render_saved_run_metrics_sidebar(metrics: Dict[str, Any], run_path: Path) -> None:
    """Surface `metrics.json` headline (continuous CSV) vs daily-reset stats for the active saved run."""
    with st.sidebar.expander("Saved run: metrics.json", expanded=False):
        st.caption(f"Folder: `{run_path.name}`")
        meta = metrics.get("_meta") if isinstance(metrics.get("_meta"), dict) else {}
        headline = meta.get("headline_primary", "(legacy file: headline = top-level totals)")
        st.markdown(f"**Headline mode:** `{headline}`")

        valn = meta.get("validation") if isinstance(meta.get("validation"), dict) else {}
        if valn:
            ok = bool(valn.get("tariff_cost_identity_passed", False))
            st.caption(
                f"Tariff check: {'passed' if ok else 'failed'} "
                f"(max abs err = {_fmt_metric(valn.get('tariff_cost_identity_max_abs_err'), nd=6)} RM)"
            )

        defs = meta.get("definitions") if isinstance(meta.get("definitions"), dict) else {}
        if defs:
            with st.expander("What do continuous vs daily-reset mean?", expanded=False):
                st.markdown(str(defs.get("continuous_csv", "")))
                st.markdown(str(defs.get("daily_reset", "")))

        st.markdown("**Continuous CSV (headline totals)**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline RM", _fmt_metric(metrics.get("baseline_total_cost_rm")))
        c2.metric("Smart RM", _fmt_metric(metrics.get("smart_total_cost_rm")))
        c3.metric("Savings RM", _fmt_metric(metrics.get("total_savings_rm")))
        p1, p2, p3 = st.columns(3)
        p1.metric("Peak import Δ kWh", _fmt_metric(metrics.get("peak_import_reduction_kwh")))
        p2.metric("Baseline peak kWh", _fmt_metric(metrics.get("baseline_peak_import_kwh")))
        p3.metric("Smart peak kWh", _fmt_metric(metrics.get("smart_peak_import_kwh")))

        bym = metrics.get("by_evaluation_mode") if isinstance(metrics.get("by_evaluation_mode"), dict) else {}
        dr = bym.get("daily_reset") if isinstance(bym.get("daily_reset"), dict) else {}

        st.markdown("**Daily-reset ablation (independent days)**")
        d1, d2, d3 = st.columns(3)
        d1.metric("Mean daily savings RM", _fmt_metric(dr.get("daily_savings_mean_rm", metrics.get("daily_savings_mean_rm"))))
        d2.metric("Median daily savings RM", _fmt_metric(dr.get("daily_savings_median_rm", metrics.get("daily_savings_median_rm"))))
        d3.metric("Conservation days", str(int(dr.get("conservation_days_count", metrics.get("conservation_days_count", 0)) or 0)))

        if dr:
            st.caption(
                f"Sum of daily savings (daily-reset table): {_fmt_metric(dr.get('sum_daily_savings_rm'))} RM "
                f"(continuous headline savings − this sum = {_fmt_metric(meta.get('gap_continuous_total_savings_rm_minus_sum_daily_savings_rm'))} RM)."
            )
            if meta.get("gap_explainer"):
                st.caption(str(meta["gap_explainer"]))


# -----------------------------
# Config + data + model loading
# -----------------------------
@st.cache_data
def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _list_results_runs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    runs = [p for p in results_root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.name)
    return runs


@st.cache_data
def load_simulation_csvs(base_csv: str, smart_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_base = pd.read_csv(base_csv)
    df_base["datetime"] = pd.to_datetime(df_base["datetime"], errors="coerce")
    df_base = df_base.set_index("datetime").sort_index()

    df_smart = pd.read_csv(smart_csv)
    df_smart["datetime"] = pd.to_datetime(df_smart["datetime"], errors="coerce")
    df_smart = df_smart.set_index("datetime").sort_index()

    # Normalize to tz-naive timestamps (legacy tz-aware CSVs are normalized)
    if getattr(df_base.index, "tz", None) is not None:
        df_base.index = df_base.index.tz_localize(None)
    if getattr(df_smart.index, "tz", None) is not None:
        df_smart.index = df_smart.index.tz_localize(None)

    # Convenience features for UI and metrics
    for df in (df_base, df_smart):
        if "hour" not in df.columns:
            df["hour"] = df.index.hour
        if "month" not in df.columns:
            df["month"] = df.index.month
        if "year" not in df.columns:
            df["year"] = df.index.year

    return df_base, df_smart


@st.cache_resource
def try_load_forecast_model(model_path: str):
    try:
        import joblib  # type: ignore

        return joblib.load(model_path)
    except Exception:
        return None


CONFIG_PATH_DEFAULT = "experiment_config.json"
_repo = repo_root()
cfg = load_json(str(_repo / CONFIG_PATH_DEFAULT))

results_root = _repo / "results"

st.sidebar.header("Simulation Controls")

if "guided_help" not in st.session_state:
    st.session_state["guided_help"] = False

st.sidebar.toggle(
    "Show guided help (tooltips + help panels)",
    key="guided_help",
    help=HELP["guided_help_toggle"],
)

st.sidebar.caption("Full thesis-style notes also live in `PROJECT_EXPLANATION_DETAILED.md` (repo file).")

with st.sidebar.expander("Experiment Settings (from config)", expanded=False):
    st.caption(f"Config path: `{_repo / CONFIG_PATH_DEFAULT}`")
    if cfg:
        st.json(cfg)
    else:
        st.warning("Config not found or could not be read. Using built-in defaults.")

section_help(
    "What is this section? (Experiment settings)",
    """
- **What this is**: the exact JSON settings used for reproducible runs (`experiment_config.json`).
- **Why it matters**: this is what you cite in your thesis as “experimental parameters”.
- **Not the same as sliders**: sidebar sliders are for interactive what-if exploration unless you regenerate CSVs to match.
""",
)

use_results_run = st.sidebar.toggle(
    "View a saved results run (recommended)",
    value=False,
    help=HELP["results_run_toggle"],
)
selected_run: Optional[Path] = None
if use_results_run:
    runs = _list_results_runs(results_root)
    if runs:
        run_names = [p.name for p in runs]
        default_idx = max(0, len(run_names) - 1)
        pick = st.sidebar.selectbox(
            "Results folder",
            run_names,
            index=default_idx,
            help=HELP["results_run_select"],
        )
        selected_run = results_root / pick
        st.sidebar.caption("Tip: this makes the dashboard match `daily_results.csv` and `metrics.json` from that run.")
    else:
        st.sidebar.warning("No `results/` runs found. Run `python run_all.py` first.")
        use_results_run = False

paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
if selected_run:
    base_csv_path = str(selected_run / "baseline_simulation_results.csv")
    smart_csv_path = str(selected_run / "smart_simulation_results.csv")
else:
    base_csv_path = str(
        resolve_repo_path(_repo, paths_cfg.get("baseline_results_csv", "baseline_simulation_results.csv"))
    )
    smart_csv_path = str(
        resolve_repo_path(_repo, paths_cfg.get("smart_results_csv", "smart_simulation_results.csv"))
    )

df_base, df_smart = load_simulation_csvs(base_csv_path, smart_csv_path)

model_path = str(resolve_repo_path(_repo, paths_cfg.get("rf_model_path", "models/rf_forecast_model.joblib")))
rf_model = try_load_forecast_model(model_path)

# Optional saved evaluation artifacts (only present for run_all.py runs)
saved_daily_results: Optional[pd.DataFrame] = None
saved_metrics: Optional[dict] = None
if selected_run is not None:
    try:
        daily_csv = selected_run / "daily_results.csv"
        metrics_json = selected_run / "metrics.json"
        if daily_csv.exists():
            saved_daily_results = pd.read_csv(daily_csv)
        if metrics_json.exists():
            saved_metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    except Exception:
        saved_daily_results = None
        saved_metrics = None

if selected_run is not None and isinstance(saved_metrics, dict) and saved_metrics:
    render_saved_run_metrics_sidebar(saved_metrics, selected_run)


# -----------------------------
# Simulation helpers
# -----------------------------
@dataclass(frozen=True)
class SimParams:
    capacity_kwh: float
    pmax_kw: float
    soc_min: float
    efficiency_rt: float
    pv_scale: float
    load_scale: float
    e_crit_kwh: float
    hoard_start_hour: int
    hoard_end_hour: int
    essential_discharge_kw: float
    soc_reserve: float
    peak_start_hour: int
    peak_end_hour: int
    peak_rate_rm_per_kwh: float
    offpeak_rate_rm_per_kwh: float


def tariff_rate_custom(
    hour: int, peak_start: int, peak_end: int, peak_rate: float, offpeak_rate: float
) -> float:
    if peak_start <= hour < peak_end:
        return peak_rate
    return offpeak_rate


def compute_kpis(
    df: pd.DataFrame,
    *,
    load_col: str = "Load_kW",
    grid_import_col: str = "Grid_Import_kW",
    cost_col: str = "Hourly_Cost_RM",
    peak_start_hour: int = 7,
    peak_end_hour: int = 22,
) -> Dict[str, float]:
    def _series(col: str) -> pd.Series:
        obj = df[col]
        # If duplicate column names exist, pandas returns a DataFrame; pick the last one.
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, -1]
        return obj

    total_cost = float(_series(cost_col).sum())
    total_load = float(_series(load_col).sum())
    total_import = float(_series(grid_import_col).sum())
    ssr = 0.0 if total_load <= 0 else (1.0 - (total_import / total_load)) * 100.0
    peak_mask = df["hour"].between(peak_start_hour, peak_end_hour - 1)
    peak_import = float(_series(grid_import_col).loc[peak_mask].sum())
    return {
        "total_cost_rm": total_cost,
        "total_load_kwh": total_load,
        "total_import_kwh": total_import,
        "ssr_pct": ssr,
        "peak_import_kwh": peak_import,
    }


def run_controller_simulation(
    df_in: pd.DataFrame,
    params: SimParams,
    *,
    controller: str,
    conservation_mode_active: bool,
    tariff_fn: Callable[[int], float],
    soc_init: Optional[float] = None,
) -> pd.DataFrame:
    batt = Battery(
        capacity_kwh=params.capacity_kwh,
        max_charge_kw=params.pmax_kw,
        max_discharge_kw=params.pmax_kw,
        soc_min=params.soc_min,
        efficiency_rt=params.efficiency_rt,
    )
    if soc_init is not None:
        batt.soc = float(np.clip(soc_init, params.soc_min, 1.0))

    out = df_in.copy()

    soc_hist = []
    p_batt_hist = []
    grid_import_hist = []
    cost_hist = []
    mode_hist = []

    for dt, row in out.iterrows():
        hr = int(row["hour"])

        if controller == "smart":
            if conservation_mode_active:
                is_peak = params.peak_start_hour <= hr < params.peak_end_hour
                if (not is_peak) and (batt.soc <= params.soc_reserve):
                    batt.max_discharge_kw = params.essential_discharge_kw
                    mode_now = True
                else:
                    batt.max_discharge_kw = params.pmax_kw
                    mode_now = False
            else:
                batt.max_discharge_kw = params.pmax_kw
                mode_now = False
        else:
            batt.max_discharge_kw = params.pmax_kw
            mode_now = False

        p_load = float(row["Load_kW"]) * params.load_scale
        p_pv = float(row["Target_Power_kW"]) * params.pv_scale
        p_net = p_load - p_pv

        p_batt = batt.update_soc(p_request_kw=p_net)
        p_grid = p_net - p_batt

        grid_import_kw = max(0.0, p_grid)
        cost_rm = grid_import_kw * float(tariff_fn(hr))

        soc_hist.append(float(batt.soc))
        p_batt_hist.append(float(p_batt))
        grid_import_hist.append(float(grid_import_kw))
        cost_hist.append(float(cost_rm))
        mode_hist.append(bool(mode_now))

    out["Load_kW_sim"] = (out["Load_kW"].astype(float) * params.load_scale).astype(float)
    out["PV_kW_sim"] = (out["Target_Power_kW"].astype(float) * params.pv_scale).astype(float)
    out["SoC_sim"] = soc_hist
    out["Batt_Action_kW_sim"] = p_batt_hist
    out["Grid_Import_kW_sim"] = grid_import_hist
    out["Hourly_Cost_RM_sim"] = cost_hist
    out["Conservation_Mode_sim"] = mode_hist

    return out


def simulate_one_day_reset_soc(
    df_day: pd.DataFrame,
    params: SimParams,
    *,
    controller: str,
    conservation_mode_active: bool,
    tariff_fn: Callable[[int], float],
    soc_init: float,
) -> Dict[str, float]:
    batt = Battery(
        capacity_kwh=params.capacity_kwh,
        max_charge_kw=params.pmax_kw,
        max_discharge_kw=params.pmax_kw,
        soc_min=params.soc_min,
        efficiency_rt=params.efficiency_rt,
    )
    batt.soc = float(np.clip(soc_init, params.soc_min, 1.0))

    total_cost = 0.0
    peak_import = 0.0
    min_soc = batt.soc
    soc_at_peak_start: Optional[float] = None

    for _, row in df_day.iterrows():
        hr = int(row["hour"])

        if controller == "smart" and conservation_mode_active:
            is_peak = params.peak_start_hour <= hr < params.peak_end_hour
            if (not is_peak) and (batt.soc <= params.soc_reserve):
                batt.max_discharge_kw = params.essential_discharge_kw
            else:
                batt.max_discharge_kw = params.pmax_kw
        else:
            batt.max_discharge_kw = params.pmax_kw

        p_load = float(row["Load_kW"]) * params.load_scale
        p_pv = float(row["Target_Power_kW"]) * params.pv_scale
        p_net = p_load - p_pv

        p_batt = batt.update_soc(p_request_kw=p_net)
        p_grid = p_net - p_batt
        grid_import_kw = max(0.0, p_grid)

        total_cost += grid_import_kw * float(tariff_fn(hr))
        if params.peak_start_hour <= hr < params.peak_end_hour:
            peak_import += grid_import_kw

        if hr == params.peak_start_hour and soc_at_peak_start is None:
            soc_at_peak_start = float(batt.soc)

        min_soc = min(min_soc, float(batt.soc))

    return {
        "daily_cost_rm": float(total_cost),
        "daily_peak_import_kwh": float(peak_import),
        "soc_at_peak_start": float(soc_at_peak_start) if soc_at_peak_start is not None else float("nan"),
        "min_soc": float(min_soc),
    }


def decide_conservation_mode_for_day(
    df_day: pd.DataFrame, params: SimParams, *, df_smart_reference: Optional[pd.DataFrame]
) -> Tuple[bool, str, Optional[float]]:
    # 1) If we have a smart reference slice for that day, use its precomputed flag
    if df_smart_reference is not None and "Conservation_Mode_Active" in df_smart_reference.columns:
        try:
            flag = bool(df_smart_reference["Conservation_Mode_Active"].iloc[0])
            return flag, "precomputed_flag", None
        except Exception:
            pass

    # 2) Else, try to recompute from RF model if present + required features exist
    if rf_model is None:
        return False, "no_model_available", None

    features = ["ghiradiation", "temp", "humidity", "precip", "cloudcover", "hour", "month"]
    if not all(c in df_day.columns for c in features):
        return False, "missing_features_for_forecast", None

    try:
        preds = rf_model.predict(df_day[features])  # type: ignore[attr-defined]
        e_pred = float(np.sum(preds))
        return (e_pred < params.e_crit_kwh), "rf_forecast", e_pred
    except Exception:
        return False, "forecast_failed", None


def safe_r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = math.sqrt(mse)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom
    return r2, rmse


# -----------------------------
# Sidebar controls (shared)
# -----------------------------
st.sidebar.markdown(
    """
Use this simulator like a guided story:\n
1) Pick a **historical day** (real Sarawak weather).\n
2) Adjust **hardware** (battery size + inverter power).\n
3) Click **Run Live Simulation** to compare Baseline vs AI.\n
"""
)

section_help(
    "What is this section? (How to use the sidebar)",
    """
- **Saved run vs sliders**: selecting a `results/...` folder loads saved CSV outputs. Sidebar sliders then mainly affect **interactive re-simulations** (Tab 2 / Tab 3 daily reset), not the saved curves unless you re-run the pipeline.
- **What-if sliders**: PV/load scaling are hypothetical scenarios (not measured smart-meter truth).
""",
)

available_dates = df_smart.index.normalize().unique()
min_date = available_dates.min().date()
max_date = available_dates.max().date()

selected_date = st.sidebar.date_input(
    "Choose a historical date",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
    help=HELP["date_picker"],
)

st.sidebar.subheader("Hardware (physical meaning)")
st.sidebar.caption("kWh = battery tank size. kW = pipe size (how fast energy can flow).")
batt_cfg = cfg.get("battery", {}) if isinstance(cfg, dict) else {}
default_capacity_kwh = float(batt_cfg.get("capacity_kwh", 10.0))
default_pmax_kw = float(batt_cfg.get("inverter_power_kw", 5.0))
default_soc_min_pct = float(batt_cfg.get("soc_min", 0.15)) * 100.0

capacity_kwh = float(
    st.sidebar.slider(
        "Battery capacity (kWh)",
        5.0,
        20.0,
        default_capacity_kwh,
        0.5,
        help=HELP["battery_capacity"],
    )
)
pmax_kw = float(
    st.sidebar.slider(
        "Inverter max power (kW)",
        1.0,
        10.0,
        default_pmax_kw,
        0.5,
        help=HELP["inverter_power"],
    )
)
soc_min_pct = float(
    st.sidebar.slider(
        "Minimum SoC (%)",
        5.0,
        30.0,
        default_soc_min_pct,
        1.0,
        help=HELP["soc_min"],
    )
)

section_help(
    "What is this section? (Hardware)",
    """
- **kWh vs kW**: kWh is how big the storage “tank” is; kW is how fast you can fill/empty it.
- **Min SoC**: a safety floor; raising it reduces usable energy and can change costs a lot.
""",
)

st.sidebar.subheader("Scenario scaling (what the house experiences)")
st.sidebar.caption("Use scaling to test a bigger/smaller PV system or heavier/lighter household demand.")
scenario_cfg = cfg.get("scenario", {}) if isinstance(cfg, dict) else {}
default_pv_scale = float(scenario_cfg.get("pv_scale", 1.0))
default_load_scale = float(scenario_cfg.get("load_scale", 1.0))

pv_scale = float(
    st.sidebar.slider(
        "PV scaling (×)",
        0.5,
        2.0,
        default_pv_scale,
        0.05,
        help=HELP["pv_scale"],
    )
)
load_scale = float(
    st.sidebar.slider(
        "Load scaling (×)",
        0.5,
        2.0,
        default_load_scale,
        0.05,
        help=HELP["load_scale"],
    )
)

section_help(
    "What is this section? (Scenario scaling)",
    """
- **PV scaling**: pretend your solar system produces more/less than the modeled `Target_Power_kW` timeline.
- **Load scaling**: pretend the household uses more/less than the synthetic `Load_kW` timeline.
""",
)

st.sidebar.subheader("Tariff (default C3)")
st.sidebar.caption("Peak hours cost more. Savings come mainly from reducing grid imports during peak.")
use_custom_tariff = st.sidebar.toggle(
    "Use custom tariff rates",
    value=False,
    help=HELP["custom_tariff_toggle"],
)
tariff_cfg = cfg.get("tariff", {}) if isinstance(cfg, dict) else {}
default_peak_start = int(tariff_cfg.get("peak_start_hour", 7))
default_peak_end = int(tariff_cfg.get("peak_end_hour", 22))
default_peak_rate = float(tariff_cfg.get("peak_rate_rm_per_kwh", 0.245))
default_offpeak_rate = float(tariff_cfg.get("offpeak_rate_rm_per_kwh", 0.139))
if use_custom_tariff:
    peak_start_hour = int(
        st.sidebar.slider(
            "Peak start hour",
            0,
            23,
            default_peak_start,
            help=HELP["peak_start"],
        )
    )
    peak_end_hour = int(
        st.sidebar.slider(
            "Peak end hour",
            1,
            24,
            default_peak_end,
            help=HELP["peak_end"],
        )
    )
    peak_rate = float(
        st.sidebar.number_input(
            "Peak rate (RM/kWh)",
            min_value=0.0,
            value=default_peak_rate,
            step=0.001,
            help=HELP["peak_rate"],
        )
    )
    offpeak_rate = float(
        st.sidebar.number_input(
            "Off-peak rate (RM/kWh)",
            min_value=0.0,
            value=default_offpeak_rate,
            step=0.001,
            help=HELP["offpeak_rate"],
        )
    )
    tariff_fn = lambda hr: tariff_rate_custom(hr, peak_start_hour, peak_end_hour, peak_rate, offpeak_rate)
else:
    tariff_fn = get_tariff_rate
    peak_start_hour, peak_end_hour = default_peak_start, default_peak_end
    peak_rate, offpeak_rate = default_peak_rate, default_offpeak_rate

section_help(
    "What is this section? (Tariff)",
    """
- **Peak vs off-peak**: the app charges grid imports using a higher price during peak hours.
- **Why savings track peak imports**: reducing purchases during expensive hours is usually how money is saved.
""",
)

st.sidebar.subheader("AI policy parameters")
st.sidebar.caption("The AI decides whether to hoard energy overnight on low-solar days.")
controller_cfg = cfg.get("controller", {}) if isinstance(cfg, dict) else {}
default_e_crit = float(controller_cfg.get("e_crit_kwh", 18.0))
default_soc_reserve_pct = float(controller_cfg.get("soc_reserve", 0.30)) * 100.0
default_essential_discharge = float(controller_cfg.get("essential_discharge_kw", 0.5))

e_crit = float(
    st.sidebar.slider(
        "Critical daily PV energy threshold E_crit (kWh)",
        0.0,
        30.0,
        default_e_crit,
        0.5,
        help=HELP["e_crit"],
    )
)
hoard_start = int(
    st.sidebar.slider(
        "Hoarding window start (hour)",
        0,
        23,
        0,
        help=HELP["hoard_start"],
    )
)
hoard_end = int(
    st.sidebar.slider(
        "Hoarding window end (hour)",
        0,
        23,
        6,
        help=HELP["hoard_end"],
    )
)
soc_reserve_pct = float(
    st.sidebar.slider(
        "SoC reserve target for peak hours (%)",
        20.0,
        80.0,
        default_soc_reserve_pct,
        1.0,
        help=HELP["soc_reserve"],
    )
)
essential_discharge_kw = float(
    st.sidebar.slider(
        "Essential discharge during hoarding (kW)",
        0.0,
        2.0,
        default_essential_discharge,
        0.1,
        help=HELP["essential_discharge"],
    )
)

section_help(
    "What is this section? (AI policy)",
    """
- **E_crit**: how strict the definition of a “low solar day” is.
- **Reserve SoC / essential discharge**: controls how strongly the AI avoids draining the battery before peak hours.
- **Hoarding window sliders**: mainly shade a region in Tab 2 for teaching; the live smart policy uses **off-peak vs peak** + **reserve SoC** (not those hoarding hours).
""",
)

params = SimParams(
    capacity_kwh=capacity_kwh,
    pmax_kw=pmax_kw,
    soc_min=soc_min_pct / 100.0,
    efficiency_rt=0.95,
    pv_scale=pv_scale,
    load_scale=load_scale,
    e_crit_kwh=e_crit,
    hoard_start_hour=min(hoard_start, hoard_end),
    hoard_end_hour=max(hoard_start, hoard_end),
    essential_discharge_kw=essential_discharge_kw,
    soc_reserve=soc_reserve_pct / 100.0,
    peak_start_hour=peak_start_hour,
    peak_end_hour=peak_end_hour,
    peak_rate_rm_per_kwh=peak_rate,
    offpeak_rate_rm_per_kwh=offpeak_rate,
)

with st.sidebar.expander("Quick glossary (tap to open)"):
    st.markdown(
        """
- **kW**: power right now (rate).\n
- **kWh**: energy over time (storage).\n
- **SoC**: state of charge (battery level).\n
- **Grid import**: buying electricity from the grid (this is what costs money).\n
"""
    )

with st.sidebar.expander("What is Conservation Mode? (tap to open)"):
    st.markdown(
        f"""
If the forecast says tomorrow’s solar will be low (**E_pred < E_crit**), the controller activates **Conservation Mode**.\n
On Conservation days, the controller preserves a **SoC reserve** for the peak tariff window.\n
- If battery SoC is at/below **{params.soc_reserve*100:.0f}%** during off-peak, discharge is limited to **{params.essential_discharge_kw:.1f} kW** (critical loads only).\n
- During peak hours, discharge is allowed normally to reduce expensive grid imports.\n
"""
    )

with st.sidebar.expander("Assumptions (tap to open)"):
    st.markdown(
        """
- **Costs use grid imports only** (export/curtailment has zero monetary compensation)\n
- **Battery starts at the minimum SoC floor** (default 15%)\n
- **Household load is synthetic** (target ~15–20 kWh/day)\n
- **Timestamps are tz-naive hourly timestamps** (timezone not stored in CSV)\n
"""
    )

# Sidebar: show AI decision preview for the selected day (if possible)
sel_day = pd.Timestamp(selected_date)
base_day_preview = df_base.loc[df_base.index.normalize() == sel_day]
smart_day_preview = df_smart.loc[df_smart.index.normalize() == sel_day] if not df_smart.empty else None
if not base_day_preview.empty:
    cons_preview, src_preview, e_pred_preview = decide_conservation_mode_for_day(
        base_day_preview, params, df_smart_reference=smart_day_preview
    )
    st.sidebar.subheader("AI decision for this day")
    st.sidebar.write("**Conservation Mode**" if cons_preview else "**Standard Mode**")
    st.sidebar.caption(f"Decision source: `{src_preview}`")
    if e_pred_preview is not None:
        st.sidebar.caption(f"E_pred = {e_pred_preview:.1f} kWh vs E_crit = {params.e_crit_kwh:.1f} kWh")


# -----------------------------
# Header
# -----------------------------
st.title(
    "AI-Predicted Solar Generation for Optimised Rule-Based Scheduling of Hybrid Solar Battery Systems: A Sarawak Simulation"
)
st.caption("Digital Twin: PV → Load → Battery → Grid (Sarawak ToU tariff).")


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Tab 1: How the House Works", "Tab 2: Live Simulation", "Tab 3: Yearly Comparative View", "Tab 4: Model Metrics"]
)


# -----------------------------
# Tab 1: Explain flow
# -----------------------------
with tab1:
    explain_tab_header(
        "tab1",
        "Explain Tab 1",
        """
### What this tab is for
A **concept map** of the digital twin: solar, load, battery, grid, and tariff.

### Honesty / common confusion (read this once)
- **Saved `results/...` vs sidebar sliders**: loading a saved run swaps in saved CSV timelines. Sidebar sliders mostly drive **interactive re-simulations** (Tab 2 / Tab 3 daily-reset KPIs), not the saved curves unless you regenerate CSVs to match.
- **“Hoarding window” shading**: the shaded hours in Tab 2 are a **teaching overlay** tied to sidebar hoarding sliders. The smart controller logic in this app is driven by **peak vs off-peak** + **reserve SoC** (see sidebar AI policy help tooltips).

### How to narrate it (examiner-friendly)
1) Start with net demand \\(P_{net}=P_{load}-P_{pv}\\).\n
2) Explain the battery as a **buffer** that moves energy across hours.\n
3) End with tariff: money is mostly about **when** you buy from the grid.\n
""",
    )

    st.subheader("How to explain this project in 60 seconds")
    st.markdown(
        """
**Problem:** Solar energy changes with clouds/monsoon, and electricity is more expensive during peak tariff hours.\n
**Goal:** Use the battery at the right time to reduce costly grid imports.\n

### The one equation that drives everything
Every hour we compute net demand:\n
\\[\nP_{net}(t) = P_{load}(t) - P_{pv}(t)\n\\]\n
- If \\(P_{net}(t) > 0\\): the house needs extra power (**deficit**) → battery tries to discharge → grid supplies the rest.\n
- If \\(P_{net}(t) < 0\\): there is extra solar (**surplus**) → battery tries to charge.\n

### What is Baseline vs AI?
- **Baseline (Self‑Consumption):** reacts only to the current hour.\n
- **AI Controller:** at midnight, forecasts tomorrow’s solar. If tomorrow looks low‑solar, it activates **Conservation Mode** to protect energy for expensive hours.\n
"""
    )

    section_help(
        "Guided help: how to use this tab",
        """
- **Keep it physical**: kW is a *rate*, kWh is *stored energy*, SoC is “how full the tank is”.
- **Don’t over-claim the forecast**: the forecast mainly decides **whether** Conservation Mode turns on; dispatch still follows rule-based battery physics.
""",
    )

    st.subheader("Flow diagram")
    st.graphviz_chart(
        """
digraph flow {
  rankdir=LR;
  PV [label="PV generation (P_pv)"];
  Load [label="House load (P_load)"];
  Net [label="Net power (P_net = P_load - P_pv)"];
  BatteryNode [label="Battery digital twin"];
  Grid [label="Grid import/export (P_grid)"];
  Cost [label="Hourly cost (Tariff)"];

  PV -> Net;
  Load -> Net;
  Net -> BatteryNode;
  BatteryNode -> Grid;
  Grid -> Cost;
}
"""
    )

    section_help(
        "Guided help: reading the flow diagram",
        """
- **PV → Net**: solar reduces what the house must source from the battery/grid.\n
- **Battery → Grid**: if the battery can’t cover a deficit, the grid fills the gap.\n
- **Grid → Cost**: imports are priced by the tariff clock (peak vs off-peak).\n
""",
    )

    st.subheader("Baseline vs AI (in plain words)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**Baseline (Self‑Consumption)**\n
- Uses the battery whenever there is a deficit, even during the night\n
- Has no awareness of tomorrow’s weather\n
"""
        )
    with c2:
        st.markdown(
            f"""
**AI Controller (Predict + Rule Override)**\n
- At midnight, predicts tomorrow’s total PV energy \\(E_{{pred}}\\)\n
- If \\(E_{{pred}} < E_{{crit}}\\), activates **Conservation Mode**\n
- On conservation days, during **off-peak** hours, if SoC is near the **reserve** target, discharge is limited so more energy remains for **peak** hours\n
- The shaded **{params.hoard_start_hour:02d}:00–{params.hoard_end_hour:02d}:00** region in Tab 2 is a **visual aid** (not an extra policy layer)\n
"""
        )

    section_help(
        "Guided help: Baseline vs AI wording",
        """
- **Baseline** is greedy: it uses the battery whenever there is a deficit, without a “tomorrow plan”.\n
- **AI** is cautious on forecasted low-solar days: it tries to keep more energy available before peak pricing.\n
""",
    )

    with st.expander("Glossary (tap to open)"):
        st.markdown(
            """
- **kW**: power right now (rate)\n
- **kWh**: energy over time (storage)\n
- **SoC**: state of charge (battery level, %)\n
- **Grid import**: electricity purchased from the grid (cost)\n
- **Peak tariff**: time window where RM/kWh is higher\n
"""
        )


# -----------------------------
# Tab 2: Live simulation
# -----------------------------
with tab2:
    explain_tab_header(
        "tab2",
        "Explain Tab 2",
        """
### What this tab does
Runs a **one-day, interactive** baseline vs AI simulation for the sidebar-selected date using your **current** hardware/tariff/AI settings.

### Honesty / interpretation
- **This is not automatically the saved `results/...` story**: saved CSVs are loaded for reference, but Tab 2 recomputes dispatch when you click **Run**.\n
- **Conservation Mode flag for the day** is derived from the forecast + \\(E_{crit}\\) logic (see metrics above the button).\n
- **Hoarding shading** on the SoC chart is a **visual aid** aligned to sidebar hoarding hours; the smart policy still uses **off-peak vs peak** + **reserve SoC**.\n
""",
    )

    st.subheader("Interactive day simulation (Baseline vs AI)")
    st.markdown(
        """
**What happens when you click Run?**\n
1) Read PV and Load for the selected day\n
2) Compute net demand \\(P_{net}(t)\\)\n
3) Battery responds within limits (capacity, max power, min SoC, efficiency)\n
4) Grid supplies the remaining deficit (grid import)\n
5) Cost is computed using the tariff (imports only)\n
"""
    )

    section_help(
        "Guided help: Tab 2 workflow",
        """
- **Pick a day** in the sidebar date picker.\n
- **Adjust sliders** to explore what-if hardware/tariff/AI settings.\n
- Click **Run** to generate fresh curves for that day (nothing is written to `results/`).\n
""",
    )

    sel_day = pd.Timestamp(selected_date)
    base_day = df_base.loc[df_base.index.normalize() == sel_day]
    if base_day.empty:
        st.error("Selected date not found in dataset.")
    else:
        smart_day_ref = None
        try:
            smart_day_ref = df_smart.loc[df_smart.index.normalize() == sel_day].copy()
        except Exception:
            smart_day_ref = None

        conservation_mode, mode_source, e_pred = decide_conservation_mode_for_day(
            base_day, params, df_smart_reference=smart_day_ref
        )

        default_live_soc = float(scenario_cfg.get("initial_soc_live", 0.50)) * 100.0
        soc_live_init_pct = float(
            st.slider(
                "Initial SoC for Live Simulation (%)",
                15.0,
                80.0,
                default_live_soc,
                1.0,
                help=HELP["tab2_initial_soc"],
            )
        )
        st.caption("To match Tab 3 daily-reset values, set this equal to Tab 3 initial SoC.")
        soc_live_init = soc_live_init_pct / 100.0

        top = st.columns(4)
        top[0].metric("AI mode for this day", "Conservation" if conservation_mode else "Standard")
        top[1].metric("Decision source", mode_source)
        top[2].metric("E_crit (kWh)", f"{params.e_crit_kwh:.1f}")
        top[3].metric("E_pred (kWh)", "—" if e_pred is None else f"{e_pred:.1f}")

        if conservation_mode:
            st.info(
                "Why AI helps today: Forecast indicates low solar, so Conservation Mode is active. "
                "The smart controller limits off-peak discharge when SoC is near the reserve target so more energy remains for peak-priced hours."
            )
        else:
            st.info(
                "Why AI looks similar today: Forecast indicates sufficient solar. Conservation Mode is not needed, so behavior is close to baseline."
            )

        if st.button("Run Live Simulation", type="primary", help=HELP["tab2_run_button"]):
            with st.spinner("Simulating baseline and AI controllers..."):
                sim_base = run_controller_simulation(
                    base_day,
                    params,
                    controller="baseline",
                    conservation_mode_active=False,
                    tariff_fn=tariff_fn,
                    soc_init=soc_live_init,
                )
                sim_smart = run_controller_simulation(
                    base_day,
                    params,
                    controller="smart",
                    conservation_mode_active=conservation_mode,
                    tariff_fn=tariff_fn,
                    soc_init=soc_live_init,
                )

            kpi_b = compute_kpis(
                sim_base,
                load_col="Load_kW_sim",
                grid_import_col="Grid_Import_kW_sim",
                cost_col="Hourly_Cost_RM_sim",
                peak_start_hour=params.peak_start_hour,
                peak_end_hour=params.peak_end_hour,
            )
            kpi_s = compute_kpis(
                sim_smart,
                load_col="Load_kW_sim",
                grid_import_col="Grid_Import_kW_sim",
                cost_col="Hourly_Cost_RM_sim",
                peak_start_hour=params.peak_start_hour,
                peak_end_hour=params.peak_end_hour,
            )

            savings = kpi_b["total_cost_rm"] - kpi_s["total_cost_rm"]
            peak_import_reduction = kpi_b["peak_import_kwh"] - kpi_s["peak_import_kwh"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Baseline cost (RM/day)", f"{kpi_b['total_cost_rm']:.2f}")
            m2.metric("AI cost (RM/day)", f"{kpi_s['total_cost_rm']:.2f}", delta=f"{-savings:.2f}", delta_color="inverse")
            m3.metric("Savings (RM/day)", f"{savings:.2f}")
            m4.metric("Peak import reduction (kWh)", f"{peak_import_reduction:.2f}")

            st.caption("Peak import reduction focuses on the expensive window, which is usually where AI provides most savings.")

            # Safety/operational metrics (SoC behavior)
            min_soc_base = float(sim_base["SoC_sim"].min()) * 100.0
            min_soc_ai = float(sim_smart["SoC_sim"].min()) * 100.0
            at_floor_base = int((sim_base["SoC_sim"] <= (params.soc_min + 1e-9)).sum())
            at_floor_ai = int((sim_smart["SoC_sim"] <= (params.soc_min + 1e-9)).sum())
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Min SoC (Baseline)", f"{min_soc_base:.1f}%")
            s2.metric("Min SoC (AI)", f"{min_soc_ai:.1f}%")
            s3.metric("Hours at SoC floor (Baseline)", f"{at_floor_base}")
            s4.metric("Hours at SoC floor (AI)", f"{at_floor_ai}")

            h = sim_base.index.hour

            st.divider()
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**Load vs PV (scaled): what the house needs vs what solar provides**")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(h, sim_base["Load_kW_sim"], label="Load", color="orange", linewidth=2)
                ax.plot(h, sim_base["PV_kW_sim"], label="PV", color="green", linewidth=2)
                ax.axvspan(params.peak_start_hour, params.peak_end_hour, alpha=0.1, label="Peak tariff window")
                ax.set_xlabel("Hour")
                ax.set_ylabel("kW")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                chart_help(
                    "Caption: orange load vs green PV (kW). Shaded band is the peak tariff window.",
                    "How to read: Load vs PV",
                    """
- **Gap between orange and green** is the hour-by-hour energy balance the battery/grid must cover.\n
- **Peak shading** reminds you which hours are priced higher.\n
""",
                )

            with r2:
                st.markdown("**Battery level (SoC): AI should stay higher overnight on Conservation days**")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(h, sim_base["SoC_sim"] * 100, label="Baseline SoC", color="gray", linestyle="--")
                ax.plot(h, sim_smart["SoC_sim"] * 100, label="AI SoC", color="blue", linewidth=2)
                ax.axhline(params.soc_min * 100, color="red", linestyle=":", label="SoC min")
                if conservation_mode:
                    ax.axvspan(params.hoard_start_hour, params.hoard_end_hour, alpha=0.1, label="Hoarding window")
                ax.set_xlabel("Hour")
                ax.set_ylabel("SoC (%)")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right")
                st.pyplot(fig)
                chart_help(
                    "Caption: SoC (%) through the day; dashed baseline vs solid AI.",
                    "How to read: SoC trajectories",
                    """
- On **conservation days**, you often want AI SoC **higher before peak** than baseline.\n
- The **hoarding window** shading is a teaching overlay; the controller logic is driven by **peak vs off-peak** + **reserve SoC**.\n
""",
                )

            r3, r4 = st.columns(2)
            with r3:
                st.markdown("**Grid import: lower during peak hours = lower cost**")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(h, sim_base["Grid_Import_kW_sim"], label="Baseline import", color="gray", linestyle="--")
                ax.plot(h, sim_smart["Grid_Import_kW_sim"], label="AI import", color="purple", linewidth=2)
                ax.axvspan(params.peak_start_hour, params.peak_end_hour, alpha=0.1)
                ax.set_xlabel("Hour")
                ax.set_ylabel("kW")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                chart_help(
                    "Caption: grid import power (kW). Lower purple line in peak hours usually means lower bills.",
                    "How to read: Grid import",
                    """
- Imports happen when **load − PV − battery** is still positive.\n
- AI savings often show up as **lower imports specifically in the shaded peak window**.\n
""",
                )

            with r4:
                st.markdown("**Hourly cost: compare what you pay each hour**")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(h, sim_base["Hourly_Cost_RM_sim"], label="Baseline cost", color="gray", linestyle="--")
                ax.plot(h, sim_smart["Hourly_Cost_RM_sim"], label="AI cost", color="red", linewidth=2)
                ax.axvspan(params.peak_start_hour, params.peak_end_hour, alpha=0.1)
                ax.set_xlabel("Hour")
                ax.set_ylabel("RM")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                chart_help(
                    "Caption: hourly RM paid for grid imports (imports × tariff).",
                    "How to read: Hourly cost",
                    """
- Cost spikes when **imports** coincide with **peak pricing**.\n
- The day total is the **sum** of these hourly bars (2 d.p. metrics can hide tiny differences).\n
""",
                )

            st.divider()
            st.markdown("**Simulation table (first 24 rows)**")
            view = pd.DataFrame(
                {
                    "hour": h,
                    "load_kw": sim_base["Load_kW_sim"].values,
                    "pv_kw": sim_base["PV_kW_sim"].values,
                    "soc_baseline": sim_base["SoC_sim"].values,
                    "soc_ai": sim_smart["SoC_sim"].values,
                    "grid_import_baseline": sim_base["Grid_Import_kW_sim"].values,
                    "grid_import_ai": sim_smart["Grid_Import_kW_sim"].values,
                    "cost_baseline": sim_base["Hourly_Cost_RM_sim"].values,
                    "cost_ai": sim_smart["Hourly_Cost_RM_sim"].values,
                }
            )
            st.dataframe(view)


# -----------------------------
# Tab 3: Yearly comparative view
# -----------------------------
with tab3:
    explain_tab_header(
        "tab3",
        "Explain Tab 3",
        """
### What this tab does
Shows **year-scale** baseline vs AI behavior using the loaded CSV timelines, plus optional **daily-reset recomputation** for headline KPIs.

### Honesty / common confusion (important)
- **Saved `results/...` vs sidebar sliders**: the plots are built from the **saved hourly series** in the loaded CSVs. If you change sliders, you are *not* automatically rewriting those CSV files.\n
- **Daily-reset KPI mismatch**: when **Use daily reset evaluation** is ON, the **top cost/savings/peak KPIs** are recomputed by simulating **each day independently** from the selected initial SoC. The **Cost/Grid import time-series plots** still come from the **saved hourly trajectories**, so they may **diverge** from the KPI totals unless your sliders match the saved run and you understand what each view is measuring.\n
""",
    )

    st.subheader("Yearly comparison (Baseline vs AI)")
    st.markdown(
        """
This tab answers three questions:\n
1) **How much money is saved** over the year?\n
2) **When** do savings happen most (e.g., monsoon months)?\n
3) **What changed operationally** (peak imports reduced, conservation days)?\n
"""
    )

    section_help(
        "Guided help: Tab 3 mental model",
        """
- **Plots** = “what the saved hourly CSV says happened” (after aggregation).\n
- **Daily-reset KPIs** = “what would happen if each day restarted at the chosen SoC, using *current* sidebar physics/tariff but *saved* conservation flags per day”.\n
""",
    )

    years = sorted(set(df_base["year"].unique()).intersection(set(df_smart["year"].unique())))
    if not years:
        st.warning("No overlapping years found between baseline and smart datasets.")
    else:
        year = int(st.selectbox("Select year", years, index=len(years) - 1, help=HELP["tab3_year"]))
        agg = st.radio(
            "Aggregation",
            ["Hourly", "Daily", "Monthly"],
            horizontal=True,
            help=HELP["tab3_agg"],
        )
        conservation_only = st.toggle(
            "Show Conservation days only",
            value=False,
            help=HELP["tab3_conservation_only"],
        )
        use_daily_reset = st.toggle(
            "Use daily reset evaluation (recommended)",
            value=True,
            help=HELP["tab3_daily_reset"],
        )
        default_daily_soc = float(scenario_cfg.get("initial_soc_daily_reset", 0.50)) * 100.0
        soc_init_pct = float(
            st.slider(
                "Initial SoC for daily reset (%)",
                15.0,
                80.0,
                default_daily_soc,
                1.0,
                help=HELP["tab3_daily_reset_soc"],
            )
        )
        soc_init = soc_init_pct / 100.0

        st.info(
            "**Read me:** Tab 3 mixes two data sources. "
            "The **time-series plots** follow the **saved hourly CSV** (optionally aggregated). "
            "When **daily reset** is ON, the **KPI row** is recomputed from per-day simulations using your **current sidebar** battery/tariff settings and the **saved** conservation-day flags. "
            + (
                "When a **saved `results/...` run** is loaded, the official headline totals and the daily-reset *ablation* stats live in that folder’s `metrics.json` (see sidebar expander **Saved run: metrics.json**)."
                if saved_metrics is not None
                else ""
            )
        )
        if saved_metrics is not None:
            smeta = saved_metrics.get("_meta") if isinstance(saved_metrics.get("_meta"), dict) else {}
            st.caption(
                f"Loaded `metrics.json` declares headline mode `{smeta.get('headline_primary', '(legacy)')}`. "
                "The KPI row in this tab still follows the interactive rules above unless your sliders match the saved config snapshot."
            )

        base_y = df_base[df_base["year"] == year].copy()
        smart_y = df_smart[df_smart["year"] == year].copy()

        daily_results: Optional[pd.DataFrame] = None
        if use_daily_reset:
            # Build daily table by simulating each day independently with same initial SoC.
            dates = pd.Index(sorted(base_y.index.normalize().unique()))
            rows = []
            for d in dates:
                base_day = base_y.loc[base_y.index.normalize() == d]
                smart_day_ref = smart_y.loc[smart_y.index.normalize() == d]
                if base_day.empty or smart_day_ref.empty:
                    continue

                cons_day = bool(smart_day_ref["Conservation_Mode_Active"].resample("1D").max().iloc[0]) if "Conservation_Mode_Active" in smart_day_ref.columns else False

                base_metrics = simulate_one_day_reset_soc(
                    base_day,
                    params,
                    controller="baseline",
                    conservation_mode_active=False,
                    tariff_fn=tariff_fn,
                    soc_init=soc_init,
                )
                ai_metrics = simulate_one_day_reset_soc(
                    base_day,
                    params,
                    controller="smart",
                    conservation_mode_active=cons_day,
                    tariff_fn=tariff_fn,
                    soc_init=soc_init,
                )

                rows.append(
                    {
                        "date": pd.Timestamp(d).date(),
                        "conservation_day": cons_day,
                        "baseline_cost_rm": base_metrics["daily_cost_rm"],
                        "ai_cost_rm": ai_metrics["daily_cost_rm"],
                        "savings_rm": base_metrics["daily_cost_rm"] - ai_metrics["daily_cost_rm"],
                        "peak_import_reduction_kwh": base_metrics["daily_peak_import_kwh"] - ai_metrics["daily_peak_import_kwh"],
                        "soc_at_peak_baseline_pct": base_metrics["soc_at_peak_start"] * 100.0,
                        "soc_at_peak_ai_pct": ai_metrics["soc_at_peak_start"] * 100.0,
                        "min_soc_baseline_pct": base_metrics["min_soc"] * 100.0,
                        "min_soc_ai_pct": ai_metrics["min_soc"] * 100.0,
                    }
                )

            daily_results = pd.DataFrame(rows)
            if conservation_only and (daily_results is not None):
                daily_results = daily_results.loc[daily_results["conservation_day"] == True]  # noqa: E712
        else:
            if conservation_only and ("Conservation_Mode_Active" in smart_y.columns):
                cons_days = smart_y["Conservation_Mode_Active"].resample("1D").max()
                cons_dates = cons_days[cons_days].index.normalize()
                base_y = base_y[base_y.index.normalize().isin(cons_dates)]
                smart_y = smart_y[smart_y.index.normalize().isin(cons_dates)]

        if agg == "Hourly":
            base_series = base_y
            smart_series = smart_y
        elif agg == "Daily":
            base_series = base_y.resample("1D").sum(numeric_only=True)
            smart_series = smart_y.resample("1D").sum(numeric_only=True)
        else:
            base_series = base_y.resample("1M").sum(numeric_only=True)
            smart_series = smart_y.resample("1M").sum(numeric_only=True)

        # KPIs (year totals)
        def kpi_from(df: pd.DataFrame) -> Dict[str, float]:
            total_cost = float(df["Hourly_Cost_RM"].sum()) if "Hourly_Cost_RM" in df.columns else float("nan")
            total_import = float(df["Grid_Import_kW"].sum()) if "Grid_Import_kW" in df.columns else float("nan")
            total_load = float(df["Load_kW"].sum()) if "Load_kW" in df.columns else float("nan")
            ssr = 0.0 if total_load <= 0 else (1.0 - total_import / total_load) * 100.0
            peak_mask = df["hour"].between(params.peak_start_hour, params.peak_end_hour - 1) if "hour" in df.columns else None
            peak_import = float(df.loc[peak_mask, "Grid_Import_kW"].sum()) if peak_mask is not None else float("nan")
            return {"cost": total_cost, "import": total_import, "ssr": ssr, "peak_import": peak_import}

        if use_daily_reset and daily_results is not None and (not daily_results.empty):
            kb = {
                "cost": float(daily_results["baseline_cost_rm"].sum()),
                "ssr": float("nan"),
                "peak_import": float("nan"),
            }
            ks = {
                "cost": float(daily_results["ai_cost_rm"].sum()),
                "ssr": float("nan"),
                "peak_import": float("nan"),
            }
            savings = float(daily_results["savings_rm"].sum())
            peak_reduction = float(daily_results["peak_import_reduction_kwh"].sum())
        else:
            kb = kpi_from(base_y)
            ks = kpi_from(smart_y)
            savings = kb["cost"] - ks["cost"]
            peak_reduction = kb["peak_import"] - ks["peak_import"]
        cols = st.columns(5)
        cols[0].metric("Baseline cost (RM)", f"{kb['cost']:.2f}")
        cols[1].metric("AI cost (RM)", f"{ks['cost']:.2f}", delta=f"{-savings:.2f}", delta_color="inverse")
        cols[2].metric("Savings (RM)", f"{savings:.2f}")
        cols[3].metric("SSR Δ (pp)", "—" if use_daily_reset else f"{(ks['ssr'] - kb['ssr']):.1f}")
        cols[4].metric("Peak import reduction (kWh)", f"{peak_reduction:.2f}")
        st.caption("Values use 2 decimals to avoid rounding small savings to zero.")
        if use_daily_reset:
            st.caption(
                "If daily reset is ON: treat the KPI row as the **daily-reset recomputation** result; "
                "treat the plots below as **saved-run timelines** unless you regenerated matching CSVs."
            )

        # Safety: how low SoC went (year) and how often it hit the floor
        if "SoC" in base_y.columns and "SoC" in smart_y.columns:
            min_base_soc = float(base_y["SoC"].min()) * 100.0
            min_smart_soc = float(smart_y["SoC"].min()) * 100.0
            floor_base = int((base_y["SoC"] <= (0.15 + 1e-9)).sum())
            floor_smart = int((smart_y["SoC"] <= (0.15 + 1e-9)).sum())
            s_cols = st.columns(4)
            s_cols[0].metric("Min SoC (Baseline)", f"{min_base_soc:.1f}%")
            s_cols[1].metric("Min SoC (AI)", f"{min_smart_soc:.1f}%")
            s_cols[2].metric("Hours at SoC floor (Baseline)", f"{floor_base}")
            s_cols[3].metric("Hours at SoC floor (AI)", f"{floor_smart}")

        section_help(
            "Guided help: KPI row vs plots",
            """
- **KPI totals** answer “how much RM / kWh (peak) changed” under the selected evaluation mode.\n
- **Plots** answer “what did the saved hourly series look like across the year” after aggregation.\n
- If those disagree, first check **daily reset ON/OFF** and whether your **sliders match** the saved run.\n
""",
        )

        st.divider()
        st.markdown("**Cost over time**")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(base_series.index, base_series["Hourly_Cost_RM"], label="Baseline", color="gray", linestyle="--")
        ax.plot(smart_series.index, smart_series["Hourly_Cost_RM"], label="AI", color="blue", linewidth=2)
        ax.set_ylabel("RM")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        chart_help(
            f"Caption: {agg.lower()} series of `Hourly_Cost_RM` from the loaded CSVs (gray baseline vs blue AI).",
            "How to read: cost over time",
            """
- In **Hourly** mode, this is literally the saved hourly cost stream.\n
- In **Daily/Monthly** mode, values are **sums** within each bucket (so units are still RM, but it is not an average price).\n
""",
        )

        st.markdown("**Grid import over time**")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(base_series.index, base_series["Grid_Import_kW"], label="Baseline import", color="gray", linestyle="--")
        ax.plot(smart_series.index, smart_series["Grid_Import_kW"], label="AI import", color="purple", linewidth=2)
        ax.set_ylabel("kWh" if agg != "Hourly" else "kW")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        chart_help(
            f"Caption: {agg.lower()} series of `Grid_Import_kW` from the loaded CSVs.",
            "How to read: grid import over time",
            """
- In **Hourly** mode, the y-axis is **kW** (power).\n
- In **Daily/Monthly** mode, the plotted values are **sums of kW** within each bucket (use as a relative comparison, not a strict “kWh unless hourly”).\n
""",
        )

        if "Conservation_Mode_Active" in smart_y.columns:
            st.markdown("**Conservation days count**")
            cons_days = smart_y["Conservation_Mode_Active"].resample("1D").max().sum()
            st.metric("Days in Conservation Mode", f"{int(cons_days)}")

        st.divider()
        st.markdown("**Top 10 savings days (most convincing examples)**")
        if use_daily_reset and daily_results is not None and (not daily_results.empty):
            top10 = daily_results.sort_values("savings_rm", ascending=False).head(10)
            st.dataframe(
                top10[
                    [
                        "date",
                        "baseline_cost_rm",
                        "ai_cost_rm",
                        "savings_rm",
                        "peak_import_reduction_kwh",
                        "soc_at_peak_baseline_pct",
                        "soc_at_peak_ai_pct",
                        "min_soc_baseline_pct",
                        "min_soc_ai_pct",
                    ]
                ]
            )
        else:
            if "Hourly_Cost_RM" in df_base.columns and "Hourly_Cost_RM" in df_smart.columns:
                base_cost_daily = df_base[df_base["year"] == year]["Hourly_Cost_RM"].resample("1D").sum()
                smart_cost_daily = df_smart[df_smart["year"] == year]["Hourly_Cost_RM"].resample("1D").sum()
                joined = pd.DataFrame({"baseline_cost_rm": base_cost_daily, "ai_cost_rm": smart_cost_daily}).dropna()
                joined["savings_rm"] = joined["baseline_cost_rm"] - joined["ai_cost_rm"]

                # Peak import reduction (kWh) per day
                base_peak_daily = (
                    df_base[df_base["year"] == year]
                    .loc[df_base["hour"].between(params.peak_start_hour, params.peak_end_hour - 1), "Grid_Import_kW"]
                    .resample("1D")
                    .sum()
                )
                smart_peak_daily = (
                    df_smart[df_smart["year"] == year]
                    .loc[df_smart["hour"].between(params.peak_start_hour, params.peak_end_hour - 1), "Grid_Import_kW"]
                    .resample("1D")
                    .sum()
                )
                joined["peak_import_reduction_kwh"] = (base_peak_daily - smart_peak_daily).reindex(joined.index)

                # SoC at peak start per day
                peak_start = params.peak_start_hour
                base_soc_at_peak = (
                    df_base[df_base["year"] == year]
                    .loc[df_base["hour"] == peak_start, "SoC"]
                    .resample("1D")
                    .first()
                )
                smart_soc_at_peak = (
                    df_smart[df_smart["year"] == year]
                    .loc[df_smart["hour"] == peak_start, "SoC"]
                    .resample("1D")
                    .first()
                )
                joined["soc_at_peak_baseline_pct"] = (base_soc_at_peak * 100.0).reindex(joined.index)
                joined["soc_at_peak_ai_pct"] = (smart_soc_at_peak * 100.0).reindex(joined.index)

                # Optional: restrict to conservation days only
                if conservation_only and ("Conservation_Mode_Active" in df_smart.columns):
                    cons_flags = df_smart[df_smart["year"] == year]["Conservation_Mode_Active"].resample("1D").max()
                    joined = joined.loc[cons_flags[cons_flags].index.intersection(joined.index)]

                top10 = joined.sort_values("savings_rm", ascending=False).head(10).copy()
                top10 = top10.reset_index().rename(columns={"datetime": "date"})
                top10["date"] = top10["index"].dt.date if "index" in top10.columns else top10["date"]
                cols_show = [
                    "date",
                    "baseline_cost_rm",
                    "ai_cost_rm",
                    "savings_rm",
                    "peak_import_reduction_kwh",
                    "soc_at_peak_baseline_pct",
                    "soc_at_peak_ai_pct",
                ]
                st.dataframe(top10[cols_show])
            else:
                st.info("Daily Top 10 table unavailable (missing cost columns).")

        st.divider()
        st.markdown("**When do savings happen? (monthly)**")
        base_m = base_y.resample("1M").sum(numeric_only=True)
        smart_m = smart_y.resample("1M").sum(numeric_only=True)
        monthly_savings = (base_m["Hourly_Cost_RM"] - smart_m["Hourly_Cost_RM"]).rename("Savings_RM")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(monthly_savings.index, monthly_savings.values, color="teal", linewidth=2)
        ax.axhline(0, color="black", linewidth=1, alpha=0.6)
        ax.set_ylabel("RM (Baseline - AI)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        chart_help(
            "Caption: monthly sum of baseline cost minus monthly sum of AI cost (from saved hourly `Hourly_Cost_RM`).",
            "How to read: monthly savings curve",
            """
- Points above 0 mean **AI was cheaper than baseline** that month (based on saved hourly costs).\n
- This is **not** the same object as the **daily-reset KPI** total unless your evaluation assumptions match.\n
""",
        )


# -----------------------------
# Tab 4: Model metrics
# -----------------------------
with tab4:
    explain_tab_header(
        "tab4",
        "Explain Tab 4",
        """
### What this tab does
Sanity-check the PV forecast model used to compute \\(E_{pred}\\) for Conservation Mode.

### Honesty / limits
- These metrics are **offline accuracy** on a historical test window; they are not a guarantee of future performance.\n
- A better/worse RMSE here does not automatically equal better/worse electricity bills—dispatch and tariffs matter too.\n
""",
    )

    st.subheader("Forecasting model accuracy (R², RMSE)")
    st.caption("Evaluated chronologically on the test window used in your report (Aug 2024 → Jan 2025).")
    st.markdown(
        """
**Why this matters:** The forecast decides whether Conservation Mode activates.\n
If the forecast is inaccurate, the controller may activate Conservation Mode on the wrong days.\n

- **RMSE (kW)**: average size of prediction error. Lower is better.\n
- **R²**: how much of the PV variation is explained. Closer to 1 is better.\n
"""
    )

    section_help(
        "Guided help: interpreting forecast metrics",
        """
- **RMSE** is in **kW** because the target is instantaneous PV power.\n
- **R²** can look “high” if PV is smooth and predictable, but you should still check **bad-day tails** (residual histogram).\n
""",
    )

    if rf_model is None:
        st.warning("`models/rf_forecast_model.joblib` (or configured `paths.rf_model_path`) not found or could not be loaded. Model metrics tab is disabled.")
    else:
        features = ["ghiradiation", "temp", "humidity", "precip", "cloudcover", "hour", "month"]
        missing = [c for c in features + ["Target_Power_kW"] if c not in df_base.columns]
        if missing:
            st.error(f"Missing columns for metrics: {missing}")
        else:
            test_start = pd.Timestamp("2024-08-08")
            test_end = pd.Timestamp("2025-01-01")
            test_df = df_base.loc[test_start:test_end].copy()

            X = test_df[features]
            y_true = test_df["Target_Power_kW"].astype(float).values
            try:
                y_pred = rf_model.predict(X)  # type: ignore[attr-defined]
            except Exception as e:
                st.error("Model prediction failed on the selected test range.")
                y_pred = None

            if y_pred is not None:
                r2, rmse = safe_r2_rmse(y_true, y_pred)
                c1, c2, c3 = st.columns(3)
                c1.metric("R²", f"{r2:.3f}")
                c2.metric("RMSE (kW)", f"{rmse:.3f}")
                c3.metric("Test samples", f"{len(y_true):,}")

                st.divider()
                st.markdown("**Predicted vs actual (sampled): should align near the diagonal**")
                sample = min(2000, len(y_true))
                idx = np.linspace(0, len(y_true) - 1, sample).astype(int)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(y_true[idx], np.asarray(y_pred)[idx], s=8, alpha=0.3)
                lim = max(float(np.max(y_true[idx])), float(np.max(np.asarray(y_pred)[idx])))
                ax.plot([0, lim], [0, lim], linestyle="--", color="black", linewidth=1)
                ax.set_xlabel("Actual PV power (kW)")
                ax.set_ylabel("Predicted PV power (kW)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                chart_help(
                    "Caption: each point is one sampled hour (actual PV vs model prediction).",
                    "How to read: predicted vs actual scatter",
                    """
- Points should cluster around the **diagonal** if predictions are well-calibrated.\n
- Systematic curvature (S-shape) can mean the model **under/over-predicts** in certain regimes.\n
""",
                )

                st.markdown("**Residual distribution (sampled): errors centered near 0 is good**")
                resid = np.asarray(y_pred)[idx] - y_true[idx]
                fig, ax = plt.subplots(figsize=(10, 3))
                sns.histplot(resid, bins=40, kde=True, ax=ax)
                ax.set_xlabel("Prediction error (kW)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                chart_help(
                    "Caption: histogram of (predicted − actual) on the sampled test hours.",
                    "How to read: residual histogram",
                    """
- A distribution centered near **0** with narrow spread is ideal.\n
- Long tails mean occasional **large miss-forecasts**—those are the risky days for Conservation Mode decisions.\n
""",
                )
