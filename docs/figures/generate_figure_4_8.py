"""
Thesis figures for Section 4.8 (Phase 5 Streamlit dashboard).

These are programmatic equivalents / layout schematics suitable for print;
they are not pixel screenshots of the running app.

Outputs (300 DPI PNG):
  - figure_4_7_dashboard_layout_schematic.png   — sidebar collapsed vs expanded + tab strip
  - figure_4_8_tab1_energy_flow.png             — same Graphviz as Tab 1 (or matplotlib fallback)
  - figure_4_9_tab2_saved_csv_day_panels.png    — four panels in Tab 2 layout from saved CSVs
  - figure_4_10_tab3_monthly_costs_kpis.png     — monthly cost series + KPI strip from metrics.json
  - figure_4_11_tab4_forecast_metrics.png       — same logic as Streamlit Tab 4 / Figure 4.5

Run from repository root:
  python docs/figures/generate_figure_4_8.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "experiment_config.json"

TAB1_DOT = """
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

FEATURES = ["ghiradiation", "temp", "humidity", "precip", "cloudcover", "hour", "month"]
TARGET = "Target_Power_kW"
TEST_START = pd.Timestamp("2024-08-08")
TEST_END = pd.Timestamp("2025-01-01")


def _load_cfg() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _latest_results_run(results_root: Path) -> Path | None:
    if not results_root.is_dir():
        return None
    runs = [p for p in results_root.iterdir() if p.is_dir() and (p / "metrics.json").is_file()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.name)
    return runs[-1]


def figure_dashboard_layout(out_path: Path) -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    ax_l.set_title("(a) Sidebar — compact (expanders closed)", fontsize=10, loc="left")
    ax_l.add_patch(Rectangle((0.3, 0.5), 1.4, 9.0, facecolor="#f0f2f6", edgecolor="#333", linewidth=1))
    ax_l.text(1.0, 9.2, "☰", ha="center", fontsize=14)
    for y, lab in [
        (8.2, "Guided help"),
        (7.0, "Experiment\nJSON"),
        (5.8, "Saved run\nselector"),
        (4.6, "Hardware"),
        (3.5, "Scenario"),
        (2.4, "Tariff"),
        (1.3, "AI policy"),
        (0.5, "Date"),
    ]:
        ax_l.text(1.0, y, lab, ha="center", va="center", fontsize=7, rotation=0)

    ax_r.set_title("(b) Sidebar — expanded groups + wide main workspace", fontsize=10, loc="left")
    ax_r.add_patch(Rectangle((0.2, 0.5), 2.6, 9.0, facecolor="#eef3f8", edgecolor="#333", linewidth=1))
    y = 8.8
    for h, title in [(1.1, "Experiment Settings"), (1.0, "Hardware"), (0.9, "Scenario scaling"), (0.85, "Tariff"), (1.0, "AI policy"), (0.6, "Date picker")]:
        ax_r.add_patch(FancyBboxPatch((0.35, y - h), 2.3, h - 0.08, boxstyle="round,pad=0.02", facecolor="white", edgecolor="#888"))
        ax_r.text(0.4, y - 0.12, title, fontsize=7.5, fontweight="bold", va="top")
        y -= h + 0.12

    ax_r.add_patch(Rectangle((3.1, 0.5), 6.7, 9.0, facecolor="#fafafa", edgecolor="#222", linewidth=1.2))
    ax_r.text(6.45, 9.35, "Main workspace — st.tabs(..., layout=\"wide\")", ha="center", fontsize=9, fontweight="bold")
    ax_r.text(
        6.45, 7.1,
        "Tab 1 — concepts + Graphviz\n"
        "Tab 2 — live day simulation\n"
        "Tab 3 — yearly comparison + KPIs\n"
        "Tab 4 — forecast R² / RMSE / diagnostics",
        ha="center", va="center", fontsize=8.2, color="#222",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bbb"),
    )
    ax_r.text(6.45, 3.8, "Charts, metrics, tables", ha="center", va="center", fontsize=8, color="#444")

    fig.suptitle(
        "Figure 4.7 — Streamlit dashboard layout (phase5_dashboard.py): sidebar vs main tabs",
        fontsize=11, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _tab1_graphviz_matplotlib_fallback(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 2.2), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2)
    ax.axis("off")
    boxes = [
        (0.2, 0.55, "PV\n$P_{pv}$"),
        (2.0, 0.55, "Load\n$P_{load}$"),
        (4.0, 0.55, "Net\n$P_{net}$"),
        (6.0, 0.55, "Battery\ntwin"),
        (8.2, 0.55, "Grid\n$P_{grid}$"),
        (10.4, 0.55, "Tariff\ncost"),
    ]
    for x, y, txt in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 1.55, 0.9, boxstyle="round,pad=0.03", facecolor="#fff8e7", edgecolor="black"))
        ax.text(x + 0.78, y + 0.45, txt, ha="center", va="center", fontsize=8)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 1.55
        x1 = boxes[i + 1][0]
        ax.annotate("", xy=(x1, 1.0), xytext=(x0, 1.0), arrowprops=dict(arrowstyle="-|>", lw=1.0))
    ax.text(7, 1.95, "Equivalent topology to Tab 1 Graphviz (fallback if system Graphviz unavailable)", ha="center", fontsize=7)
    fig.suptitle("Figure 4.8 — Energy balance flow (Tab 1 concept diagram)", fontsize=11, y=1.05)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_tab1_graphviz(out_path: Path) -> None:
    tmp = OUT_DIR / "_tab1_graphviz_tmp"
    try:
        import graphviz

        src = graphviz.Source(TAB1_DOT.strip())
        src.render(filename=str(tmp), format="png", cleanup=True)
        produced = Path(str(tmp) + ".png")
        if produced.is_file():
            produced.replace(out_path)
        else:
            raise FileNotFoundError("graphviz render produced no PNG")
        gv = Path(str(tmp) + ".gv")
        if gv.is_file():
            gv.unlink(missing_ok=True)
    except Exception:
        _tab1_graphviz_matplotlib_fallback(out_path)
    finally:
        for p in (tmp, Path(str(tmp) + ".gv")):
            if p.is_file():
                p.unlink(missing_ok=True)


def _pick_conservation_day(df_smart: pd.DataFrame) -> pd.Timestamp:
    if "Conservation_Mode_Active" not in df_smart.columns:
        return pd.Timestamp(df_smart.index.min().normalize())
    s = df_smart["Conservation_Mode_Active"].astype(bool)
    if not s.any():
        return pd.Timestamp(df_smart.index.min().normalize()) + pd.Timedelta(days=180)
    day = df_smart.loc[s].index.normalize().unique()
    day = sorted(day)[max(0, len(day) // 3)]
    return pd.Timestamp(day)


def figure_tab2_panels(
    df_b: pd.DataFrame,
    df_s: pd.DataFrame,
    day: pd.Timestamp,
    peak_start: int,
    peak_end: int,
    out_path: Path,
    *,
    csv_note: str,
) -> None:
    d0 = day.normalize()
    d1 = d0 + pd.Timedelta(days=1)
    b = df_b.loc[d0:d1 - pd.Timedelta(seconds=1)]
    s = df_s.loc[d0:d1 - pd.Timedelta(seconds=1)]
    if b.empty or s.empty:
        raise ValueError("Empty day slice for Tab 2 figure")
    common = b.index.intersection(s.index)
    b = b.loc[common].sort_index()
    s = s.loc[common].sort_index()
    h = b.index.hour

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=150)
    cons = bool(s["Conservation_Mode_Active"].iloc[0]) if "Conservation_Mode_Active" in s.columns else False

    ax = axes[0, 0]
    ax.plot(h, b["Load_kW"], color="darkorange", lw=2, label="Load")
    ax.plot(h, b["Target_Power_kW"], color="green", lw=2, label="PV")
    ax.axvspan(peak_start, peak_end, alpha=0.12, color="gray")
    ax.set_title("Load vs PV (saved hourly series)", fontsize=9)
    ax.set_xlabel("Hour")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(h, b["SoC"] * 100, color="gray", ls="--", lw=1.5, label="Baseline SoC")
    ax.plot(h, s["SoC"] * 100, color="royalblue", lw=2, label="Smart SoC")
    ax.axvspan(peak_start, peak_end, alpha=0.12, color="gray")
    ax.set_title("SoC (%) — saved continuous run", fontsize=9)
    ax.set_xlabel("Hour")
    ax.set_ylabel("SoC (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(h, b["Grid_Import_kW"], color="gray", ls="--", lw=1.5, label="Baseline import")
    ax.plot(h, s["Grid_Import_kW"], color="purple", lw=2, label="Smart import")
    ax.axvspan(peak_start, peak_end, alpha=0.12, color="gray")
    ax.set_title("Grid import (kW)", fontsize=9)
    ax.set_xlabel("Hour")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(h, b["Hourly_Cost_RM"], color="gray", ls="--", lw=1.5, label="Baseline cost")
    ax.plot(h, s["Hourly_Cost_RM"], color="crimson", lw=2, label="Smart cost")
    ax.axvspan(peak_start, peak_end, alpha=0.12, color="gray")
    ax.set_title("Hourly cost (RM)", fontsize=9)
    ax.set_xlabel("Hour")
    ax.set_ylabel("RM")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    cap = (
        f"Figure 4.9 — Tab 2–style four-panel view using saved CSV trajectories (not a live re-sim). "
        f"Day {d0.date()} (Conservation_Mode_Active={'true' if cons else 'false'} on smart CSV). "
        f"Data: {csv_note}. Streamlit Tab 2 recomputes when Run is clicked; curves match batch outputs when sliders match the saved run."
    )
    fig.suptitle(cap, fontsize=8.5, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_tab3_monthly_and_kpis(
    df_b: pd.DataFrame,
    df_s: pd.DataFrame,
    year: int,
    metrics: dict,
    out_path: Path,
    *,
    run_folder: str | None,
) -> None:
    mb = df_b[df_b["year"] == year]
    ms = df_s[df_s["year"] == year]
    if mb.empty:
        raise ValueError(f"No rows for year {year}")

    mb = mb.copy()
    ms = ms.copy()
    mb.index = pd.to_datetime(mb.index)
    ms.index = pd.to_datetime(ms.index)
    try:
        b_m = mb["Hourly_Cost_RM"].resample("ME").sum()
        s_m = ms["Hourly_Cost_RM"].resample("ME").sum()
    except Exception:
        b_m = mb["Hourly_Cost_RM"].resample("M").sum()
        s_m = ms["Hourly_Cost_RM"].resample("M").sum()

    fig, (ax_ts, ax_k) = plt.subplots(2, 1, figsize=(10, 6), dpi=150, gridspec_kw={"height_ratios": [2.2, 1]})
    ax_ts.plot(b_m.index, b_m.values, marker="o", label="Baseline monthly RM", color="gray")
    ax_ts.plot(s_m.index, s_m.values, marker="o", label="Smart monthly RM", color="darkblue")
    ax_ts.set_ylabel("Sum Hourly_Cost_RM (RM / month)")
    ax_ts.set_title(f"Figure 4.10 — Tab 3–style monthly aggregation ({year})", fontsize=10, loc="left")
    ax_ts.legend()
    ax_ts.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    cont = metrics.get("by_evaluation_mode", {}).get("continuous_csv", metrics)
    lines = [
        "Continuous CSV headline (metrics.json):",
        f"  Baseline total cost (RM): {cont.get('baseline_total_cost_rm', float('nan')):.2f}",
        f"  Smart total cost (RM):    {cont.get('smart_total_cost_rm', float('nan')):.2f}",
        f"  Total savings (RM):       {cont.get('total_savings_rm', float('nan')):.2f}",
        f"  Peak import reduction:    {cont.get('peak_import_reduction_kwh', float('nan')):.2f} kWh (proxy)",
        "",
        "Daily-reset (excerpt):",
    ]
    dr = metrics.get("by_evaluation_mode", {}).get("daily_reset", {})
    lines += [
        f"  Mean daily savings (RM/day): {dr.get('daily_savings_mean_rm', float('nan')):.3f}",
        f"  Conservation days:          {metrics.get('conservation_days_count', '—')}",
        f"  Sum daily savings (RM):     {dr.get('sum_daily_savings_rm', float('nan')):.2f}",
    ]
    if run_folder:
        lines.insert(0, f"Saved run folder: {run_folder}")
    ax_k.axis("off")
    ax_k.text(0.02, 0.95, "\n".join(lines), transform=ax_k.transAxes, va="top", fontsize=8, family="monospace")

    note = (
        "Plots use loaded hourly CSVs; KPI block is copied from metrics.json for the selected results run. "
        "Tab 3 can recompute daily-reset KPIs from sidebar sliders while plots still show saved series — see dashboard help."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=7.5, color="#444", wrap=True)

    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _safe_r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = math.sqrt(mse)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom
    return r2, rmse


def figure_tab4_forecast(df: pd.DataFrame, model, out_path: Path) -> None:
    df = df.dropna(subset=FEATURES + [TARGET])
    test_df = df.loc[TEST_START:TEST_END].copy()
    X = test_df[FEATURES]
    y_true = test_df[TARGET].astype(float).values
    y_pred = model.predict(X)
    r2, rmse = _safe_r2_rmse(y_true, y_pred)
    n = len(y_true)
    sample = min(2000, n)
    idx = np.linspace(0, n - 1, sample).astype(int)
    yt = y_true[idx]
    yp = np.asarray(y_pred)[idx]
    resid = yp - yt

    fig = plt.figure(figsize=(11, 4.8), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(yt, yp, s=8, alpha=0.3, color="#1f4e79")
    lim = max(float(np.max(yt)), float(np.max(yp)))
    ax0.plot([0, lim], [0, lim], linestyle="--", color="black", linewidth=1)
    ax0.set_xlabel("Actual PV power (kW)")
    ax0.set_ylabel("Predicted PV power (kW)")
    ax0.set_title("(a) Predicted vs actual (sampled)", fontsize=10, loc="left")
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1])
    sns.histplot(resid, bins=40, kde=True, ax=ax1, color="#2e5aac", edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Prediction error (kW)  (predicted − actual)")
    ax1.set_title("(b) Residual distribution (sampled)", fontsize=10, loc="left")
    ax1.grid(True, axis="y", alpha=0.3)

    cap = (
        f"Same window as Dashboard Tab 4: {TEST_START.date()} → {TEST_END.date()} "
        f"({n:,} rows). R² = {r2:.4f}, RMSE = {rmse:.4f} kW."
    )
    fig.text(0.5, 0.02, cap, ha="center", fontsize=8, color="#333")

    fig.suptitle(
        "Figure 4.11 — Forecast metrics view (equivalent to Streamlit Tab 4)",
        fontsize=11, y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    cfg = _load_cfg()
    paths = cfg.get("paths", {})
    tariff = cfg.get("tariff", {})
    peak_start = int(tariff.get("peak_start_hour", 7))
    peak_end = int(tariff.get("peak_end_hour", 22))

    base_p = REPO_ROOT / paths.get("baseline_results_csv", "baseline_simulation_results.csv")
    smart_p = REPO_ROOT / paths.get("smart_results_csv", "smart_simulation_results.csv")
    proc_p = REPO_ROOT / paths.get("phase1_processed_csv", "data/processed/sarawak_weather_processed.csv")
    model_p = REPO_ROOT / paths.get("rf_model_path", "models/rf_forecast_model.joblib")

    df_b = pd.read_csv(base_p, parse_dates=["datetime"]).set_index("datetime").sort_index()
    df_s = pd.read_csv(smart_p, parse_dates=["datetime"]).set_index("datetime").sort_index()
    for d in (df_b, df_s):
        d["year"] = d.index.year
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)

    csv_note = f"{base_p.name} + {smart_p.name} (paths from experiment_config.json)"

    figure_dashboard_layout(OUT_DIR / "figure_4_7_dashboard_layout_schematic.png")
    figure_tab1_graphviz(OUT_DIR / "figure_4_8_tab1_energy_flow.png")

    day = _pick_conservation_day(df_s)
    figure_tab2_panels(
        df_b,
        df_s,
        day,
        peak_start,
        peak_end,
        OUT_DIR / "figure_4_9_tab2_saved_csv_day_panels.png",
        csv_note=csv_note,
    )

    run = _latest_results_run(REPO_ROOT / "results")
    if run:
        with (run / "metrics.json").open(encoding="utf-8") as f:
            metrics = json.load(f)
        run_name = run.name
    else:
        run_name = None
        metrics = {
            "conservation_days_count": "— (no results/*/metrics.json)",
            "by_evaluation_mode": {
                "continuous_csv": {
                    "baseline_total_cost_rm": float(df_b["Hourly_Cost_RM"].sum()),
                    "smart_total_cost_rm": float(df_s["Hourly_Cost_RM"].sum()),
                    "total_savings_rm": float(df_b["Hourly_Cost_RM"].sum() - df_s["Hourly_Cost_RM"].sum()),
                    "peak_import_reduction_kwh": float("nan"),
                },
                "daily_reset": {
                    "daily_savings_mean_rm": float("nan"),
                    "sum_daily_savings_rm": float("nan"),
                },
            },
        }

    year = 2024 if bool((df_b["year"] == 2024).any()) else int(df_b["year"].mode().iloc[0])
    figure_tab3_monthly_and_kpis(
        df_b,
        df_s,
        year,
        metrics,
        OUT_DIR / "figure_4_10_tab3_monthly_costs_kpis.png",
        run_folder=run_name,
    )

    df_proc = pd.read_csv(proc_p, parse_dates=["datetime"]).set_index("datetime").sort_index()
    df_proc["hour"] = df_proc.index.hour
    df_proc["month"] = df_proc.index.month
    model = joblib.load(model_p)
    figure_tab4_forecast(df_proc, model, OUT_DIR / "figure_4_11_tab4_forecast_metrics.png")

    print("Wrote Phase 5 / dashboard thesis figures to", OUT_DIR)
    print("  figure_4_7_dashboard_layout_schematic.png")
    print("  figure_4_8_tab1_energy_flow.png")
    print("  figure_4_9_tab2_saved_csv_day_panels.png")
    print("  figure_4_10_tab3_monthly_costs_kpis.png")
    print("  figure_4_11_tab4_forecast_metrics.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
