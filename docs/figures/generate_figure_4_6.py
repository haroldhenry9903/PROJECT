"""
Figures for thesis Section 4.6 (Phase 3 forecasting).

Outputs:
  - figure_4_4_chronological_train_test_timeline.png
  - figure_4_5_forecast_diagnostics_tab4_window.png
  - phase3_best_params_for_table_4_8.txt  (from saved model attributes)

Run from repository root:
  python docs/figures/generate_figure_4_6.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "experiment_config.json"

FEATURES = ["ghiradiation", "temp", "humidity", "precip", "cloudcover", "hour", "month"]
TARGET = "Target_Power_kW"
SPLIT = pd.Timestamp("2024-08-08")
TEST_END_TAB4 = pd.Timestamp("2025-01-01")


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _safe_r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Match phase5_dashboard.safe_r2_rmse."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = math.sqrt(mse)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom
    return r2, rmse


def load_engineered_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    return df.dropna(subset=FEATURES + [TARGET])


def figure_train_test_timeline(df: pd.DataFrame, out_path: Path) -> None:
    t0, t1 = df.index.min(), df.index.max()
    fig, ax = plt.subplots(figsize=(10.5, 2.4), dpi=150)
    ax.set_xlim(t0, t1)
    y0, h = 0.35, 0.3
    ax.set_ylim(0, 1)

    train_end = SPLIT - pd.Timedelta(seconds=1)
    # Train bar
    ax.axvspan(t0, train_end, ymin=y0, ymax=y0 + h, color="#2e5aac", alpha=0.85, clip_on=False)
    # Test bar (chronological test: split → end of dataset)
    ax.axvspan(SPLIT, t1, ymin=y0, ymax=y0 + h, color="#c45c26", alpha=0.85, clip_on=False)
    # Dashboard / report metrics window (subset of test)
    ax.axvspan(SPLIT, TEST_END_TAB4, ymin=y0, ymax=y0 + h, color="#f4c430", alpha=0.35, clip_on=False)

    ax.axvline(SPLIT, color="black", linewidth=1.2, linestyle="-", ymax=0.95)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        t0 + (train_end - t0) / 2, y0 + h / 2,
        "Train\n(< 2024-08-08)\n" + r"($\leq$ 2024-08-07 last hour)",
        transform=trans, ha="center", va="center", fontsize=8, color="white", fontweight="bold",
    )
    ax.text(
        SPLIT + (t1 - SPLIT) / 2, y0 + h / 2,
        "Test\n($\\geq$ 2024-08-08 → dataset end)",
        transform=trans, ha="center", va="center", fontsize=8, color="white", fontweight="bold",
    )

    ax.text(
        SPLIT + (TEST_END_TAB4 - SPLIT) / 2, y0 + h + 0.06,
        "Tab 4 / report metrics window\n2024-08-08 → 2025-01-01",
        transform=trans, ha="center", va="bottom", fontsize=7.5, color="#333",
    )

    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Time (hourly index)", fontsize=9)

    fig.suptitle(
        "Figure 4.4 — Chronological train/test split for Phase 3 (no random shuffle)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_forecast_diagnostics(model, df: pd.DataFrame, out_path: Path) -> tuple[float, float]:
    """Replicate Streamlit Tab 4: slice, sample, scatter, residual histogram."""
    test_df = df.loc[SPLIT:TEST_END_TAB4].copy()
    test_df = test_df.dropna(subset=FEATURES + [TARGET])
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
        f"Same window as Streamlit Tab 4: {SPLIT.date()} → {TEST_END_TAB4.date()} "
        f"({n:,} hourly rows after dropna). "
        f"Metrics (full window, not subsampled): R² = {r2:.4f}, RMSE = {rmse:.4f} kW."
    )
    fig.text(0.5, 0.02, cap, ha="center", fontsize=8, color="#333", wrap=True)

    fig.suptitle(
        "Figure 4.5 — PV forecast diagnostics (Random Forest, dashboard-equivalent view)",
        fontsize=11, y=0.98,
    )
    fig.subplots_adjust(bottom=0.18, top=0.88)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return r2, rmse


def write_table_4_8_snippet(model, out_path: Path) -> None:
    ne = int(getattr(model, "n_estimators", -1))
    md = getattr(model, "max_depth", None)
    md_s = "None" if md is None else str(int(md))
    lines = [
        "Fill Table 4.8 from the saved model (last Phase 3 run):",
        "",
        f"n_estimators (selected): {ne}",
        f"max_depth (selected): {md_s}",
        "",
        "Grid (searched): n_estimators in {50, 100, 200}; max_depth in {10, 20, None}",
        "cv: 3 | scoring: neg_root_mean_squared_error | n_jobs: -1",
        "",
        "If you retrain, also copy the console line:",
        'Best hyperparameters found: {...}',
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cfg = _load_config()
    paths = cfg.get("paths", {})
    csv_path = REPO_ROOT / paths.get("phase1_processed_csv", "data/processed/sarawak_weather_processed.csv")
    model_path = REPO_ROOT / paths.get("rf_model_path", "models/rf_forecast_model.joblib")

    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))
    if not model_path.is_file():
        raise FileNotFoundError(str(model_path))

    df = load_engineered_df(csv_path)
    model = joblib.load(model_path)

    p_timeline = OUT_DIR / "figure_4_4_chronological_train_test_timeline.png"
    p_diag = OUT_DIR / "figure_4_5_forecast_diagnostics_tab4_window.png"
    p_table = OUT_DIR / "phase3_best_params_for_table_4_8.txt"

    figure_train_test_timeline(df, p_timeline)
    r2, rmse = figure_forecast_diagnostics(model, df, p_diag)
    write_table_4_8_snippet(model, p_table)

    print(f"Wrote:\n  {p_timeline}\n  {p_diag}\n  {p_table}")
    print(f"Tab-4-window metrics: R2={r2:.4f}, RMSE={rmse:.4f} kW, n_estimators={model.n_estimators}, max_depth={model.max_depth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
