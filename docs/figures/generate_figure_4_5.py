"""
Generate thesis figures for Section 4.5 (Phase 2 baseline / digital twin).

Outputs (300 DPI PNG):
  - figure_4_2_battery_energy_flow.png
  - figure_4_3_synthetic_load_one_week.png

Run from repository root:
  python docs/figures/generate_figure_4_5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "experiment_config.json"


def _load_baseline_csv() -> Path:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = json.load(f)
    rel = cfg.get("paths", {}).get("baseline_results_csv", "baseline_simulation_results.csv")
    p = REPO_ROOT / rel
    if not p.is_file():
        raise FileNotFoundError(f"Baseline CSV not found: {p}")
    return p


def figure_battery_energy_flow(out_path: Path, efficiency_rt: float = 0.95) -> None:
    """Schematic: P_load, P_pv, P_net, battery + limits, losses, grid + import clip."""
    eta_oneway = efficiency_rt**0.5
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # --- Nodes (x, y, w, h) in axis units ---
    def box(xy, w, h, text, fontsize=9, fc="#f5f5f5", ec="black"):
        r = FancyBboxPatch(
            xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0, facecolor=fc, edgecolor=ec,
        )
        ax.add_patch(r)
        cx, cy = xy[0] + w / 2, xy[1] + h / 2
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, wrap=True)

    # Load & PV
    box((0.15, 3.35), 1.35, 0.65, r"$P_{\mathrm{load}}$" + "\n(residential)", 10)
    box((0.15, 2.0), 1.35, 0.65, r"$P_{\mathrm{pv}}$" + "\n(Target_Power_kW)", 10)

    # Sum junction
    ax.add_patch(mpatches.Circle((2.55, 2.825), 0.18, fc="white", ec="black", lw=1))
    ax.text(2.55, 2.825, r"$\Sigma$", ha="center", va="center", fontsize=11)

    # Battery enclosure
    bat_x, bat_y, bat_w, bat_h = 3.15, 1.35, 3.35, 2.55
    outer = FancyBboxPatch(
        (bat_x, bat_y), bat_w, bat_h,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=1.2, facecolor="#eef3f8", edgecolor="#334455",
    )
    ax.add_patch(outer)
    ax.text(
        bat_x + bat_w / 2, bat_y + bat_h - 0.22,
        "Battery digital twin (kW limits, kWh capacity)",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#223344",
    )

    # SoC tank
    tank_x, tank_w = bat_x + 0.35, 0.55
    tank_y0, tank_h = bat_y + 0.45, 1.55
    ax.add_patch(Rectangle((tank_x, tank_y0), tank_w, tank_h, fill=False, lw=1.0, ec="black"))
    soc_min_y = tank_y0 + 0.12 * tank_h
    soc_top_y = tank_y0 + 0.88 * tank_h
    ax.add_patch(Rectangle((tank_x + 0.04, soc_min_y), tank_w - 0.08, soc_top_y - soc_min_y, fc="#6fa8dc", ec="none", alpha=0.85))
    ax.plot([tank_x, tank_x + tank_w], [soc_min_y, soc_min_y], color="black", lw=0.8)
    ax.text(tank_x + tank_w + 0.08, soc_min_y, r"$SoC_{\min}$", ha="left", va="center", fontsize=8)
    ax.text(tank_x + tank_w + 0.08, tank_y0 + tank_h, "100%", ha="left", va="center", fontsize=8)
    ax.text(tank_x + tank_w / 2, tank_y0 + tank_h + 0.18, "SoC (tank level)", ha="center", va="bottom", fontsize=8)

    # Inverter limits text
    ax.text(
        bat_x + 1.15, bat_y + 0.55,
        r"$P_{\mathrm{chg}}\leq P_{\mathrm{inv}}$, $P_{\mathrm{dis}}\leq P_{\mathrm{inv}}$" + "\n"
        r"usable energy $\propto (SoC - SoC_{\min})$, headroom $\propto (SoC_{\max}-SoC)$",
        ha="left", va="bottom", fontsize=8, linespacing=1.25,
    )

    # Loss annotation
    ax.text(
        bat_x + 1.15, bat_y + 1.45,
        r"one-way $\eta_{\mathrm{chg}}=\eta_{\mathrm{dis}}=\sqrt{\eta_{\mathrm{rt}}}$"
        f"\n(default $\\eta_{{\\mathrm{{rt}}}}={efficiency_rt:g}$ → "
        f"$\\sqrt{{\\eta_{{\\mathrm{{rt}}}}}}\\approx {eta_oneway:.3f}$)",
        ha="left", va="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#888", linewidth=0.6),
    )

    ax.text(
        bat_x + 1.15, bat_y + 2.25,
        r"$P_{\mathrm{batt}} = f_{\mathrm{batt}}(P_{\mathrm{net}})$  (signed: + discharge to AC, $-$ charge)",
        ha="left", va="center", fontsize=8.5,
    )

    # Grid
    box((7.05, 2.55), 1.55, 0.85, "Grid\n" + r"$P_{\mathrm{grid}} = P_{\mathrm{net}} - P_{\mathrm{batt}}$", 9)
    box((7.05, 1.15), 1.55, 0.85, r"Import (economics)" + "\n" + r"$\max(0, P_{\mathrm{grid}})$", 9, fc="#fdeaea")

    # Arrows helper
    def arrow(p0, p1, text=None, text_offset=(0, 0), rad=0.0):
        arr = FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=12, linewidth=1.0,
            color="black", connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arr)
        if text:
            mx = (p0[0] + p1[0]) / 2 + text_offset[0]
            my = (p0[1] + p1[1]) / 2 + text_offset[1]
            ax.text(mx, my, text, fontsize=8, ha="center", va="bottom")

    arrow((1.5, 3.68), (2.37, 2.95), r"$P_{\mathrm{load}}$", (0, 0.12))
    arrow((1.5, 2.33), (2.37, 2.7), r"$P_{\mathrm{pv}}$", (0, -0.12))
    ax.text(2.85, 3.05, r"$P_{\mathrm{net}}=P_{\mathrm{load}}-P_{\mathrm{pv}}$", fontsize=8.5)

    arrow((2.73, 2.825), (3.15, 2.45), None)
    ax.text(2.95, 2.35, r"$P_{\mathrm{net}}$", fontsize=8.5, ha="center")

    arrow((6.5, 2.45), (7.05, 2.98), r"$P_{\mathrm{grid}}$", (0, 0.1))
    arrow((7.82, 2.55), (7.82, 2.02), None)

    # Small loss squiggles (decorative, near battery label)
    ax.annotate(
        "", xy=(bat_x + 0.95, bat_y + 2.05), xytext=(bat_x + 0.55, bat_y + 1.75),
        arrowprops=dict(arrowstyle="-|>", color="#c00", lw=0.8, shrinkA=0, shrinkB=0),
    )
    ax.text(bat_x + 1.0, bat_y + 1.9, "losses", fontsize=7, color="#a00", ha="left", va="bottom")

    ax.set_title(
        "Figure 4.2 — Battery energy flow and sign convention (baseline hourly loop)",
        fontsize=11, pad=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_synthetic_load_week(out_path: Path, csv_path: Path) -> None:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    start = df.index.min()
    end = start + pd.Timedelta(days=7)
    wk = df.loc[start:end - pd.Timedelta(hours=1), ["Load_kW"]].copy()
    wk["dow"] = wk.index.day_name()

    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
    ax.plot(wk.index, wk["Load_kW"], color="#1f4e79", lw=0.9, label="Load_kW")

    # Shade weekend calendar days touched by the window
    day0 = wk.index.min().normalize()
    day1 = wk.index.max().normalize() + pd.Timedelta(days=1)
    for d in pd.date_range(day0, day1, freq="D"):
        if int(d.weekday()) >= 5:
            ax.axvspan(d, d + pd.Timedelta(days=1), color="#f4cccc", alpha=0.35, lw=0)

    ax.text(
        start + pd.Timedelta(hours=30), wk["Load_kW"].max() * 0.92,
        "weekday pattern",
        fontsize=8, color="#333",
    )
    first_sat = day0 + pd.Timedelta(days=(5 - int(day0.weekday())) % 7)
    if first_sat < day0:
        first_sat = first_sat + pd.Timedelta(days=7)
    ax.text(
        first_sat + pd.Timedelta(hours=12), wk["Load_kW"].max() * 0.92,
        "weekend uplift (+10% energy)",
        fontsize=8, color="#333", ha="center",
    )

    ax.set_ylabel("Load (kW)")
    ax.set_xlabel("Date/time (hourly)")
    ax.margins(x=0.01)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_title(
        "Figure 4.3 — Seven-day excerpt of synthetic residential Load_kW",
        fontsize=11, pad=8,
    )
    note = (
        "Series is synthetic (generate_synthetic_load); shaded bands are weekend days; "
        "daily target energy is randomised in band then hourly shape is re-scaled."
    )
    ax.text(0.0, -0.22, note, transform=ax.transAxes, fontsize=7.5, va="top", color="#444")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    csv_path = _load_baseline_csv()
    eta_rt = 0.95
    if CONFIG_PATH.is_file():
        with CONFIG_PATH.open(encoding="utf-8") as f:
            eta_rt = float(json.load(f).get("battery", {}).get("efficiency_rt", eta_rt))

    p_bat = OUT_DIR / "figure_4_2_battery_energy_flow.png"
    p_load = OUT_DIR / "figure_4_3_synthetic_load_one_week.png"
    figure_battery_energy_flow(p_bat, efficiency_rt=eta_rt)
    figure_synthetic_load_week(p_load, csv_path=csv_path)
    print(f"Wrote:\n  {p_bat}\n  {p_load}\nfrom baseline CSV:\n  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
