"""
Figure for thesis Section 4.7 (Phase 4 smart optimizer — hourly discharge limit policy).

Output:
  - figure_4_6_smart_policy_hourly_flowchart.png

Mirrors the control logic in phase4_smart_optimizer.py (after the midnight
conservation_mode update).

Run from repository root:
  python docs/figures/generate_figure_4_7.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, RegularPolygon

OUT_DIR = Path(__file__).resolve().parent


def _box(ax, xy, w, h, text, fontsize=8.5, fc="#eef3f8"):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, facecolor=fc, edgecolor="#222",
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _diamond(ax, center, r, text, fontsize=8):
    cx, cy = center
    poly = RegularPolygon(
        (cx, cy), numVertices=4, radius=r, orientation=math.pi / 4,
        facecolor="#fff4e0", edgecolor="#222", linewidth=1.0,
    )
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, xy0, xy1, label=None, rad=0.0, fs=7.5):
    arr = FancyArrowPatch(
        xy0, xy1, arrowstyle="-|>", mutation_scale=14, linewidth=1.0,
        color="#111", connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if label:
        mx, my = (xy0[0] + xy1[0]) / 2, (xy0[1] + xy1[1]) / 2
        ax.text(mx, my + 0.12, label, ha="center", va="bottom", fontsize=fs, color="#333")


def figure_smart_policy_flowchart(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 7.2), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    ax.text(
        6, 10.05,
        "Hourly discharge limit (before battery.update_soc) — matches phase4_smart_optimizer.py",
        ha="center", va="top", fontsize=9, fontweight="bold", color="#1a1a1a",
    )
    ax.text(
        6, 9.55,
        "Day-level mode is set at midnight from $E_{\\mathrm{pred}}$ vs $E_{\\mathrm{crit}}$; "
        "here conservation_mode is already known for the calendar day.",
        ha="center", va="top", fontsize=7.5, color="#444", style="italic",
    )

    # --- Decision 1: conservation ---
    _diamond(ax, (6, 7.85), 0.95, "conservation_mode\nactive?", fontsize=8)

    # No → Standard
    _arrow(ax, (5.35, 7.85), (2.35, 6.55), "No", rad=0.05)
    _box(
        ax, (0.35, 5.75), 3.6, 1.1,
        r"Standard operating day" "\n" r"$\mathrm{max\_discharge\_kw} \leftarrow P_{\mathrm{inv}}$",
        fontsize=8.5, fc="#e8f4ea",
    )

    # Yes → Peak?
    _arrow(ax, (6.65, 7.85), (9.35, 6.95), "Yes", rad=-0.05)
    _diamond(
        ax, (9.8, 6.05), 0.88,
        "Peak window?\n"
        r"$\mathrm{peak\_start} \leq h < \mathrm{peak\_end}$"
        "\n(default 7–22 h)",
        fontsize=7.2,
    )

    # Peak Yes → P_inv (right)
    _arrow(ax, (10.55, 5.55), (11.0, 4.35), "Yes", rad=0.0)
    _box(
        ax, (9.15, 3.45), 2.85, 1.05,
        "Peak: full discharge headroom\n" r"$\mathrm{max\_discharge\_kw} \leftarrow P_{\mathrm{inv}}$",
        fontsize=8, fc="#e3f0ff",
    )

    # Peak No (off-peak) → SoC diamond
    _arrow(ax, (9.05, 5.35), (6.9, 4.5), "No (off-peak)", rad=-0.12)
    _diamond(ax, (5.5, 3.55), 0.82, "$\\mathrm{SoC} \\leq \\mathrm{SoC}_{\\mathrm{res}}$ ?", fontsize=8)

    # SoC Yes → P_ess
    _arrow(ax, (4.75, 3.0), (3.0, 2.0), "Yes", rad=0.0)
    _box(
        ax, (0.9, 1.15), 3.0, 1.0,
        "Hoard for peak\n" r"$\mathrm{max\_discharge\_kw} \leftarrow P_{\mathrm{ess}}$",
        fontsize=8, fc="#fdeaea",
    )

    # SoC No → P_inv
    _arrow(ax, (6.25, 3.0), (7.55, 2.0), "No", rad=0.0)
    _box(
        ax, (7.05, 1.15), 2.85, 1.0,
        "Off-peak but above reserve\n" r"$\mathrm{max\_discharge\_kw} \leftarrow P_{\mathrm{inv}}$",
        fontsize=8, fc="#f5f5f5",
    )

    ax.plot([1.5, 11.2], [0.55, 0.55], color="#555", linewidth=0.8, linestyle=":")
    _box(
        ax, (2.2, 0.02), 7.6, 0.48,
        r"Then: $P_{\mathrm{net}}=P_{\mathrm{load}}-P_{\mathrm{pv}}$ $\rightarrow$ update_soc $\rightarrow$ "
        r"$P_{\mathrm{grid}}$ $\rightarrow$ $\max(0,\cdot)$ import $\rightarrow$ ToU tariff (same as Phase 2)",
        fontsize=7.5, fc="#fafafa",
    )

    merge_y = 0.52
    _arrow(ax, (2.15, 5.75), (4.2, merge_y), None, rad=0.18)
    _arrow(ax, (10.55, 3.45), (7.8, merge_y), None, rad=-0.12)
    _arrow(ax, (2.4, 1.15), (4.0, merge_y), None, rad=0.08)
    _arrow(ax, (8.45, 1.15), (7.0, merge_y), None, rad=-0.08)

    ax.text(
        6, -0.35,
        r"$P_{\mathrm{inv}}$: nominal inverter discharge cap (battery.inverter_power_kw); "
        r"$P_{\mathrm{ess}}$: essential_discharge_kw (often 0); "
        r"$\mathrm{SoC}_{\mathrm{res}}$: soc_reserve in experiment_config.json (controller).",
        ha="center", va="top", fontsize=7, color="#333",
    )

    fig.suptitle(
        "Figure 4.6 — Smart policy state machine: Conservation vs standard, peak vs off-peak, reserve discharge cap",
        fontsize=11, y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    out = OUT_DIR / "figure_4_6_smart_policy_hourly_flowchart.png"
    figure_smart_policy_flowchart(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
