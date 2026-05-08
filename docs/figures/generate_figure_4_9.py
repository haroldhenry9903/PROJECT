"""
Thesis figures for Section 4.9 (Testing and verification).

Generates:
  - figure_4_12_run_all_verification_pipeline.png  — automated checks in run_all.py
  - figure_4_13_results_bundle_artefacts.png     — reproducible results folder layout
  - figure_4_14_phase1_data_validation_checklist.png — C1–C4 checklist (§4.9.1)

Copies batch evaluation plots from the newest results/.../figures/ run (if present):
  - figure_4_15_system_eval_total_cost_comparison.png
  - figure_4_16_system_eval_daily_savings_hist.png
  - figure_4_17_system_eval_daily_peak_import_reduction_hist.png
  - figure_4_18_system_eval_soc_distribution.png

§4.9.3: cite existing Tab 4 exports (e.g. figure_4_5_forecast_diagnostics_tab4_window.png or
figure_4_11_tab4_forecast_metrics.png) — no duplicate here.

Run from repository root:
  python docs/figures/generate_figure_4_9.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results"


def _latest_run_with_figures() -> Path | None:
    if not RESULTS_ROOT.is_dir():
        return None
    runs = [p for p in RESULTS_ROOT.iterdir() if p.is_dir() and (p / "figures" / "total_cost_comparison.png").is_file()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.name)
    return runs[-1]


def figure_verification_pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.8), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    def box(xy, w, h, title, body, fc="#eef3f8"):
        x, y = xy
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.0, facecolor=fc, edgecolor="#222",
            )
        )
        ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.1, body, ha="center", va="center", fontsize=7.3, linespacing=1.35)

    def arrow(p0, p1):
        ax.add_patch(
            FancyArrowPatch(
                p0, p1, arrowstyle="-|>", mutation_scale=16, linewidth=1.1, color="#111",
                connectionstyle="arc3,rad=0",
            )
        )

    ax.text(
        5, 10.55,
        "Figure 4.12 — Automated verification chain in run_all.py (fail-fast before metrics)",
        ha="center", va="top", fontsize=11, fontweight="bold",
    )

    box(
        (0.6, 8.85), 8.8, 1.05,
        "Inputs",
        "Load baseline & smart CSVs → copy to results/…/\n"
        "Normalize DatetimeIndex to timezone-naive\n"
        "Guard: baseline.index.tz must match smart.index.tz",
        "#e8f4ea",
    )
    arrow((5, 8.85), (5, 8.35))

    box(
        (0.6, 6.55), 8.8, 1.75,
        "(1) validate_timeseries",
        "Required columns: SoC, Grid_Import_kW, Hourly_Cost_RM\n"
        "Index: DatetimeIndex, tz-naive\n"
        "SoC ∈ [soc_min − 1e−9, 1 + 1e−9]\n"
        "Grid_Import_kW ≥ −1e−9 ; Hourly_Cost_RM ≥ −1e−9\n"
        "Timestep: successive Δt = 1 hour exactly",
        "#fff4e0",
    )
    arrow((5, 6.55), (5, 6.05))

    box(
        (0.6, 4.25), 8.8, 1.75,
        "(2) validate_energy_balance",
        r"$P_{\mathrm{net}} = P_{\mathrm{load}} - P_{\mathrm{pv}}$  (Target_Power_kW)"
        "\n"
        r"$P_{\mathrm{grid,unc}} = P_{\mathrm{net}} - P_{\mathrm{batt}}$"
        "\n"
        r"expected import = $\max(0, P_{\mathrm{grid,unc}})$"
        "\n"
        r"max |expected − stored Grid_Import_kW| ≤ $10^{-6}$ kW",
        "#e3f0ff",
    )
    arrow((5, 4.25), (5, 3.75))

    box(
        (0.6, 1.95), 8.8, 1.55,
        "(3) validate_tariff_cost_identity",
        "Spot rows: hour ∈ {peak_start−1, peak_start, peak_end−1, peak_end} (mod 24)\n"
        "Check Hourly_Cost_RM ≈ Grid_Import_kW × λ(hour)\n"
        "Default tol: max abs error ≤ 1e−5 RM\n"
        "Embeds into metrics.json → _meta.validation",
        "#fdeaea",
    )
    arrow((5, 1.95), (5, 1.45))

    box(
        (0.6, 0.35), 8.8, 0.95,
        "Then",
        "compute_daily_results → compute_metrics → metrics.json\n"
        "save_figures → figures/*.png (+ daily_results.csv, config_snapshot.json)",
        "#f5f5f5",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_results_bundle(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(5, 9.55, "Figure 4.13 — Reproducible evidence bundle (results/YYYYMMDD_HHMM/)", ha="center", fontsize=11, fontweight="bold")

    root = FancyBboxPatch((1.0, 1.2), 8.0, 7.6, boxstyle="round,pad=0.04", facecolor="#f0f4f8", edgecolor="#333", linewidth=1.2)
    ax.add_patch(root)
    ax.text(5, 8.55, "results / RUN_STAMP /", ha="center", fontsize=10, fontweight="bold")

    y = 7.85
    for name, note in [
        ("config_snapshot.json", "Frozen copy of experiment JSON"),
        ("baseline_simulation_results.csv", "Continuous hourly baseline"),
        ("smart_simulation_results.csv", "Continuous hourly smart"),
        ("daily_results.csv", "Daily-reset per-day table"),
        ("metrics.json", "KPIs + _meta definitions + validation summary"),
        ("figures/", "total_cost_comparison, daily_* hist, soc_distribution"),
    ]:
        ax.add_patch(FancyBboxPatch((1.4, y), 7.2, 0.62, boxstyle="round,pad=0.02", facecolor="white", edgecolor="#666"))
        ax.text(1.55, y + 0.31, f"• {name}", ha="left", va="center", fontsize=8.5, family="monospace")
        ax.text(4.8, y + 0.31, note, ha="left", va="center", fontsize=7.5, color="#444")
        y -= 0.78

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_phase1_checklist(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "Figure 4.14 — Phase 1 data integrity checks (phase1a_data_cleaning.py)", ha="center", fontsize=11, fontweight="bold")

    rows = [
        ("C1 — Missingness", "Post-clean missing count = 0 (numeric weather columns)."),
        ("C2 — Hour grid", "Resample 1H + asfreq; uniform hourly stepping for downstream simulation."),
        ("C3 — Non-negative radiation", "precip, solarradiation, ghiradiation, dniradiation, difradiation clipped to ≥ 0."),
        ("C4 — EDA (optional)", "Qualitative plots under eda_metrics/ (correlation, GHI, temperature trends)."),
    ]
    y = 0.78
    for code, desc in rows:
        ax.text(0.05, y, code, ha="left", va="top", fontsize=9, fontweight="bold")
        ax.text(0.05, y - 0.06, desc, ha="left", va="top", fontsize=8, color="#333", wrap=True)
        y -= 0.2

    ax.text(0.5, 0.06, "C1–C3: hard integrity targets. C4: narrative support, not an automated assertion in run_all.py.", ha="center", fontsize=7.5, style="italic", color="#555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def copy_eval_figures(run: Path) -> list[tuple[str, str]]:
    src_dir = run / "figures"
    mapping = [
        ("total_cost_comparison.png", "figure_4_15_system_eval_total_cost_comparison.png"),
        ("daily_savings_hist.png", "figure_4_16_system_eval_daily_savings_hist.png"),
        ("daily_peak_import_reduction_hist.png", "figure_4_17_system_eval_daily_peak_import_reduction_hist.png"),
        ("soc_distribution.png", "figure_4_18_system_eval_soc_distribution.png"),
    ]
    copied = []
    for src_name, dest_name in mapping:
        s = src_dir / src_name
        if s.is_file():
            shutil.copy2(s, OUT_DIR / dest_name)
            copied.append((src_name, dest_name))
    return copied


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    figure_verification_pipeline(OUT_DIR / "figure_4_12_run_all_verification_pipeline.png")
    figure_results_bundle(OUT_DIR / "figure_4_13_results_bundle_artefacts.png")
    figure_phase1_checklist(OUT_DIR / "figure_4_14_phase1_data_validation_checklist.png")

    run = _latest_run_with_figures()
    if run:
        copied = copy_eval_figures(run)
        print(f"Copied evaluation PNGs from: {run.name}")
        for a, b in copied:
            print(f"  {a} -> {b}")
        if len(copied) < 4:
            print("Warning: not all four evaluation PNGs were found in that run's figures/ folder.")
    else:
        print("No results/*/figures/total_cost_comparison.png found — skipped figure_4_15–4_18 copies.")
        print("Run: python run_all.py   then re-run this script.")

    print("Wrote:")
    print("  figure_4_12_run_all_verification_pipeline.png")
    print("  figure_4_13_results_bundle_artefacts.png")
    print("  figure_4_14_phase1_data_validation_checklist.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
