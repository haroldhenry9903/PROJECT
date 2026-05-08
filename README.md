# AI-Driven Solar Battery Optimizer — Sarawak Grid-Tied Home Simulation

This repository implements a **deterministic, hour-by-hour simulation** of a **grid-tied residential system** in **Sarawak (Kuching)** with **solar PV**, a **home battery**, and a **time-of-use (ToU) electricity tariff**. It compares:

| Strategy | Idea |
|----------|------|
| **Baseline** | Reactive self-consumption: the battery always tries to cover net load at each hour. |
| **Smart (AI-assisted)** | Uses a **Random Forest** forecast of next-day PV energy to trigger **Conservation Mode** and a **peak-reserve** discharge policy on “low solar” days. |

The physics (battery limits, efficiency, grid import clipping) are shared; only the **control policy** differs. Economic analysis uses **grid imports only** (no feed-in tariff). See `ASSUMPTIONS.md` for the full list of modelling assumptions.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `experiment_config.json` | **Single source of truth** for paths, battery, tariff, controller knobs, seeds, pipeline flags, and evaluation metadata. |
| `core_battery_model.py` | Battery **digital twin**: `Battery` class and `update_soc(p_request_kw)`. |
| `phase1a_data_cleaning.py` | Raw weather CSV → cleaned hourly series (`data/processed/..._cleaned.csv`). |
| `phase1b_physics_synthesis.py` | Cleaned data + **pvlib** model chain → `Target_Power_kW` (`data/processed/sarawak_weather_processed.csv`). |
| `phase1_data_prep.py` | **Legacy** monolithic prep (optional); see its docstring. |
| `phase2_baseline_sim.py` | Baseline simulation → `baseline_simulation_results.csv` (or path from config). |
| `phase3_ai_training.py` | Train Random Forest PV forecaster → `models/rf_forecast_model.joblib`. |
| `phase4_smart_optimizer.py` | Smart policy simulation → `smart_simulation_results.csv`. |
| `run_all.py` | Orchestrates phases 2→4, validates outputs, writes **timestamped** `results/YYYYMMDD_HHMM/`. |
| `phase5_dashboard.py` | **Streamlit** UI: explain, live day, yearly comparison, model metrics. |
| `repo_paths.py` | Resolves paths relative to the repo root. |
| `data/raw/` | Source CSVs (e.g. `sarawak_weather20_25.csv`; legacy `sarawak_weather_data.csv` for old prep). |
| `data/processed/` | Cleaned + synthesized datasets used by training and simulations. |
| `models/` | Trained forecaster (`rf_forecast_model.joblib`). |
| `results/YYYYMMDD_HHMM/` | Full artifact bundle from `run_all.py` (CSVs, `metrics.json`, figures, `config_snapshot.json`). |
| `eda_metrics/` | Optional EDA plots from Phase 1a (if generated). |
| `tools/` | Utilities (e.g. PDF text extraction). |

Phases 2–5 read default paths from `experiment_config.json` under `paths`.

---

## Prerequisites

- **Python 3.10+** recommended (match the environment you use to train/save `joblib` models).
- **Git** (optional) for versioning thesis experiments.

Install dependencies from the repo root:

```powershell
python -m pip install -r requirements.txt
```

Core scientific stack includes **pandas**, **numpy**, **scikit-learn**, **matplotlib**, **pvlib** (Phase 1b), **streamlit** (dashboard). Keep **scikit-learn** aligned with `requirements.txt` when reloading saved models to avoid version warnings.

---

## Quick start (thesis reproducibility)

### One-command pipeline (recommended)

Creates `results/YYYYMMDD_HHMM/` containing:

| Artifact | Description |
|----------|-------------|
| `baseline_simulation_results.csv` | Full continuous baseline run (copied from project root output paths in config). |
| `smart_simulation_results.csv` | Full continuous smart run. |
| `daily_results.csv` | **Per-day** re-simulation with shared initial SoC (fair day-isolated comparison). |
| `metrics.json` | KPIs: continuous CSV sums **and** daily-reset statistics; includes `_meta` definitions. |
| `figures/*.png` | Cost comparison, daily savings / peak-reduction histograms, SoC distributions. |
| `config_snapshot.json` | Copy of the config used for that run. |

```powershell
python run_all.py
```

Use a custom config:

```powershell
python run_all.py experiment_config.json
```

### Optional steps inside `run_all.py`

In `experiment_config.json`, under `pipeline`:

- `"run_phase1": true` — runs `phase1a_data_cleaning.py` then `phase1b_physics_synthesis.py` first (requires appropriate **raw** CSVs under `data/raw/`).
- `"run_phase3_train_model": true` — retrains the Random Forest before Phase 4.

Default is `false` for both (assumes processed data and an existing model).

### Dashboard

From the repository root:

```powershell
streamlit run phase5_dashboard.py
```

Enable **“View a saved results run”** and pick the matching `results/YYYYMMDD_HHMM/` folder so plots and KPIs align with `metrics.json` for that run.

---

## Manual pipeline (step by step)

### Phase 1 — Data: cleaning + PV synthesis

**Canonical split pipeline:**

```powershell
python phase1a_data_cleaning.py
python phase1b_physics_synthesis.py
```

Typical outputs:

- `data/processed/sarawak_weather_cleaned.csv` — hourly, gap-filled weather aligned to Kuching.
- `data/processed/sarawak_weather_processed.csv` — adds **`Target_Power_kW`** via **pvlib** `ModelChain` for a 5 kWp system near Kuching.

**Legacy:** `phase1_data_prep.py` for a single-script path from `data/raw/sarawak_weather_data.csv` (see script docstring).

### Phase 2 — Baseline simulation

```powershell
python phase2_baseline_sim.py
```

Supports `--config experiment_config.json`. Writes synthetic **load** (seeded), applies `Battery` to net power, applies **tariff** from config, exports baseline CSV.

### Phase 3 — Train PV forecaster

```powershell
python phase3_ai_training.py
python phase3_ai_training.py --config experiment_config.json
```

Uses processed CSV from `paths.phase1_processed_csv` and saves to `paths.rf_model_path`. **Features:** `ghiradiation`, `temp`, `humidity`, `precip`, `cloudcover`, `hour`, `month`. **Target:** `Target_Power_kW`. **Chronological split:** train before `2024-08-08`, test from `2024-08-08` onward. Tuning via `GridSearchCV` on Random Forest hyperparameters.

### Phase 4 — Smart optimizer

```powershell
python phase4_smart_optimizer.py
```

Loads baseline CSV (for consistent load), loads RF model, applies midnight forecast + Conservation Mode + peak-reserve policy.

### Phase 5 — Dashboard

```powershell
streamlit run phase5_dashboard.py
```

---

## Configuration reference (`experiment_config.json`)

| Section | Role |
|---------|------|
| `paths` | `phase1_processed_csv`, `baseline_results_csv`, `smart_results_csv`, `rf_model_path`. |
| `seed.numpy_seed` | Fixes synthetic load generation (deterministic reruns). |
| `battery` | `capacity_kwh`, `inverter_power_kw`, `soc_min`, `efficiency_rt` (round-trip; internally split √(η) per direction in `core_battery_model.py`). |
| `tariff` | `peak_start_hour`, `peak_end_hour`, `peak_rate_rm_per_kwh`, `offpeak_rate_rm_per_kwh` — **same** tariff for baseline and smart cost columns. |
| `controller` | `e_crit_kwh` (low-solar threshold), `soc_reserve`, `essential_discharge_kw`. |
| `scenario` | `pv_scale`, `load_scale` (dashboard / daily re-sim), `initial_soc_live`, `initial_soc_daily_reset`. |
| `evaluation` | `headline_primary`, `daily_reset_soc_fraction_key` — documented in `metrics.json._meta` and copied to `config_snapshot.json`. |
| `pipeline` | `run_phase1`, `run_phase3_train_model`. |

---

## Understanding `metrics.json`

`run_all.py` reports **two complementary views**:

1. **Continuous CSV (headline annual bill style)** — Sums `Hourly_Cost_RM`, `Grid_Import_kW`, `Load_kW` on the **full** timelines (SoC **carries** across days as in the actual year-long runs).
2. **Daily reset** — Replays each calendar day in isolation with the **same** starting SoC; uses the saved **`Conservation_Mode_Active`** flag per day for the smart branch. Answers “on a typical isolated day, does AI win?”

Legacy top-level keys remain for backward compatibility; **`by_evaluation_mode`** and **`_meta`** spell out definitions and the **gap** between continuous total savings and the sum of daily savings (expected non-zero; see `_meta.gap_explainer`).

For SSR, total import, and cost under ToU, read `ASSUMPTIONS.md` — cost can improve when peak imports fall even if total kWh imported changes modestly.

---

## Validations in `run_all.py`

After simulations, the pipeline checks:

- Required columns; **tz-naive** hourly `DatetimeIndex`.
- SoC ∈ [soc_min, 1.0]; non-negative grid import and hourly cost.
- **Hourly** step continuity (1 h).
- **Energy balance** vs `Batt_Action_kW` and PV/load columns.
- **Tariff identity** spot-check: `Hourly_Cost_RM` ≈ `Grid_Import_kW` × rate(hour).

Failures raise errors so broken runs do not silently become thesis numbers.

---

## Key assumptions (short list)

- **No export revenue** — costing uses imports only.
- **Synthetic load** — shaped residential profile + noise; deterministic with fixed seed.
- **Tz-naive** hourly timestamps in simulation CSVs (pipeline normalizes where needed).
- **Tariff** — single definition in config for both strategies.

Full detail: **`ASSUMPTIONS.md`**.

---

## Documentation

| File | Content |
|------|---------|
| `PROJECT_EXPLANATION_DETAILED.md` | Long-form methodology, algorithms, dashboard tabs, thesis defence notes. |
| `ASSUMPTIONS.md` | Conventions, evaluation modes, metrics interpretation. |

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| `joblib` / sklearn warnings | Match package versions to `requirements.txt`. |
| Tab 2 vs Tab 3 mismatch | Align **initial SoC** (live slider vs `initial_soc_daily_reset`). |
| Phase 1 fails | Raw CSV path/columns; `phase1a` resampling and tz handling. |
| Empty or missing `results/` | Run `python run_all.py` from repo root; confirm Phase 2/4 write paths in config. |

---

## Repo hygiene (what is tracked vs generated)

To keep the repository lightweight and reviewable, this repo tracks **source + configs + documentation**, and ignores large/generated artifacts.

**Tracked (committed):**
- Python source: `phase*.py`, `core_battery_model.py`, `run_all.py`, `repo_paths.py`
- Configuration: `experiment_config.json`, `requirements.txt`
- Documentation: `README.md`, `ASSUMPTIONS.md`, `PROJECT_EXPLANATION_DETAILED.md`
- Utilities: `tools/` (e.g. `tools/pdf_to_text.py`)

**Not tracked (generated / large):**
- Run artifacts: `results/YYYYMMDD_HHMM/` (CSVs, `metrics.json`, figures, `config_snapshot.json`)
- Root outputs: `baseline_simulation_results.csv`, `smart_simulation_results.csv`
- Trained models: `models/*.joblib`
- Generated plots: `docs/figures/*.png`, `eda_metrics/*.png`
- Datasets: `data/raw/`, `data/processed/`

**How to reproduce (after you have the dataset):**
- Install deps: `python -m pip install -r requirements.txt`
- Run pipeline: `python run_all.py` (writes `results/YYYYMMDD_HHMM/`)
- Launch dashboard: `streamlit run phase5_dashboard.py` and load a saved `results/YYYYMMDD_HHMM/` folder

**Submission bundle (recommended for examiners):**
- Zip one complete run folder: `results/YYYYMMDD_HHMM/`
- Optionally include `models/` and the required `data/` files used for that run

---

## Citation / reproducibility checklist

- [ ] Fixed `experiment_config.json` (or archived `config_snapshot.json` from `results/`).
- [ ] Same `seed.numpy_seed` for comparable load traces.
- [ ] Record Python and `scikit-learn` versions used when training the RF model.
- [ ] For figures, note whether KPIs come from **continuous** or **daily-reset** metrics.
