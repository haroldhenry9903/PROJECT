# Project explanation (detailed)

This document supplements **`README.md`** with methodology, equations, file-level behaviour, and guidance for writing up or defending the work in a thesis or report. **`ASSUMPTIONS.md`** remains the canonical list of modelling assumptions and metrics interpretation.

---

## 1. Research question (what is being tested?)

The project asks whether **forecast-informed battery control** can reduce **electricity cost** under a **ToU tariff** compared to **reactive self-consumption**, for a **grid-tied home** in **Sarawak** using **historical weather** to drive PV (and a **physics-based** PV model for the “true” `Target_Power_kW`).

Important scope boundaries:

- **Simulation**, not a live deployment: load is **synthetic**; PV comes from processed weather + **pvlib** (Phase 1b) or the same columns consumed by Phase 2.
- **AI does not change physics**: the same `Battery` class and grid-import accounting apply to baseline and smart runs; the smart run adds a **policy layer** (discharge power limits and a daily Conservation Mode flag).
- **Economics**: only **grid imports** are billed; **export/curtailment has zero value**.

---

## 2. Mental model: one hour of simulation

At each hour \(t\):

1. **Inputs** — Load \(P_{load}(t)\) (synthetic in Phase 2/4) and PV \(P_{pv}(t)\) (`Target_Power_kW` from the processed dataset).
2. **Net demand** — \(P_{net}(t) = P_{load}(t) - P_{pv}(t)\).
3. **Battery** — `Battery.update_soc(p_request_kw=p_net)` returns \(P_{batt}(t)\) subject to power limits, SoC bounds, and efficiency (see §4).
4. **Grid** — \(P_{grid}(t) = P_{net}(t) - P_{batt}(t)\); **import** for costing is \(P_{import}(t) = \max(0, P_{grid}(t))\).
5. **Cost** — \(Cost(t) = P_{import}(t) \times \text{TariffRate}(\text{hour}(t))\) using **`experiment_config.json`** `tariff` for **both** baseline and smart.
6. **Outputs** — Store SoC, battery action, grid import, hourly cost, and (for smart) conservation flags.

All dashboard aggregates and `metrics.json` KPIs trace back to these hourly rows.

---

## 3. Terms (plain language)

| Term | Meaning |
|------|---------|
| **kW** | Instantaneous power. |
| **kWh** | Energy over an interval; with **1 h** steps, kW sums approximate kWh for that interval. |
| **SoC** | State of charge, fraction in \([soc_{min}, 1]\). |
| **Peak window** | Hours where `peak_rate_rm_per_kwh` applies: `peak_start_hour` through `peak_end_hour` exclusive (see `tariff` in config). |
| **Grid import** | Purchased energy from the grid (the quantity that generates cost). |
| **Conservation Mode** | A **day-level** mode (decided at midnight) that enables the peak-reserve discharge restriction during off-peak on “low predicted solar” days. |

---

## 4. Physics engine: `core_battery_model.py`

### 4.1 Sign convention

- **`p_request_kw > 0`** — Net **deficit**: house needs power; battery should **discharge** (positive \(P_{batt}\) returned when helping).
- **`p_request_kw < 0`** — Net **surplus**: PV exceeds load; battery **charges** (negative \(P_{batt}\)).

### 4.2 Constraints

Discharge is limited by **max discharge power**, **SoC above `soc_min`**, and efficiency. Charge is limited by **max charge power**, **SoC below 1**, and efficiency. Surplus not absorbed is implicitly **curtailed/exported** with **no income**.

### 4.3 Round-trip efficiency

If round-trip efficiency is \(\eta_{rt}\) (e.g. 0.95), the implementation uses **per-direction** efficiency \(\sqrt{\eta_{rt}}\) so that charge and discharge losses combine consistently over a full cycle.

---

## 5. Data pipeline (Phase 1)

### 5.1 Phase 1a — `phase1a_data_cleaning.py`

- Reads a **raw** weather CSV (paths resolved via `repo_paths`).
- Parses `datetime`, aligns to a **strict hourly** grid, handles missing values (forward-fill / interpolation as implemented).
- May localize or resample so the timeline is continuous at **1 H** resolution.
- Output: **`sarawak_weather_cleaned.csv`** (or the filename configured in the script) with columns needed for Phase 1b (e.g. GHI/DNI/DHI, temperature).

### 5.2 Phase 1b — `phase1b_physics_synthesis.py`

- Loads the cleaned CSV.
- Builds a **pvlib** `Location` for **Kuching** and a **PVSystem** (e.g. 5 kWp, tilt/azimuth as in code).
- Runs **`ModelChain`** to obtain AC power, converts to **`Target_Power_kW`**, clips negatives.
- Strips timezone for downstream CSV compatibility where applicable.
- Output: **`data/processed/sarawak_weather_processed.csv`** — the main input for Phase 2 and Phase 3.

### 5.3 Legacy — `phase1_data_prep.py`

Single-file alternative documented in its docstring; use when reproducing an older workflow from `sarawak_weather_data.csv`.

---

## 6. Phase 2 — Baseline (`phase2_baseline_sim.py`)

- Loads **`paths.phase1_processed_csv`**.
- Generates **deterministic synthetic residential load** using `seed.numpy_seed`.
- Each hour: compute \(P_{net}\), call `Battery.update_soc`, compute **`Grid_Import_kW`** and **`Hourly_Cost_RM`** from config tariff.
- **No lookahead**: the battery never “plans” for tomorrow.

Typical CSV columns include: `Load_kW`, `Target_Power_kW`, `SoC`, `Batt_Action_kW`, `Grid_Import_kW`, `Hourly_Cost_RM`, `datetime`.

---

## 7. Phase 3 — Random Forest forecaster (`phase3_ai_training.py`)

### 7.1 Purpose

Provide **hourly PV predictions** from **weather + calendar features** so Phase 4 can estimate **next-day total PV energy** and classify **low-solar days**.

### 7.2 Features and target

- **Features:** `ghiradiation`, `temp`, `humidity`, `precip`, `cloudcover`, `hour`, `month`.
- **Target:** `Target_Power_kW`.

### 7.3 Train / test split

- **Chronological** (no random shuffle): training ends before **`2024-08-08`**; test is **on or after** that date — appropriate for time-series generalisation claims.

### 7.4 Model

- **RandomForestRegressor** with **`GridSearchCV`** (`n_estimators`, `max_depth`; scoring via negative RMSE; `cv=3`).
- Best estimator saved with **`joblib`** to **`paths.rf_model_path`** (default `models/rf_forecast_model.joblib`).

### 7.5 Thesis framing

Perfect hourly PV forecasts are **not** required: Conservation Mode depends on comparing **predicted daily energy** to **`e_crit_kwh`**. The relevant question is whether the classifier is **stable enough** around that threshold (Tab 4 shows RMSE/R² for transparency).

---

## 8. Phase 4 — Smart optimizer (`phase4_smart_optimizer.py`)

### 8.1 Inputs

- **Baseline CSV** — ensures **identical load** and PV trace between strategies.
- **Trained RF model** — hourly predictions for the next 24 h when deciding the day mode.

### 8.2 Midnight decision

At **hour 0**, using the next 24 feature rows:

1. Predict \(\hat{P}_{pv}(h)\) for each hour.
2. \(E_{pred} \approx \sum_h \hat{P}_{pv}(h)\) (kW × 1 h ≈ kWh).
3. If \(E_{pred} < E_{crit}\) (from `controller.e_crit_kwh`) ⇒ **Conservation Mode** for that calendar day (stored e.g. as **`Conservation_Mode_Active`**).

### 8.3 Hourly policy (Conservation days)

Let peak hours be \([H_{peak,start}, H_{peak,end})\) from `tariff`.

- **Peak:** restore nominal **`max_discharge_kw`** (inverter limit) so the battery can reduce expensive imports.
- **Off-peak:** if **`SoC ≤ soc_reserve`**, cap discharge to **`essential_discharge_kw`** (often `0` to **hold** energy for peak).

Non-conservation days: baseline-like discharge cap (full inverter power when discharging).

### 8.4 Costing

Same **`Hourly_Cost_RM`** definition as Phase 2 (config tariff).

---

## 9. Orchestration — `run_all.py`

### 9.1 Order of operations

1. Optional Phase 1a + 1b if `pipeline.run_phase1` is true.
2. **Phase 2** baseline (subprocess with `--config`).
3. Optional **Phase 3** retrain if `pipeline.run_phase3_train_model` is true.
4. **Phase 4** smart.
5. Copy baseline/smart CSVs into **`results/YYYYMMDD_HHMM/`**.
6. **Validate** series (§9.2).
7. Build **`daily_results.csv`** via `compute_daily_results` (§10).
8. **`metrics.json`** via `compute_metrics` (§11).
9. **`save_figures`** → `figures/*.png`.
10. Write **`config_snapshot.json`**.

### 9.2 Validations

- SoC bounds; non-negative import and cost; **1 h** regular timestamps.
- **Energy balance** identity tying `Load_kW`, `Target_Power_kW`, `Batt_Action_kW`, `Grid_Import_kW`.
- **Tariff spot-check** on representative hours for both CSVs.

---

## 10. Daily-reset evaluation (`daily_results.csv`)

**Motivation:** In the **continuous** year-long CSVs, SoC **carries over** across midnight. That is realistic for an annual bill, but it confuses **per-day** questions (“Did AI help *on that day*?”) and interacts badly with filters like “Conservation days only” (spillover from previous days).

**Daily reset:** For each calendar day, re-simulate **baseline** and **smart** **starting from the same** `scenario.initial_soc_daily_reset`, while the **smart** branch still uses the **saved** `Conservation_Mode_Active` for that day from Phase 4.

**Columns (conceptually):** per-day costs, peak-window imports, savings, peak import reduction, conservation flag, optional min SoC statistics — see actual CSV headers after a run.

---

## 11. `metrics.json` structure (how to read it)

- **Legacy top-level keys** — e.g. total costs, total savings, SSR, peak import reduction, daily savings mean/median/min/max — preserved for older notebooks/scripts.
- **`by_evaluation_mode`** — separates:
  - **`continuous_csv`** — sums from full timelines (`Hourly_Cost_RM`, etc.); matches **stateful** annual behaviour.
  - **`daily_reset`** — statistics from **`daily_results.csv`** isolated days.
- **`_meta`** — `schema_version`, `headline_primary`, **definitions** of each mode, **`daily_reset_soc_init`**, **`gap_continuous_total_savings_rm_minus_sum_daily_savings_rm`**, and **`gap_explainer`**.

A **non-zero gap** between continuous total savings and the sum of daily savings is **expected**: the two modes answer **different counterfactual questions** (see `ASSUMPTIONS.md`).

**SSR / total import vs money:** Under ToU, the bill is driven by **when** you import. The smart policy may trade **slightly more** off-peak energy for **less** peak energy — cost can fall even if SSR or total kWh imported does not improve.

---

## 12. Dashboard — `phase5_dashboard.py`

### 12.1 Sidebar and data sources

- Loads **`experiment_config.json`**.
- Can load baseline/smart CSVs from the **project root paths** or from a selected **`results/YYYYMMDD_HHMM/`** run (recommended for consistency with `metrics.json`).

Parameters (battery, tariff, `pv_scale` / `load_scale`, controller knobs) drive **on-the-fly** simulations in interactive tabs.

### 12.2 Tab 1 — Conceptual explanation

Narrative layer: net load, battery, tariff, baseline vs AI policy — suitable for non-specialist readers.

### 12.3 Tab 2 — Live one-day simulation

- Runs **baseline** and **AI** for a **selected date** over 24 h.
- **Initial SoC** is user-controlled — **must match** `initial_soc_daily_reset` if you want consistency with Tab 3 daily-reset logic for that comparison.
- Metrics: **cost per day**, **savings**, **peak import reduction** for that day.

**Plots:** load vs PV, SoC trajectories, grid import, hourly cost — use to **show mechanism** (e.g. higher SoC entering peak on conservation days).

### 12.4 Tab 3 — Yearly / aggregate comparison

- Aggregation: hourly / daily / monthly views as implemented.
- **Daily-reset toggle** — aligns interpretation with `daily_results.csv` / `metrics.json` daily-reset section.
- **Conservation days filter** — interpret with care in **continuous** mode (spillover); cleaner with **daily reset**.
- **Top savings days** — good candidates to **replay in Tab 2** with matched SoC.

### 12.5 Tab 4 — Model metrics

Shows **RMSE** and **R²** for the RF on the held-out chronological test split; residual and scatter plots if implemented — supports the claim that the forecaster is **fit for purpose** for a **threshold-based** policy, not necessarily perfect point forecasting.

---

## 13. Defending results in viva / examination

Strong lines of reasoning:

1. **Fair tariff** — Same `tariff` block for baseline and smart; spot-checks in `run_all.py`.
2. **Fair load and PV** — Smart uses baseline CSV for load; same weather/PV trace.
3. **Two evaluation modes** — Report **continuous** headline savings for “annual bill” intuition; use **daily reset** for “typical day” and pedagogy.
4. **Peak-focused narrative** — Under ToU, **peak import reduction** is often the clearest mechanistic story alongside **total savings (RM)**.

Common examiner questions:

| Question | Answer sketch |
|----------|----------------|
| Why is Tab 2 ≠ Tab 3? | Different **initial SoC** or **mode** (single day vs aggregate / daily reset). |
| Why filter conservation days only? | In **continuous** runs, spillover distorts day labels; prefer **daily reset** for isolated day interpretation. |
| Is synthetic load a weakness? | Yes — frame as **controlled** scenario; cite seed and methodology. |
| Why RF not deep learning? | Interpretability, data size, baseline strength; can be future work. |

---

## 14. Recommended workflow (thesis run)

1. Pin **`experiment_config.json`** (battery, tariff, controller, seed).
2. `python run_all.py` → archive **`results/YYYYMMDD_HHMM/`** (includes **`config_snapshot.json`**).
3. `streamlit run phase5_dashboard.py` → load that results folder.
4. Export figures from `results/.../figures/` and cite **`metrics.json`** with explicit **continuous vs daily-reset** wording.

---

## 15. Limitations and extensions (discussion section prompts)

- **Load** is synthetic; real households differ (EV, AC, occupancy).
- **Tariff** is simplified (single peak block; real utilities may have tiers, demand charges, or blocks).
- **No distribution network constraints**, **no uncertainty quantification** on forecasts in the controller (single point forecast).
- **Classification robustness** near **`e_crit_kwh`** could be studied with sensitivity analysis.
- **Alternative policies**: MPC, stochastic DP, reinforcement learning — compare against this **transparent** rule-based AI baseline.

This document is intentionally detailed; trim sections when pasting into a thesis to avoid duplication with your own narrative chapters.
