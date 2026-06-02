# Project explanation (detailed)

This document supplements **[`../README.md`](../README.md)** with methodology, equations, file-level behaviour, and guidance for writing up or defending the work in a thesis or report. **[`ASSUMPTIONS.md`](ASSUMPTIONS.md)** (sibling of this file) remains the canonical list of modelling assumptions and metrics interpretation.

---

## 1. Research question (what is being tested?)

The project asks whether **forecast-informed battery control** can reduce **electricity cost** under a **ToU tariff** compared to **reactive self-consumption**, for a **grid-tied home** in each of **five Sarawak regions** (Kuching, Bintulu, Kapit, Miri, Sri Aman) — and whether the choice of **forecasting algorithm** (Random Forest vs SVR) materially changes the answer. PV in every region is driven by **historical weather** and a **physics-based** PV model (`Target_Power_kW`) instantiated per region with its own lat/lon.

Important scope boundaries:

- **Simulation**, not a live deployment: load is **synthetic**; PV comes from processed weather + **pvlib** (Phase 1b) or the same columns consumed by Phase 2.
- **AI does not change physics**: the same `Battery` class and grid-import accounting apply to baseline and smart runs; the smart run adds a **policy layer** (discharge power limits and a daily Conservation Mode flag).
- **AI is a swap-in component**: the smart controller calls `model.predict(X)` on a 7-column feature window. Both an RF (`RandomForestRegressor`) and an SVM (`Pipeline(StandardScaler, SVR)`) are trained and reported per region; the controller is indifferent to which is plugged in. The thesis uses this to argue the savings stem from the **policy + ToU asymmetry**, not from a particular ML algorithm.
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

### 5.1 Phase 1 (current) — `phase1_split_multi_region.py`

- Reads the canonical multi-region CSV (`swkdataweather.csv`) listed in `paths.raw_multi_region_csv`.
- Parses `datetime`, aligns to a **strict hourly** grid per region, handles missing values (forward-fill / interpolation).
- For every region in `regions[]` writes `data/processed/<region>_weather_cleaned.csv` with the columns needed by Phase 1b (e.g. GHI/DNI/DHI, temperature, humidity, precip, cloudcover).
- Tested by `tests/test_multi_region_split.py` — 30 tests cover row counts, no-NaN guarantees, datetime monotonicity, and per-region column completeness; they run on every CI invocation.

### 5.2 Phase 1b — `phase1b_physics_synthesis.py`

- Loads the per-region cleaned CSV (`data/processed/<region>_weather_cleaned.csv`).
- Builds a **pvlib** `Location` parametrised by **the region's `lat`/`lon`** from `experiment_config.json` and a **PVSystem** (e.g. 5 kWp, tilt/azimuth as in code).
- Runs **`ModelChain`** to obtain AC power, converts to **`Target_Power_kW`**, clips negatives.
- Strips timezone for downstream CSV compatibility where applicable.
- Output: **`data/processed/<region>_weather_processed.csv`** for each region in `regions`. Phase 2 and Phase 3 read these by region name.

### 5.3 Legacy single-region scripts (archived)

- `archive/legacy/phase1a_data_cleaning.py` — the original single-region cleaner that operated on one weather CSV at a time. Superseded by `phase1_split_multi_region.py`.
- `archive/legacy/phase1_data_prep.py` — a monolithic single-file alternative documented in its docstring; use only when reproducing an older workflow from `sarawak_weather_data.csv`.

---

## 6. Phase 2 — Baseline (`phase2_baseline_sim.py`)

- Loads **`paths.phase1_processed_csv`**.
- Generates **deterministic synthetic residential load** using `seed.numpy_seed`.
- Each hour: compute \(P_{net}\), call `Battery.update_soc`, compute **`Grid_Import_kW`** and **`Hourly_Cost_RM`** from config tariff.
- **No lookahead**: the battery never “plans” for tomorrow.

Typical CSV columns include: `Load_kW`, `Target_Power_kW`, `SoC`, `Batt_Action_kW`, `Grid_Import_kW`, `Hourly_Cost_RM`, `datetime`.

---

## 7. Phase 3 — PV forecasters: Random Forest + SVM (`phase3_ai_training.py`)

### 7.1 Purpose

Provide **hourly PV predictions** from **weather + calendar features** so Phase 4 can estimate **next-day total PV energy** and classify **low-solar days**. Two algorithms are trained side-by-side so the thesis can answer the **algorithm-sensitivity** question: does the headline savings number change when you swap the forecaster?

### 7.2 Features and target

- **Features:** `ghiradiation`, `temp`, `humidity`, `precip`, `cloudcover`, `hour`, `month` (7 columns, identical for RF and SVM).
- **Target:** `Target_Power_kW`.

### 7.3 Train / test split + cross-validation

- **Chronological** held-out split (no random shuffle): training ends before **`2024-08-08`**; test is **on or after** that date — appropriate for time-series generalisation claims.
- **`sklearn.model_selection.TimeSeriesSplit(n_splits=5)`** inside `GridSearchCV` for every hyperparameter selection — strictly forward-only folds, never validating on past hours after training on the future. The motivation is documented in `docs/VALIDATION.md` §2.

### 7.4 Models

- **Random Forest** — `RandomForestRegressor(random_state=42)` with a wide `GridSearchCV` grid over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`. Negative RMSE scoring. Best estimator saved to `models/rf_forecast_model_<region>.joblib`.
- **SVM/SVR** — `Pipeline([StandardScaler, SVR(kernel='rbf')])` with a grid over `C`, `gamma`, `epsilon`. To keep training tractable, grid search runs on a stratified `phase3.svm_subsample` row sample (default `8000`); the best estimator is then **refit on the full training set** before serialisation. Saved to `models/svm_forecast_model_<region>.joblib`.

Per-region artefacts (one set per region in `regions[]`):

- `models/cv_report_<region>.json` — `cv_rmse_mean`, `cv_rmse_std`, per-fold scores, best params, `test_{rmse,mae,r2,mape}` on the held-out window — for **both** algorithms.
- `models/cv_results_{rf,svm}_<region>.csv` — the full `GridSearchCV.cv_results_` table for auditing.
- `figures/training/<region>/{rf,svm}_fold_scores.png` — per-fold RMSE bar chart with the mean as a dashed line.
- `figures/training/<region>/{rf,svm}_learning_curve.png` — train vs validation RMSE versus training-set size (helps argue under-/over-fitting).
- `figures/training/<region>/{rf,svm}_validation_curve.png` — RMSE vs the headline hyperparameter (`max_depth` for RF, `C` for SVM).
- `figures/training/<region>/compare_<region>.png` — side-by-side test-window scoreboard for RF vs SVM.

### 7.5 Thesis framing

Perfect hourly PV forecasts are **not** required: Conservation Mode depends on comparing **predicted daily energy** to **`e_crit_kwh`**. The relevant question is whether the classifier is **stable enough** around that threshold (Tab 4 shows RMSE/R² for transparency). Reporting **both** algorithms also lets the thesis demonstrate that the savings story is **algorithm-robust** rather than RF-specific — `summary.csv` from `run_all.py` contains one row per `(region, algo)` cell and `verdict.txt` calls out which algorithm wins per region and whether the gap is statistically significant.

### 7.6 Closed loophole — perfect foresight vs realistic forecast error

#### 7.6.1 The loophole that previously existed

`Target_Power_kW` is computed in `phase1b_physics_synthesis.py` by running
**pvlib** on `ghiradiation`, `dniradiation`, `difradiation`, `temp`,
`wind_speed`, plus solar position from `latitude/longitude/hour/month`. The RF
in Phase 3 is then trained on a strict subset of those same columns
(`ghiradiation`, `temp`, `humidity`, `precip`, `cloudcover`, `hour`, `month`).

In the original Phase 4, every midnight the controller did:

```python
X_24h = df.iloc[i : i + 24][features]   # ground-truth weather!
predictions = rf_model.predict(X_24h)
```

Because `df` contained the **measured** weather of the next 24 hours, the RF
was effectively asked to *invert a deterministic physics model on its true
inputs* — which it does very well. The smart controller therefore enjoyed
**perfect day-ahead knowledge** of tomorrow's PV. The reported savings were
an upper bound, not a realistic deployment number.

#### 7.6.2 The fix

`phase4_smart_optimizer.py` now exposes `apply_forecast_noise(...)`, called at
every midnight before `rf_model.predict(...)` whenever
`forecast.enabled = true` in `experiment_config.json`. The noise model is:

\[
  \begin{aligned}
    \tilde{G}(t)        &= \max\!\big(0,\; G(t)\,(1 + b_{ghi})\, e^{\,\varepsilon_g(t)}\big), \quad \varepsilon_g \sim \mathcal{N}(0, \sigma_{ghi}^2),\\
    \tilde{P}(t)        &= \max\!\big(0,\; P(t)\, e^{\,\varepsilon_p(t)}\big), \quad \varepsilon_p \sim \mathcal{N}(0, \sigma_{precip}^2),\\
    \tilde{T}(t)        &= T(t) + \mathcal{N}(0, \sigma_T^2),\\
    \tilde{H}(t)        &= \mathrm{clip}\!\big(H(t) + \mathcal{N}(0, \sigma_H^2),\, 0,\, 100\big),\\
    \tilde{C}(t)        &= \mathrm{clip}\!\big(C(t) + \mathcal{N}(0, \sigma_{cc}^2),\, 0,\, 100\big),
  \end{aligned}
\]

with calendar features `hour`, `month` left untouched. The RNG is seeded from
`forecast.seed` (precedence over `seed.forecast_seed` and the numpy seed).

#### 7.6.3 Sensitivity sweep (multi-seed, in-process)

`run_all.py` automatically re-runs Phase 4 across the configured grid
(`forecast.sweep_sigmas`, default
\(\sigma_{ghi} \in \{0.00, 0.10, 0.20, 0.30, 0.40\}\)) with
`forecast.sweep_seeds` (default `10`) seeded trials per σ. The sweep runs
**in-process** by importing `phase4_smart_optimizer.simulate(...)` directly,
so the baseline frame and the trained RF are loaded once for the whole
pipeline (the previous sub-process-per-trial approach did not scale beyond
single-seed runs).

Outputs:

- `results/<stamp>/forecast_sensitivity_raw.csv` — one row per `(σ, seed)`
  trial with the per-trial KPIs.
- `results/<stamp>/forecast_sensitivity.csv` — per-σ summary with mean
  values plus `*_std`, `*_min`, `*_max` siblings on the bottom-line KPIs
  (annual savings, peak-import reduction, conservation-day count, daily MAPE).
- `results/<stamp>/figures/forecast_sensitivity.png` — `plt.errorbar` of
  mean savings ±1 std vs σ, with a vertical dashed line at the active
  `forecast.ghi_noise_sigma`.
- `results/<stamp>/metrics.json._meta.forecast` — the active forecast block,
  echoed for the headline run.
- One new column `Forecast_Energy_kWh_Predicted` in
  `smart_simulation_results.csv` so the predicted-vs-actual daily PV gap is
  auditable per row.

**σ = 0.00** is the perfect-foresight upper bound and lets you reproduce the
old "AI vs baseline" headline. The other rows tell the honest story: how much
of those savings survive realistic day-ahead error, and the error bars
quantify how much variance is attributable to the noise draw versus the σ
level itself. With `sweep_seeds=1`, the summary CSV reproduces the previous
single-seed sweep exactly.

The dashboard surfaces these artifacts in two places:

- The **sidebar** gains a `Forecast realism (σ used for the headline)`
  expander showing `_meta.forecast` and the perfect-foresight gap.
- **Tab 4** renders the per-σ table and the errorbar figure inline.
- **Tab 2's RF-fallback path** was also patched to call
  `apply_forecast_noise(...)` before `rf_model.predict(...)` (with a date-
  deterministic RNG seed), removing a residual mini-loophole in the
  interactive day-preview when no saved smart flag exists for the selected
  day.

#### 7.6.4 Backward compatibility

If the `forecast` section is missing or `enabled` is set to `false`, Phase 4
takes the legacy code path and reproduces the original
`smart_simulation_results.csv` to within 1e-9 on every numeric column. Old
notebooks, dashboards, and scripts therefore keep working.

#### 7.6.5 What this fix does *not* claim

This is a synthetic noise model — it does **not** replace an archived
numerical-weather-prediction (NWP) error distribution. It removes the obvious
oracle, but real day-ahead errors are autocorrelated and regime-dependent
(cloudy days are forecast much worse than clear days). Future work: feed the
RF actual archived day-ahead forecasts for Kuching, or retrain the RF using
**lagged-only** features so it has to forecast rather than invert pvlib.

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

`run_all.py` loops over the Cartesian product **regions × models** from `experiment_config.json` (filterable on the CLI with `--region` and `--algo`). For each `(region, algo)` cell:

1. Optional Phase 1 + 1b if `pipeline.run_phase1` is true (run only once for the region, not per algo).
2. **Phase 2** baseline (in-process call into `phase2_baseline_sim.simulate` with `--region`).
3. Optional **Phase 3** retrain if `pipeline.run_phase3_train_model` is true — trains **both** RF and SVM for that region.
4. **Phase 4** smart via `phase4_smart_optimizer.simulate(df, model, cfg)`, where `model` is the requested `(region, algo)` joblib.
5. Copy baseline/smart CSVs into **`results/YYYYMMDD_HHMM/<region>/<algo>/`**.
6. **Validate** series (§9.2).
7. Build **`daily_results.csv`** via `compute_daily_results` (§10).
8. Optional **σ-sweep** (`forecast.sweep_sigmas` × `forecast.sweep_seeds`), skipped if `--no-sweeps` is passed. Produces `forecast_sensitivity{,_raw}.csv` and `figures/forecast_sensitivity.png`.
9. **`metrics.json`** via `compute_metrics` (§11) including `statistical_tests` (Wilcoxon + paired-t + perfect-foresight ceiling % when the σ-sweep ran).
10. **`save_figures`** → `figures/*.png` (cost comparison, daily-savings hist, peak-import-reduction hist, SoC distribution).

After all cells finish, two top-level files are written at the root of `results/YYYYMMDD_HHMM/`:

- **`summary.csv`** — one row per `(region, algo)` cell with the headline KPIs (total savings, mean daily saving, peak-import reduction, Wilcoxon p-value, paired-t p-value, perfect-foresight %). The table you cite in the thesis comparison chapter.
- **`verdict.txt`** — plain-text bullet list summarising, per region, which algorithm captured more savings and whether the difference reached p < 0.05.
- **`config_snapshot.json`** — copy of the config used for the run.

### 9.2 Validations

- SoC bounds; non-negative import and cost; **1 h** regular timestamps.
- **Energy balance** identity tying `Load_kW`, `Target_Power_kW`, `Batt_Action_kW`, `Grid_Import_kW` (tolerance `≤ 1e-6 kW`).
- **Tariff spot-check** on representative hours for both CSVs (tolerance `≤ 1e-5 RM`).

Each validation raises a hard error on failure so broken runs cannot silently become thesis numbers. The exact formulas and tolerances are in `docs/VALIDATION.md`.

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

A **non-zero gap** between continuous total savings and the sum of daily savings is **expected**: the two modes answer **different counterfactual questions** (see [`ASSUMPTIONS.md`](ASSUMPTIONS.md)).

**SSR / total import vs money:** Under ToU, the bill is driven by **when** you import. The smart policy may trade **slightly more** off-peak energy for **less** peak energy — cost can fall even if SSR or total kWh imported does not improve.

---

## 12. Dashboard — `phase5_dashboard.py`

### 12.1 Sidebar and data sources

- Loads **`experiment_config.json`**.
- Exposes **region** and **algo** selectors that pick which `(region, algo)` cell of a saved run to display. The run picker filters to directories named `YYYYMMDD_HHMM` only — utility folders like `results/_per_region/` are hidden from the dropdown.
- Can load baseline/smart CSVs from the **project root paths** or from the selected **`results/YYYYMMDD_HHMM/<region>/<algo>/`** subfolder (recommended for consistency with `metrics.json`).

Parameters (battery, tariff, `pv_scale` / `load_scale`, controller knobs) drive **on-the-fly** simulations in interactive tabs.

### 12.2 Tab 1 — Conceptual explanation

Narrative layer: net load, battery, tariff, baseline vs AI policy — suitable for non-specialist readers.

### 12.3 Tab 2 — Live one-day simulation

- Runs **baseline** and **AI** for a **selected date** over 24 h, using the currently selected `(region, algo)` model.
- **Initial SoC** is user-controlled — **must match** `initial_soc_daily_reset` if you want consistency with Tab 3 daily-reset logic for that comparison.
- Metrics: **cost per day**, **savings**, **peak import reduction** for that day.

**Plots:** load vs PV, SoC trajectories, grid import, hourly cost — use to **show mechanism** (e.g. higher SoC entering peak on conservation days).

### 12.4 Tab 3 — Yearly / aggregate comparison

- Aggregation: hourly / daily / monthly views as implemented.
- **Daily-reset toggle** — aligns interpretation with `daily_results.csv` / `metrics.json` daily-reset section.
- **Conservation days filter** — interpret with care in **continuous** mode (spillover); cleaner with **daily reset**.
- **Top savings days** — good candidates to **replay in Tab 2** with matched SoC.

### 12.5 Tab 4 — Model and pipeline diagnostics (six sections)

Tab 4 is the **thesis-grade** tab. It is split into six numbered sections so each can be cited individually in the report:

1. **Model comparison** — RF vs SVM side-by-side: held-out `test_rmse`, `test_mae`, `test_r2`, `test_mape`, plus a winner-callout per region pulled from `summary.csv`.
2. **Training figures** — embeds `figures/training/<region>/{rf,svm}_fold_scores.png`, `_learning_curve.png`, `_validation_curve.png`, and the per-region `compare_<region>.png` so the examiner can see the CV evidence behind the headline numbers.
3. **Held-out accuracy** — the `models/cv_report_<region>.json` block for the selected `(region, algo)` cell with `cv_rmse_mean ± cv_rmse_std`, fold scores, and the test-window scoreboard.
4. **Simulation validation** — renders the four validation formulas from `docs/VALIDATION.md` as LaTeX with the live `validation.tariff_cost_identity_max_abs_err` and `validation.energy_balance_max_abs_err` from `metrics.json` and a PASS/FAIL chip per identity.
5. **Statistical evaluation** — Wilcoxon signed-rank, paired-t-test, mean daily saving (RM), and (when the σ-sweep ran) **% of perfect-foresight ceiling**. The ceiling tile is cosmetically hidden when its value is missing or `NaN` so legacy runs do not show an empty card.
6. **Forecast sensitivity sweep** — only rendered when `forecast_sensitivity.csv` / `.png` exist for the selected cell. Displays the per-σ table and the errorbar figure (`mean ± 1 std` across seeds, with the active headline σ marked).

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
| Why RF and SVM, not deep learning? | RF gives an interpretable baseline and stable feature importance; SVM is a different inductive bias (kernel similarity, scaled features) that stress-tests the controller. The fact that **both** give comparable savings — see `summary.csv` and `verdict.txt` — argues the gains come from the **policy + ToU asymmetry**, not the algorithm. Deep learning is reasonable future work but the data volume (~5 years hourly) makes the marginal value low. |
| How do you know it generalises beyond Kuching? | The same pipeline is run for **five regions** (Kuching, Bintulu, Kapit, Miri, Sri Aman). The `summary.csv` row per region shows the savings hold across the geographies; `verdict.txt` flags any region where the smart-vs-baseline gap is not statistically significant. |

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
- **No distribution network constraints**. Forecast uncertainty is now modelled
  via a synthetic, seeded noise model (see §7.6) and reported as a sensitivity
  curve, but a true probabilistic / NWP-grounded forecast is future work.
- **Classification robustness** near **`e_crit_kwh`** could be studied with sensitivity analysis.
- **Alternative policies**: MPC, stochastic DP, reinforcement learning — compare against this **transparent** rule-based AI baseline.

This document is intentionally detailed; trim sections when pasting into a thesis to avoid duplication with your own narrative chapters.
