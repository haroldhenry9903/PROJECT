# Validation & Accuracy Justification

This document lists **every formula** used to justify the accuracy of (a) the
PV-power forecaster and (b) the rule-based smart-controller simulation.
Each formula is paired with the **code location** where it is implemented and
the **numeric tolerance** that the pipeline enforces at run time. The aim is
that an examiner can read this document alongside the code and verify that
each claim in the thesis is checked by an assertion in `run_all.py`.

> File references use the form `file.py:line` and point at the function (or
> the line where the assertion lives) in this repository.

---

## 1. Forecast-model accuracy

For each `(region, algorithm)` pair we report four scores on a chronologically
held-out test window (rows with `datetime >= 2024-08-08`) and on a 5-fold
`TimeSeriesSplit` cross-validation. Code path:
`phase3_ai_training.py:_evaluate_on_test` and
`phase3_ai_training.py:train_one_model` (cv_fold_scores).

Let `y_i` be the actual `Target_Power_kW` for hour `i` and `ŷ_i` the model's
prediction. Let `\bar{y}` be the mean of `y_i` over the evaluation window.

### 1.1 Root-Mean-Square Error (RMSE)

$$
\mathrm{RMSE} \;=\; \sqrt{\;\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \;}
$$

Same physical units as the target (kW). The metric most sensitive to large
errors; we report this both as `cv_rmse_mean ± cv_rmse_std` (5-fold) and as
`test_rmse` (held-out window) in `models/cv_report_{region}.json`.

### 1.2 Mean Absolute Error (MAE)

$$
\mathrm{MAE} \;=\; \frac{1}{n} \sum_{i=1}^{n} \bigl| y_i - \hat{y}_i \bigr|
$$

Robust to outliers. Reported as `test_mae`.

### 1.3 Coefficient of Determination (R²)

$$
R^{2} \;=\; 1 \;-\; \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

1.0 = perfect fit; 0.0 = no better than predicting the mean. Reported as
`test_r2`.

### 1.4 Mean Absolute Percentage Error (MAPE, daytime hours only)

$$
\mathrm{MAPE} \;=\; \frac{100\%}{n^{\star}} \sum_{i\,:\,y_i > 0.05} \frac{\bigl| y_i - \hat{y}_i \bigr|}{y_i}
$$

We restrict the sum to hours where the actual PV power exceeds 0.05 kW
(~ 1 % of nameplate) because MAPE explodes at night when `y_i = 0`. `n*` is
the count of qualifying hours. Reported as `test_mape`.

### 1.5 Forecast-skill MAPE on daily PV totals

Used inside `run_all.py:_forecast_mape_daily` to evaluate the Phase-4
midnight-prediction quality (not the same as the hourly MAPE above):

$$
\mathrm{MAPE_{daily}} \;=\; \frac{100\%}{D^{\star}} \sum_{d\,:\,E_d^{\text{actual}}>0} \frac{\bigl| E_d^{\text{pred}} - E_d^{\text{actual}} \bigr|}{E_d^{\text{actual}}}
$$

Where `E_d^pred` is the daily total predicted PV (kWh) at midnight of day `d`
and `E_d^actual` is the realised total. `D*` excludes any zero-PV day.

---

## 2. Cross-validation strategy (why 5-fold and which 5-fold)

We use `sklearn.model_selection.TimeSeriesSplit(n_splits=5)` rather than
`KFold(shuffle=True)`. The reason is **causality**: every CV fold must train
on past hours only and validate on later hours, so the score reflects the
deployment scenario (forecast tomorrow given everything observed up to today).
Shuffling would leak future weather into past folds and inflate every score.

Per-fold RMSE is logged in `cv_fold_scores` (`models/cv_report_{region}.json`)
and visualised in `figures/training/{region}/{algo}_fold_scores.png` as a bar
chart with the mean as a dashed horizontal line. The dashboard's "verdict
line" picks the algorithm with the lowest `test_rmse` for each region.

---

## 3. Simulation integrity (energy balance, tariff, SoC bounds)

The smart-controller simulation is implemented in
`phase4_smart_optimizer.py:simulate`. After every run, `run_all.py` runs
**four** post-conditions that fail loudly if the implementation drifts.

### 3.1 Net-power identity

$$
P_{\text{net}}(t) \;=\; P_{\text{load}}(t) \;-\; P_{\text{pv}}(t)
$$

Positive `P_net` = deficit (need to discharge battery or buy from grid).
Negative `P_net` = surplus (charge battery; remaining is curtailed).

### 3.2 Grid-import identity

$$
P_{\text{grid,import}}(t) \;=\; \max \bigl( 0, \;\; P_{\text{net}}(t) - P_{\text{batt}}(t) \bigr)
$$

`P_batt` is the actual battery flow returned by `core_battery_model.Battery.update_soc`.
Validated in `run_all.py:validate_energy_balance` (`run_all.py:132-157`) with
the tolerance:

> `max_abs_err ≤ 1e-6 kW` (else `ValueError` is raised, the run aborts).

This proves that for **every** of the ~43,872 hourly rows, the implementation
matches the equation to floating-point precision.

### 3.3 Tariff–cost identity

$$
\text{Cost}_{\text{hour}}(t) \;=\; P_{\text{grid,import}}(t) \cdot \tau(h_t),
\qquad
\tau(h) \;=\;
\begin{cases}
\tau_{\text{peak}}     & \text{if } h_{\text{peak,start}} \le h < h_{\text{peak,end}} \\
\tau_{\text{offpeak}}  & \text{otherwise}
\end{cases}
$$

Tariff rates come from `experiment_config.json#tariff` (Sarawak C3 defaults:
peak 07:00–22:00 at 0.245 RM/kWh, off-peak 0.139 RM/kWh). Spot-checked at
representative hours by `run_all.py:validate_tariff_cost_identity`
(`run_all.py:160-211`):

> `max_abs_err ≤ 1e-5 RM` (else `RuntimeError` is raised).

### 3.4 SoC bounds + hourly continuity

$$
\text{SoC}(t) \in \bigl[ \text{SoC}_{\min},\; 1.0 \bigr],
\qquad
t_{k+1} - t_k = 1 \text{ hour}
$$

Both checked by `run_all.py:validate_timeseries` (`run_all.py:98-129`).
Violations of either condition halt the run (no silent fallback).

---

## 4. System-level KPIs

These are the headline numbers reported in `metrics.json` and the dashboard.

### 4.1 Self-Sufficiency Ratio (SSR, %)

$$
\mathrm{SSR}\,\% \;=\; \left( 1 \;-\; \frac{\sum_t P_{\text{grid,import}}(t)}{\sum_t P_{\text{load}}(t)} \right) \times 100
$$

Fraction of the household's energy that did NOT come from the grid. Implemented
in `run_all.py:_kpis` (`run_all.py:341-354`).

### 4.2 Annual cost savings (RM)

$$
\Delta\mathrm{Cost} \;=\; \mathrm{Cost}_{\text{baseline}} \;-\; \mathrm{Cost}_{\text{smart}}
$$

Positive = AI saved money. Implemented in `run_all.py:compute_metrics`.

### 4.3 Peak-window import reduction (kWh)

$$
\Delta P_{\text{import,peak}} \;=\; \sum_{t \in \text{peak}} P_{\text{grid,import}}^{\text{base}}(t) \;-\; \sum_{t \in \text{peak}} P_{\text{grid,import}}^{\text{smart}}(t)
$$

This is usually where the AI policy creates value (off-peak energy stored for
peak-hour use). Implemented in `run_all.py:_kpis`.

---

## 5. Evaluating "how good is the simulation, really?"

The KPIs above describe the simulation against itself. Sections 5.1 and 5.2
test it against an **external** standard.

### 5.1 Paired statistical tests on daily savings

We treat each calendar day's savings `Δ_d = base_d - smart_d` as a paired
observation and report two tests:

**Wilcoxon signed-rank** (`scipy.stats.wilcoxon`)

$$
W \;=\; \sum_{d=1}^{D} \mathrm{rank}\bigl(\,|\Delta_d|\,\bigr) \cdot \mathrm{sgn}(\Delta_d)
$$

Non-parametric; does not assume normality of daily savings. The reported
`p_value` is two-sided.

**Paired t-test** (`scipy.stats.ttest_rel`)

$$
t \;=\; \frac{\bar{\Delta}}{s_{\Delta} / \sqrt{D}}
\qquad \text{with } df = D - 1
$$

Parametric; valid when daily savings are approximately normal (with `D ≥ 30`
the Central Limit Theorem covers most practical cases).

Both `p_value`s are written under `metrics.json#statistical_tests`. A
p-value below 0.05 establishes that AI savings are statistically significant
across the 365 paired days.

### 5.2 Percentage of perfect-foresight ceiling achieved

`run_all.py:run_forecast_sensitivity_sweep` already runs the smart controller
with `forecast.ghi_noise_sigma = 0` (perfect oracle) as the ceiling. We
report:

$$
\eta_{\text{forecast}} \;=\; \frac{\Delta\mathrm{Cost}_{\text{realistic-}\sigma}}{\Delta\mathrm{Cost}_{\sigma=0}} \times 100\%
$$

A value of 100 % means the noisy-forecast controller captured *all* of the
savings that an oracle could capture. Lower values quantify how much money
forecast error costs. Reported under
`metrics.json#statistical_tests.pct_of_perfect_foresight_ceiling`.

---

## 6. Reproducibility

Every randomised step uses a seeded RNG:

* Synthetic load — `experiment_config.json#seed.numpy_seed = 42`
  (`phase2_baseline_sim.py:generate_synthetic_load`).
* Forecast noise — `experiment_config.json#forecast.seed = 1337`
  (`phase4_smart_optimizer.py:_resolve_forecast_seed`).
* RF — `random_state=42` in `phase3_ai_training.py:build_estimator`.

With identical config and identical input CSVs, two consecutive `python run_all.py`
invocations produce byte-identical artifacts (modulo the timestamped output
folder name).

---

## 7. Quick "is the simulation healthy?" checklist

When opening a new `results/<stamp>/metrics.json`, look at these fields:

| Key | Healthy value | Where it's enforced |
|-----|---------------|---------------------|
| `_meta.validation.tariff_cost_identity_passed` | `true` | `run_all.py:209` |
| `_meta.validation.tariff_cost_identity_max_abs_err` | `< 1e-5` | `run_all.py:209` |
| `total_savings_rm` | > 0 (AI must beat baseline) | `run_all.py:compute_metrics` |
| `_meta.forecast.enabled` | `true` (else you're reporting the upper bound) | `phase4_smart_optimizer.py:172` |
| `statistical_tests.wilcoxon_signed_rank.p_value` | `< 0.05` | `run_all.py:compute_metrics` (Step 7) |
| `statistical_tests.pct_of_perfect_foresight_ceiling` | typically 70–95 % | `run_all.py:compute_metrics` (Step 7) |

If any of these is missing or red, the simulation result should not be cited
in the thesis without explanation.
