# Assumptions (Simulation + Evaluation)

This project is a **simulation** of a grid-tied home in Sarawak with PV + battery + time-of-use tariff.
The goal is to compare **Baseline (Self-Consumption)** vs **AI Controller (Conservation Mode)** fairly.

## Energy and sign conventions
- Net power is defined as:
  \[
  P_{net}(t) = P_{load}(t) - P_{pv}(t)
  \]
  - \(P_{net} > 0\): deficit → discharge battery then import from grid
  - \(P_{net} < 0\): surplus → charge battery, remaining is curtailed/exported

## Grid export
- Economic cost is computed using **grid imports only**.
- Export/curtailment is assumed to have **zero monetary compensation**.

## Battery model
- Battery is modeled as an LFP-like device with:
  - capacity (kWh)
  - inverter power limits (kW charge/discharge)
  - minimum SoC floor (default 15%) and maximum SoC (100%)
  - round-trip efficiency (default 95%, implemented as sqrt split for charge/discharge)
- Initial SoC depends on the **evaluation mode**:
  - Whole-run CSVs (`baseline_simulation_results.csv`, `smart_simulation_results.csv`, or their per-region copies under `results/<stamp>/<region>/<algo>/`) start from the battery model default unless overridden by the script.
  - Daily-reset evaluation (`daily_results.csv`) starts from `experiment_config.json` (`scenario.initial_soc_daily_reset`) for both Baseline and AI, to ensure a fair per-day comparison.

## Household load
- Household load is **synthetic** (a shaped daily profile with noise), not a measured smart-meter dataset.
- Daily energy is targeted around **15–20 kWh/day**, with a slight weekend multiplier.
- Determinism: the synthetic load is deterministic when using the same `seed.numpy_seed` from `experiment_config.json`.

## AI controller behavior
- The "AI" forecaster is **plug-and-play**: anything implementing
  `model.predict(X)` on the 7-column feature window (`ghiradiation`, `temp`,
  `humidity`, `precip`, `cloudcover`, `hour`, `month`) works. The project
  ships and reports two: a **Random Forest** (`models/rf_forecast_model_<region>.joblib`)
  and an **SVR/SVM pipeline** (`models/svm_forecast_model_<region>.joblib`).
  Both are trained per region on the same chronological split with
  `TimeSeriesSplit(n_splits=5)` cross-validation, so the smart-controller code
  path is identical regardless of which estimator is chosen at the CLI
  (`phase4_smart_optimizer.py --algo rf|svm`) or the dashboard sidebar.
- At every midnight, the next-24h weather features are passed through a
  configurable **forecast-error model** (see *Forecast error model* below)
  **before** being given to the forecaster. The controller therefore operates
  on a *noisy day-ahead forecast*, not the ground-truth weather. Setting all
  noise sigmas/biases to zero (or `forecast.enabled = false`) reproduces the
  perfect-foresight upper bound for sensitivity studies.
- Conservation Mode activates when the **predicted** next-day energy
  \(\hat{E}_{pred}\) falls below \(E_{crit}\). With realistic forecast noise,
  this is a noisy classifier (false alarms and missed low-solar days are both
  possible) — exactly as it would be in a real deployment.
- On Conservation days, the controller uses a **Peak-Reserve Policy**:
  - During **off-peak**, if SoC is at/below the reserve target, discharge is restricted to an **essential discharge limit**.
  - During **peak**, normal discharge is allowed to reduce expensive grid imports.

## Forecast error model
- The smart controller queries the RF only with **noisy** day-ahead features.
  This closes the loophole that `Target_Power_kW` (the simulation truth) is a
  deterministic function of the same features the RF was trained on; without
  the noise model the RF would effectively be a perfect-foresight oracle that
  inverts pvlib.
- Distributions (per column, applied independently each day with a dedicated
  seeded `numpy` RNG):
  - `ghiradiation`: \( x' = \max(0, x \cdot (1 + b_{ghi}) \cdot e^{\mathcal{N}(0,\sigma_{ghi})}) \).
  - `precip`: \( x' = \max(0, x \cdot e^{\mathcal{N}(0,\sigma_{precip})}) \).
  - `temp`: \( x' = x + \mathcal{N}(0, \sigma_T) \) (degC).
  - `humidity`: \( x' = \mathrm{clip}(x + \mathcal{N}(0, \sigma_H), 0, 100) \).
  - `cloudcover`: \( x' = \mathrm{clip}(x + \mathcal{N}(0, \sigma_{cc}), 0, 100) \).
  - `hour`, `month` are never perturbed (calendar features are exact).
- Defaults (`experiment_config.json#forecast`): `ghi_noise_sigma=0.20`,
  `ghi_bias=0.0`, `cloudcover_noise_sigma=0.10`, `temp_noise_sigma=1.5`,
  `humidity_noise_sigma=5.0`, `precip_noise_sigma=0.5`, `clip_nonneg=true`,
  `seed=1337`. The `seed` precedence is `forecast.seed` >
  `seed.forecast_seed` > `seed.numpy_seed + 1`.
- Determinism: with a fixed seed and unchanged config, the noisy features are
  reproducible across runs (covered by `tests/test_forecast_noise.py`).
- Sensitivity sweep: `run_all.py` automatically re-runs Phase 4 across the
  configured grid (`forecast.sweep_sigmas`, default
  σ ∈ {0.00, 0.10, 0.20, 0.30, 0.40}) with `forecast.sweep_seeds` (default
  `10`) seeded trials per σ, in-process. It writes:
  - `forecast_sensitivity_raw.csv` — one row per `(σ, seed)` trial.
  - `forecast_sensitivity.csv` — per-σ summary with mean values plus
    `*_std`, `*_min`, `*_max` siblings.
  - `figures/forecast_sensitivity.png` — `plt.errorbar` plot of mean savings
    **±1 std across seeds** vs σ, with a vertical dashed line at the active
    `forecast.ghi_noise_sigma`.
  The σ=0 row is the perfect-foresight upper bound; the gap to higher σ
  rows quantifies how much of the headline savings depends on forecast
  quality, and the error bars expose how much variance is attributable to
  the noise draw versus the σ level itself. With `sweep_seeds=1` the
  summary CSV reproduces the previous single-seed sweep exactly (mean =
  the single trial value, std = 0) for backward compatibility.
- Limitation: this synthetic noise model is not a substitute for an archived
  numerical-weather-prediction (NWP) error distribution. It correctly removes
  oracular savings but does not capture autocorrelated, regime-dependent
  forecast errors. Reporting the σ-sweep alongside the headline number is the
  honest framing.

## Timezone
- For the reproducible pipeline (`run_all.py`), CSV timestamps are treated as **tz-naive hourly** timestamps.
- Validation checks enforce:
  - Hourly step continuity (1-hour intervals)
  - Baseline/Smart timestamp timezone consistency (both tz-naive in the default pipeline)

## Tariff single source of truth
- Tariff rates and peak window come from `experiment_config.json.tariff`.
- Both Baseline (Phase 2) and AI (Phase 4) compute hourly costs from the same tariff definition to avoid unfair comparisons.

## Metrics: two evaluation modes (read this for thesis tables)

`run_all.py` writes `results/.../metrics.json` with **two complementary views** of performance. The **legacy top-level keys** (`baseline_total_cost_rm`, `total_savings_rm`, `daily_savings_mean_rm`, …) are unchanged for backward compatibility; **`_meta`** and **`by_evaluation_mode`** spell out what each number means.

1. **Continuous CSV headline (primary for annual bill impact)**  
   Totals are computed by summing stored columns on the **full** timelines copied from `baseline_simulation_results.csv` and `smart_simulation_results.csv` (notably `Hourly_Cost_RM`, `Grid_Import_kW`, `Load_kW`). Peak-window import uses the configured peak hours on the datetime index. This matches how the year-long digital twin actually ran when those CSVs were produced (SoC carries across days).

2. **Daily-reset ablation (pedagogical / fair day-isolated comparison)**  
   `daily_results.csv` recomputes each calendar day **in isolation** with the same starting SoC (`scenario.initial_soc_daily_reset`) for baseline and smart, while taking the smart branch’s conservation flag from the saved smart CSV for that day. The **mean** of per-day savings answers “on a typical isolated day, does AI win?” The **median** is often more representative than the mean when the distribution is skewed or contains many near-zero days: half of all days fall below the median daily savings, half above. A **negative mean** with **positive annual savings** can still be consistent because the **annual headline** includes cross-day coupling and a different counterfactual than summing independent one-day replays.

3. **SSR / total imports vs money under ToU**  
   Self-supply rate (SSR) and **total** grid imports are energy-centric. Time-of-use billing cares **when** energy is bought. The smart policy may accept **slightly more** total import (or lower SSR) if those extra imports occur at **cheap off-peak** rates while **reducing peak-window imports** enough to lower the **sum of `Hourly_Cost_RM`**. So cost can improve even if SSR or total kWh imported does not.

4. **Gap field**  
   `metrics.json` includes `gap_continuous_total_savings_rm_minus_sum_daily_savings_rm` under `_meta`: the difference between continuous headline savings and the sum of daily-reset per-day savings. A non-zero gap is **expected** and should be interpreted using `_meta.gap_explainer`, not treated as a bug by default.

Reproducibility: `experiment_config.json` may include an `evaluation` block (`headline_primary`, `daily_reset_soc_fraction_key`) copied into the run’s `config_snapshot.json` and reflected in `metrics.json._meta.headline_primary`.

