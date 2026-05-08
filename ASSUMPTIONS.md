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
  - Whole-run CSVs (`baseline_simulation_results.csv`, `smart_simulation_results.csv`) start from the battery model default unless overridden by the script.
  - Daily-reset evaluation (`daily_results.csv`) starts from `experiment_config.json` (`scenario.initial_soc_daily_reset`) for both Baseline and AI, to ensure a fair per-day comparison.

## Household load
- Household load is **synthetic** (a shaped daily profile with noise), not a measured smart-meter dataset.
- Daily energy is targeted around **15–20 kWh/day**, with a slight weekend multiplier.
- Determinism: the synthetic load is deterministic when using the same `seed.numpy_seed` from `experiment_config.json`.

## AI controller behavior
- The “AI” is a Random Forest forecaster that predicts the next-day PV profile.
- Conservation Mode activates when predicted next-day energy \(E_{pred}\) falls below \(E_{crit}\).
- On Conservation days, the controller uses a **Peak-Reserve Policy**:
  - During **off-peak**, if SoC is at/below the reserve target, discharge is restricted to an **essential discharge limit**.
  - During **peak**, normal discharge is allowed to reduce expensive grid imports.

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

