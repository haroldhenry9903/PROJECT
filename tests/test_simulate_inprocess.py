"""In-process tests for `simulate()` and the dashboard's Tab 2 forecast helper.

These tests are intentionally hermetic:
  - They do NOT depend on the trained `models/rf_forecast_model.joblib` file.
  - They do NOT depend on `data/processed/...` CSVs.
  - They do NOT require a streamlit runtime; we test the logical helper used
    by Tab 2 directly.

A tiny `_FakeForecaster` stands in for the real Random Forest. Its `predict`
output is a deterministic function of the input features, which gives us a
testbed for determinism, mutation, and noise propagation without requiring a
trained model.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from phase4_smart_optimizer import (
    FEATURE_COLUMNS,
    apply_forecast_noise,
    simulate,
)


class _FakeForecaster:
    """Tiny stand-in for the trained Random Forest.

    Predicts hourly PV power as a smooth function of ghiradiation alone:
        pv_hat = clip(ghi / 200, 0, 5)  # kW
    plus a small calendar-aware adjustment that depends only on `hour` so
    that midnight-window sums show realistic spread across days.
    """

    feature_names_in_: np.ndarray = np.array(FEATURE_COLUMNS, dtype=object)
    n_features_in_: int = len(FEATURE_COLUMNS)

    def predict(self, X) -> np.ndarray:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURE_COLUMNS)
        ghi = df["ghiradiation"].astype(float).to_numpy()
        hour = df["hour"].astype(float).to_numpy()
        diurnal = np.where((hour >= 6) & (hour <= 18), 1.0, 0.0)
        return np.clip(ghi / 200.0, 0.0, 5.0) * diurnal


def _make_window(hours: int = 72, seed: int = 0) -> pd.DataFrame:
    """Build a small in-memory baseline-output dataframe for `simulate()`.

    Includes the columns simulate() expects (`Load_kW`, `Target_Power_kW`, the
    7 RF features) plus a tz-naive hourly DatetimeIndex aligned to midnight.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 00:00", periods=hours, freq="h")
    h = idx.hour.to_numpy()
    # Realistic-ish tropical GHI shape peaking around 12:00 with daily variation.
    daily_amp = 700.0 + rng.normal(0, 80, hours)
    ghi = np.maximum(0.0, daily_amp * np.cos((h - 12.0) * np.pi / 12.0))
    return pd.DataFrame(
        {
            "Load_kW": 0.6 + 0.4 * np.sin(h * np.pi / 12.0) + 0.1 * rng.normal(size=hours),
            "Target_Power_kW": np.clip(ghi / 200.0, 0.0, 5.0),
            "ghiradiation": ghi,
            "temp": 26.0 + 2.0 * np.sin(h * np.pi / 12.0),
            "humidity": np.clip(70.0 + 10.0 * rng.normal(size=hours), 0.0, 100.0),
            "precip": np.zeros(hours),
            "cloudcover": np.clip(40.0 + 20.0 * rng.normal(size=hours), 0.0, 100.0),
            "hour": h,
            "month": idx.month,
        },
        index=idx,
    )


def _baseline_cfg(forecast_enabled: bool = False, **forecast_overrides) -> dict:
    cfg = {
        "battery": {"capacity_kwh": 10.0, "inverter_power_kw": 5.0, "soc_min": 0.15, "efficiency_rt": 0.95},
        "tariff": {"peak_start_hour": 7, "peak_end_hour": 22, "peak_rate_rm_per_kwh": 0.245, "offpeak_rate_rm_per_kwh": 0.139},
        "controller": {"e_crit_kwh": 18.0, "soc_reserve": 0.50, "essential_discharge_kw": 0.0},
        "seed": {"numpy_seed": 42},
    }
    if forecast_enabled or forecast_overrides:
        cfg["forecast"] = {
            "enabled": forecast_enabled,
            "ghi_noise_sigma": 0.20,
            "ghi_bias": 0.0,
            "cloudcover_noise_sigma": 5.0,
            "temp_noise_sigma": 1.5,
            "humidity_noise_sigma": 5.0,
            "precip_noise_sigma": 0.5,
            "clip_nonneg": True,
            "seed": 1337,
        }
        cfg["forecast"].update(forecast_overrides)
    return cfg


def test_simulate_does_not_mutate_input():
    df = _make_window()
    before = df.copy(deep=True)
    rf = _FakeForecaster()
    simulate(df, rf, _baseline_cfg(forecast_enabled=True))
    pd.testing.assert_frame_equal(df, before)


def test_simulate_is_deterministic_with_same_cfg():
    df = _make_window()
    rf = _FakeForecaster()
    cfg = _baseline_cfg(forecast_enabled=True)
    a = simulate(df, rf, cfg)
    b = simulate(df, rf, cfg)
    pd.testing.assert_frame_equal(a, b)


def test_simulate_returns_required_columns():
    out = simulate(_make_window(), _FakeForecaster(), _baseline_cfg(forecast_enabled=True))
    required = {
        "SoC",
        "Batt_Action_kW",
        "Grid_Import_kW",
        "Hourly_Cost_RM",
        "Conservation_Mode_Active",
        "Forecast_Energy_kWh_Predicted",
    }
    assert required.issubset(set(out.columns))


def test_simulate_legacy_path_matches_disabled_forecast():
    """`forecast.enabled=False` and a missing `forecast` block must produce
    bit-identical numeric output (the perfect-foresight legacy path)."""
    df = _make_window()
    rf = _FakeForecaster()
    cfg_disabled = _baseline_cfg(forecast_enabled=False)
    cfg_no_block = _baseline_cfg(forecast_enabled=False)
    cfg_no_block.pop("forecast", None)

    a = simulate(df, rf, cfg_disabled)
    b = simulate(df, rf, cfg_no_block)
    for col in ["SoC", "Batt_Action_kW", "Grid_Import_kW", "Hourly_Cost_RM"]:
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(), atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(a["Conservation_Mode_Active"].to_numpy(), b["Conservation_Mode_Active"].to_numpy())


def test_simulate_noise_changes_predicted_energy():
    """Two different `ghi_noise_sigma` values produce different daily forecast
    energies for the same calendar day. This is the core invariant that
    proves the dashboard Tab 2 fallback now responds to forecast noise."""
    df = _make_window()
    rf = _FakeForecaster()
    cfg_low = _baseline_cfg(forecast_enabled=True, ghi_noise_sigma=0.05)
    cfg_high = _baseline_cfg(forecast_enabled=True, ghi_noise_sigma=0.40)
    out_low = simulate(df, rf, cfg_low)
    out_high = simulate(df, rf, cfg_high)
    e_low = out_low["Forecast_Energy_kWh_Predicted"].dropna().unique()
    e_high = out_high["Forecast_Energy_kWh_Predicted"].dropna().unique()
    assert not np.allclose(np.sort(e_low), np.sort(e_high))


def test_simulate_soc_within_bounds():
    out = simulate(_make_window(hours=72), _FakeForecaster(), _baseline_cfg(forecast_enabled=True))
    soc = out["SoC"].astype(float).to_numpy()
    assert (soc >= 0.15 - 1e-9).all()
    assert (soc <= 1.0 + 1e-9).all()


def test_simulate_grid_import_nonnegative_and_cost_consistent():
    out = simulate(_make_window(hours=48), _FakeForecaster(), _baseline_cfg(forecast_enabled=True))
    gi = out["Grid_Import_kW"].astype(float).to_numpy()
    cost = out["Hourly_Cost_RM"].astype(float).to_numpy()
    hours = out.index.hour.to_numpy()
    assert (gi >= -1e-9).all()
    rates = np.where((hours >= 7) & (hours < 22), 0.245, 0.139)
    np.testing.assert_allclose(cost, gi * rates, atol=1e-12)


def test_simulate_truncated_last_day_keeps_conservation_off():
    """When the dataset doesn't have a full 24h window after the final
    midnight, the controller must keep conservation off and the forecast
    energy NaN for that tail (matches legacy behaviour)."""
    df = _make_window(hours=30)  # final midnight is at idx 24, only 6 rows after
    out = simulate(df, _FakeForecaster(), _baseline_cfg(forecast_enabled=True))
    tail = out.iloc[24:]
    assert tail["Forecast_Energy_kWh_Predicted"].isna().all()
    assert (tail["Conservation_Mode_Active"] == False).all()


def test_simulate_seed_changes_predictions_under_noise():
    df = _make_window()
    rf = _FakeForecaster()
    a = simulate(df, rf, _baseline_cfg(forecast_enabled=True, seed=1337))
    b = simulate(df, rf, _baseline_cfg(forecast_enabled=True, seed=2024))
    e_a = a["Forecast_Energy_kWh_Predicted"].dropna().unique()
    e_b = b["Forecast_Energy_kWh_Predicted"].dropna().unique()
    assert not np.allclose(np.sort(e_a), np.sort(e_b))


def test_apply_forecast_noise_then_predict_round_trip():
    """Sanity-check: when forecast is enabled, e_pred at sigma=0 (no noise)
    matches the legacy path's e_pred for the same window."""
    df = _make_window(hours=24)
    rf = _FakeForecaster()
    a = simulate(df, rf, _baseline_cfg(forecast_enabled=True, ghi_noise_sigma=0.0,
                                       cloudcover_noise_sigma=0.0,
                                       temp_noise_sigma=0.0,
                                       humidity_noise_sigma=0.0,
                                       precip_noise_sigma=0.0))
    b = simulate(df, rf, _baseline_cfg(forecast_enabled=False))
    np.testing.assert_allclose(
        a["Forecast_Energy_kWh_Predicted"].dropna().to_numpy(),
        b["Forecast_Energy_kWh_Predicted"].dropna().to_numpy(),
        atol=1e-12,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
