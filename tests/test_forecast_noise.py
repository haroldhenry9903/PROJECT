"""Unit tests for the forecast-noise injection that closes the perfect-foresight
loophole in the smart battery controller.

These tests intentionally avoid spinning up the full Phase 4 simulation; they
verify the contract of `apply_forecast_noise` in isolation:

    1. Determinism given the same seed.
    2. The input dataframe is never mutated.
    3. With `forecast.enabled = false`, the function is a no-op (legacy /
       perfect-foresight path).
    4. Non-negativity for `ghiradiation` and `precip` after multiplicative noise.
    5. Bounds [0, 100] for `humidity` and `cloudcover` after additive noise.
    6. Calendar features `hour` and `month` are never perturbed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase4_smart_optimizer import _resolve_forecast_seed, apply_forecast_noise


FEATURES = ["ghiradiation", "temp", "humidity", "precip", "cloudcover", "hour", "month"]


def _sample_window(n: int = 24) -> pd.DataFrame:
    """A 24-hour fixture window with realistic-ish tropical values."""
    return pd.DataFrame(
        {
            "ghiradiation": np.linspace(0.0, 900.0, n),
            "temp": np.linspace(24.0, 32.0, n),
            "humidity": np.linspace(60.0, 95.0, n),
            "precip": np.array([0.0] * (n - 4) + [1.0, 5.0, 0.5, 0.0])[:n],
            "cloudcover": np.linspace(20.0, 80.0, n),
            "hour": np.arange(n) % 24,
            "month": np.full(n, 6),
        }
    )


_DEFAULT_CFG = {
    "enabled": True,
    "ghi_noise_sigma": 0.20,
    "ghi_bias": 0.0,
    "cloudcover_noise_sigma": 5.0,
    "temp_noise_sigma": 1.5,
    "humidity_noise_sigma": 5.0,
    "precip_noise_sigma": 0.5,
    "clip_nonneg": True,
    "seed": 1337,
}


def test_disabled_section_is_noop():
    df = _sample_window()
    out = apply_forecast_noise(df, forecast_cfg={"enabled": False}, rng=np.random.default_rng(0))
    pd.testing.assert_frame_equal(df, out)


def test_empty_section_is_noop():
    df = _sample_window()
    out = apply_forecast_noise(df, forecast_cfg={}, rng=np.random.default_rng(0))
    pd.testing.assert_frame_equal(df, out)


def test_does_not_mutate_input():
    df = _sample_window()
    before = df.copy(deep=True)
    apply_forecast_noise(df, forecast_cfg=_DEFAULT_CFG, rng=np.random.default_rng(123))
    pd.testing.assert_frame_equal(df, before)


def test_determinism_with_same_seed():
    df = _sample_window()
    a = apply_forecast_noise(df, forecast_cfg=_DEFAULT_CFG, rng=np.random.default_rng(2025))
    b = apply_forecast_noise(df, forecast_cfg=_DEFAULT_CFG, rng=np.random.default_rng(2025))
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_produce_different_outputs():
    df = _sample_window()
    a = apply_forecast_noise(df, forecast_cfg=_DEFAULT_CFG, rng=np.random.default_rng(1))
    b = apply_forecast_noise(df, forecast_cfg=_DEFAULT_CFG, rng=np.random.default_rng(2))
    assert not np.allclose(a["ghiradiation"].values, b["ghiradiation"].values)


def test_zero_sigma_zero_bias_is_a_noop_for_features():
    df = _sample_window()
    cfg = {
        "enabled": True,
        "ghi_noise_sigma": 0.0,
        "ghi_bias": 0.0,
        "cloudcover_noise_sigma": 0.0,
        "temp_noise_sigma": 0.0,
        "humidity_noise_sigma": 0.0,
        "precip_noise_sigma": 0.0,
        "clip_nonneg": True,
        "seed": 7,
    }
    out = apply_forecast_noise(df, forecast_cfg=cfg, rng=np.random.default_rng(7))
    for col in ["ghiradiation", "temp", "humidity", "precip", "cloudcover", "hour", "month"]:
        np.testing.assert_allclose(out[col].to_numpy(), df[col].to_numpy(), atol=1e-12, err_msg=col)


def test_ghi_and_precip_remain_nonnegative():
    df = _sample_window()
    cfg = dict(_DEFAULT_CFG, ghi_noise_sigma=1.5, precip_noise_sigma=2.0, ghi_bias=-0.99)
    for seed in [0, 1, 2, 3, 4, 5, 99, 1234, 9999]:
        out = apply_forecast_noise(df, forecast_cfg=cfg, rng=np.random.default_rng(seed))
        assert (out["ghiradiation"] >= 0.0).all(), f"GHI went negative at seed {seed}"
        assert (out["precip"] >= 0.0).all(), f"precip went negative at seed {seed}"


def test_humidity_and_cloudcover_clipped_to_unit_percent():
    df = _sample_window()
    cfg = dict(_DEFAULT_CFG, humidity_noise_sigma=200.0, cloudcover_noise_sigma=200.0)
    for seed in [0, 1, 2, 7, 42, 123, 9999]:
        out = apply_forecast_noise(df, forecast_cfg=cfg, rng=np.random.default_rng(seed))
        assert ((out["humidity"] >= 0.0) & (out["humidity"] <= 100.0)).all()
        assert ((out["cloudcover"] >= 0.0) & (out["cloudcover"] <= 100.0)).all()


def test_calendar_features_never_perturbed():
    df = _sample_window()
    out = apply_forecast_noise(df, forecast_cfg=_DEFAULT_CFG, rng=np.random.default_rng(11))
    np.testing.assert_array_equal(out["hour"].to_numpy(), df["hour"].to_numpy())
    np.testing.assert_array_equal(out["month"].to_numpy(), df["month"].to_numpy())


def test_ghi_bias_shifts_mean():
    df = _sample_window()
    cfg_pos = dict(_DEFAULT_CFG, ghi_noise_sigma=1e-9, ghi_bias=+0.10, clip_nonneg=True)
    out_pos = apply_forecast_noise(df, forecast_cfg=cfg_pos, rng=np.random.default_rng(0))
    np.testing.assert_allclose(
        out_pos["ghiradiation"].to_numpy(),
        df["ghiradiation"].to_numpy() * 1.10,
        rtol=1e-6,
    )


def test_resolve_forecast_seed_precedence():
    cfg = {"forecast": {"seed": 11}, "seed": {"forecast_seed": 22, "numpy_seed": 5}}
    assert _resolve_forecast_seed(cfg) == 11

    cfg = {"forecast": {}, "seed": {"forecast_seed": 22, "numpy_seed": 5}}
    assert _resolve_forecast_seed(cfg) == 22

    cfg = {"seed": {"numpy_seed": 5}}
    assert _resolve_forecast_seed(cfg) == 6

    cfg = {}
    assert _resolve_forecast_seed(cfg) == 43


def test_handles_empty_window():
    df = _sample_window().head(0)
    out = apply_forecast_noise(df, forecast_cfg=_DEFAULT_CFG, rng=np.random.default_rng(0))
    assert len(out) == 0
    assert list(out.columns) == list(df.columns)


def test_missing_optional_columns_does_not_error():
    df = _sample_window()[["ghiradiation", "temp", "hour", "month"]]
    out = apply_forecast_noise(df, forecast_cfg=_DEFAULT_CFG, rng=np.random.default_rng(0))
    assert set(out.columns) == set(df.columns)
    np.testing.assert_array_equal(out["hour"].to_numpy(), df["hour"].to_numpy())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
