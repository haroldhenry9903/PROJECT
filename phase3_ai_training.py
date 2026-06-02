"""Phase 3: Train per-region PV forecasters and audit them.

This module is the single source of truth for forecasting model training in
the project. It addresses the supervisor feedback:

* Train BOTH a Random Forest and an SVM (SVR) for each region.
* Use a *time-aware* 5-fold cross validation (`TimeSeriesSplit`) so we never
  leak future weather into past folds.
* Sweep a *wider* hyperparameter grid for each algorithm (see config).
* Persist per-fold CV scores and the full `GridSearchCV.cv_results_` table
  so the dashboard / thesis can quote mean ± std without recomputation.
* Emit "training process" figures (learning curve, validation curve, per-fold
  bar chart) that we will show in the dashboard's Tab 4.

Output artifacts (paths come from `experiment_config.json#paths`):

* ``models/{algo}_forecast_model_{region}.joblib`` — refit on the full
  training window using the best hyperparameters.
* ``models/cv_report_{region}.json`` — one JSON file per region containing a
  block per algorithm with ``best_params``, ``cv_rmse_mean``, ``cv_rmse_std``,
  ``cv_fold_scores``, ``test_rmse``, ``test_mae``, ``test_r2``, ``test_mape``,
  ``n_train_full``, ``n_train_used_for_gridsearch``, ``notes``.
* ``models/cv_results_{algo}_{region}.csv`` — full ``cv_results_`` table.
* ``figures/training/{region}/{algo}_learning_curve.png``
* ``figures/training/{region}/{algo}_validation_curve.png``
* ``figures/training/{region}/{algo}_fold_scores.png``
* ``figures/training/{region}/compare_{region}.png`` (RF vs SVM overlay)

CLI examples::

    # Train both algos for the active region only (default):
    python phase3_ai_training.py

    # Train both algos for every region in the config:
    python phase3_ai_training.py --all-regions

    # Train just one combo (smoke test):
    python phase3_ai_training.py --region kuching --algo svm \
        --svm-subsample 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    cross_val_score,
    learning_curve,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from repo_paths import repo_root, resolve_repo_path


FEATURES = ["ghiradiation", "temp", "humidity", "precip", "cloudcover", "hour", "month"]
TARGET = "Target_Power_kW"
CHRONO_SPLIT_DATE = "2024-08-08"


# ----------------------------- config helpers --------------------------------

def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_template(repo: Path, template: str, **kwargs: str) -> Path:
    return resolve_repo_path(repo, template.format(**kwargs))


# ----------------------------- data loading ----------------------------------

def _processed_csv_path(repo: Path, cfg: dict, region: str) -> Path:
    """Return the per-region processed CSV path (falls back to legacy single-region path)."""
    paths_cfg = cfg.get("paths", {}) or {}
    template = paths_cfg.get("processed_csv_template")
    if template:
        p = _resolve_template(repo, template, region=region)
        if p.exists():
            return p
    # Legacy fallback for single-region projects.
    legacy_rel = paths_cfg.get("phase1_processed_csv", "data/processed/sarawak_weather_processed.csv")
    return resolve_repo_path(repo, legacy_rel)


def load_region_dataframe(repo: Path, cfg: dict, region: str) -> pd.DataFrame:
    """Load the per-region processed CSV (the output of `phase1_split_multi_region.py`).

    Adds engineered `hour` / `month` columns and ensures the datetime index is
    tz-naive and sorted ascending. Rows with NaN in any required column are
    dropped (typical cause: leading/trailing window after resampling).
    """
    csv_path = _processed_csv_path(repo, cfg, region)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed CSV for region '{region}' not found at {csv_path}. "
            "Run `python phase1_split_multi_region.py` first."
        )
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def chronological_split(df: pd.DataFrame, split_date: str = CHRONO_SPLIT_DATE):
    """Chronological train / test split shared by every model in the project."""
    train_mask = df.index < split_date
    test_mask = df.index >= split_date
    X = df[FEATURES]
    y = df[TARGET].astype(float)
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# ----------------------------- estimator factory -----------------------------

def build_estimator(algo: str):
    """Return an UNFITTED sklearn estimator for the requested algorithm.

    Implementation note on parallelism: we deliberately set ``n_jobs=1`` on
    the inner estimator. The outer ``GridSearchCV`` (and ``learning_curve``,
    ``cross_val_score``) already runs with ``n_jobs=-1``; nesting two
    ``n_jobs=-1`` layers oversubscribes the CPU and a 540-fit RF grid would
    take 60+ minutes instead of 5-10.
    """
    if algo == "rf":
        return RandomForestRegressor(random_state=42, n_jobs=1)
    if algo == "svm":
        # StandardScaler is essential for SVR-RBF (otherwise gamma is meaningless).
        return Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    raise ValueError(f"Unknown algorithm: {algo!r} (expected 'rf' or 'svm').")


def algo_param_grid(cfg: dict, algo: str, *, smoke: bool = False) -> dict[str, list]:
    """Return the hyperparameter grid for the chosen algorithm.

    ``smoke=True`` returns a tiny grid (a few combinations only) so the smoke
    test on Kuching takes minutes, not an hour. The full grids defined in
    ``experiment_config.json#models`` are reserved for the nightly run.
    """
    grids = (cfg.get("models", {}) or {})
    if smoke:
        if algo == "rf":
            return {
                "n_estimators": [100, 200],
                "max_depth": [10, 20],
                "min_samples_split": [2],
                "min_samples_leaf": [1],
            }
        if algo == "svm":
            return {
                "svr__kernel": ["rbf"],
                "svr__C": [1.0, 10.0],
                "svr__gamma": ["scale"],
                "svr__epsilon": [0.1],
            }
        raise ValueError(algo)
    if algo == "rf":
        return dict(grids.get("rf_grid", {}))
    if algo == "svm":
        return dict(grids.get("svm_grid", {}))
    raise ValueError(algo)


# ----------------------------- metrics ---------------------------------------

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, ignoring near-zero ground-truth values.

    PV power is zero at night, so a naive MAPE would explode. We compute it
    only on hours with y_true > 0.05 kW (≈ 1% of nameplate). With no such
    hours we return NaN; the dashboard renders this as an em-dash.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0.05
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0)


def _evaluate_on_test(estimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = np.asarray(estimator.predict(X_test), dtype=float)
    y_true = np.asarray(y_test, dtype=float)
    return {
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_r2": float(r2_score(y_true, y_pred)),
        "test_mape": _mape(y_true, y_pred),
    }


# ----------------------------- subsampling -----------------------------------

def _chronological_subsample(X: pd.DataFrame, y: pd.Series, n: int, *, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Take roughly `n` rows spread evenly across the chronological train set.

    SVR with RBF kernel scales O(n²)-O(n³) in training, so we never want to
    feed the GridSearch the full 30k+ row train window. Evenly-spaced sampling
    keeps each season represented in every CV fold (vs random sampling, which
    can starve early folds).
    """
    if n >= len(X):
        return X, y
    idx = np.linspace(0, len(X) - 1, num=n).astype(int)
    return X.iloc[idx], y.iloc[idx]


# ----------------------------- training plots --------------------------------

def _save_learning_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv,
    train_sizes: list[float],
    out_path: Path,
    title: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render and save a train-vs-CV RMSE curve. Returns the raw arrays so
    `compare_{region}.png` can overlay multiple algorithms cheaply.
    """
    sizes_abs, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.array(train_sizes, dtype=float),
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        shuffle=False,
    )
    train_rmse_mean = -train_scores.mean(axis=1)
    train_rmse_std = train_scores.std(axis=1)
    val_rmse_mean = -val_scores.mean(axis=1)
    val_rmse_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sizes_abs, train_rmse_mean, "o-", color="#2E86AB", label="Train RMSE")
    ax.fill_between(sizes_abs, train_rmse_mean - train_rmse_std, train_rmse_mean + train_rmse_std,
                    color="#2E86AB", alpha=0.15)
    ax.plot(sizes_abs, val_rmse_mean, "s--", color="#F18F01", label="CV RMSE (5 folds)")
    ax.fill_between(sizes_abs, val_rmse_mean - val_rmse_std, val_rmse_mean + val_rmse_std,
                    color="#F18F01", alpha=0.15)
    ax.set_xlabel("Training set size (rows)")
    ax.set_ylabel("RMSE (kW)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return sizes_abs, val_rmse_mean, val_rmse_std


def _save_validation_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    param_name: str,
    param_range: list,
    cv,
    out_path: Path,
    title: str,
) -> None:
    # `validation_curve` requires float-castable values for the X axis. We
    # plot a string axis when the values are non-numeric (e.g. `'scale'`).
    pr_numeric = [v if isinstance(v, (int, float)) else None for v in param_range]
    # Replace `None` (e.g. `max_depth=None`) with a sentinel for plotting only.
    plot_x = [str(v) if v is not None else "None" for v in param_range]

    train_scores, val_scores = validation_curve(
        estimator, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    train_rmse_mean = -train_scores.mean(axis=1)
    val_rmse_mean = -val_scores.mean(axis=1)
    val_rmse_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(plot_x, train_rmse_mean, "o-", color="#2E86AB", label="Train RMSE")
    ax.plot(plot_x, val_rmse_mean, "s--", color="#F18F01", label="CV RMSE (5 folds)")
    ax.fill_between(plot_x, val_rmse_mean - val_rmse_std, val_rmse_mean + val_rmse_std,
                    color="#F18F01", alpha=0.15)
    ax.set_xlabel(param_name)
    ax.set_ylabel("RMSE (kW)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_fold_bar(fold_scores: list[float], *, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [f"F{i+1}" for i in range(len(fold_scores))]
    bars = ax.bar(xs, fold_scores, color="#2E86AB", alpha=0.85)
    mean_val = float(np.mean(fold_scores))
    std_val = float(np.std(fold_scores, ddof=0))
    ax.axhline(mean_val, color="black", linestyle="--", linewidth=1,
               label=f"mean = {mean_val:.3f} ± {std_val:.3f}")
    for b, v in zip(bars, fold_scores):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (kW)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_compare_curve(
    rf_curve: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    svm_curve: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    *,
    out_path: Path,
    region: str,
) -> None:
    if rf_curve is None and svm_curve is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if rf_curve is not None:
        x, y, ystd = rf_curve
        ax.plot(x, y, "o-", color="#2E86AB", label="RF CV RMSE")
        ax.fill_between(x, y - ystd, y + ystd, color="#2E86AB", alpha=0.15)
    if svm_curve is not None:
        x, y, ystd = svm_curve
        ax.plot(x, y, "s--", color="#F18F01", label="SVM CV RMSE")
        ax.fill_between(x, y - ystd, y + ystd, color="#F18F01", alpha=0.15)
    ax.set_xlabel("Training set size (rows)")
    ax.set_ylabel("RMSE (kW)")
    ax.set_title(f"RF vs SVM learning curves — {region}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------- main training routine -------------------------

@dataclass
class TrainResult:
    algo: str
    region: str
    best_params: dict[str, Any]
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_fold_scores: list[float]
    test_rmse: float
    test_mae: float
    test_r2: float
    test_mape: float
    n_train_full: int
    n_train_used_for_gridsearch: int
    notes: str
    learning_curve: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    seconds_elapsed: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "cv_rmse_mean": float(self.cv_rmse_mean),
            "cv_rmse_std": float(self.cv_rmse_std),
            "cv_fold_scores": [float(x) for x in self.cv_fold_scores],
            "test_rmse": float(self.test_rmse),
            "test_mae": float(self.test_mae),
            "test_r2": float(self.test_r2),
            "test_mape": float(self.test_mape),
            "n_train_full": int(self.n_train_full),
            "n_train_used_for_gridsearch": int(self.n_train_used_for_gridsearch),
            "seconds_elapsed": float(self.seconds_elapsed),
            "notes": self.notes,
        }


def train_one_model(
    algo: str,
    region: str,
    cfg: dict,
    *,
    repo: Path | None = None,
    svm_subsample_override: int | None = None,
    skip_figures: bool = False,
    smoke: bool = False,
) -> TrainResult:
    """Train + tune + evaluate one (algo, region) pair end-to-end."""
    if repo is None:
        repo = repo_root()
    paths_cfg = cfg.get("paths", {}) or {}
    models_cfg = cfg.get("models", {}) or {}
    cv_splits = int(models_cfg.get("cv_splits", 5))
    train_sizes_frac = list(models_cfg.get("learning_curve_train_sizes", [0.1, 0.28, 0.46, 0.64, 0.82, 1.0]))

    t_total = time.perf_counter()
    df = load_region_dataframe(repo, cfg, region)
    X_train, X_test, y_train, y_test = chronological_split(df)
    n_train_full = int(len(X_train))
    if n_train_full == 0:
        raise RuntimeError(f"Empty training set for region {region!r}; chronological split removed everything.")

    cv = TimeSeriesSplit(n_splits=cv_splits)

    # ---- gridsearch input (RF uses full; SVM uses a subsample for tractability)
    if algo == "svm":
        n_target = int(svm_subsample_override if svm_subsample_override is not None
                        else models_cfg.get("svm_subsample_rows", 8000))
        X_gs, y_gs = _chronological_subsample(X_train, y_train, n_target)
        notes = (
            f"SVR-RBF GridSearchCV ran on a chronological subsample of {len(X_gs)} rows "
            f"from {n_train_full} train rows to keep CV tractable; best params then refit "
            f"on the full training window."
        )
    else:
        X_gs, y_gs = X_train, y_train
        notes = "GridSearchCV used the full training window."
    n_train_used_for_gridsearch = int(len(X_gs))

    # ---- grid search
    estimator = build_estimator(algo)
    param_grid = algo_param_grid(cfg, algo, smoke=smoke)

    n_combinations = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    n_fits = n_combinations * cv_splits
    print(f"[{region}/{algo}] GridSearchCV: {n_combinations} param combos × {cv_splits} folds = {n_fits} fits "
          f"on {n_train_used_for_gridsearch} rows...")
    t_gs = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        gs = GridSearchCV(
            estimator,
            param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            refit=True,
            verbose=1,
        )
        gs.fit(X_gs, y_gs)
    print(f"[{region}/{algo}] GridSearchCV finished in {time.perf_counter()-t_gs:.1f}s; best = {gs.best_params_}")

    # Persist the full cv_results_ table for reviewers.
    cv_results_path = _resolve_template(
        repo, paths_cfg.get("cv_results_template", "models/cv_results_{algo}_{region}.csv"),
        algo=algo, region=region,
    )
    cv_results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gs.cv_results_).to_csv(cv_results_path, index=False)

    # ---- refit best params on the FULL training window so deployment-time
    # accuracy on the test split reflects the production model, not the
    # subsample. For RF, the gs.best_estimator_ is already trained on
    # X_train; for SVM we re-fit on X_train (slow but only once per region).
    if algo == "svm":
        if smoke:
            # In smoke mode we keep the GridSearchCV best_estimator_ trained on
            # the subsample only. Skipping the full-window refit saves ~10 min
            # per region. The saved model still implements the `.predict` API,
            # so run_all.py / phase4 can exercise the pipeline end-to-end.
            final_model = gs.best_estimator_
            print(f"[{region}/{algo}] smoke mode: skipping full-window refit "
                  f"(model trained on {n_train_used_for_gridsearch} subsample rows)")
        else:
            final_model = build_estimator(algo).set_params(**gs.best_params_)
            print(f"[{region}/{algo}] refitting SVR on full {n_train_full} train rows...")
            t_rf = time.perf_counter()
            final_model.fit(X_train, y_train)
            print(f"[{region}/{algo}] refit done in {time.perf_counter()-t_rf:.1f}s")
    else:
        final_model = gs.best_estimator_
        # Re-enable thread parallelism on the saved RF for fast predict() at
        # inference time (Phase 4 / dashboard). Training used n_jobs=1 only to
        # avoid nested oversubscription with GridSearchCV's n_jobs=-1.
        try:
            final_model.set_params(n_jobs=-1)
        except Exception:
            pass

    # ---- per-fold CV scores using best estimator (so reported mean ± std is for the chosen hyperparameters)
    best_idx = int(gs.best_index_)
    cv_fold_scores = [
        float(-gs.cv_results_[f"split{i}_test_score"][best_idx])
        for i in range(cv_splits)
    ]
    cv_rmse_mean = float(np.mean(cv_fold_scores))
    cv_rmse_std = float(np.std(cv_fold_scores, ddof=0))

    # ---- test metrics
    test_metrics = _evaluate_on_test(final_model, X_test, y_test)

    # ---- persist model
    model_path = _resolve_template(
        repo, paths_cfg.get("model_template", "models/{algo}_forecast_model_{region}.joblib"),
        algo=algo, region=region,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"[{region}/{algo}] saved model -> {model_path}")

    # ---- back-compat: if this is the active region + algo='rf', also write to the legacy single-region path.
    active_region = str(cfg.get("regions", {}).get("active", region))
    legacy_rf_path_rel = paths_cfg.get("rf_model_path")
    if algo == "rf" and region == active_region and legacy_rf_path_rel:
        legacy_path = resolve_repo_path(repo, legacy_rf_path_rel)
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, legacy_path)
        print(f"[{region}/{algo}] also wrote legacy alias -> {legacy_path}")

    # ---- figures (learning curve, validation curve, per-fold bars)
    lc_arrays = None
    if not skip_figures:
        figures_dir = resolve_repo_path(repo, f"figures/training/{region}")
        figures_dir.mkdir(parents=True, exist_ok=True)
        # Same data slice we used for GridSearch (keeps wall-clock honest for SVM).
        lc_X, lc_y = (X_gs, y_gs) if algo == "svm" else (X_train, y_train)
        try:
            lc_arrays = _save_learning_curve(
                build_estimator(algo).set_params(**gs.best_params_),
                lc_X, lc_y,
                cv=TimeSeriesSplit(n_splits=cv_splits),
                train_sizes=train_sizes_frac,
                out_path=figures_dir / f"{algo}_learning_curve.png",
                title=f"Learning curve — {region}/{algo.upper()}",
            )
        except Exception as exc:
            print(f"[{region}/{algo}] WARN: learning_curve failed: {exc}")

        # Validation curve over the most informative hyperparameter.
        vc_param = models_cfg.get(
            "validation_curve_rf_param" if algo == "rf" else "validation_curve_svm_param",
            "max_depth" if algo == "rf" else "svr__C",
        )
        vc_range = list(models_cfg.get(
            "validation_curve_rf_range" if algo == "rf" else "validation_curve_svm_range",
            [5, 10, 20, 30] if algo == "rf" else [0.1, 1.0, 10.0, 100.0],
        ))
        try:
            _save_validation_curve(
                build_estimator(algo),
                lc_X, lc_y,
                param_name=vc_param,
                param_range=vc_range,
                cv=TimeSeriesSplit(n_splits=cv_splits),
                out_path=figures_dir / f"{algo}_validation_curve.png",
                title=f"Validation curve ({vc_param}) — {region}/{algo.upper()}",
            )
        except Exception as exc:
            print(f"[{region}/{algo}] WARN: validation_curve failed: {exc}")

        try:
            _save_fold_bar(
                cv_fold_scores,
                out_path=figures_dir / f"{algo}_fold_scores.png",
                title=f"5-fold CV RMSE per fold — {region}/{algo.upper()}",
            )
        except Exception as exc:
            print(f"[{region}/{algo}] WARN: fold-bar plot failed: {exc}")

    return TrainResult(
        algo=algo,
        region=region,
        best_params={str(k): (v if v is None or isinstance(v, (int, float, str, bool)) else str(v))
                     for k, v in gs.best_params_.items()},
        cv_rmse_mean=cv_rmse_mean,
        cv_rmse_std=cv_rmse_std,
        cv_fold_scores=cv_fold_scores,
        n_train_full=n_train_full,
        n_train_used_for_gridsearch=n_train_used_for_gridsearch,
        notes=notes,
        learning_curve=lc_arrays,
        seconds_elapsed=time.perf_counter() - t_total,
        **test_metrics,
    )


def _write_cv_report(repo: Path, cfg: dict, region: str, results: dict[str, TrainResult]) -> Path:
    """Write the per-region CV report, merging with any pre-existing JSON.

    Merging lets you re-train one algorithm at a time (e.g. ``--algo rf``)
    without erasing the other algorithm's block from a previous run. The
    top-level metadata (cv_strategy, features, etc.) is always rewritten
    from the current config so a re-train with new settings can never end
    up with stale metadata pointing at fresh algorithm rows.
    """
    paths_cfg = cfg.get("paths", {}) or {}
    out_path = _resolve_template(
        repo, paths_cfg.get("cv_report_template", "models/cv_report_{region}.json"),
        region=region,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_algos: dict[str, Any] = {}
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(prior, dict) and isinstance(prior.get("algorithms"), dict):
                existing_algos = dict(prior["algorithms"])
        except Exception:
            existing_algos = {}

    merged_algos: dict[str, Any] = dict(existing_algos)
    for algo, tr in results.items():
        merged_algos[algo] = tr.to_dict()

    payload = {
        "region": region,
        "cv_strategy": "TimeSeriesSplit",
        "cv_splits": int((cfg.get("models", {}) or {}).get("cv_splits", 5)),
        "chronological_split_date": CHRONO_SPLIT_DATE,
        "features": list(FEATURES),
        "target": TARGET,
        "algorithms": merged_algos,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def train_region(
    region: str,
    cfg: dict,
    *,
    algos: list[str] | None = None,
    repo: Path | None = None,
    svm_subsample_override: int | None = None,
    skip_figures: bool = False,
    smoke: bool = False,
) -> dict[str, TrainResult]:
    """Train every requested algorithm for a single region and write the CV report.

    Returns the per-algo `TrainResult`s so callers (e.g. `run_all.py`) can
    summarise across regions without re-loading the JSON.
    """
    if repo is None:
        repo = repo_root()
    algos = list(algos or (cfg.get("models", {}) or {}).get("algorithms", ["rf", "svm"]))

    results: dict[str, TrainResult] = {}
    for algo in algos:
        results[algo] = train_one_model(
            algo, region, cfg,
            repo=repo,
            svm_subsample_override=svm_subsample_override,
            skip_figures=skip_figures,
            smoke=smoke,
        )

    # RF vs SVM comparison plot (only if both were trained AND figures are enabled)
    if not skip_figures and "rf" in results and "svm" in results:
        try:
            _save_compare_curve(
                results["rf"].learning_curve,
                results["svm"].learning_curve,
                out_path=resolve_repo_path(repo, f"figures/training/{region}/compare_{region}.png"),
                region=region,
            )
        except Exception as exc:
            print(f"[{region}] WARN: compare plot failed: {exc}")

    report_path = _write_cv_report(repo, cfg, region, results)
    print(f"[{region}] CV report -> {report_path}")
    return results


# ----------------------------- CLI -------------------------------------------

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="experiment_config.json")
    p.add_argument("--region", default=None,
                   help="region name (default: regions.active in config)")
    p.add_argument("--all-regions", action="store_true",
                   help="train every region listed in the config (overrides --region)")
    p.add_argument("--algo", default=None, choices=("rf", "svm"),
                   help="if provided, train only this algorithm; default = all in config")
    p.add_argument("--svm-subsample", type=int, default=None,
                   help="override models.svm_subsample_rows (smoke-test knob)")
    p.add_argument("--skip-figures", action="store_true", help="skip training-process figures")
    p.add_argument("--smoke", action="store_true",
                   help="use small built-in grids (a few combos per algo) for fast iteration; "
                        "do NOT use for thesis numbers — the full grids live in config")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = repo_root()
    cfg = _load_config(resolve_repo_path(repo, args.config))

    region_list = list((cfg.get("regions", {}) or {}).get("list", []))
    if not region_list:
        raise RuntimeError(
            "experiment_config.json#regions.list is empty. Define your regions before training."
        )
    if args.all_regions:
        regions_to_run = [r["name"] for r in region_list]
    else:
        active = args.region or str(cfg.get("regions", {}).get("active", region_list[0]["name"]))
        regions_to_run = [active]

    algos = [args.algo] if args.algo else None

    summary_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for region in regions_to_run:
        print(f"\n=== Training region: {region} ===")
        results = train_region(
            region, cfg,
            algos=algos,
            repo=repo,
            svm_subsample_override=args.svm_subsample,
            skip_figures=args.skip_figures,
            smoke=args.smoke,
        )
        for algo, tr in results.items():
            summary_rows.append({
                "region": region,
                "algo": algo,
                "cv_rmse_mean": tr.cv_rmse_mean,
                "cv_rmse_std": tr.cv_rmse_std,
                "test_rmse": tr.test_rmse,
                "test_r2": tr.test_r2,
                "seconds": round(tr.seconds_elapsed, 1),
            })
    print(f"\nAll training done in {time.perf_counter()-t0:.1f}s")
    if summary_rows:
        print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
