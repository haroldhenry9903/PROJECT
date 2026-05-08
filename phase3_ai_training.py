import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from repo_paths import repo_root, resolve_repo_path


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_and_engineer_features(filepath: str) -> pd.DataFrame:
    """
    Step 5: Time-Series Data preparation
    1. Load data
    2. Feature Engineering: Create 'hour' and 'month' from datetime
    """
    print(f"Loading processed data from {filepath}...")
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    print("Engineering temporal features (hour, month)...")
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    
    return df

def split_time_series_data(df: pd.DataFrame):
    """
    Splits the data chronologically into train and test sets.
    Features (X): ghiradiation, temp, humidity, precip, cloudcover, hour, month.
    Target (y): Target_Power_kW.
    Train: Jan 1, 2023 - Aug 7, 2024
    Test: Aug 8, 2024 - Jan 1, 2025
    """
    features = ['ghiradiation', 'temp', 'humidity', 'precip', 'cloudcover', 'hour', 'month']
    target = 'Target_Power_kW'
    
    # We must ensure no missing values exist in our features or target
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    print("Performing chronological train-test split...")
    # Training Set: Before 2024-08-08
    train_mask = df.index < '2024-08-08'
    
    # Testing Set: On or after 2024-08-08
    test_mask = df.index >= '2024-08-08'
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_save_path: str):
    """
    Step 6: Model Training & Tuning
    1. RandomForestRegressor initialization
    2. GridSearchCV for hyperparameter tuning
    3. Train the model
    4. Evaluate (RMSE, R2)
    5. Save best model with joblib
    """
    print("Initializing Random Forest Regressor and GridSearchCV...")
    rf = RandomForestRegressor(random_state=42)
    
    # Grid search parameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None]
    }
    
    # Set up Grid Search with cross-validation (cv=3 for speed on standard datasets)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        n_jobs=-1, # Use all available CPU cores
        verbose=1
    )
    
    print("Training the model (this may take a few moments)...")
    grid_search.fit(X_train, y_train)
    
    # Extract the absolute best model
    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    
    print("Evaluating best model on Testing Set...")
    # Predict on unseen test data
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 30)
    print("Model Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f} kW")
    print(f"R-squared (R2): {r2:.4f}")
    print("-" * 30)
    
    print(f"Saving the best model to {model_save_path}...")
    joblib.dump(best_model, model_save_path)
    print("Phase 3: AI Training completed successfully!")

def main():
    repo = repo_root()
    config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "experiment_config.json"
    cfg_path = resolve_repo_path(repo, config_path)
    cfg = _load_config(str(cfg_path))
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}

    input_filepath = str(
        resolve_repo_path(repo, paths.get("phase1_processed_csv", "data/processed/sarawak_weather_processed.csv"))
    )
    model_save_path = str(resolve_repo_path(repo, paths.get("rf_model_path", "models/rf_forecast_model.joblib")))

    # 1. Load and engineer features
    df = load_and_engineer_features(input_filepath)
    
    # 2. Chronological Split
    X_train, X_test, y_train, y_test = split_time_series_data(df)
    
    # 3. Train, Evaluate, and Save
    train_and_evaluate_model(X_train, X_test, y_train, y_test, model_save_path)

if __name__ == "__main__":
    main()

