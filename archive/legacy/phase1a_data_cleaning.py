import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from repo_paths import repo_root, resolve_repo_path

def load_and_clean_data(input_file: str, output_file: str, eda_dir: str):
    print(f"Loading raw data from {input_file}...")
    df_raw = pd.read_csv(input_file)
    
    print("\n--- Data Trust Report: Raw Data ---")
    print(f"Total Rows: {len(df_raw)}")
    missing_before = df_raw.isnull().sum()
    print("Missing Values before cleaning:")
    print(missing_before[missing_before > 0] if missing_before.sum() > 0 else "No missing values!")
    
    os.makedirs(eda_dir, exist_ok=True)
    
    print("\n[Step 1] Parsing datetime and setting index...")
    df = df_raw.copy()
    
    # Fallback parsing strategy in case the dataset has mixed date formats
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed', utc=False)
    # Remove timezone info if it exists so we can localize it fresh
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)

    df.set_index('datetime', inplace=True)
    
    print("[Step 2] Localizing timezone to Asia/Kuching...")
    df = df.tz_localize('Asia/Kuching', ambiguous='infer', nonexistent='shift_forward')
        
    print("[Step 3] Resampling to 1H frequency to ensure strict continuous timeline...")
    # This guarantees no missing hours in the dataset
    df = df.resample('1H').asfreq()
    
    print("[Step 4] Handling missing data via explicit Forward Fill and Interpolation...")
    # Interpolate linearly for gaps up to 3 hours
    df = df.interpolate(method='linear', limit=3)
    # Forward-fill for any remaining longer gaps
    df = df.bfill().ffill()  # Use bfill to catch any leading NaNs
    
    print("[Step 5] Enforcing physical baseline constraints (>= 0.0) for radiative parameters...")
    cols_to_clip = ['precip', 'solarradiation', 'ghiradiation', 'dniradiation', 'difradiation']
    for col in cols_to_clip:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)
            
    print("\n--- Data Trust Report: Cleaned Data ---")
    print(f"Total Rows after processing: {len(df)}")
    missing_after = df.isnull().sum()
    print(f"Missing Values remaining: {missing_after.sum()} (Should be 0)")
    
    # ---------------------------------------------------------
    # EDA Plots for "Trust & Transparency"
    # ---------------------------------------------------------
    print("\nGenerating Exploratory Data Analysis (EDA) Plots to prove data validity...")
    
    # 1. Correlation Matrix
    test_cols = ['temp', 'humidity', 'precip', 'cloudcover', 'solarradiation', 'ghiradiation']
    test_cols = [c for c in test_cols if c in df.columns]
    corr = df[test_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Meteorological Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'correlation_matrix.png'))
    plt.close()
    
    # 2. Daily GHI Distribution
    plt.figure(figsize=(10, 4))
    if 'ghiradiation' in df.columns:
        df_plot = df.dropna(subset=['ghiradiation'])
        sns.histplot(df_plot['ghiradiation'][df_plot['ghiradiation'] > 0], bins=60, kde=True, color='orange')
        plt.title("Distribution of Active Global Horizontal Irradiance (GHI > 0)")
        plt.xlabel("Global Horizontal Irradiance (W/m2)")
        plt.ylabel("Frequency (Hours)")
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, 'ghi_distribution.png'))
        plt.close()

    # 3. Monthly Average Temp to show seasonal variation (5 years)
    if 'temp' in df.columns:
        monthly_temp = df['temp'].resample('ME').mean() # Ensure we use 'ME' instead of 'M' to avoid Warnings
        plt.figure(figsize=(12, 4))
        plt.plot(monthly_temp.index, monthly_temp.values, marker='o', color='red', linestyle='-')
        plt.title("Monthly Average Temperature over Entire Dataset (Seasonal Validation)")
        plt.ylabel("Temperature (°C)")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, 'monthly_temperature_trend.png'))
        plt.close()
        
    print(f"Saving explicitly cleaned and trusted dataset to {output_file}...")
    df.to_csv(output_file)
    print("Phase 1A: Data Cleaning and Verification completed successfully!")

if __name__ == '__main__':
    _repo = repo_root()
    load_and_clean_data(
        input_file=str(resolve_repo_path(_repo, "data/raw/sarawak_weather20_25.csv")),
        output_file=str(resolve_repo_path(_repo, "data/processed/sarawak_weather_cleaned.csv")),
        eda_dir=str(resolve_repo_path(_repo, "eda_metrics")),
    )
