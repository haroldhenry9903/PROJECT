import json
import sys
from pathlib import Path

# Add repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import joblib
from copy import deepcopy

from core_battery_model import Battery
from phase4_smart_optimizer import simulate as phase4_simulate
from repo_paths import repo_root, resolve_repo_path

def main():
    repo = repo_root()
    
    # Load config
    with open(repo / "experiment_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    regions = cfg["regions"]["list"]
    algos = cfg["models"]["algorithms"]
    
    # Create perfect foresight config
    perf_cfg = deepcopy(cfg)
    perf_cfg["forecast"]["enabled"] = True
    perf_cfg["forecast"]["ghi_noise_sigma"] = 0.0
    perf_cfg["forecast"]["ghi_bias"] = 0.0
    perf_cfg["forecast"]["cloudcover_noise_sigma"] = 0.0
    perf_cfg["forecast"]["temp_noise_sigma"] = 0.0
    perf_cfg["forecast"]["humidity_noise_sigma"] = 0.0
    perf_cfg["forecast"]["precip_noise_sigma"] = 0.0
    
    results = []
    
    # Load summary to get baseline & headline smart savings
    summary_path = repo / "results" / "20260513_1705" / "summary.csv"
    summary_df = pd.read_csv(summary_path)
    
    for r in regions:
        rname = r["name"]
        for algo in algos:
            print(f"Calculating perfect foresight for {rname} ({algo})...")
            
            # Paths
            base_csv = repo / f"results/_per_region/{rname}_baseline.csv"
            model_path = repo / f"models/{algo}_forecast_model_{rname}.joblib"
            
            # Load baseline df
            df_base = pd.read_csv(base_csv)
            df_base["datetime"] = pd.to_datetime(df_base["datetime"], errors="coerce")
            df_base = df_base.set_index("datetime").sort_index()
            
            # Load model
            model = joblib.load(model_path)
            
            # Run simulation
            df_perf = phase4_simulate(df_base, model, perf_cfg, verbose=False)
            
            # Calculate metrics
            perf_cost = float(df_perf["Hourly_Cost_RM"].astype(float).sum())
            base_cost = float(df_base["Hourly_Cost_RM"].astype(float).sum())
            perf_savings = base_cost - perf_cost
            
            # Find the row in summary_df
            match_row = summary_df.loc[(summary_df["region"] == rname) & (summary_df["algo"] == algo)]
            if len(match_row) > 0:
                headline_savings = float(match_row.iloc[0]["annual_savings_rm"])
                pct_ceiling = (headline_savings / perf_savings) * 100.0 if perf_savings > 0 else 0.0
                results.append({
                    "region": rname,
                    "algo": algo,
                    "baseline_cost": base_cost,
                    "headline_savings": headline_savings,
                    "perfect_savings": perf_savings,
                    "pct_ceiling": pct_ceiling
                })
                print(f"  Base: {base_cost:.2f}, Headline: {headline_savings:.2f}, Perfect: {perf_savings:.2f}, Pct: {pct_ceiling:.2f}%")
            else:
                print(f"  No match found in summary for {rname} ({algo})")
                
    # Output markdown table
    print("\n--- RESULTS TABLE ---")
    print("| Region | Algo | Baseline Cost (RM) | Headline Savings (RM) | Perfect Foresight Savings (RM) | Pct of Perfect Ceiling (%) |")
    print("|---|---|---|---|---|---|")
    for res in results:
        print(f"| {res['region'].title()} | {res['algo'].upper()} | {res['baseline_cost']:.2f} | {res['headline_savings']:.2f} | {res['perfect_savings']:.2f} | {res['pct_ceiling']:.2f}% |")

if __name__ == "__main__":
    main()
