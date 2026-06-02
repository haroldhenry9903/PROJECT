import json
import sys
from pathlib import Path

# Add repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from repo_paths import repo_root

def main():
    repo = repo_root()
    regions = ["kuching", "miri", "bintulu", "kapit", "sri_aman"]
    
    real_data = []
    
    for r in regions:
        path = repo / f"models/cv_report_{r}.json"
        if not path.exists():
            print(f"File not found: {path}")
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for algo in ["rf", "svm"]:
            block = data["algorithms"][algo]
            bp = block["best_params"]
            
            # Format params
            if algo == "rf":
                param_str = f"est={bp['n_estimators']}, depth={bp['max_depth']}, split={bp['min_samples_split']}, leaf={bp['min_samples_leaf']}"
            else:
                param_str = f"C={bp['svr__C']}, gamma={bp['svr__gamma']}, eps={bp['svr__epsilon']}"
                
            real_data.append({
                "Region": r.title() if r != "sri_aman" else "Sri Aman",
                "Algo": algo.upper(),
                "Best Params": param_str,
                "RMSE": block["test_rmse"],
                "MAE": block["test_mae"],
                "R2": block["test_r2"],
                "MAPE": block["test_mape"]
            })
            
    print("\n=== ACTUAL VERIFIED METRICS FROM CV REPORTS ===")
    print("| Region | Algo | Best Params | Test RMSE | Test MAE | Test R2 | Test MAPE (%) |")
    print("|---|---|---|---|---|---|---|")
    for res in real_data:
        print(f"| {res['Region']} | {res['Algo']} | {res['Best Params']} | {res['RMSE']:.4f} | {res['MAE']:.4f} | {res['R2']:.4f} | {res['MAPE']:.2f}% |")
    
    # Draft data for comparison
    draft_data = [
        {"Region": "Kuching", "Algo": "RF", "Best Params": "est=200, depth=20, split=2, leaf=4", "RMSE": 0.043, "MAE": 0.012, "R2": 0.998, "MAPE": 3.06},
        {"Region": "Kuching", "Algo": "SVM", "Best Params": "C=100, gamma=0.01, eps=0.01", "RMSE": 0.044, "MAE": 0.015, "R2": 0.998, "MAPE": 0.97},
        {"Region": "Miri", "Algo": "RF", "Best Params": "est=400, depth=∞ (null), split=2, leaf=4", "RMSE": 0.028, "MAE": 0.008, "R2": 0.999, "MAPE": 2.02},
        {"Region": "Miri", "Algo": "SVM", "Best Params": "C=100, gamma=0.01, eps=0.01", "RMSE": 0.029, "MAE": 0.010, "R2": 0.999, "MAPE": 1.33},
        {"Region": "Bintulu", "Algo": "RF", "Best Params": "est=200, depth=∞, split=2, leaf=2", "RMSE": 0.024, "MAE": 0.008, "R2": 1.000, "MAPE": 2.05},
        {"Region": "Bintulu", "Algo": "SVM", "Best Params": "C=100, gamma=0.01, eps=0.01", "RMSE": 0.018, "MAE": 0.009, "R2": 1.000, "MAPE": 1.36},
        {"Region": "Kapit", "Algo": "RF", "Best Params": "est=400, depth=20, split=2, leaf=4", "RMSE": 0.026, "MAE": 0.008, "R2": 0.999, "MAPE": 2.22},
        {"Region": "Kapit", "Algo": "SVM", "Best Params": "C=100, gamma=0.01, eps=0.01", "RMSE": 0.026, "MAE": 0.009, "R2": 0.999, "MAPE": 1.28},
        {"Region": "Sri Aman", "Algo": "RF", "Best Params": "est=400, depth=∞, split=2, leaf=2", "RMSE": 0.030, "MAE": 0.008, "R2": 0.999, "MAPE": 1.93},
        {"Region": "Sri Aman", "Algo": "SVM", "Best Params": "C=100, gamma=0.01, eps=0.01", "RMSE": 0.032, "MAE": 0.010, "R2": 0.999, "MAPE": 1.08}
    ]
    df_draft = pd.DataFrame(draft_data)
    
    print("\n=== COMPARISON REPORT ===")
    for idx, row in df.iterrows():
        reg = row["Region"]
        algo = row["Algo"]
        draft_row = df_draft[(df_draft["Region"] == reg) & (df_draft["Algo"] == algo)].iloc[0]
        
        bp_diff = "" if row["Best Params"] == draft_row["Best Params"] else f" (Draft: '{draft_row['Best Params']}')"
        rmse_diff = "" if abs(row["RMSE"] - draft_row["RMSE"]) < 0.0015 else f" (Draft: {draft_row['RMSE']})"
        mae_diff = "" if abs(row["MAE"] - draft_row["MAE"]) < 0.0015 else f" (Draft: {draft_row['MAE']})"
        r2_diff = "" if abs(row["R2"] - draft_row["R2"]) < 0.0015 else f" (Draft: {draft_row['R2']})"
        mape_diff = "" if abs(row["MAPE"] - draft_row["MAPE"]) < 0.05 else f" (Draft: {draft_row['MAPE']})"
        
        if bp_diff or rmse_diff or mae_diff or r2_diff or mape_diff:
            print(f"Discrepancies for {reg} ({algo}):")
            if bp_diff:
                print(f"  Best Params: Real '{row['Best Params']}'{bp_diff}")
            if rmse_diff:
                print(f"  RMSE: Real {row['RMSE']:.4f}{rmse_diff}")
            if mae_diff:
                print(f"  MAE: Real {row['MAE']:.4f}{mae_diff}")
            if r2_diff:
                print(f"  R2: Real {row['R2']:.4f}{r2_diff}")
            if mape_diff:
                print(f"  MAPE: Real {row['MAPE']:.2f}%{mape_diff}%")
        else:
            print(f"✓ {reg} ({algo}) matches perfectly.")

if __name__ == "__main__":
    main()
