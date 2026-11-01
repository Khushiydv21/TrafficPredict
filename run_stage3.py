"""
Stage 3 Runner Script: Model Training, Evaluation & Multi-Horizon Forecasting
Author: Khushi Yadav
Project: TrafficPredict - Website Traffic Forecasting System

Execute this file using:
python run_stage3.py
"""

import os
import sys
import json
import joblib
from pathlib import Path

# Add project root directory to Python path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
os.environ["MPLCONFIGDIR"] = str(PROJECT_DIR / ".mpl_cache")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import chronological_train_test_split
from src.forecast import (
    evaluate_all_models,
    train_companion_visitor_models,
    generate_future_forecasts,
    FEATURE_COLS
)


def main():
    print("=" * 85)
    print("  TRAFFICPREDICT - STAGE 3: MODEL TRAINING, EVALUATION & MULTI-HORIZON FORECASTING")
    print("=" * 85)

    data_path = PROJECT_DIR / "data" / "features_website_traffic.csv"
    models_dir = PROJECT_DIR / "models"
    figures_dir = PROJECT_DIR / "reports" / "figures"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Step 1: Load Feature Dataset
    print(f"\n[Step 1] Loading engineered feature dataset from: {data_path}")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"  --> Loaded {len(df):,} records with {df.shape[1]} columns.")

    # Step 2: Chronological Train-Test Split (90-day Holdout)
    print("\n[Step 2] Chronological Train-Test Split (No Data Leakage):")
    train_df, test_df = chronological_train_test_split(df, test_days=90)
    print(f"  * Training set : {len(train_df):,} days ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"  * Holdout test : {len(test_df):,} days ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    # Step 3: Train and Evaluate All 4 Models across Horizons
    print("\n[Step 3] Training & Evaluating 4 Forecasting Models on Holdout Horizons (7, 15, 30 Days):")
    horizons = [7, 15, 30]
    metrics_results, trained_models = evaluate_all_models(train_df, test_df, horizons=horizons)

    # Display comparison table
    print("\n" + "-" * 85)
    print(f"{'Algorithm / Model':<28} | {'Horizon':<8} | {'MAE':<8} | {'RMSE':<8} | {'MAPE (%)':<10} | {'R²':<8}")
    print("-" * 85)
    for model_name, data in metrics_results.items():
        for h in horizons:
            m = data["horizons"][str(h)]
            print(f"{model_name:<28} | {str(h)+' Days':<8} | {m['MAE']:<8.2f} | {m['RMSE']:<8.2f} | {m['MAPE']:<10.2f} | {m['R2']:<8.4f}")
        print("-" * 85)

    # Identify Best Model based on lowest average MAPE
    avg_mapes = {
        name: np.mean([data["horizons"][str(h)]["MAPE"] for h in horizons])
        for name, data in metrics_results.items()
    }
    best_model_name = min(avg_mapes, key=avg_mapes.get)
    best_model = trained_models[best_model_name]
    print(f"\n  >>> BEST OVERALL MODEL: {best_model_name} (Average MAPE: {avg_mapes[best_model_name]:.2f}%)")

    # Step 4: Train Companion Visitor Models
    print("\n[Step 4] Fitting Companion Models for Visitor Segments (Unique, First-Time, Returning):")
    companion_models = train_companion_visitor_models(train_df)
    print("  --> Companion models fitted successfully on training partition.")

    # Step 5: Generate Out-of-Sample Future Forecasts
    print("\n[Step 5] Generating Future Out-of-Sample Forecasts (Beyond 2020-08-19):")
    future_df = generate_future_forecasts(df, best_model, companion_models, max_horizon=30)
    print(f"  * Generated forecasts from {future_df['Date'].iloc[0]} to {future_df['Date'].iloc[-1]} (30 days ahead).")
    print("\n  Sample 7-Day Future Traffic Forecast:")
    print(future_df.head(7).to_string(index=False))

    # Step 6: Save Artifacts for Dashboard Integration
    print(f"\n[Step 6] Saving Models, Metrics JSON, and Forecasts in: {models_dir}")
    
    # 6.1 Save metrics.json
    metrics_file = models_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_results, f, indent=2)
    print(f"  --> Saved: {metrics_file.name}")

    # 6.2 Save future forecasts
    forecast_file = models_dir / "future_forecasts.json"
    future_df.to_json(forecast_file, orient="records", indent=2)
    print(f"  --> Saved: {forecast_file.name}")

    # 6.3 Save best model and companions
    joblib.dump(best_model, models_dir / "best_model.joblib")
    joblib.dump(companion_models, models_dir / "companion_models.joblib")
    print("  --> Saved: best_model.joblib and companion_models.joblib")

    # Step 7: Generate Visual Comparison Charts
    print(f"\n[Step 7] Generating Publication-Quality Figures in: {figures_dir}")
    sns.set_theme(style="whitegrid")

    # Chart 1: Actual vs Predicted (30 Days Holdout)
    plt.figure(figsize=(12, 5))
    h30_dates = pd.to_datetime(metrics_results[best_model_name]["predictions"]["30"]["dates"])
    actual_30 = metrics_results[best_model_name]["predictions"]["30"]["actual"]
    pred_30 = metrics_results[best_model_name]["predictions"]["30"]["predicted"]
    
    plt.plot(h30_dates, actual_30, label="Actual Page Loads", color="#1f77b4", marker="o", linewidth=2)
    plt.plot(h30_dates, pred_30, label=f"Predicted ({best_model_name})", color="#ff7f0e", marker="s", linestyle="--", linewidth=2)
    plt.title(f"30-Day Holdout Forecast: Actual vs Predicted ({best_model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Page Loads", fontsize=11)
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figures_dir / "actual_vs_predicted_30d.png", dpi=150)
    plt.close()

    # Chart 2: Model MAE Comparison by Horizon
    plt.figure(figsize=(10, 5))
    bar_data = []
    for m_name in metrics_results:
        for h in [7, 15, 30]:
            bar_data.append({
                "Model": m_name,
                "Horizon": f"{h} Days",
                "MAE": metrics_results[m_name]["horizons"][str(h)]["MAE"]
            })
    bdf = pd.DataFrame(bar_data)
    sns.barplot(data=bdf, x="Horizon", y="MAE", hue="Model", palette="tab10")
    plt.title("Model Error (MAE) Comparison across 7, 15, and 30-Day Horizons", fontsize=13, fontweight="bold")
    plt.ylabel("Mean Absolute Error (Page Loads)", fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_mae.png", dpi=150)
    plt.close()

    # Chart 3: Future 30-Day Out-of-Sample Forecast
    plt.figure(figsize=(12, 5))
    fut_dates = pd.to_datetime(future_df["Date"])
    plt.plot(fut_dates, future_df["Page.Loads"], marker="o", color="#2ca02c", linewidth=2, label="Forecasted Page Loads")
    plt.title("Out-of-Sample Forecast: Next 30 Days Traffic (Page.Loads)", fontsize=13, fontweight="bold")
    plt.xlabel("Forecast Date", fontsize=11)
    plt.ylabel("Page Loads", fontsize=11)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "future_forecast_30d.png", dpi=150)
    plt.close()

    print("  --> Saved comparison charts in reports/figures/")
    print("\n" + "=" * 85)
    print("  STAGE 3 COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    main()
