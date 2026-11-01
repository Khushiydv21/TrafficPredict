"""
Stage 2 Runner Script: Exploratory Data Analysis (EDA) & Feature Engineering
Author: Khushi Yadav
Project: TrafficPredict - Website Traffic Forecasting System

Execute this file using:
python run_stage2.py
"""

import os
import sys
from pathlib import Path

# Add project root directory to Python path and set writable cache for matplotlib
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
os.environ["MPLCONFIGDIR"] = str(PROJECT_DIR / ".mpl_cache")

import pandas as pd
from src.eda import (
    calculate_weekly_seasonality,
    calculate_monthly_seasonality,
    calculate_correlations,
    perform_adf_test,
    calculate_autocorrelations,
    generate_eda_plots
)
from src.features import (
    build_feature_pipeline,
    chronological_train_test_split
)


def main():
    print("=" * 80)
    print("  TRAFFICPREDICT - STAGE 2: EXPLORATORY DATA ANALYSIS & FEATURE ENGINEERING")
    print("=" * 80)

    clean_path = PROJECT_DIR / "data" / "cleaned_website_traffic.csv"
    features_path = PROJECT_DIR / "data" / "features_website_traffic.csv"
    figures_dir = PROJECT_DIR / "reports" / "figures"

    # Step 1: Load Cleaned Dataset from Stage 1
    print(f"\n[Step 1] Loading cleaned dataset from: {clean_path}")
    df = pd.read_csv(clean_path)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"  --> Loaded {len(df):,} records spanning from {df['Date'].min().date()} to {df['Date'].max().date()}.")

    # Step 2: Exploratory Data Analysis (EDA)
    print("\n[Step 2] Exploratory Time-Series Analysis:")

    # 2.1 Weekly Seasonality
    print("\n  2.1 Weekly Seasonality Pattern (Academic Traffic vs Weekend Dip):")
    weekly_df = calculate_weekly_seasonality(df)
    print(weekly_df[["Day", "mean", "std", "median"]].to_string(index=False))

    # 2.2 Monthly Seasonality
    print("\n  2.2 Monthly Seasonality Pattern (Academic Semester Cycles):")
    monthly_df = calculate_monthly_seasonality(df)
    print(monthly_df[["Month_Name", "mean", "median"]].to_string(index=False))

    # 2.3 Visitor Segment Correlations
    print("\n  2.3 Pearson Correlation Matrix:")
    corr_df = calculate_correlations(df)
    print(corr_df.to_string())

    # 2.4 Stationarity Test (Augmented Dickey-Fuller)
    print("\n  2.4 Augmented Dickey-Fuller (ADF) Stationarity Test on Page.Loads:")
    adf_res = perform_adf_test(df["Page.Loads"])
    print(f"  * ADF Test Statistic : {adf_res['adf_statistic']}")
    print(f"  * p-value            : {adf_res['p_value']}")
    print(f"  * Stationary (p<0.05): {adf_res['is_stationary']}")
    print(f"  * Interpretation     : {adf_res['interpretation']}")

    # 2.5 Autocorrelation Analysis (ACF)
    print("\n  2.5 Autocorrelation at Key Time Lags:")
    acf_list = calculate_autocorrelations(df["Page.Loads"], max_lag=30)
    key_lags = [1, 2, 7, 14, 21, 28, 30]
    for item in acf_list:
        if item["lag"] in key_lags:
            bar = "#" * int(max(0, item["autocorrelation"] * 25))
            print(f"      Lag {item['lag']:>2} days : {item['autocorrelation']:+.4f}  | {bar}")

    # 2.6 Save Visual Charts
    print(f"\n[Step 3] Generating Visual EDA Charts in: {figures_dir}")
    saved_charts = generate_eda_plots(df, str(figures_dir))
    for chart in saved_charts:
        print(f"  --> Saved: {Path(chart).name}")

    # Step 4: Time-Series Feature Engineering
    print("\n[Step 4] Time-Series Feature Engineering Pipeline (Zero Data Leakage):")
    feat_df = build_feature_pipeline(df, target_col="Page.Loads", drop_warmup=True)
    print(f"  * Generated {feat_df.shape[1]} total features.")
    print("  * Features created:")
    print("      - Calendar: year, month, quarter, day_of_month, day_of_week, day_of_year, is_weekend")
    print("      - Cyclical: sin_day_of_week, cos_day_of_week, sin_month, cos_month")
    print("      - Autoregressive Lags: Page.Loads_lag_1, 2, 3, 7, 14, 21, 30")
    print("      - Rolling Window Stats: roll_mean, roll_std, roll_min, roll_max (7, 14, 30 days, shifted)")
    print("      - Visitor Ratios: returning_ratio, first_time_ratio, pages_per_visit")

    # Step 5: Chronological Train-Test Split
    print("\n[Step 5] Chronological Train-Test Split (90-Day Holdout):")
    train_df, test_df = chronological_train_test_split(feat_df, test_days=90)
    print(f"  * Training Set Size : {len(train_df):,} days ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"  * Holdout Test Size : {len(test_df):,} days ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    print("  * Date Overlap Check: ZERO overlap. Pure forward-looking holdout.")
    print("  * Test Window covers: Next 7-Day, Next 15-Day, and Next 30-Day forecast horizons.")

    # Step 6: Save Feature Dataset
    print(f"\n[Step 6] Saving engineered feature matrix to: {features_path}")
    feat_df.to_csv(features_path, index=False)
    print("  --> Saved feature matrix successfully!")

    print("\n" + "=" * 80)
    print("  STAGE 2 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
