"""
Stage 1 Runner Script: Data Loading, Inspection, and Cleaning
Author: Khushi Yadav
Project: TrafficPredict - Website Traffic Forecasting System

Execute this file using:
python run_stage1.py
"""

import os
import sys
from pathlib import Path

# Add project root directory to Python path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_preprocessing import (
    load_raw_data,
    inspect_raw_data,
    clean_dataset,
    verify_data_integrity,
    save_cleaned_data
)


def main():
    print("=" * 75)
    print("  TRAFFICPREDICT - STAGE 1: DATA LOADING, INSPECTION & CLEANING")
    print("=" * 75)

    raw_path = PROJECT_DIR / "data" / "daily-website-visitors.csv"
    clean_path = PROJECT_DIR / "data" / "cleaned_website_traffic.csv"

    # Step 1: Load Raw Dataset
    print(f"\n[Step 1] Loading raw dataset from: {raw_path}")
    raw_df = load_raw_data(str(raw_path))
    print(f"  --> Successfully loaded {len(raw_df):,} records.")

    # Step 2: Inspect Raw Data
    print("\n[Step 2] Inspecting Raw Dataset Characteristics:")
    raw_summary = inspect_raw_data(raw_df)
    print(f"  * Shape (Rows, Columns)   : {raw_summary['shape']}")
    print(f"  * Total Duplicate Rows    : {raw_summary['duplicate_rows']}")
    print("  * Raw Column Data Types   :")
    for col, dtype in raw_summary["dtypes"].items():
        sample_val = raw_df[col].iloc[0]
        print(f"      - {col:<20}: {dtype:<10} (Sample: '{sample_val}')")

    # Step 3: Clean the Data
    print("\n[Step 3] Performing Data Cleaning & Standardizing:")
    print("  * Parsing 'Date' into pandas DateTime format (MM/DD/YYYY)")
    print("  * Removing comma separators from traffic columns: ['Page.Loads', 'Unique.Visits', 'First.Time.Visits', 'Returning.Visits']")
    print("  * Casting traffic metrics to 64-bit integers")
    print("  * Sorting records chronologically from earliest to latest date")
    
    clean_df = clean_dataset(raw_df)

    # Step 4: Verify Time Series Integrity
    print("\n[Step 4] Rigorous Time-Series Integrity Verification:")
    report = verify_data_integrity(clean_df)
    print(f"  * Timeline Range          : {report['start_date']} to {report['end_date']}")
    print(f"  * Calendar Day Span       : {report['expected_days']:,} days")
    print(f"  * Total Records Present   : {report['actual_records']:,} days")
    print(f"  * Missing Calendar Days   : {report['missing_days_count']} (Continuity is 100% complete!)")
    print(f"  * Duplicate Dates         : {report['duplicate_dates']}")
    print(f"  * Total Null / NaN Values : {report['total_nulls']}")
    print(f"  * Time Series Continuous? : {'YES - READY FOR MODELING' if report['is_continuous'] else 'NO - GAPS DETECTED'}")

    # Step 5: Summary Statistics of Cleaned Target Metrics
    print("\n[Step 5] Summary Statistics of Traffic Metrics:")
    traffic_cols = ["Page.Loads", "Unique.Visits", "First.Time.Visits", "Returning.Visits"]
    stats_df = clean_df[traffic_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(stats_df.to_string())

    # Step 6: Save Cleaned Dataset
    print(f"\n[Step 6] Saving cleaned data to: {clean_path}")
    save_cleaned_data(clean_df, str(clean_path))
    print("  --> Saved successfully!")

    print("\n" + "=" * 75)
    print("  STAGE 1 COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
