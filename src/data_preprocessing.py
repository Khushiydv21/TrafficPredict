"""
Stage 1: Data Preprocessing & Integrity Verification Module
Author: Khushi Yadav
Project: TrafficPredict - Website Traffic Forecasting

This module handles:
1. Loading the historical website traffic dataset.
2. Inspecting raw column data types, shapes, and sample rows.
3. Converting string formatted numbers (with comma separators) into integers.
4. Parsing and standardizing calendar dates into DateTime objects.
5. Checking missing values, duplicates, and daily date continuity.
6. Chronological sorting to ensure temporal consistency without data leakage.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw CSV dataset from the specified file path.
    
    Args:
        file_path (str): Relative or absolute path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded raw pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def inspect_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs initial inspection on raw DataFrame.
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
        
    Returns:
        dict: Inspection summary containing shape, columns, null counts, and duplicates.
    """
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    return summary


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw traffic dataset according to strict time-series specifications:
    1. Strips comma separators and converts traffic metrics into integers.
    2. Converts Date column to datetime64[ns].
    3. Sorts rows chronologically by Date.
    4. Resets DataFrame index.
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned and sorted DataFrame.
    """
    # Create an explicit copy to avoid modifying original reference
    df_clean = df.copy()

    # 1. Convert Date to datetime format (US format: mm/dd/yyyy)
    df_clean["Date"] = pd.to_datetime(df_clean["Date"], format="%m/%d/%Y")

    # 2. Traffic columns that contain commas (e.g., '1,582')
    traffic_columns = [
        "Page.Loads",
        "Unique.Visits",
        "First.Time.Visits",
        "Returning.Visits"
    ]

    for col in traffic_columns:
        if col in df_clean.columns:
            # Convert to string first, remove commas, then convert to int64
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .astype(int)
            )

    # 3. Sort chronologically by Date to maintain correct temporal sequence
    df_clean = df_clean.sort_values("Date").reset_index(drop=True)

    return df_clean


def verify_data_integrity(df_clean: pd.DataFrame) -> Dict[str, Any]:
    """
    Rigorously verifies time series integrity:
    1. Null value counts
    2. Duplicated dates
    3. Daily date continuity (ensures no missing dates in timeline)
    4. Start date and End date
    
    Args:
        df_clean (pd.DataFrame): Cleaned DataFrame.
        
    Returns:
        dict: Integrity report metrics.
    """
    start_date = df_clean["Date"].min()
    end_date = df_clean["Date"].max()
    expected_days = (end_date - start_date).days + 1
    actual_records = len(df_clean)

    # Generate complete date range to test continuity
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    missing_dates = full_date_range.difference(df_clean["Date"])

    report = {
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "expected_days": int(expected_days),
        "actual_records": int(actual_records),
        "missing_days_count": len(missing_dates),
        "is_continuous": (len(missing_dates) == 0 and expected_days == actual_records),
        "duplicate_dates": int(df_clean["Date"].duplicated().sum()),
        "total_nulls": int(df_clean.isnull().sum().sum()),
    }
    return report


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves cleaned dataframe to CSV.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe.
        output_path (str): Destination path for CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
