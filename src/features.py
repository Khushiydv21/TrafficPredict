"""
Stage 2: Feature Engineering & Time-Series Preprocessing Module
Author: Khushi Yadav
Project: TrafficPredict - Website Traffic Forecasting System

This module creates robust, leak-free time-series features:
1. Calendar / Temporal Features: Day of week, Month, Quarter, Weekend indicators.
2. Cyclical Encodings: Sine/Cosine transformations of periodic calendar cycles.
3. Lag Features: Historical traffic observations (Lag 1, 2, 3, 7, 14, 21, 30).
4. Rolling Window Statistics: Moving average and volatility over 7, 14, 30 days (shift=1).
5. Visitor Interaction Ratios: Returning ratio, new visitor ratio, pages per visit.
6. Strict Chronological Train-Test Split: No shuffling, preventing future lookahead.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def create_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Extracts time-based calendar attributes and cyclical sine/cosine encodings.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a datetime Date column.
        date_col (str): Name of the date column.
        
    Returns:
        pd.DataFrame: DataFrame with new calendar features.
    """
    df_out = df.copy()
    
    dt = df_out[date_col].dt
    df_out["year"] = dt.year
    df_out["month"] = dt.month
    df_out["quarter"] = dt.quarter
    df_out["day_of_month"] = dt.day
    df_out["day_of_week"] = dt.dayofweek  # Monday=0, Sunday=6
    df_out["day_of_year"] = dt.dayofyear
    df_out["is_weekend"] = df_out["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical Sine and Cosine encodings
    # Preserves circular continuity (e.g. Sunday (6) is adjacent to Monday (0))
    df_out["sin_day_of_week"] = np.sin(2 * np.pi * df_out["day_of_week"] / 7)
    df_out["cos_day_of_week"] = np.cos(2 * np.pi * df_out["day_of_week"] / 7)
    df_out["sin_month"] = np.sin(2 * np.pi * (df_out["month"] - 1) / 12)
    df_out["cos_month"] = np.cos(2 * np.pi * (df_out["month"] - 1) / 12)

    return df_out


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "Page.Loads",
    lags: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Creates autoregressive lag features representing historical traffic.
    
    Args:
        df (pd.DataFrame): Input DataFrame sorted chronologically.
        target_col (str): Name of the target column to lag.
        lags (list of int): Lag intervals in days. Default: [1, 2, 3, 7, 14, 21, 30].
        
    Returns:
        pd.DataFrame: DataFrame with lag features.
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14, 21, 30]

    df_out = df.copy()
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df_out[col_name] = df_out[target_col].shift(lag)

    return df_out


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Page.Loads",
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Creates rolling window moving statistics (mean, std, min, max).
    Crucial: Rolling statistics are applied on shift(1) to strictly prevent
    data leakage (ensures the model never includes today's traffic in the rolling window).
    
    Args:
        df (pd.DataFrame): Input DataFrame sorted chronologically.
        target_col (str): Target column name.
        windows (list of int): Rolling window sizes in days. Default: [7, 14, 30].
        
    Returns:
        pd.DataFrame: DataFrame with rolling features.
    """
    if windows is None:
        windows = [7, 14, 30]

    df_out = df.copy()
    # Shift by 1 day to ensure we only look at past history
    shifted_target = df_out[target_col].shift(1)

    for w in windows:
        df_out[f"{target_col}_roll_mean_{w}"] = shifted_target.rolling(window=w).mean()
        df_out[f"{target_col}_roll_std_{w}"] = shifted_target.rolling(window=w).std()
        df_out[f"{target_col}_roll_min_{w}"] = shifted_target.rolling(window=w).min()
        df_out[f"{target_col}_roll_max_{w}"] = shifted_target.rolling(window=w).max()

    return df_out


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates visitor behavior and engagement ratios.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame with traffic columns.
        
    Returns:
        pd.DataFrame: DataFrame with ratio features.
    """
    df_out = df.copy()

    # Avoid division by zero with clip(lower=1)
    unique = df_out["Unique.Visits"].clip(lower=1)
    df_out["returning_ratio"] = df_out["Returning.Visits"] / unique
    df_out["first_time_ratio"] = df_out["First.Time.Visits"] / unique
    df_out["pages_per_visit"] = df_out["Page.Loads"] / unique

    return df_out


def build_feature_pipeline(
    df: pd.DataFrame,
    target_col: str = "Page.Loads",
    drop_warmup: bool = True
) -> pd.DataFrame:
    """
    Orchestrates complete feature generation pipeline.
    
    Args:
        df (pd.DataFrame): Cleaned dataset from Stage 1.
        target_col (str): Target column for lags and rolling windows.
        drop_warmup (bool): If True, drops initial rows with NaN values resulting
                           from the maximum lag/window (30 days). No data fabricated.
                           
    Returns:
        pd.DataFrame: Feature matrix ready for time-series modeling.
    """
    # 1. Calendar features
    feat_df = create_calendar_features(df, date_col="Date")

    # 2. Lag features
    feat_df = create_lag_features(feat_df, target_col=target_col)

    # 3. Rolling window features (leak-free)
    feat_df = create_rolling_features(feat_df, target_col=target_col)

    # 4. Interaction / visitor ratios
    feat_df = create_interaction_features(feat_df)

    # 5. Clean warmup rows (first 30 days have NaNs due to 30-day lag/window)
    if drop_warmup:
        initial_len = len(feat_df)
        feat_df = feat_df.dropna().reset_index(drop=True)
        dropped = initial_len - len(feat_df)
        print(f"  * Warmup window dropped: {dropped} days (initial 30-day lag buffer).")
        print(f"  * Modeling dataset size : {len(feat_df):,} complete records.")

    return feat_df


def chronological_train_test_split(
    df: pd.DataFrame,
    test_days: int = 90
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset chronologically into Train and Test subsets.
    Prevents temporal data leakage.
    
    Args:
        df (pd.DataFrame): Processed feature DataFrame sorted by Date.
        test_days (int): Number of most recent calendar days for hold-out test set.
                         Default: 90 days (covers 7-, 15-, and 30-day evaluation).
                         
    Returns:
        tuple: (train_df, test_df)
    """
    if test_days >= len(df):
        raise ValueError("test_days cannot exceed or equal total dataset size.")

    train_df = df.iloc[:-test_days].copy().reset_index(drop=True)
    test_df = df.iloc[-test_days:].copy().reset_index(drop=True)

    return train_df, test_df
