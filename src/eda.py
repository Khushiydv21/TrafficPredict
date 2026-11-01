"""
Stage 2: Exploratory Data Analysis (EDA) Module
Author: Khushi Yadav
Project: TrafficPredict - Website Traffic Forecasting System

This module performs:
1. Long-term trend analysis over the 2014-2020 timeline.
2. Weekly seasonality profiling (weekday vs weekend patterns).
3. Monthly & seasonal academic cycle profiling.
4. Correlation analysis between Page Loads and visitor segments.
5. Stationarity testing (Augmented Dickey-Fuller test).
6. Autocorrelation (ACF) computation across multiple lag horizons.
7. Automated saving of publication-quality charts into reports/figures/.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_weekly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes average traffic statistics grouped by day of week.
    Preserves natural calendar order (Monday to Sunday).
    """
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly = (
        df.groupby("Day")["Page.Loads"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reindex(day_order)
        .reset_index()
    )
    weekly["mean"] = weekly["mean"].round(1)
    weekly["std"] = weekly["std"].round(1)
    return weekly


def calculate_monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes average traffic grouped by calendar month (1 to 12).
    """
    df_temp = df.copy()
    df_temp["Month_Num"] = df_temp["Date"].dt.month
    df_temp["Month_Name"] = df_temp["Date"].dt.strftime("%B")

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    monthly = (
        df_temp.groupby(["Month_Num", "Month_Name"])["Page.Loads"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .sort_values("Month_Num")
    )
    monthly["mean"] = monthly["mean"].round(1)
    return monthly


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson correlation matrix for traffic and visitor metrics.
    """
    metrics = ["Page.Loads", "Unique.Visits", "First.Time.Visits", "Returning.Visits"]
    return df[metrics].corr().round(4)


def perform_adf_test(series: pd.Series) -> Dict[str, Any]:
    """
    Performs Augmented Dickey-Fuller (ADF) stationarity test.
    Gracefully imports from statsmodels if available.
    """
    try:
        import warnings
        from statsmodels.tsa.stattools import adfuller
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(series.dropna(), autolag="AIC")
        return {
            "adf_statistic": round(float(result[0]), 4),
            "p_value": round(float(result[1]), 5),
            "used_lag": int(result[2]),
            "n_obs": int(result[3]),
            "critical_values": {k: round(v, 4) for k, v in result[4].items()},
            "is_stationary": bool(result[1] < 0.05),
            "interpretation": (
                "Stationary (p < 0.05): Mean and variance are stable over time."
                if result[1] < 0.05
                else "Non-Stationary (p >= 0.05): Has strong trend/seasonality; differencing may be required for ARIMA."
            )
        }
    except ImportError:
        # Fallback approximation if statsmodels is not present
        return {
            "adf_statistic": None,
            "p_value": None,
            "interpretation": "statsmodels library required for full ADF test."
        }


def calculate_autocorrelations(series: pd.Series, max_lag: int = 30) -> List[Dict[str, Any]]:
    """
    Computes Pearson autocorrelation coefficients for lags 1 through max_lag.
    """
    acf_results = []
    for lag in range(1, max_lag + 1):
        corr = series.autocorr(lag=lag)
        acf_results.append({
            "lag": lag,
            "autocorrelation": round(float(corr), 4)
        })
    return acf_results


def generate_eda_plots(df: pd.DataFrame, figures_dir: str) -> List[str]:
    """
    Generates and saves exploratory data analysis charts.
    """
    os.makedirs(figures_dir, exist_ok=True)
    saved_files = []
    sns.set_theme(style="whitegrid")

    # 1. Overall Traffic Over Time
    plt.figure(figsize=(14, 5))
    plt.plot(df["Date"], df["Page.Loads"], color="#1f77b4", linewidth=1, label="Daily Page Loads")
    plt.plot(df["Date"], df["Page.Loads"].rolling(30).mean(), color="#ff7f0e", linewidth=2.5, label="30-Day Moving Avg")
    plt.title("Historical Website Page Loads (2014 - 2020) with 30-Day Trend", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Page Loads", fontsize=11)
    plt.legend(loc="upper left")
    plt.tight_layout()
    p1 = os.path.join(figures_dir, "traffic_timeline.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    saved_files.append(p1)

    # 2. Weekly Seasonality (Day of Week Boxplot)
    plt.figure(figsize=(10, 5))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    palette = ["#2ca02c" if d not in ["Saturday", "Sunday"] else "#d62728" for d in day_order]
    sns.boxplot(data=df, x="Day", y="Page.Loads", order=day_order, palette=palette, hue="Day", legend=False)
    plt.title("Weekly Seasonality: Traffic Distribution by Day of Week", fontsize=13, fontweight="bold")
    plt.xlabel("Day of Week", fontsize=11)
    plt.ylabel("Daily Page Loads", fontsize=11)
    plt.tight_layout()
    p2 = os.path.join(figures_dir, "weekly_seasonality.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    saved_files.append(p2)

    # 3. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = calculate_correlations(df)
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".3f", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix: Traffic & Visitor Segments", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p3 = os.path.join(figures_dir, "correlation_heatmap.png")
    plt.savefig(p3, dpi=150)
    plt.close()
    saved_files.append(p3)

    return saved_files
