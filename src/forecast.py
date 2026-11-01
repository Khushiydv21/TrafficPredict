"""
Stage 3: Multi-Horizon Forecasting and Metric Generation Module
Author: Khushi Yadav
Project: TrafficPredict - Website Traffic Forecasting System

This module coordinates:
1. Training all 4 candidate models: Baseline, Linear (Ridge), Random Forest, and SARIMA.
2. Horizon-by-horizon evaluation (7, 15, and 30 days) computing MAE, RMSE, MAPE, and R^2.
3. Out-of-sample forward forecasting (7, 15, and 30 days) for:
   - Page.Loads
   - Unique.Visits
   - First.Time.Visits
   - Returning.Visits
4. Persisting models, metrics.json, and future_forecasts.json for Streamlit dashboard integration.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import Ridge

from src.models import (
    compute_metrics,
    SeasonalNaiveBaseline,
    TimeSeriesLinearModel,
    TimeSeriesRandomForest,
    TimeSeriesSARIMA
)
from src.features import create_calendar_features


FEATURE_COLS = [
    'year', 'month', 'quarter', 'day_of_month', 'day_of_week', 'day_of_year', 'is_weekend',
    'sin_day_of_week', 'cos_day_of_week', 'sin_month', 'cos_month',
    'Page.Loads_lag_1', 'Page.Loads_lag_2', 'Page.Loads_lag_3', 'Page.Loads_lag_7',
    'Page.Loads_lag_14', 'Page.Loads_lag_21', 'Page.Loads_lag_30',
    'Page.Loads_roll_mean_7', 'Page.Loads_roll_std_7', 'Page.Loads_roll_min_7', 'Page.Loads_roll_max_7',
    'Page.Loads_roll_mean_14', 'Page.Loads_roll_std_14', 'Page.Loads_roll_min_14', 'Page.Loads_roll_max_14',
    'Page.Loads_roll_mean_30', 'Page.Loads_roll_std_30', 'Page.Loads_roll_min_30', 'Page.Loads_roll_max_30'
]


def evaluate_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizons: List[int] = [7, 15, 30]
) -> Dict[str, Any]:
    """
    Trains candidate models on train_df and evaluates on test_df across horizons.
    """
    X_train, y_train = train_df[FEATURE_COLS], train_df["Page.Loads"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["Page.Loads"]

    models = {
        "Seasonal Baseline": SeasonalNaiveBaseline(season_length=7).fit(y_train),
        "Linear Regression (Ridge)": TimeSeriesLinearModel(alpha=1.0).fit(X_train, y_train),
        "Random Forest Regressor": TimeSeriesRandomForest(n_estimators=120, max_depth=10).fit(X_train, y_train),
        "SARIMA(1,1,1)x(1,1,1,7)": TimeSeriesSARIMA().fit(y_train)
    }

    results = {}
    for name, model in models.items():
        results[name] = {"horizons": {}, "predictions": {}}
        for h in horizons:
            y_true = y_test.iloc[:h].values
            if name == "Seasonal Baseline":
                y_pred = model.predict(h)
            elif name == "SARIMA(1,1,1)x(1,1,1,7)":
                y_pred = model.predict(h)
            else:
                y_pred = model.predict(X_test.iloc[:h])

            # Ensure non-negative traffic
            y_pred = np.clip(y_pred, a_min=0, a_max=None)
            m = compute_metrics(y_true, y_pred)
            results[name]["horizons"][str(h)] = m
            results[name]["predictions"][str(h)] = {
                "dates": test_df["Date"].iloc[:h].dt.strftime("%Y-%m-%d").tolist(),
                "actual": [int(x) for x in y_true],
                "predicted": [int(round(x)) for x in y_pred]
            }

    return results, models


def train_companion_visitor_models(df: pd.DataFrame) -> Dict[str, Ridge]:
    """
    Fits synchronized linear regressors to project Unique.Visits,
    First.Time.Visits, and Returning.Visits from Page.Loads and day-of-week.
    """
    features = ["Page.Loads", "sin_day_of_week", "cos_day_of_week", "is_weekend"]
    companion_models = {}

    for target in ["Unique.Visits", "First.Time.Visits", "Returning.Visits"]:
        reg = Ridge(alpha=1.0)
        reg.fit(df[features], df[target])
        companion_models[target] = reg

    return companion_models


def generate_future_forecasts(
    last_known_df: pd.DataFrame,
    best_model: Any,
    companion_models: Dict[str, Ridge],
    max_horizon: int = 30
) -> pd.DataFrame:
    """
    Recursively forecasts up to max_horizon out-of-sample days beyond the dataset.
    """
    last_date = pd.to_datetime(last_known_df["Date"].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=max_horizon, freq="D")
    
    # Track historical Page Loads history to update lags dynamically
    history = last_known_df["Page.Loads"].tolist()
    records = []

    for curr_date in future_dates:
        # Build calendar features for current future date
        row = {
            "Date": curr_date,
            "year": curr_date.year,
            "month": curr_date.month,
            "quarter": curr_date.quarter,
            "day_of_month": curr_date.day,
            "day_of_week": curr_date.dayofweek,
            "day_of_year": curr_date.dayofyear,
            "is_weekend": int(curr_date.dayofweek in [5, 6]),
            "sin_day_of_week": float(np.sin(2 * np.pi * curr_date.dayofweek / 7)),
            "cos_day_of_week": float(np.cos(2 * np.pi * curr_date.dayofweek / 7)),
            "sin_month": float(np.sin(2 * np.pi * (curr_date.month - 1) / 12)),
            "cos_month": float(np.cos(2 * np.pi * (curr_date.month - 1) / 12))
        }

        # Build dynamic autoregressive lags
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            row[f"Page.Loads_lag_{lag}"] = history[-lag]

        # Build dynamic rolling statistics on history
        for w in [7, 14, 30]:
            window_slice = history[-w:]
            row[f"Page.Loads_roll_mean_{w}"] = float(np.mean(window_slice))
            row[f"Page.Loads_roll_std_{w}"] = float(np.std(window_slice))
            row[f"Page.Loads_roll_min_{w}"] = float(np.min(window_slice))
            row[f"Page.Loads_roll_max_{w}"] = float(np.max(window_slice))

        feat_row = pd.DataFrame([row])[FEATURE_COLS]
        pred_loads = float(best_model.predict(feat_row)[0])
        pred_loads = max(0.0, pred_loads)
        history.append(pred_loads)

        # Predict companion visitor metrics
        comp_in = pd.DataFrame([{
            "Page.Loads": pred_loads,
            "sin_day_of_week": row["sin_day_of_week"],
            "cos_day_of_week": row["cos_day_of_week"],
            "is_weekend": row["is_weekend"]
        }])

        pred_unique = max(0.0, float(companion_models["Unique.Visits"].predict(comp_in)[0]))
        pred_first = max(0.0, float(companion_models["First.Time.Visits"].predict(comp_in)[0]))
        pred_returning = max(0.0, float(companion_models["Returning.Visits"].predict(comp_in)[0]))

        records.append({
            "Date": curr_date.strftime("%Y-%m-%d"),
            "Day": curr_date.strftime("%A"),
            "Page.Loads": int(round(pred_loads)),
            "Unique.Visits": int(round(pred_unique)),
            "First.Time.Visits": int(round(pred_first)),
            "Returning.Visits": int(round(pred_returning))
        })

    return pd.DataFrame(records)
