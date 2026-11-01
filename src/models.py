"""
Stage 3: Time-Series Machine Learning Models Module
Author: Khushi Yadav
Project: TrafficPredict - Website Traffic Forecasting System

Implements:
1. SeasonalNaiveBaseline (predicts same day last week y_{t-7})
2. TimeSeriesLinearModel (Ridge/LinearRegression with time & lag features)
3. TimeSeriesRandomForest (Random Forest Regressor with time-series feature matrix)
4. TimeSeriesSARIMA (Seasonal ARIMA model via statsmodels)
5. Evaluation metrics calculation: MAE, RMSE, MAPE, R^2
6. Multi-step recursive and direct forecasting for 7, 15, and 30 days.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression & time-series error metrics:
    - MAE: Mean Absolute Error (in page loads)
    - RMSE: Root Mean Squared Error (in page loads)
    - MAPE: Mean Absolute Percentage Error (%)
    - R2: Coefficient of Determination
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # Avoid zero division in MAPE
    non_zero = y_true != 0
    mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4)
    }


class SeasonalNaiveBaseline:
    """
    Baseline time-series model that predicts traffic based on 7-day seasonality:
    y_hat_t = y_{t-7} (the observed value on the same day of the previous week).
    """
    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.last_season_values_ = None

    def fit(self, y: pd.Series):
        self.last_season_values_ = y.iloc[-self.season_length:].values
        return self

    def predict(self, horizon: int) -> np.ndarray:
        # Repeat the last observed seasonal cycle across the desired horizon
        reps = int(np.ceil(horizon / self.season_length))
        preds = np.tile(self.last_season_values_, reps)[:horizon]
        return preds


class TimeSeriesLinearModel:
    """Linear regression model with L2 regularization (Ridge) on time features."""
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = X.columns.tolist()
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_cols])


class TimeSeriesRandomForest:
    """Random Forest Regressor capturing non-linear interactions across time features."""
    def __init__(self, n_estimators: int = 150, max_depth: int = 12, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = X.columns.tolist()
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_cols])


class TimeSeriesSARIMA:
    """Seasonal ARIMA statistical model for univariate time series."""
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model_ = None

    def fit(self, y: pd.Series):
        # Fit on recent window for speed and stability
        y_train = y.iloc[-365:] if len(y) > 365 else y
        model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model_ = model.fit(disp=False)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        forecast = self.fitted_model_.forecast(steps=horizon)
        return forecast.values
