# 📈 TrafficPredict: Website Traffic Forecasting System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14%2B-00599C.svg)](https://www.statsmodels.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Predicting Website Traffic Using Machine Learning and Time Series Forecasting**  
> A production-grade end-to-end Data Science project and interactive Streamlit web dashboard.  
> *Inspired by the modular architecture and design of [AquaPredict](https://github.com/arambh8960/AquaPredict).*

---

## 🌟 Executive Summary

TrafficPredict is a complete website traffic forecasting system built on **2,167 days** of continuous historical visitor records (14 September 2014 to 19 August 2020). 

It implements strict **zero-data-leakage** time-series preprocessing, cyclical feature engineering, and benchmarks **4 distinct forecasting paradigms** across **Next 7-day, 15-day, and 30-day horizons** to predict `Page.Loads`, while synchronizing projections for `Unique.Visits`, `First.Time.Visits`, and `Returning.Visits`.

---

## 📊 Benchmark Results Summary

Evaluated on an out-of-sample chronological holdout test set ($N=90\text{ days}$):

| Algorithm / Model | Horizon | MAE (Loads) | RMSE (Loads) | MAPE (%) | Accuracy (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Baseline** | 7 Days | 388.29 | 486.87 | 8.70% | 91.30% | 0.3892 |
| **Seasonal Baseline** | 15 Days | 491.27 | 568.03 | 11.24% | 88.76% | 0.0818 |
| **Seasonal Baseline** | 30 Days | 584.83 | 660.83 | 14.13% | 85.87% | 0.0123 |
| **Linear Regression (Ridge)** 🏆 | **7 Days** | **222.60** | **283.62** | **5.00%** | **95.00%** | **0.7927** |
| **Linear Regression (Ridge)** 🏆 | **15 Days** | **234.36** | **315.86** | **5.28%** | **94.72%** | **0.7161** |
| **Linear Regression (Ridge)** 🏆 | **30 Days** | **254.65** | **336.03** | **5.90%** | **94.10%** | **0.7446** |
| **Random Forest Regressor** | 7 Days | 398.26 | 477.08 | 8.94% | 91.06% | 0.4135 |
| **Random Forest Regressor** | 15 Days | 320.05 | 393.93 | 7.33% | 92.67% | 0.5584 |
| **Random Forest Regressor** | 30 Days | 278.99 | 342.48 | 6.54% | 93.46% | 0.7347 |
| **SARIMA(1,1,1)×(1,1,1)₇** | 7 Days | 237.08 | 323.77 | 5.16% | 94.84% | 0.7299 |
| **SARIMA(1,1,1)×(1,1,1)₇** | 15 Days | 355.63 | 449.99 | 7.86% | 92.14% | 0.4237 |
| **SARIMA(1,1,1)×(1,1,1)₇** | 30 Days | 555.08 | 663.13 | 13.20% | 86.80% | 0.0054 |

**Key Takeaway:** **Linear Regression with L2 Regularization (Ridge)** paired with cyclical trigonometric calendar encodings achieved the highest accuracy across all horizons (Average MAPE of **5.39%**, yielding $>94.6\%$ accuracy).

---

## 🗂️ Project Repository Structure

```text
TrafficPredict/
├── app.py                          # Streamlit Multi-Page Web Dashboard
├── data/
│   ├── daily-website-visitors.csv  # Raw dataset (2,167 daily records)
│   ├── cleaned_website_traffic.csv # Cleaned dataset (Stage 1)
│   └── features_website_traffic.csv# 41 engineered features matrix (Stage 2)
├── models/
│   ├── metrics.json                # Serialized evaluation metrics & curve data
│   ├── future_forecasts.json       # Forward out-of-sample multi-horizon forecasts
│   ├── best_model.joblib           # Trained Best Model artifact
│   └── companion_models.joblib     # Visitor cohort projection models
├── reports/
│   └── figures/                    # High-resolution publication plots
│       ├── traffic_timeline.png
│       ├── weekly_seasonality.png
│       ├── correlation_heatmap.png
│       ├── model_comparison_mae.png
│       ├── actual_vs_predicted_30d.png
│       └── future_forecast_30d.png
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Stage 1: Cleaning & continuity verification
│   ├── eda.py                      # Stage 2: Seasonality, ADF test, ACF analysis
│   ├── features.py                 # Stage 2: Lags, rolling windows, cyclical features
│   ├── models.py                   # Stage 3: Baseline, Ridge, RF, SARIMA classes
│   └── forecast.py                 # Stage 3: Multi-horizon evaluation & projections
├── run_stage1.py                   # Stage 1 execution runner
├── run_stage2.py                   # Stage 2 execution runner
├── run_stage3.py                   # Stage 3 execution runner
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Git exclusion rules
├── VIVA_PREPARATION_GUIDE.md       # Technical Viva / Interview Questions & Answers
└── README.md                       # Complete Project Documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Khushiydv21/TrafficPredict.git
cd TrafficPredict
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the End-to-End Pipeline
You can run any stage individually:
```bash
# Stage 1: Data Preprocessing & Integrity Verification
python run_stage1.py

# Stage 2: Exploratory Data Analysis & Feature Engineering
python run_stage2.py

# Stage 3: Model Training, Multi-Horizon Evaluation & Forecasting
python run_stage3.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to view the interactive dashboard.

---

## 🖥️ Streamlit Dashboard Walkthrough

Inspired by [AquaPredict](https://aquapredict8960.streamlit.app/), the application features 5 dedicated modules:

1. **🏠 Home Dashboard:**
   - Real-time KPI metric cards (Total Page Loads, Daily Average, Peak Traffic, New Visitor Share).
   - 6-Year interactive Plotly timeline with 7-day and 30-day moving averages and zoom range sliders.
   - Visitor composition donut chart.
   - Expandable summary statistics table.

2. **📈 Traffic Prediction:**
   - Multi-horizon forecast selector (**7 Days**, **15 Days**, **30 Days**).
   - Algorithm selector (Ridge, Random Forest, SARIMA, Seasonal Baseline).
   - Interactive trajectory plot with automated **weekend highlight shading**.
   - Stacked visitor segment projections (First-Time vs Returning).
   - Instant **CSV Export** button.

3. **📊 Model Comparison:**
   - Full performance benchmarking matrix across MAE, RMSE, MAPE, and $R^2$.
   - Grouped bar charts contrasting model error and accuracy across horizons.
   - Interactive Actual vs Predicted trajectory inspector for the 90-day test set.

4. **🔍 Data Analytics:**
   - **Weekly Seasonality:** Boxplot highlighting weekday academic demand vs weekend slump.
   - **Monthly Cycles:** Bar chart illustrating semester exam peaks (April/November) and summer drops (July/August).
   - **Visitor Segments:** Scatter plot of Returning vs First-Time visitors with bubble size mapping page loads.
   - **Correlation Matrix:** Heatmap showing strong linear correlation between page loads and visitor types.

5. **ℹ️ About Project:**
   - Complete technical documentation of project goals, dataset specs, methodology, technology stack, and limitations.

---

## 💡 Key Data Science Discoveries

1. **Strong Weekly Seasonality:** Tuesday is consistently the highest traffic day (averaging **4,955 page loads**), whereas Saturday drops to **2,501 page loads** (~50% decrease).
2. **Lag 7 Autocorrelation ($+0.8971$):** Traffic exactly 7 days prior is the single most predictive feature in the dataset.
3. **Academic Semester Cyclicality:** Traffic surges during university examination periods (April & November) and drops sharply during summer recess (July & August).
4. **ADF Stationarity:** Augmented Dickey-Fuller test yielded $t = -4.2838$ and $p = 0.00047$, rejecting the unit-root null hypothesis.

---

## 👩‍💻 Author & Acknowledgements

* **Developer:** **Khushi Yadav**
* **GitHub Profile:** [@Khushiydv21](https://github.com/Khushiydv21)
* **Design Reference:** Inspired by the Streamlit MLOps architecture of [AquaPredict](https://github.com/arambh8960/AquaPredict) by [@arambh8960](https://github.com/arambh8960).
