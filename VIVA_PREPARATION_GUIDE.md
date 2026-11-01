# 🎓 TrafficPredict: Viva & Technical Placement Interview Guide

> **Prepared for Khushi Yadav**  
> Use this guide to excel in your project viva, faculty presentations, and Data Science / Machine Learning job interviews.

---

### Q1. Why can't we use standard `train_test_split(shuffle=True)` for time-series forecasting?
**Answer:**  
Standard `train_test_split(shuffle=True)` randomly shuffles rows. In time-series data, records have a strict temporal sequence ($t_1, t_2, t_3, \dots$). If we shuffle, the training set will contain future days (e.g. Wednesday), and the test set will contain past days (e.g. Tuesday). The model would learn from future knowledge to predict the past—this is severe **temporal data leakage** and produces unrealistically optimistic test scores that fail in production. We must use a **chronological split** where the training set strictly precedes the holdout test set in time.

---

### Q2. How did you prevent data leakage when calculating rolling window statistics?
**Answer:**  
If we calculate a 7-day rolling average directly on day $t$ without shifting:
$$\text{Rolling Mean}_t = \frac{y_t + y_{t-1} + \dots + y_{t-6}}{7}$$
The target value $y_t$ is inside the feature itself! At test time, $y_t$ is unknown because that is what we are trying to predict.  
To prevent this, we applied `.shift(1)` before computing rolling statistics:
$$\text{Rolling Mean}_t = \frac{y_{t-1} + y_{t-2} + \dots + y_{t-7}}{7}$$
This guarantees that rolling features strictly use historical observations.

---

### Q3. Why did Linear Regression (Ridge) outperform Random Forest and SARIMA across all horizons?
**Answer:**  
There are four core mathematical and structural reasons:
1. **Harmonic / Sinusoidal Seasonality:** Weekly traffic patterns follow continuous cyclical oscillations. Our trigonometric sine and cosine encodings (`sin_day_of_week`, `cos_day_of_week`) provide smooth basis functions that linear models can map with exact weight coefficients.
2. **L2 Regularization (Ridge) Controls Collinearity:** Autoregressive lags (like Lag 7 and Lag 14) are highly correlated ($r = 0.89$ and $0.78$). Ridge regression applies an $L_2$ penalty ($\lambda \sum \beta_j^2$) that shrinks weights smoothly rather than blowing up coefficients due to multicollinearity.
3. **Extrapolation Limitations of Tree Ensembles:** Random Forest partitions feature space with orthogonal splits (step functions) bounded by training leaf nodes. It cannot extrapolate continuous trends beyond its training bounds as effectively as regularized linear models.
4. **Error Accumulation in SARIMA:** While SARIMA performs well on short horizons (5.16% MAPE at 7 days), it relies strictly on recursive residual compounding without exogenous calendar regressors, causing its error to compound over 30 days (13.20% MAPE).

---

### Q4. What is the Augmented Dickey-Fuller (ADF) test, and what did your p-value ($0.00047$) signify?
**Answer:**  
The ADF test checks whether a time series is stationary or contains a unit root:
- **Null Hypothesis ($H_0$):** The series has a unit root (it is non-stationary and has a time-dependent trend/variance).
- **Alternative Hypothesis ($H_1$):** The series is stationary.

Our test yielded a test statistic of **$-4.2838$** and a **$p\text{-value} = 0.00047$**. Since $p < 0.05$, we decisively reject the null hypothesis. This proves the series is stationary around a constant mean with recurring seasonal cycles, meaning differencing ($d=1$) was not mandatory for stationarity, though seasonal differencing ($D=1$) helped SARIMA.

---

### Q5. Why did we use trigonometric sine and cosine encodings instead of One-Hot Encoding for calendar features?
**Answer:**  
Calendar variables like Day of Week (0 to 6) and Month (1 to 12) are periodic and circular:
- Day 6 (Sunday) is immediately followed by Day 0 (Monday).
- Month 12 (December) is immediately followed by Month 1 (January).

If we use numeric integers ($0, 1, \dots, 6$), the distance between Sunday (6) and Monday (0) appears to be 6 units apart.  
By applying:
$$\sin\left(\frac{2\pi \cdot \text{day}}{7}\right) \quad \text{and} \quad \cos\left(\frac{2\pi \cdot \text{day}}{7}\right)$$
we map the 7 days onto a continuous unit circle where Sunday and Monday are adjacent points in 2D space.

---

### Q6. Why did you drop the first 30 days of data instead of imputing missing values?
**Answer:**  
Because our feature engineering pipeline generated up to **30-day lags** (`lag_30`) and **30-day rolling averages** (`roll_mean_30`), the first 30 days of the dataset naturally have `NaN` values (there are no records prior to September 14, 2014 to calculate a 30-day lag).  
In time-series modeling, imputing artificial values (like mean or zero) for past lags introduces synthetic noise and distorts the true historical relationship. Dropping those 30 warmup rows left us with **2,137 complete, authentic records** with zero fabricated data.

---

### Q7. What are the differences between MAE, RMSE, and MAPE?
**Answer:**  
- **MAE (Mean Absolute Error):** $\frac{1}{n} \sum |y - \hat{y}|$. Measures average magnitude of errors in the same units (page loads). It is robust to extreme outliers.
- **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$. Squaring errors gives disproportionately high weight to large errors, penalizing severe forecast misses.
- **MAPE (Mean Absolute Percentage Error):** $\frac{100\%}{n} \sum \left|\frac{y - \hat{y}}{y}\right|$. Expresses error as a percentage of actual traffic, providing a unit-free metric easily understood by business executives (e.g. 5.0% MAPE = 95% accuracy).

---

### Q8. How did you forecast companion visitor metrics (`Unique.Visits`, `First.Time.Visits`, `Returning.Visits`)?
**Answer:**  
Through EDA, we discovered extremely high Pearson correlations between `Page.Loads` and visitor segments ($r = 0.9887$ with Unique Visits, $r = 0.9807$ with First-Time Visits). We trained companion regularized linear models using `Page.Loads`, `sin_day_of_week`, `cos_day_of_week`, and `is_weekend` as predictors. When future `Page.Loads` are forecasted, the companion models project synchronized visitor counts that preserve historical ratio consistency.

---

### Q9. What are the practical business applications of 7-, 15-, and 30-day traffic forecasts?
**Answer:**  
1. **Cloud Server Auto-scaling & Cost Optimization:** Provisioning server instances, database read-replicas, and CDN bandwidth ahead of peak days (like Tuesday) to prevent downtime, while downscaling on weekends to save cloud costs.
2. **Ad Revenue & Monetization Planning:** Forecasting impressions for programmatic ad inventory sales and campaign scheduling.
3. **Marketing Campaign Timing:** Aligning product launches and promotional emails with natural high-engagement days.

---

### Q10. How would you deploy and monitor this system in a real-world MLOps pipeline?
**Answer:**  
1. **Containerization:** Package the Streamlit app and dependencies inside a Docker container.
2. **Automated Retraining:** Set up an Apache Airflow or GitHub Actions cron job that runs weekly, ingests the latest week of visitor logs, executes `run_stage1.py`, `run_stage2.py`, and `run_stage3.py`, and updates `best_model.joblib`.
3. **Data Drift & Concept Drift Monitoring:** Track rolling MAPE over time. If error exceeds a threshold (e.g. $>10\%$), trigger an alert to investigate algorithm updates or external market shifts.
