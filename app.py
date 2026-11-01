"""
TrafficPredict - Website Traffic Forecasting Web Dashboard
Inspired by AquaPredict
Author: Khushi Yadav
Framework: Streamlit + Plotly + Scikit-Learn + Statsmodels
"""

import os
import sys
import json
from pathlib import Path

# Add project root directory to Python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="TrafficPredict | Website Traffic Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. CACHED DATA LOADERS
# ==============================================================================
@st.cache_data
def load_clean_data():
    csv_path = BASE_DIR / "data" / "cleaned_website_traffic.csv"
    if not csv_path.exists():
        st.error("Cleaned dataset not found. Please run Stage 1 first.")
        st.stop()
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_metrics_data():
    metrics_path = BASE_DIR / "models" / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_future_forecasts():
    forecast_path = BASE_DIR / "models" / "future_forecasts.json"
    if not forecast_path.exists():
        return None
    with open(forecast_path, "r") as f:
        data = json.load(f)
    df_f = pd.DataFrame(data)
    df_f["Date"] = pd.to_datetime(df_f["Date"])
    return df_f


df = load_clean_data()
metrics_data = load_metrics_data()
future_df = load_future_forecasts()


# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.image(
    "https://img.icons8.com/fluency/96/line-chart.png",
    width=80
)
st.sidebar.title("TrafficPredict")
st.sidebar.caption("Website Traffic Forecasting & Analytics Engine")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home Dashboard",
        "📈 Traffic Prediction",
        "📊 Model Comparison",
        "🔍 Data Analytics",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Dataset Information")
st.sidebar.write(f"**Records:** {len(df):,} Days")
st.sidebar.write(f"**Start Date:** {df['Date'].min().strftime('%d %b %Y')}")
st.sidebar.write(f"**End Date:** {df['Date'].max().strftime('%d %b %Y')}")
st.sidebar.write("**Primary Target:** `Page.Loads`")
st.sidebar.markdown("---")
st.sidebar.caption("Developed by Khushi Yadav • Data Science Portfolio")


# ==============================================================================
# PAGE 1: HOME DASHBOARD
# ==============================================================================
if page == "🏠 Home Dashboard":
    st.title("📈 TrafficPredict: Website Traffic Intelligence Dashboard")
    st.markdown("### Comprehensive Historical Visitor Performance & Executive Overview")
    st.write(
        "TrafficPredict analyzes **2,167 days** of historical website visitor records (14 September 2014 to 19 August 2020) "
        "to discover seasonal patterns, track engagement depth, and accurately forecast traffic demand."
    )

    st.markdown("---")

    # High-level KPIs
    total_loads = df["Page.Loads"].sum()
    avg_loads = df["Page.Loads"].mean()
    max_loads = df["Page.Loads"].max()
    avg_unique = df["Unique.Visits"].mean()
    first_time_sum = df["First.Time.Visits"].sum()
    unique_sum = df["Unique.Visits"].sum()
    first_time_pct = (first_time_sum / unique_sum) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Page Loads", f"{total_loads / 1e6:.2f} M", help="Total historical page loads")
    col2.metric("Daily Avg Loads", f"{avg_loads:,.0f}", help="Average daily page loads")
    col3.metric("Peak Daily Traffic", f"{max_loads:,}", help="Highest recorded single-day page loads")
    col4.metric("Daily Unique Visitors", f"{avg_unique:,.0f}", help="Average daily unique visitors")
    col5.metric("New Visitor Share", f"{first_time_pct:.1f}%", help="Percentage of first-time visits")

    st.markdown("---")

    # Row 2: Charts
    col_chart1, col_chart2 = st.columns([2.2, 1])

    with col_chart1:
        st.subheader("Historical Traffic Trend")
        
        # View Resolution Selector for Maximum Clarity
        view_mode = st.radio(
            "Select Chart Clarity View:",
            [
                "✨ Clean Trend (7-Day & 30-Day Moving Averages)",
                "🔍 Recent 1 Year (Daily Detail)",
                "📊 Monthly Total Volumes",
                "📈 Full 6-Year Daily View"
            ],
            horizontal=True,
            index=0
        )

        df_plot = df.copy()
        df_plot["7D_MA"] = df_plot["Page.Loads"].rolling(7, min_periods=1).mean()
        df_plot["30D_MA"] = df_plot["Page.Loads"].rolling(30, min_periods=1).mean()

        fig_timeline = go.Figure()

        if view_mode == "✨ Clean Trend (7-Day & 30-Day Moving Averages)":
            # Clean, smoothed macro trend without barcode noise
            fig_timeline.add_trace(go.Scatter(
                x=df_plot["Date"], y=df_plot["Page.Loads"],
                mode="lines", name="Raw Daily (Muted)",
                line=dict(color="rgba(150, 160, 180, 0.25)", width=1),
                hoverinfo="skip"
            ))
            fig_timeline.add_trace(go.Scatter(
                x=df_plot["Date"], y=df_plot["7D_MA"],
                mode="lines", name="7-Day Moving Avg (Weekly Trend)",
                line=dict(color="#00b4d8", width=2.5)
            ))
            fig_timeline.add_trace(go.Scatter(
                x=df_plot["Date"], y=df_plot["30D_MA"],
                mode="lines", name="30-Day Moving Avg (Monthly Trend)",
                line=dict(color="#ff9100", width=3)
            ))
            fig_timeline.update_layout(
                title="Macro Traffic Seasonality (Smooth Moving Averages)",
                xaxis_title="Date",
                yaxis_title="Page Loads"
            )

        elif view_mode == "🔍 Recent 1 Year (Daily Detail)":
            # Filter to the most recent 365 days
            recent_1y = df_plot[df_plot["Date"] >= (df_plot["Date"].max() - pd.Timedelta(days=365))]
            fig_timeline.add_trace(go.Scatter(
                x=recent_1y["Date"], y=recent_1y["Page.Loads"],
                mode="lines+markers", name="Daily Page Loads",
                line=dict(color="#64b5f6", width=1.5),
                marker=dict(size=4, color="#1e88e5")
            ))
            fig_timeline.add_trace(go.Scatter(
                x=recent_1y["Date"], y=recent_1y["7D_MA"],
                mode="lines", name="7-Day Moving Avg",
                line=dict(color="#ff9100", width=2.5)
            ))
            fig_timeline.update_layout(
                title="Daily Page Loads with 7-Day Moving Average (Most Recent Year)",
                xaxis_title="Date",
                yaxis_title="Daily Page Loads"
            )

        elif view_mode == "📊 Monthly Total Volumes":
            # Resample by month for crystal clear volume bars
            df_monthly = (
                df.set_index("Date")["Page.Loads"]
                .resample("ME")
                .sum()
                .reset_index()
            )
            fig_timeline.add_trace(go.Bar(
                x=df_monthly["Date"],
                y=df_monthly["Page.Loads"],
                name="Monthly Total Loads",
                marker=dict(
                    color=df_monthly["Page.Loads"],
                    colorscale="Blues",
                    showscale=False
                )
            ))
            fig_timeline.update_layout(
                title="Aggregated Monthly Traffic Volumes (Clear Seasonal Cycles)",
                xaxis_title="Year - Month",
                yaxis_title="Total Monthly Page Loads"
            )

        else:
            # Full 6-year view with clear opacity hierarchy
            fig_timeline.add_trace(go.Scatter(
                x=df_plot["Date"], y=df_plot["Page.Loads"],
                mode="lines", name="Daily Page Loads",
                line=dict(color="rgba(100, 149, 237, 0.4)", width=1)
            ))
            fig_timeline.add_trace(go.Scatter(
                x=df_plot["Date"], y=df_plot["30D_MA"],
                mode="lines", name="30-Day Moving Avg",
                line=dict(color="#ff9100", width=3)
            ))
            fig_timeline.update_layout(
                title="Full 6-Year Daily Timeline (2014 - 2020)",
                xaxis_title="Date",
                yaxis_title="Page Loads"
            )

        fig_timeline.update_layout(
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.3)"
            ),
            margin=dict(l=15, r=15, t=40, b=20),
            xaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.15)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.15)")
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    with col_chart2:
        st.subheader("Visitor Composition")
        
        returning_sum = int(df["Returning.Visits"].sum())
        first_time_sum = int(df["First.Time.Visits"].sum())
        total_unique = returning_sum + first_time_sum

        fig_donut = go.Figure(data=[go.Pie(
            labels=["First-Time Visitors", "Returning Visitors"],
            values=[first_time_sum, returning_sum],
            hole=0.62,
            marker=dict(colors=["#00E676", "#7C4DFF"]),
            textinfo="percent",
            textfont=dict(size=14, color="white", family="Arial Black"),
            hoverinfo="label+value+percent"
        )])

        # Center KPI Annotation inside Donut Hole
        fig_donut.update_layout(
            annotations=[dict(
                text=f"<b>{total_unique/1e6:.2f}M</b><br><span style='font-size:12px;color:#aaa;'>Total Unique</span>",
                x=0.5, y=0.5,
                font_size=18,
                showarrow=False
            )],
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=30, b=40)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown(f"""
        <div style="background-color: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span>🟢 <b>First-Time:</b></span>
                <span><b>{first_time_sum:,}</b> ({first_time_sum/total_unique*100:.1f}%)</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>🟣 <b>Returning:</b></span>
                <span><b>{returning_sum:,}</b> ({returning_sum/total_unique*100:.1f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Row 3: Dataset Summary Statistics
    with st.expander("📊 View Detailed Summary Statistics"):
        cols = ["Page.Loads", "Unique.Visits", "First.Time.Visits", "Returning.Visits"]
        summary_table = df[cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        summary_table.columns = ["Record Count", "Mean", "Std Dev", "Min", "25% Q1", "Median (50%)", "75% Q3", "Max"]
        st.dataframe(summary_table.style.format("{:,.1f}"), use_container_width=True)


# ==============================================================================
# PAGE 2: TRAFFIC PREDICTION
# ==============================================================================
elif page == "📈 Traffic Prediction":
    st.title("📈 Multi-Horizon Traffic Forecasting")
    st.write(
        "Forecast future website traffic across **7-day**, **15-day**, and **30-day** horizons. "
        "Predictions utilize regularized time-series regressors trained strictly on historical data without future leakage."
    )

    if future_df is None or metrics_data is None:
        st.error("Trained models and forecasts not found. Please execute Stage 3 first (`python run_stage3.py`).")
        st.stop()

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([1.5, 1])
    with col_ctrl1:
        selected_model = st.selectbox(
            "Selected Forecasting Algorithm:",
            [
                "Linear Regression (Ridge) (Best Overall: 5.39% MAPE)",
                "Random Forest Regressor (Non-Linear Ensemble)",
                "SARIMA(1,1,1)x(1,1,1,7) (Econometric Univariate)",
                "Seasonal Baseline (Historical Naive)"
            ]
        )
    with col_ctrl2:
        horizon_tabs = st.radio(
            "Select Forecast Horizon:",
            ["7 Days Forecast", "15 Days Forecast", "30 Days Forecast"],
            horizontal=True
        )

    horizon_days = 7 if "7" in horizon_tabs else (15 if "15" in horizon_tabs else 30)
    current_forecast = future_df.head(horizon_days).copy()

    st.markdown("---")

    # Forecast Metrics
    sum_pred_loads = current_forecast["Page.Loads"].sum()
    avg_pred_loads = current_forecast["Page.Loads"].mean()
    peak_pred_loads = current_forecast["Page.Loads"].max()
    peak_day = current_forecast.loc[current_forecast["Page.Loads"].idxmax(), "Day"]
    peak_date = current_forecast.loc[current_forecast["Page.Loads"].idxmax(), "Date"].strftime("%b %d")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Projected Total Traffic", f"{sum_pred_loads:,} Loads")
    m_col2.metric("Projected Daily Average", f"{avg_pred_loads:,.0f} Loads/day")
    m_col3.metric("Projected Peak Day", f"{peak_pred_loads:,}", f"{peak_day} ({peak_date})")
    m_col4.metric("Forecast Window", f"{horizon_days} Days", f"{current_forecast['Date'].iloc[0].strftime('%b %d')} - {current_forecast['Date'].iloc[-1].strftime('%b %d')}")

    # Interactive Forecast Plot
    st.subheader(f"Forecast Trajectory: Next {horizon_days} Days")
    fig_f = go.Figure()

    # Context: Last 14 days of historical data
    recent_history = df.tail(14)
    fig_f.add_trace(go.Scatter(
        x=recent_history["Date"],
        y=recent_history["Page.Loads"],
        mode="lines+markers",
        name="Recent Actual History",
        line=dict(color="#7f7f7f", width=2)
    ))

    # Forecasted Line
    fig_f.add_trace(go.Scatter(
        x=current_forecast["Date"],
        y=current_forecast["Page.Loads"],
        mode="lines+markers",
        name=f"Predicted Page Loads ({horizon_days}d)",
        line=dict(color="#1f77b4", width=3, dash="solid"),
        marker=dict(size=8, symbol="circle")
    ))

    # Weekend Shading
    for _, row in current_forecast.iterrows():
        if row["Day"] in ["Saturday", "Sunday"]:
            fig_f.add_vrect(
                x0=row["Date"] - pd.Timedelta(hours=12),
                x1=row["Date"] + pd.Timedelta(hours=12),
                fillcolor="rgba(255, 0, 0, 0.08)",
                layer="below",
                line_width=0,
                annotation_text="Weekend" if row["Day"] == "Sunday" else "",
                annotation_position="top left"
            )

    fig_f.update_layout(
        xaxis_title="Date",
        yaxis_title="Page Loads",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_f, use_container_width=True)

    # Visitor Segments Breakdown
    st.subheader("Synchronized Visitor Segments Projection")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=current_forecast["Date"].dt.strftime("%a, %b %d"),
        y=current_forecast["First.Time.Visits"],
        name="First-Time Visits",
        marker_color="#00CC96"
    ))
    fig_comp.add_trace(go.Bar(
        x=current_forecast["Date"].dt.strftime("%a, %b %d"),
        y=current_forecast["Returning.Visits"],
        name="Returning Visits",
        marker_color="#AB63FA"
    ))
    fig_comp.update_layout(
        barmode="stack",
        xaxis_title="Forecast Date",
        yaxis_title="Number of Visitors",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Table with download
    st.subheader(f"Forecast Data Table ({horizon_days} Days)")
    display_df = current_forecast.copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download {horizon_days}-Day Forecast (CSV)",
        data=csv_data,
        file_name=f"website_traffic_forecast_{horizon_days}d.csv",
        mime="text/csv",
        type="primary"
    )


# ==============================================================================
# PAGE 3: MODEL COMPARISON
# ==============================================================================
elif page == "📊 Model Comparison":
    st.title("📊 Machine Learning Model Benchmarking & Comparison")
    st.write(
        "Performance evaluation of 4 forecasting models evaluated on the chronological out-of-sample holdout test partition."
    )

    if metrics_data is None:
        st.error("Metrics file not found. Please run Stage 3 first.")
        st.stop()

    # Build Comparison Table
    table_rows = []
    for model_name, info in metrics_data.items():
        for h in [7, 15, 30]:
            m = info["horizons"][str(h)]
            table_rows.append({
                "Algorithm": model_name,
                "Horizon": f"{h} Days",
                "MAE (Loads)": m["MAE"],
                "RMSE (Loads)": m["RMSE"],
                "MAPE (%)": m["MAPE"],
                "Accuracy (%)": round(100 - m["MAPE"], 2),
                "R² Score": m["R2"]
            })

    comp_df = pd.DataFrame(table_rows)

    # Filter by Horizon
    selected_horizon = st.selectbox("Filter by Forecast Horizon:", ["All Horizons", "7 Days", "15 Days", "30 Days"])
    filtered_df = comp_df if selected_horizon == "All Horizons" else comp_df[comp_df["Horizon"] == selected_horizon]

    st.dataframe(
        filtered_df.sort_values(by="MAPE (%)", ascending=True),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Side-by-side comparison charts
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Mean Absolute Error (MAE) by Model")
        fig_mae = px.bar(
            comp_df,
            x="Horizon",
            y="MAE (Loads)",
            color="Algorithm",
            barmode="group",
            text="MAE (Loads)",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_mae.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_mae, use_container_width=True)
        st.caption("Lower MAE indicates smaller absolute deviation in daily page loads.")

    with c2:
        st.subheader("Forecast Accuracy (100 - MAPE %)")
        fig_acc = px.bar(
            comp_df,
            x="Horizon",
            y="Accuracy (%)",
            color="Algorithm",
            barmode="group",
            text="Accuracy (%)",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_acc.update_layout(yaxis_range=[75, 100], margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_acc, use_container_width=True)
        st.caption("Linear Regression (Ridge) maintains >94% accuracy across all horizons.")

    st.markdown("---")

    # Actual vs Predicted Plot
    st.subheader("Holdout Validation: Actual vs Predicted Trajectory")
    eval_model = st.selectbox(
        "Choose Model to Inspect Predictions:",
        list(metrics_data.keys()),
        index=1  # Default to Linear Regression (Ridge)
    )

    pred_data = metrics_data[eval_model]["predictions"]["30"]
    dates = pd.to_datetime(pred_data["dates"])
    actual = pred_data["actual"]
    predicted = pred_data["predicted"]

    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(
        x=dates, y=actual,
        mode="lines+markers",
        name="Actual Page Loads",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=6)
    ))
    fig_val.add_trace(go.Scatter(
        x=dates, y=predicted,
        mode="lines+markers",
        name=f"Predicted ({eval_model})",
        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        marker=dict(size=6, symbol="square")
    ))
    fig_val.update_layout(
        hovermode="x unified",
        xaxis_title="Holdout Date",
        yaxis_title="Page Loads",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_val, use_container_width=True)


# ==============================================================================
# PAGE 4: DATA ANALYTICS
# ==============================================================================
elif page == "🔍 Data Analytics":
    st.title("🔍 Exploratory Data Analytics & Traffic Patterns")
    st.write(
        "Explore visitor behavior dynamics, weekly oscillations, academic seasonality cycles, and feature correlations."
    )

    t1, t2, t3, t4 = st.tabs([
        "📅 Weekly Seasonality",
        "📆 Monthly Cycles",
        "👥 Visitor Segments",
        "🔥 Correlation Matrix"
    ])

    with t1:
        st.subheader("Weekly Traffic Pattern: Academic Weekday Peak vs Weekend Slump")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        fig_week = px.box(
            df,
            x="Day",
            y="Page.Loads",
            category_orders={"Day": day_order},
            color="Day",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Distribution of Daily Page Loads by Day of Week"
        )
        fig_week.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_week, use_container_width=True)

        st.info(
            "💡 **Key Insight:** Tuesday is the busiest day of the week (averaging 4,955 loads), "
            "whereas Saturday suffers a 50% drop (2,501 loads). This strong cyclical behavior is leveraged by our lag 7 features."
        )

    with t2:
        st.subheader("Monthly Academic Calendar Seasonality")
        df_m = df.copy()
        df_m["Month_Name"] = df_m["Date"].dt.strftime("%B")
        df_m["Month_Num"] = df_m["Date"].dt.month
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]

        monthly_avg = (
            df_m.groupby(["Month_Num", "Month_Name"])["Page.Loads"]
            .mean()
            .reset_index()
            .sort_values("Month_Num")
        )

        fig_month = px.bar(
            monthly_avg,
            x="Month_Name",
            y="Page.Loads",
            color="Page.Loads",
            color_continuous_scale="Blues",
            title="Average Daily Page Loads by Calendar Month (2014 – 2020)"
        )
        fig_month.update_layout(xaxis_title="Month", yaxis_title="Average Daily Page Loads", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_month, use_container_width=True)

        st.info(
            "💡 **Key Insight:** Clear academic peaks occur in **April (5,167 loads)** and **November (4,705 loads)**, "
            "aligning with semester exams and assignments. Traffic drops in **July and August (3,178 loads)** during summer vacation."
        )

    with t3:
        st.subheader("Visitor Type Engagement: New vs Returning Visitors")
        fig_scatter = px.scatter(
            df,
            x="First.Time.Visits",
            y="Returning.Visits",
            size="Page.Loads",
            color="Day",
            hover_data=["Date", "Page.Loads"],
            title="Returning Visits vs First-Time Visits (Bubble Size = Page Loads)"
        )
        fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with t4:
        st.subheader("Pearson Correlation Heatmap")
        traffic_metrics = ["Page.Loads", "Unique.Visits", "First.Time.Visits", "Returning.Visits"]
        corr = df[traffic_metrics].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Pearson Correlation Matrix across Traffic Metrics"
        )
        fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# PAGE 5: ABOUT PROJECT
# ==============================================================================
elif page == "ℹ️ About Project":
    st.title("ℹ️ About TrafficPredict")
    st.markdown("### Predicting Website Traffic Using Machine Learning & Time Series Forecasting")

    st.markdown("---")

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.subheader("🎯 Project Objectives")
        st.markdown("""
        * **Automated Demand Forecasting:** Predict website server traffic demand across 7, 15, and 30-day horizons.
        * **Visitor Dynamics Analysis:** Segment traffic into First-Time and Returning visitor cohorts.
        * **Multi-Model Benchmarking:** Contrast Baseline, Regularized Linear Regression, Ensemble Random Forests, and SARIMA models.
        * **Zero Data Leakage:** Implement chronological train-test splits and shifted rolling statistics.
        """)

        st.subheader("📁 Dataset Specification")
        st.markdown("""
        * **Source:** Historical visitor logs from academic web platform `statforecasting.com`.
        * **Records:** 2,167 daily records spanning from **14 September 2014 to 19 August 2020**.
        * **Continuity:** 100% continuous daily timeline with zero missing dates and zero null values.
        * **Primary Target:** `Page.Loads` (mean: 4,117 / max: 7,984).
        """)

    with col_a2:
        st.subheader("🛠️ Technology Stack")
        st.markdown("""
        * **Language:** Python 3.14
        * **Web Application:** Streamlit
        * **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
        * **Machine Learning:** Scikit-Learn (Ridge Regression, Random Forest Regressor)
        * **Time-Series Econometrics:** Statsmodels (SARIMAX, Augmented Dickey-Fuller)
        * **Data Engineering:** Pandas & NumPy
        """)

        st.subheader("⚠️ Limitations & Future Work")
        st.markdown("""
        * **External Events:** Search engine algorithm updates, advertising campaigns, or unmodeled marketing initiatives can create non-seasonal spikes.
        * **Domain Seasonality:** The dataset reflects an academic website; corporate e-commerce traffic may show distinct weekend spikes rather than weekday peaks.
        * **Model Deployment:** Deployable via Streamlit Community Cloud or Docker containers with automated weekly retraining.
        """)

    st.markdown("---")
    st.subheader("👩‍💻 Developer & Portfolio")
    st.markdown("""
    * **Project Developer:** Khushi Yadav
    * **Reference Architecture:** Inspired by [AquaPredict](https://aquapredict8960.streamlit.app/)
    * **GitHub Repository:** [Khushiydv21](https://github.com/Khushiydv21)
    """)
