"""
TrafficPredict - Vercel Serverless Web Application
Framework: Flask + Plotly Serverless Engine
Author: Khushi Yadav
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, request, Response, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "trafficpredict-secret-key-2026")


def get_data():
    csv_path = BASE_DIR / "data" / "cleaned_website_traffic.csv"
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_metrics():
    with open(BASE_DIR / "models" / "metrics.json", "r") as f:
        return json.load(f)


def get_forecasts():
    with open(BASE_DIR / "models" / "future_forecasts.json", "r") as f:
        data = json.load(f)
    df_f = pd.DataFrame(data)
    df_f["Date"] = pd.to_datetime(df_f["Date"])
    return df_f


@app.route("/")
def index():
    df = get_data()
    metrics_data = get_metrics()
    future_df = get_forecasts()

    # 1. KPIs
    total_loads = df["Page.Loads"].sum()
    avg_loads = df["Page.Loads"].mean()
    max_loads = df["Page.Loads"].max()
    avg_unique = df["Unique.Visits"].mean()
    first_time_sum = df["First.Time.Visits"].sum()
    unique_sum = df["Unique.Visits"].sum()
    first_time_pct = (first_time_sum / unique_sum) * 100
    returning_sum = df["Returning.Visits"].sum()

    kpis = {
        "total_loads": f"{total_loads / 1e6:.2f} M",
        "avg_loads": f"{avg_loads:,.0f}",
        "max_loads": f"{max_loads:,}",
        "avg_unique": f"{avg_unique:,.0f}",
        "first_time_pct": f"{first_time_pct:.1f}%",
        "first_time_sum": f"{first_time_sum:,}",
        "returning_sum": f"{returning_sum:,}",
        "total_unique": f"{unique_sum / 1e6:.2f} M"
    }

    # 2. Timeline Chart (7-Day & 30-Day Moving Averages)
    df_plot = df.copy()
    df_plot["7D_MA"] = df_plot["Page.Loads"].rolling(7, min_periods=1).mean()
    df_plot["30D_MA"] = df_plot["Page.Loads"].rolling(30, min_periods=1).mean()

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["Page.Loads"],
        mode="lines", name="Raw Daily (Muted)",
        line=dict(color="rgba(150, 160, 180, 0.25)", width=1),
        hoverinfo="skip"
    ))
    fig_timeline.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["7D_MA"],
        mode="lines", name="7-Day Moving Avg",
        line=dict(color="#00b4d8", width=2.5)
    ))
    fig_timeline.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["30D_MA"],
        mode="lines", name="30-Day Moving Avg",
        line=dict(color="#ff9100", width=3)
    ))
    fig_timeline.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    timeline_html = pio.to_html(fig_timeline, full_html=False, include_plotlyjs=False)

    # 3. Donut Chart
    fig_donut = go.Figure(data=[go.Pie(
        labels=["First-Time Visitors", "Returning Visitors"],
        values=[first_time_sum, returning_sum],
        hole=0.62,
        marker=dict(colors=["#00E676", "#7C4DFF"]),
        textinfo="percent",
        textfont=dict(size=14, color="white")
    )])
    fig_donut.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{unique_sum/1e6:.2f}M</b><br><span style='font-size:12px;color:#aaa;'>Total Unique</span>",
            x=0.5, y=0.5, font_size=18, showarrow=False
        )],
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    donut_html = pio.to_html(fig_donut, full_html=False, include_plotlyjs=False)

    # 4. Forecast Chart (Next 30 Days)
    fig_forecast = go.Figure()
    recent = df.tail(14)
    fig_forecast.add_trace(go.Scatter(
        x=recent["Date"], y=recent["Page.Loads"],
        mode="lines+markers", name="Recent Actual",
        line=dict(color="#888", width=2)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=future_df["Date"], y=future_df["Page.Loads"],
        mode="lines+markers", name="Predicted Page Loads (30d)",
        line=dict(color="#00D2FF", width=3),
        marker=dict(size=6)
    ))
    fig_forecast.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    forecast_html = pio.to_html(fig_forecast, full_html=False, include_plotlyjs=False)

    # 5. Visitor Segment Projections
    fig_visitors = go.Figure()
    fig_visitors.add_trace(go.Bar(
        x=future_df["Date"].dt.strftime("%b %d"),
        y=future_df["First.Time.Visits"],
        name="First-Time Visits", marker_color="#00CC96"
    ))
    fig_visitors.add_trace(go.Bar(
        x=future_df["Date"].dt.strftime("%b %d"),
        y=future_df["Returning.Visits"],
        name="Returning Visits", marker_color="#AB63FA"
    ))
    fig_visitors.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    visitors_html = pio.to_html(fig_visitors, full_html=False, include_plotlyjs=False)

    # 6. Model Benchmarking Table Data
    comp_rows = []
    for model_name, info in metrics_data.items():
        for h in [7, 15, 30]:
            m = info["horizons"][str(h)]
            comp_rows.append({
                "model": model_name,
                "horizon": f"{h} Days",
                "mae": m["MAE"],
                "rmse": m["RMSE"],
                "mape": m["MAPE"],
                "accuracy": round(100 - m["MAPE"], 2),
                "r2": m["R2"]
            })

    # 7. Actual vs Predicted Validation Chart (30 Days)
    best_p = metrics_data["Linear Regression (Ridge)"]["predictions"]["30"]
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(
        x=pd.to_datetime(best_p["dates"]), y=best_p["actual"],
        mode="lines+markers", name="Actual Page Loads",
        line=dict(color="#1f77b4", width=2.5)
    ))
    fig_val.add_trace(go.Scatter(
        x=pd.to_datetime(best_p["dates"]), y=best_p["predicted"],
        mode="lines+markers", name="Ridge Predicted",
        line=dict(color="#ff7f0e", width=2.5, dash="dash")
    ))
    fig_val.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    val_html = pio.to_html(fig_val, full_html=False, include_plotlyjs=False)

    # 8. Weekly Seasonality Boxplot
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig_weekly = px.box(
        df, x="Day", y="Page.Loads",
        category_orders={"Day": day_order},
        color="Day", template="plotly_dark"
    )
    fig_weekly.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10)
    )
    weekly_html = pio.to_html(fig_weekly, full_html=False, include_plotlyjs=False)

    forecast_table = future_df[["Date", "Day", "Page.Loads", "Unique.Visits", "First.Time.Visits", "Returning.Visits"]].to_dict(orient="records")

    return render_template(
        "index.html",
        kpis=kpis,
        timeline_html=timeline_html,
        donut_html=donut_html,
        forecast_html=forecast_html,
        visitors_html=visitors_html,
        val_html=val_html,
        weekly_html=weekly_html,
        comp_rows=comp_rows,
        forecast_table=forecast_table
    )


@app.route("/download/forecast.csv")
def download_csv():
    future_df = get_forecasts()
    csv_str = future_df.to_csv(index=False)
    return Response(
        csv_str,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=traffic_forecast_30d.csv"}
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
