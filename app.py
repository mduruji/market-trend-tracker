import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Quantitative Market Trend & Risk Tracker",
    page_icon="📈",
    layout="wide",
)

DB_PATH = "market_data.duckdb"


@st.cache_data(ttl=3600)
def load_dashboard_data():
    """Loads transformed metrics and correlation matrix from DuckDB."""
    con = duckdb.connect(DB_PATH, read_only=True)
    analytics_df = con.execute(
        "SELECT * FROM processed_analytics ORDER BY Date ASC;"
    ).df()
    corr_df = con.execute("SELECT * FROM correlation_matrix;").df()
    con.close()
    return analytics_df, corr_df


try:
    df, corr_matrix = load_dashboard_data()
except Exception as e:
    st.error(
        f"Unable to read DuckDB tables. Please execute `python analytics.py` first. Error: {e}"
    )
    st.stop()

# --- Sidebar Controls ---
st.sidebar.title("Portfolio Settings")
all_tickers = sorted(df["Ticker"].unique().tolist())
selected_ticker = st.sidebar.selectbox(
    "Focus Asset", all_tickers, index=0 if "SPY" in all_tickers else 0
)

min_date = df["Date"].min().to_pydatetime()
max_date = df["Date"].max().to_pydatetime()
date_range = st.sidebar.date_input(
    "Analysis Window",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(
        date_range[1]
    )
    filtered_df = df[
        (df["Date"] >= start_dt)
        & (df["Date"] <= end_dt)
        & (df["Ticker"] == selected_ticker)
    ].copy()
    cohort_df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()
else:
    filtered_df = df[df["Ticker"] == selected_ticker].copy()
    cohort_df = df.copy()

# --- Top Banner & KPIs ---
st.title("Quantitative Risk & Return Monitor")
st.caption(
    "Automated data pipeline transforming raw market records into risk, drawdown, and correlation metrics."
)

latest_record = filtered_df.dropna(subset=["Close"]).iloc[-1]
total_return = latest_record["Cumulative_Return"] * 100
current_vol = latest_record["Rolling_Vol"] * 100
current_sharpe = latest_record["Rolling_Sharpe"]
max_dd = filtered_df["Drawdown"].min() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cumulative Return", f"{total_return:+.2f}%")
col2.metric(
    "Annualized Volatility (60D)",
    f"{current_vol:.2f}%" if pd.notnull(current_vol) else "N/A",
)
col3.metric(
    "Sharpe Ratio (Rf=4%)",
    f"{current_sharpe:.2f}" if pd.notnull(current_sharpe) else "N/A",
)
col4.metric("Max Peak-to-Trough Loss", f"{max_dd:.2f}%")

st.markdown("---")

# --- Visual Tabs ---
tab1, tab2 = st.tabs(["Performance & Drawdown", "Risk & Asset Correlations"])

with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        fig_perf = px.line(
            cohort_df,
            x="Date",
            y="Cumulative_Return",
            color="Ticker",
            title="Benchmark Comparison: Cumulative Return",
            labels={"Cumulative_Return": "Cumulative Growth (Base 0)"},
        )
        fig_perf.update_layout(
            hovermode="x unified", legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_right:
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=filtered_df["Date"],
                y=filtered_df["Drawdown"] * 100,
                mode="lines",
                fill="tozeroy",
                line=dict(color="#d9534f", width=1.5),
                name=f"{selected_ticker} Drawdown",
            )
        )
        fig_dd.update_layout(
            title=f"{selected_ticker}: Historical Drawdown Profile (%)",
            yaxis_title="Decline from Peak (%)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

with tab2:
    col_risk1, col_risk2 = st.columns(2)

    with col_risk1:
        fig_sharpe = px.line(
            filtered_df.dropna(subset=["Rolling_Sharpe"]),
            x="Date",
            y="Rolling_Sharpe",
            title=f"{selected_ticker}: 60-Day Rolling Sharpe Ratio",
            labels={"Rolling_Sharpe": "Sharpe Ratio"},
        )
        fig_sharpe.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="green",
            annotation_text="Target (1.0)",
        )
        fig_sharpe.add_hline(y=0.0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_sharpe, use_container_width=True)

    with col_risk2:
        # Isolate correlation values from the table
        corr_data = corr_matrix.set_index("Ticker")
        fig_corr = px.imshow(
            corr_data.values,
            x=corr_data.columns.tolist(),
            y=corr_data.index.tolist(),
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            title="Cross-Asset Log Return Correlation",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with st.expander("Inspect Underlying DuckDB Table"):
    st.dataframe(
        filtered_df[
            [
                "Date",
                "Ticker",
                "Close",
                "Log_Return",
                "Rolling_Vol",
                "Rolling_Sharpe",
                "Drawdown",
            ]
        ].tail(50),
        use_container_width=True,
    )