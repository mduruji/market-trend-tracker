import logging
from typing import Tuple
import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
DB_PATH = "market_data.duckdb"
ANNUALIZATION_FACTOR = 252  # Trading days in a calendar year
DEFAULT_RISK_FREE_RATE = 0.04  # 4% annual risk-free rate


def load_raw_data(db_path: str = DB_PATH) -> pd.DataFrame:
    """Extracts raw prices sorted chronologically."""
    con = duckdb.connect(db_path, read_only=True)
    query = """
        SELECT Date, Ticker, Close 
        FROM raw_prices 
        ORDER BY Ticker, Date ASC;
    """
    df = con.execute(query).df()
    con.close()
    return df


def calculate_quantitative_metrics(
    df: pd.DataFrame,
    rf_rate: float = DEFAULT_RISK_FREE_RATE,
    rolling_window: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates returns, volatility, rolling Sharpe ratio, and running drawdowns.

    Also produces a cross-asset return correlation matrix.
    """
    df = df.copy()
    daily_rf = rf_rate / ANNUALIZATION_FACTOR

    # Ensure chronological sort before windowing
    df = df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    # 1. Log Returns: ln(P_t / P_{t-1})
    df["Log_Return"] = df.groupby("Ticker")["Close"].transform(
        lambda s: np.log(s / s.shift(1))
    )

    # 2. Cumulative Simple Return: (P_t / P_0) - 1
    df["Cumulative_Return"] = df.groupby("Ticker")["Close"].transform(
        lambda s: (s / s.iloc[0]) - 1.0
    )

    # 3. Rolling Annualized Volatility
    df["Rolling_Vol"] = (
        df.groupby("Ticker")["Log_Return"].transform(
            lambda s: s.rolling(rolling_window).std()
        )
        * np.sqrt(ANNUALIZATION_FACTOR)
    )

    # 4. Rolling 60-Day Sharpe Ratio: (Mean(R - Rf) / Std(R)) * sqrt(252)
    def compute_sharpe(s: pd.Series) -> pd.Series:
        excess_return = s - daily_rf
        mean_excess = excess_return.rolling(rolling_window).mean()
        std_dev = s.rolling(rolling_window).std()
        return (mean_excess / std_dev) * np.sqrt(ANNUALIZATION_FACTOR)

    df["Rolling_Sharpe"] = df.groupby("Ticker")["Log_Return"].transform(
        compute_sharpe
    )

    # 5. Drawdown Engine: Peak-to-Trough Decline
    def compute_drawdown(s: pd.Series) -> pd.Series:
        running_max = s.cummax()
        return (s - running_max) / running_max

    df["Drawdown"] = df.groupby("Ticker")["Close"].transform(compute_drawdown)

    # 6. Correlation Matrix over daily log returns
    pivot_returns = df.pivot(index="Date", columns="Ticker", values="Log_Return")
    corr_matrix = pivot_returns.corr().reset_index()

    return df, corr_matrix


def run_analytics_pipeline(db_path: str = DB_PATH):
    """Executes quantitative calculations and saves results back to DuckDB."""
    logging.info("Reading raw pricing for quantitative modeling...")
    raw_df = load_raw_data(db_path)

    if raw_df.empty:
        logging.error("No raw data found to process.")
        return

    metrics_df, corr_df = calculate_quantitative_metrics(raw_df)

    con = duckdb.connect(db_path)
    con.execute(
        "CREATE OR REPLACE TABLE processed_analytics AS SELECT * FROM metrics_df;"
    )
    con.execute(
        "CREATE OR REPLACE TABLE correlation_matrix AS SELECT * FROM corr_df;"
    )
    con.close()
    logging.info(
        "Quantitative analytics saved successfully to DuckDB tables: 'processed_analytics' and 'correlation_matrix'."
    )


if __name__ == "__main__":
    run_analytics_pipeline()