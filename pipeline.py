import logging
from typing import List
import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
DB_PATH = "market_data.duckdb"


def fetch_ticker_data(
    ticker: str, period: str = "3y", interval: str = "1d"
) -> pd.DataFrame:
    """Fetches and normalizes historical pricing data for a single asset."""
    logging.info(f"Downloading data for {ticker}...")
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            logging.warning(f"No records returned for {ticker}.")
            return pd.DataFrame()

        # Flatten yfinance MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] for col in raw.columns]

        df = raw.reset_index()

        # Keep standard pricing columns
        expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in expected_cols if c in df.columns]].copy()
        df["Ticker"] = ticker

        # Strip timezone offsets and enforce numeric types
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Close", "Date"])
        return df
    except Exception as e:
        logging.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def run_pipeline(tickers: List[str] = DEFAULT_TICKERS, db_path: str = DB_PATH):
    """Orchestrates data extraction and persists records into DuckDB."""
    all_data = []
    for ticker in tickers:
        df = fetch_ticker_data(ticker)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        logging.error("Pipeline aborted: no valid data collected.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Explicitly enforce column order to match the SQL table schema
    target_columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    combined_df = combined_df[target_columns]

    con = duckdb.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_prices (
            Date TIMESTAMP,
            Ticker VARCHAR,
            Open DOUBLE,
            High DOUBLE,
            Low DOUBLE,
            Close DOUBLE,
            Volume DOUBLE,
            PRIMARY KEY (Date, Ticker)
        );
    """
    )

    # Explicitly name target columns in the INSERT statement
    con.register("temp_df", combined_df)
    con.execute(
        """
        DELETE FROM raw_prices 
        WHERE (Date, Ticker) IN (SELECT Date, Ticker FROM temp_df);
    """
    )
    con.execute(
        """
        INSERT INTO raw_prices (Date, Ticker, Open, High, Low, Close, Volume)
        SELECT Date, Ticker, Open, High, Low, Close, Volume FROM temp_df;
    """
    )
    total_records = con.execute("SELECT COUNT(*) FROM raw_prices;").fetchone()[0]
    con.close()

    logging.info(
        f"Pipeline complete. Database '{db_path}' holds {total_records} records."
    )


if __name__ == "__main__":
    run_pipeline()