from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def _load_from_sqlite(db_path: Path, table_name: str) -> pd.DataFrame:
    """
    Load daily bars from SQLite.

    Preferred table:
    - daily_prices(trade_date, ts_code, open, high, low, close, volume)

    Fallback:
    - kline_data where frequency='daily'
    """
    with sqlite3.connect(db_path) as conn:
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()

        if table_exists and table_name == "daily_prices":
            sql = """
            SELECT
                trade_date AS date,
                ts_code AS stock,
                open,
                high,
                low,
                close,
                volume
            FROM daily_prices
            """
            return pd.read_sql_query(sql, conn)

        # Fallback for kline_data schema
        fallback_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kline_data'"
        ).fetchone()
        if fallback_exists:
            cols = [c[1] for c in conn.execute("PRAGMA table_info(kline_data)").fetchall()]
            # New schema: datetime + frequency
            if ("datetime" in cols) and ("frequency" in cols):
                sql = """
                SELECT
                    substr(datetime, 1, 10) AS date,
                    ts_code AS stock,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM kline_data
                WHERE frequency='daily'
                """
                df = pd.read_sql_query(sql, conn)
                if not df.empty:
                    return df
            # Legacy schema: trade_time + timeframe
            if ("trade_time" in cols) and ("timeframe" in cols):
                sql = """
                SELECT
                    substr(trade_time, 1, 10) AS date,
                    ts_code AS stock,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM kline_data
                WHERE timeframe='daily'
                """
                df = pd.read_sql_query(sql, conn)
                if not df.empty:
                    return df
                tf = conn.execute("SELECT DISTINCT timeframe FROM kline_data ORDER BY timeframe").fetchall()
                tf_text = ",".join([x[0] for x in tf if x and x[0]]) or "UNKNOWN"
                raise ValueError(
                    f"SQLite has kline_data but no daily rows. available timeframe={tf_text}. "
                    f"Please select a DB with daily_prices or kline_data daily."
                )

    raise ValueError(f"No usable table found in sqlite file: {db_path}")


def load_data(
    data_source: str,
    data_path: Path,
    sqlite_table: str = "daily_prices",
    sample_stocks: int | None = None,
) -> pd.DataFrame:
    """
    Load merged multi-stock daily data into a single dataframe.
    Expected output columns:
    date, stock, open, high, low, close, volume
    """
    data_source = data_source.lower().strip()
    if data_source == "csv":
        df = pd.read_csv(data_path)
    elif data_source == "parquet":
        df = pd.read_parquet(data_path)
    elif data_source == "sqlite":
        df = _load_from_sqlite(data_path, sqlite_table)
    else:
        raise ValueError("data_source must be one of: csv, parquet, sqlite")

    required = {"date", "stock", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "stock", "open", "high", "low", "close", "volume"])

    if sample_stocks is not None and sample_stocks > 0:
        # Deterministic subset for quick local iteration.
        chosen = sorted(df["stock"].astype(str).unique())[:sample_stocks]
        df = df[df["stock"].isin(chosen)]

    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    return df


def load_stock_info_map(data_source: str, data_path: Path) -> pd.DataFrame:
    """
    Load stock -> industry/market mapping for feature engineering and constraints.

    Returns columns:
    - stock
    - industry
    - market
    """
    data_source = data_source.lower().strip()
    if data_source != "sqlite":
        return pd.DataFrame(columns=["stock", "industry", "market"])

    with sqlite3.connect(data_path) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stock_info'"
        ).fetchone()
        if not exists:
            return pd.DataFrame(columns=["stock", "industry", "market"])

        info = pd.read_sql_query(
            """
            SELECT
                ts_code AS stock,
                industry,
                market
            FROM stock_info
            """,
            conn,
        )
    info["stock"] = info["stock"].astype(str)
    return info.drop_duplicates(subset=["stock"], keep="last").reset_index(drop=True)


def load_stock_info_snapshot(
    data_source: str,
    data_path: Path,
    snapshot_at: str | None = None,
) -> pd.DataFrame:
    """
    Load a fixed stock snapshot for enrichment.

    - If snapshot_at is None: use latest row per ts_code by updated_at (or arbitrary single row if no updated_at parsing).
    - If snapshot_at is provided: only use rows with updated_at <= snapshot_at.

    This is a static snapshot and is NOT a full as-of history mapping.
    Returned columns: stock, symbol, name, market, industry, list_date
    """
    data_source = data_source.lower().strip()
    if data_source != "sqlite":
        return pd.DataFrame(columns=["stock", "symbol", "name", "market", "industry", "list_date"])

    with sqlite3.connect(data_path) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stock_info'"
        ).fetchone()
        if not exists:
            return pd.DataFrame(columns=["stock", "symbol", "name", "market", "industry", "list_date"])

        cols = [r[1] for r in conn.execute("PRAGMA table_info(stock_info)").fetchall()]
        base_cols = ["ts_code AS stock", "symbol", "name", "market", "industry", "list_date"]
        has_updated_at = "updated_at" in cols
        if has_updated_at:
            base_cols.append("updated_at")
        sql = f"SELECT {', '.join(base_cols)} FROM stock_info"
        info = pd.read_sql_query(sql, conn)

    if info.empty:
        return pd.DataFrame(columns=["stock", "symbol", "name", "market", "industry", "list_date"])

    info["stock"] = info["stock"].astype(str)
    if "updated_at" in info.columns:
        info["updated_at"] = pd.to_datetime(info["updated_at"], errors="coerce")
        if snapshot_at is not None:
            cutoff = pd.Timestamp(snapshot_at)
            info = info[info["updated_at"].notna() & (info["updated_at"] <= cutoff)].copy()
        info = info.sort_values(["stock", "updated_at"]).drop_duplicates(subset=["stock"], keep="last")
    else:
        info = info.drop_duplicates(subset=["stock"], keep="last")

    return info[["stock", "symbol", "name", "market", "industry", "list_date"]].reset_index(drop=True)
