import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


KLINE_DDL_SQL = """
CREATE TABLE IF NOT EXISTS kline_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_code TEXT NOT NULL,
    datetime TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    amount REAL,
    amplitude REAL,
    pct_change REAL,
    change REAL,
    turnover REAL,
    frequency TEXT NOT NULL
);
"""

KLINE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_kline_unique
ON kline_data (ts_code, datetime, frequency);
"""

FEATURE_DDL_SQL = """
CREATE TABLE IF NOT EXISTS feature_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_code TEXT NOT NULL,
    datetime TEXT NOT NULL,
    frequency TEXT NOT NULL,
    return_1 REAL,
    return_5 REAL,
    ma_5 REAL,
    ma_10 REAL,
    ma_20 REAL,
    ema_12 REAL,
    ema_26 REAL,
    dif REAL,
    dea REAL,
    macd REAL,
    rsi_14 REAL,
    atr_14 REAL,
    vol_z_20 REAL,
    updated_at TEXT NOT NULL
);
"""

FEATURE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_unique
ON feature_data (ts_code, datetime, frequency);
"""

STOCK_INFO_DDL_SQL = """
CREATE TABLE IF NOT EXISTS stock_info (
    ts_code TEXT PRIMARY KEY,
    symbol TEXT,
    name TEXT,
    market TEXT,
    industry TEXT,
    list_date TEXT,
    updated_at TEXT NOT NULL
);
"""


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows}


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(KLINE_DDL_SQL)
    conn.execute(KLINE_INDEX_SQL)
    conn.execute(FEATURE_DDL_SQL)
    conn.execute(FEATURE_INDEX_SQL)
    conn.execute(STOCK_INFO_DDL_SQL)

    # Lightweight migration for old kline_data schema.
    existing = _table_columns(conn, "kline_data")
    add_cols = {
        "amplitude": "REAL",
        "pct_change": "REAL",
        "change": "REAL",
        "turnover": "REAL",
    }
    for col, col_type in add_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE kline_data ADD COLUMN {col} {col_type}")

    conn.commit()


def upsert_stock_info_batch(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "updated_at" not in df.columns:
        df = df.copy()
        df["updated_at"] = now_text

    required = ["ts_code", "symbol", "name", "market", "industry", "list_date", "updated_at"]
    for col in required:
        if col not in df.columns:
            df[col] = None

    records = list(df[required].itertuples(index=False, name=None))
    before = conn.total_changes
    conn.executemany(
        """
        INSERT OR REPLACE INTO stock_info
        (ts_code, symbol, name, market, industry, list_date, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()
    return conn.total_changes - before


def get_max_datetime(conn: sqlite3.Connection, ts_code: str, frequency: str) -> Optional[str]:
    row = conn.execute(
        "SELECT MAX(datetime) FROM kline_data WHERE ts_code = ? AND frequency = ?",
        (ts_code, frequency),
    ).fetchone()
    return row[0] if row and row[0] else None


def insert_batch(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    for col in ["amplitude", "pct_change", "change", "turnover"]:
        if col not in df.columns:
            df[col] = None

    records = list(
        df[
            [
                "ts_code",
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "amplitude",
                "pct_change",
                "change",
                "turnover",
                "frequency",
            ]
        ].itertuples(index=False, name=None)
    )
    before = conn.total_changes
    conn.executemany(
        """
        INSERT OR IGNORE INTO kline_data
        (ts_code, datetime, open, high, low, close, volume, amount,
         amplitude, pct_change, change, turnover, frequency)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()
    return conn.total_changes - before


def get_feature_max_datetime(conn: sqlite3.Connection, ts_code: str, frequency: str) -> Optional[str]:
    row = conn.execute(
        "SELECT MAX(datetime) FROM feature_data WHERE ts_code = ? AND frequency = ?",
        (ts_code, frequency),
    ).fetchone()
    return row[0] if row and row[0] else None


def _load_kline_for_feature_update(
    conn: sqlite3.Connection,
    ts_code: str,
    frequency: str,
    last_feature_dt: Optional[str],
    lookback_rows: int,
) -> pd.DataFrame:
    cols = (
        "ts_code, datetime, open, high, low, close, volume, amount, "
        "amplitude, pct_change, change, turnover, frequency"
    )
    if not last_feature_dt:
        sql = f"""
        SELECT {cols}
        FROM kline_data
        WHERE ts_code = ? AND frequency = ?
        ORDER BY datetime
        """
        return pd.read_sql_query(sql, conn, params=(ts_code, frequency))

    prev_sql = f"""
    SELECT {cols}
    FROM kline_data
    WHERE ts_code = ? AND frequency = ? AND datetime < ?
    ORDER BY datetime DESC
    LIMIT ?
    """
    prev = pd.read_sql_query(prev_sql, conn, params=(ts_code, frequency, last_feature_dt, lookback_rows))
    if not prev.empty:
        prev = prev.iloc[::-1].reset_index(drop=True)

    cur_sql = f"""
    SELECT {cols}
    FROM kline_data
    WHERE ts_code = ? AND frequency = ? AND datetime >= ?
    ORDER BY datetime
    """
    cur = pd.read_sql_query(cur_sql, conn, params=(ts_code, frequency, last_feature_dt))

    if prev.empty and cur.empty:
        return pd.DataFrame()
    if prev.empty:
        return cur
    if cur.empty:
        return prev
    return pd.concat([prev, cur], ignore_index=True)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"]).sort_values("datetime")
    if out.empty:
        return pd.DataFrame()

    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["close", "high", "low", "volume"])
    if out.empty:
        return pd.DataFrame()

    out["return_1"] = out["close"].pct_change(1)
    out["return_5"] = out["close"].pct_change(5)
    out["ma_5"] = out["close"].rolling(5).mean()
    out["ma_10"] = out["close"].rolling(10).mean()
    out["ma_20"] = out["close"].rolling(20).mean()

    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["dif"] = out["ema_12"] - out["ema_26"]
    out["dea"] = out["dif"].ewm(span=9, adjust=False).mean()
    out["macd"] = (out["dif"] - out["dea"]) * 2.0

    diff = out["close"].diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()

    vol_ma20 = out["volume"].rolling(20).mean()
    vol_std20 = out["volume"].rolling(20).std()
    out["vol_z_20"] = (out["volume"] - vol_ma20) / vol_std20.replace(0, np.nan)

    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    keep = [
        "ts_code",
        "datetime",
        "frequency",
        "return_1",
        "return_5",
        "ma_5",
        "ma_10",
        "ma_20",
        "ema_12",
        "ema_26",
        "dif",
        "dea",
        "macd",
        "rsi_14",
        "atr_14",
        "vol_z_20",
        "updated_at",
    ]
    return out[keep]


def upsert_feature_batch(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    feature_cols = [
        "ts_code",
        "datetime",
        "frequency",
        "return_1",
        "return_5",
        "ma_5",
        "ma_10",
        "ma_20",
        "ema_12",
        "ema_26",
        "dif",
        "dea",
        "macd",
        "rsi_14",
        "atr_14",
        "vol_z_20",
        "updated_at",
    ]
    payload = df[feature_cols].copy()
    payload = payload.where(pd.notna(payload), None)
    records = list(
        payload.itertuples(index=False, name=None)
    )
    before = conn.total_changes
    conn.executemany(
        """
        INSERT OR REPLACE INTO feature_data
        (ts_code, datetime, frequency, return_1, return_5, ma_5, ma_10, ma_20,
         ema_12, ema_26, dif, dea, macd, rsi_14, atr_14, vol_z_20, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()
    return conn.total_changes - before


def refresh_features_for_symbol_frequency(
    conn: sqlite3.Connection,
    ts_code: str,
    frequency: str,
    lookback_rows: int = 300,
) -> int:
    last_feature_dt = get_feature_max_datetime(conn, ts_code, frequency)
    src = _load_kline_for_feature_update(conn, ts_code, frequency, last_feature_dt, lookback_rows)
    if src.empty:
        return 0
    feat = _compute_features(src)
    if feat.empty:
        return 0

    # In incremental mode, keep overlap (lookback rows) to refresh rolling features safely.
    if last_feature_dt:
        feat = feat[feat["datetime"] >= last_feature_dt]
    return upsert_feature_batch(conn, feat)
