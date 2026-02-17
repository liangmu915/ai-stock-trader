import argparse
import logging
import sqlite3
import time
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import requests

API_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def ts_code_to_secid(ts_code: str) -> str:
    """
    Convert TS code (e.g. 000001.SZ, 600000.SH) to Eastmoney secid.
    SZ -> 0.{code}, SH -> 1.{code}
    """
    if "." not in ts_code:
        raise ValueError(f"Invalid ts_code format: {ts_code}")
    code, market = ts_code.upper().split(".", 1)
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"Invalid stock code: {ts_code}")
    if market == "SZ":
        return f"0.{code}"
    if market == "SH":
        return f"1.{code}"
    raise ValueError(f"Unsupported market in ts_code: {ts_code}")


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS minute_5m (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            datetime TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_minute_5m_code_dt
        ON minute_5m (ts_code, datetime)
        """
    )
    conn.commit()


def get_max_datetime(conn: sqlite3.Connection, ts_code: str) -> Optional[str]:
    row = conn.execute(
        "SELECT MAX(datetime) FROM minute_5m WHERE ts_code = ?",
        (ts_code,),
    ).fetchone()
    return row[0] if row and row[0] else None


def date_str_yyyymmdd(date_text: str) -> str:
    dt = datetime.strptime(date_text, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y%m%d")


def max_date_str(a: str, b: str) -> str:
    return a if a >= b else b


def fetch_eastmoney_5m(
    ts_code: str,
    beg: str,
    end: str,
    best_effort: bool = True,
    timeout: int = 20,
) -> pd.DataFrame:
    """
    Fetch 5-minute K-line data from Eastmoney and return standardized DataFrame.
    Output columns:
    ts_code, datetime, open, high, low, close, volume, amount
    """
    secid = ts_code_to_secid(ts_code)
    # For minute K-line, Eastmoney is more stable with wide server-side range.
    # We then filter locally by the requested beg/end date window.
    params = {
        "secid": secid,
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "5",
        "fqt": "1",
        "beg": "0",
        "end": "20500000",
        "lmt": "1000000",
    }

    try:
        resp = requests.get(API_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logging.error("Request failed for %s: %s", ts_code, exc)
        return pd.DataFrame()
    except ValueError as exc:
        logging.error("JSON parse failed for %s: %s", ts_code, exc)
        return pd.DataFrame()

    data = payload.get("data") if isinstance(payload, dict) else None
    klines = data.get("klines") if isinstance(data, dict) else None
    if not klines:
        logging.info("Empty kline response for %s in range %s-%s", ts_code, beg, end)
        return pd.DataFrame()

    rows: List[Tuple[str, str, float, float, float, float, float, float]] = []
    for item in klines:
        parts = item.split(",")
        if len(parts) < 7:
            continue
        try:
            dt_text = parts[0].strip()
            open_px = float(parts[1])
            close_px = float(parts[2])
            high_px = float(parts[3])
            low_px = float(parts[4])
            volume = float(parts[5])
            amount = float(parts[6])
            rows.append((ts_code, dt_text, open_px, high_px, low_px, close_px, volume, amount))
        except (TypeError, ValueError):
            continue

    if not rows:
        return pd.DataFrame()

    all_df = pd.DataFrame(
        rows,
        columns=["ts_code", "datetime", "open", "high", "low", "close", "volume", "amount"],
    )
    all_df["datetime"] = pd.to_datetime(all_df["datetime"], errors="coerce")
    all_df = all_df.dropna(subset=["datetime"]).sort_values("datetime")
    if all_df.empty:
        return pd.DataFrame()

    avail_start = all_df["datetime"].min().strftime("%Y-%m-%d %H:%M:%S")
    avail_end = all_df["datetime"].max().strftime("%Y-%m-%d %H:%M:%S")
    logging.info("Available 5m window for %s from API: %s -> %s", ts_code, avail_start, avail_end)

    start_dt = pd.to_datetime(f"{beg} 00:00:00", format="%Y%m%d %H:%M:%S")
    end_dt = pd.to_datetime(f"{end} 23:59:59", format="%Y%m%d %H:%M:%S")
    df = all_df[(all_df["datetime"] >= start_dt) & (all_df["datetime"] <= end_dt)]
    if df.empty:
        if best_effort:
            logging.warning(
                "No 5m bars in requested window for %s (%s-%s). "
                "Falling back to all available data from API window.",
                ts_code,
                beg,
                end,
            )
            df = all_df
        else:
            logging.info(
                "Kline payload exists but no 5m bars in requested window for %s: %s-%s",
                ts_code,
                beg,
                end,
            )
            return pd.DataFrame()

    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def save_to_db(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    records = list(
        df[
            ["ts_code", "datetime", "open", "high", "low", "close", "volume", "amount"]
        ].itertuples(index=False, name=None)
    )
    conn.executemany(
        """
        INSERT OR IGNORE INTO minute_5m
        (ts_code, datetime, open, high, low, close, volume, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()
    return conn.total_changes


def incremental_fetch_and_save(
    conn: sqlite3.Connection,
    ts_code: str,
    beg: str,
    end: str,
    best_effort: bool,
) -> int:
    last_dt = get_max_datetime(conn, ts_code)
    fetch_beg = beg

    if last_dt:
        last_beg = date_str_yyyymmdd(last_dt)
        fetch_beg = max_date_str(beg, last_beg)
        logging.info("Incremental mode for %s, last datetime in DB: %s", ts_code, last_dt)

    df = fetch_eastmoney_5m(ts_code=ts_code, beg=fetch_beg, end=end, best_effort=best_effort)
    if df.empty:
        return 0

    if last_dt:
        df = df[df["datetime"] > last_dt]
        if df.empty:
            logging.info("No new bars for %s", ts_code)
            return 0

    before = conn.total_changes
    save_to_db(conn, df)
    inserted = conn.total_changes - before
    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Eastmoney 5-minute A-share data and store in SQLite."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="000001.SZ,600000.SH",
        help="Comma-separated TS codes, e.g. 000001.SZ,600000.SH",
    )
    parser.add_argument("--beg", type=str, default="20250101", help="Begin date, e.g. 20250101")
    parser.add_argument("--end", type=str, default="20250131", help="End date, e.g. 20250131")
    parser.add_argument("--db", type=str, default="market.db", help="SQLite DB path")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between requests")
    parser.add_argument(
        "--strict-window",
        action="store_true",
        help="Only keep requested --beg/--end window. Default is best-effort max data.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    if not symbols:
        raise ValueError("No symbols provided.")

    with sqlite3.connect(args.db) as conn:
        init_db(conn)

        for ts_code in symbols:
            try:
                inserted = incremental_fetch_and_save(
                    conn=conn,
                    ts_code=ts_code,
                    beg=args.beg,
                    end=args.end,
                    best_effort=not args.strict_window,
                )
                logging.info("Done %s, inserted rows: %d", ts_code, inserted)
            except Exception as exc:
                logging.error("Failed %s: %s", ts_code, exc)
            time.sleep(args.sleep)

    logging.info("All tasks finished. SQLite file: %s", args.db)


if __name__ == "__main__":
    main()
