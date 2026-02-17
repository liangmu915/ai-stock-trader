import argparse
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import requests

API_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
FAILED_LOG_FILE = "failed_symbols.txt"


def ts_code_to_secid(ts_code: str) -> str:
    code, market = ts_code.upper().split(".", 1)
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"Invalid ts_code: {ts_code}")
    if market == "SZ":
        return f"0.{code}"
    if market == "SH":
        return f"1.{code}"
    raise ValueError(f"Unsupported market for ts_code: {ts_code}")


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_prices (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
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
        CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_prices_unique
        ON daily_prices (ts_code, trade_date)
        """
    )
    conn.commit()


def get_stock_list(db_path: str, csv_path: Optional[str] = None) -> List[str]:
    """
    Load ts_code list from CSV or SQLite stock_info table.
    Priority: CSV > stock_info.
    """
    symbols: List[str] = []

    if csv_path:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        with p.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            if "ts_code" not in [h.strip() for h in header]:
                raise ValueError("CSV must contain a 'ts_code' column.")
            idx = [h.strip() for h in header].index("ts_code")
            for line in f:
                parts = [x.strip() for x in line.strip().split(",")]
                if idx < len(parts) and parts[idx]:
                    symbols.append(parts[idx].upper())
    else:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT ts_code FROM stock_info ORDER BY ts_code").fetchall()
        symbols = [r[0].upper() for r in rows if r and r[0]]

    symbols = sorted(set(symbols))
    if not symbols:
        raise ValueError("No symbols found. Provide --csv or ensure stock_info table has data.")
    return symbols


def request_with_retry(
    session: requests.Session,
    url: str,
    params: dict,
    max_retries: int = 5,
    connect_timeout: float = 5.0,
    read_timeout: float = 15.0,
) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=(connect_timeout, read_timeout))
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            print(f"[Timeout] attempt {attempt}/{max_retries}: {exc}", file=sys.stderr)
        except requests.RequestException as exc:
            last_exc = exc
            print(f"[RequestError] attempt {attempt}/{max_retries}: {exc}", file=sys.stderr)

        if attempt < max_retries:
            backoff = 0.6 * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
            time.sleep(backoff)

    raise RuntimeError(f"Request failed after {max_retries} retries: {last_exc}")


def fetch_daily_data(
    session: requests.Session,
    ts_code: str,
    start_date: str,
    end_date: str,
) -> List[Tuple[str, str, float, float, float, float, float, float]]:
    """
    Fetch daily adjusted kline data from Eastmoney.
    Returns tuples:
    (ts_code, trade_date, open, high, low, close, volume, amount)
    """
    secid = ts_code_to_secid(ts_code)
    params = {
        "secid": secid,
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",   # daily
        "fqt": "1",     # forward adjusted
        "beg": start_date,
        "end": end_date,
        "lmt": "1000000",
    }

    data = request_with_retry(session, API_URL, params)
    payload = data.get("data") if isinstance(data, dict) else None
    klines = payload.get("klines") if isinstance(payload, dict) else None
    if not klines:
        return []

    out: List[Tuple[str, str, float, float, float, float, float, float]] = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 7:
            continue
        try:
            trade_date = parts[0].strip()
            open_px = float(parts[1])
            close_px = float(parts[2])
            high_px = float(parts[3])
            low_px = float(parts[4])
            volume = float(parts[5])
            amount = float(parts[6])
            out.append((ts_code, trade_date, open_px, high_px, low_px, close_px, volume, amount))
        except (ValueError, TypeError):
            continue

    return out


def get_latest_trade_date(conn: sqlite3.Connection, ts_code: str) -> Optional[str]:
    row = conn.execute(
        "SELECT MAX(trade_date) FROM daily_prices WHERE ts_code = ?",
        (ts_code,),
    ).fetchone()
    return row[0] if row and row[0] else None


def save_to_db(
    conn: sqlite3.Connection,
    rows: List[Tuple[str, str, float, float, float, float, float, float]],
) -> int:
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        """
        INSERT OR IGNORE INTO daily_prices
        (ts_code, trade_date, open, high, low, close, volume, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def append_failed(ts_code: str, reason: str, failed_file: str = FAILED_LOG_FILE) -> None:
    with open(failed_file, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}\t{ts_code}\t{reason}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download recent N years of daily A-share data.")
    parser.add_argument("--db", type=str, default="market_kline.db", help="SQLite db path")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV path containing ts_code column")
    parser.add_argument("--failed-log", type=str, default=FAILED_LOG_FILE, help="Failed symbol log file")
    parser.add_argument("--years", type=int, default=15, help="How many recent years to fetch (default: 15)")
    args = parser.parse_args()

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365 * max(1, args.years))
    start_date = start_dt.strftime("%Y%m%d")
    end_date = end_dt.strftime("%Y%m%d")

    with sqlite3.connect(args.db) as conn:
        ensure_table(conn)
        symbols = get_stock_list(db_path=args.db, csv_path=args.csv)

        total = len(symbols)
        ok_count = 0
        fail_count = 0
        total_inserted = 0
        begin_ts = time.time()

        session = requests.Session()
        try:
            print(f"Start daily crawl: symbols={total}, range={start_date}-{end_date}")
            for i, ts_code in enumerate(symbols, start=1):
                elapsed = time.time() - begin_ts
                avg = elapsed / i if i else 0
                eta = avg * (total - i)
                print(
                    f"[{i}/{total}] {ts_code} | elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
                )

                try:
                    latest = get_latest_trade_date(conn, ts_code)
                    req_start = start_date
                    if latest:
                        latest_compact = latest.replace("-", "")[:8]
                        if latest_compact > req_start:
                            req_start = latest_compact

                    rows = fetch_daily_data(
                        session=session,
                        ts_code=ts_code,
                        start_date=req_start,
                        end_date=end_date,
                    )

                    if latest:
                        rows = [r for r in rows if r[1].replace("-", "") > latest.replace("-", "")[:8]]

                    inserted = save_to_db(conn, rows)
                    total_inserted += inserted
                    ok_count += 1
                except Exception as exc:
                    fail_count += 1
                    err = str(exc)
                    print(f"[ERROR] {ts_code}: {err}", file=sys.stderr)
                    append_failed(ts_code, err, args.failed_log)
                    continue
                finally:
                    time.sleep(random.uniform(0.3, 0.8))
        finally:
            session.close()

        total_runtime = time.time() - begin_ts
        print("\n===== Summary =====")
        print(f"Total processed : {total}")
        print(f"Success count   : {ok_count}")
        print(f"Failure count   : {fail_count}")
        print(f"Rows inserted   : {total_inserted}")
        print(f"Total runtime   : {format_seconds(total_runtime)}")
        print(f"Failed log file : {args.failed_log}")


if __name__ == "__main__":
    main()
