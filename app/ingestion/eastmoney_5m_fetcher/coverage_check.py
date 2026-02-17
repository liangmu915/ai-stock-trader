import argparse
from typing import Optional, Tuple

import requests

API_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def ts_code_to_secid(ts_code: str) -> str:
    code, market = ts_code.upper().split(".", 1)
    if market == "SZ":
        return f"0.{code}"
    if market == "SH":
        return f"1.{code}"
    raise ValueError(f"Unsupported ts_code: {ts_code}")


def fetch_coverage(ts_code: str, klt: str) -> Tuple[int, Optional[str], Optional[str]]:
    secid = ts_code_to_secid(ts_code)
    params = {
        "secid": secid,
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": klt,
        "fqt": "1",
        "beg": "0",
        "end": "20500000",
        "lmt": "1000000",
    }
    resp = requests.get(API_URL, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    klines = ((payload or {}).get("data") or {}).get("klines") or []
    if not klines:
        return 0, None, None
    first_dt = klines[0].split(",")[0]
    last_dt = klines[-1].split(",")[0]
    return len(klines), first_dt, last_dt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Eastmoney kline coverage for symbols.")
    parser.add_argument("--symbols", required=True, help="Comma-separated TS codes, e.g. 000001.SZ,600000.SH")
    parser.add_argument("--klt", default="60", help="Kline period, e.g. 5,15,30,60,101")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"klt={args.klt}")
    for ts_code in symbols:
        try:
            rows, start, end = fetch_coverage(ts_code, args.klt)
            print(f"{ts_code}: rows={rows}, start={start}, end={end}")
        except Exception as exc:
            print(f"{ts_code}: ERROR {exc}")


if __name__ == "__main__":
    main()
