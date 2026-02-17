import argparse
from typing import Optional, Tuple

import akshare as ak


def _symbol_from_ts(ts_code: str) -> str:
    code, _market = ts_code.upper().split(".", 1)
    return code


def fetch_coverage(ts_code: str, klt: str) -> Tuple[int, Optional[str], Optional[str]]:
    symbol = _symbol_from_ts(ts_code)

    if klt == "101":
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date="19900101",
            end_date="20500101",
            adjust="",
        )
        if df is None or df.empty:
            return 0, None, None
        time_col = "日期" if "日期" in df.columns else df.columns[0]
        start = str(df[time_col].iloc[0])[:10]
        end = str(df[time_col].iloc[-1])[:10]
        return len(df), start, end

    if klt not in {"1", "5", "15", "30", "60"}:
        raise ValueError("klt for AkShare must be one of: 1,5,15,30,60,101")

    df = ak.stock_zh_a_hist_min_em(
        symbol=symbol,
        period=klt,
        start_date="1970-01-01 09:30:00",
        end_date="2050-01-01 15:00:00",
        adjust="",
    )
    if df is None or df.empty:
        return 0, None, None
    time_col = "时间" if "时间" in df.columns else df.columns[0]
    start = str(df[time_col].iloc[0])
    end = str(df[time_col].iloc[-1])
    return len(df), start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AkShare kline coverage for symbols.")
    parser.add_argument("--symbols", required=True, help="Comma-separated TS codes, e.g. 000001.SZ,600000.SH")
    parser.add_argument("--klt", default="60", help="Kline period for AkShare: 1,5,15,30,60,101(daily)")
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
