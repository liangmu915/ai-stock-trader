import logging
from datetime import datetime, timedelta

from src.data_fetcher import fetch_daily_kline, fetch_minute_kline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    daily = fetch_daily_kline("000001.SZ", "20250101", "20250131")
    print("daily rows:", len(daily))
    print("daily sample:", daily[:3])

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    mins = fetch_minute_kline("000001.SZ", start_date, end_date, "5min")
    print("5min range:", start_date, "->", end_date)
    print("5min rows:", len(mins))
    print("5min sample:", mins[:3])


if __name__ == "__main__":
    main()
