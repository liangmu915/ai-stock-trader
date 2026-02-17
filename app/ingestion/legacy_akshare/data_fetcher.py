import logging
import time
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple

import akshare as ak

logger = logging.getLogger(__name__)

KlineRecord = Tuple[
    str,
    str,
    str,
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]


def _to_float(value: Any) -> Optional[float]:
    """Convert value to float safely; return None for invalid values."""
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_date_range(start_date: str, end_date: str) -> None:
    """Validate date string format and ordering."""
    try:
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
    except ValueError as exc:
        raise ValueError("start_date and end_date must be in YYYYMMDD format.") from exc

    if start_dt > end_dt:
        raise ValueError("start_date cannot be later than end_date.")


def _extract_symbol(ts_code: str) -> str:
    """Extract 6-digit symbol from ts_code, e.g. 000001 from 000001.SZ."""
    symbol = ts_code.split(".")[0].strip()
    if not (len(symbol) == 6 and symbol.isdigit()):
        raise ValueError("ts_code must contain a 6-digit code, e.g. 000001.SZ or 000001.")
    return symbol


def fetch_minute_kline(
    ts_code: str,
    start_date: str,
    end_date: str,
    timeframe: str,
) -> List[KlineRecord]:
    """
    Fetch A-share minute K-line data from AkShare.

    Args:
        ts_code: Stock code, e.g. "000001.SZ" or "000001".
        start_date: Start date in YYYYMMDD format.
        end_date: End date in YYYYMMDD format.
        timeframe: "1min" or "5min".

    Returns:
        [(ts_code, trade_time, timeframe, open, high, low, close, volume), ...]
    """
    period_map = {"1min": "1", "5min": "5"}
    if timeframe not in period_map:
        raise ValueError("timeframe must be '1min' or '5min'.")

    _validate_date_range(start_date, end_date)
    symbol = _extract_symbol(ts_code)

    start_time = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]} 09:30:00"
    end_time = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]} 15:00:00"

    logger.info(
        "Fetching minute kline from akshare: ts_code=%s, symbol=%s, start=%s, end=%s, timeframe=%s",
        ts_code,
        symbol,
        start_date,
        end_date,
        timeframe,
    )

    while True:
        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=period_map[timeframe],
                start_date=start_time,
                end_date=end_time,
                adjust="",
            )
            break
        except Exception as exc:
            logger.warning("Minute kline fetch failed, retrying in 1 second: %s", exc)
            time.sleep(1)

    if df is None or df.empty:
        today = datetime.now().date()
        if datetime.strptime(end_date, "%Y%m%d").date() < today - timedelta(days=30):
            logger.warning(
                "No minute data returned for ts_code=%s between %s and %s. "
                "AkShare minute endpoint may not keep long history for this range.",
                ts_code,
                start_date,
                end_date,
            )
        else:
            logger.info("No minute data returned for ts_code=%s between %s and %s", ts_code, start_date, end_date)
        return []

    rename_map = {
        "\u65f6\u95f4": "trade_time",
        "\u5f00\u76d8": "open",
        "\u6700\u9ad8": "high",
        "\u6700\u4f4e": "low",
        "\u6536\u76d8": "close",
        "\u6210\u4ea4\u91cf": "volume",
    }
    df = df.rename(columns=rename_map)

    required_cols = ("trade_time", "open", "high", "low", "close", "volume")
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"AkShare response missing required columns: {missing_cols}")

    df["trade_time"] = df["trade_time"].astype(str).str.slice(0, 19)
    df = df.sort_values("trade_time")

    rows: List[KlineRecord] = []
    for row in df.to_dict("records"):
        rows.append(
            (
                ts_code,
                str(row.get("trade_time")),
                timeframe,
                _to_float(row.get("open")),
                _to_float(row.get("high")),
                _to_float(row.get("low")),
                _to_float(row.get("close")),
                _to_float(row.get("volume")),
            )
        )

    logger.info("Fetched %d minute bars for ts_code=%s", len(rows), ts_code)
    return rows


def fetch_daily_kline(
    ts_code: str,
    start_date: str,
    end_date: str,
) -> List[KlineRecord]:
    """
    Fetch A-share daily K-line data from AkShare.

    Args:
        ts_code: Stock code, e.g. "000001.SZ" or "000001".
        start_date: Start date in YYYYMMDD format.
        end_date: End date in YYYYMMDD format.

    Returns:
        [(ts_code, trade_time, "daily", open, high, low, close, volume), ...]
    """
    _validate_date_range(start_date, end_date)
    symbol = _extract_symbol(ts_code)

    logger.info(
        "Fetching daily kline from akshare: ts_code=%s, symbol=%s, start=%s, end=%s",
        ts_code,
        symbol,
        start_date,
        end_date,
    )

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="",
        )
    except Exception:
        logger.exception("Failed to fetch daily kline from akshare for ts_code=%s", ts_code)
        raise

    if df is None or df.empty:
        logger.info("No daily data returned for ts_code=%s between %s and %s", ts_code, start_date, end_date)
        return []

    rename_map = {
        "\u65e5\u671f": "trade_time",
        "\u5f00\u76d8": "open",
        "\u6700\u9ad8": "high",
        "\u6700\u4f4e": "low",
        "\u6536\u76d8": "close",
        "\u6210\u4ea4\u91cf": "volume",
    }
    df = df.rename(columns=rename_map)

    required_cols = ("trade_time", "open", "high", "low", "close", "volume")
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"AkShare response missing required columns: {missing_cols}")

    df["trade_time"] = df["trade_time"].astype(str).str.slice(0, 10)
    df = df.sort_values("trade_time")

    rows: List[KlineRecord] = []
    for row in df.to_dict("records"):
        rows.append(
            (
                ts_code,
                str(row.get("trade_time")),
                "daily",
                _to_float(row.get("open")),
                _to_float(row.get("high")),
                _to_float(row.get("low")),
                _to_float(row.get("close")),
                _to_float(row.get("volume")),
            )
        )

    logger.info("Fetched %d daily bars for ts_code=%s", len(rows), ts_code)
    return rows
