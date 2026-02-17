from typing import List

import pandas as pd


def parse_kline_rows(ts_code: str, frequency: str, klines: List[str]) -> pd.DataFrame:
    """
    Parse Eastmoney kline rows into normalized schema:
    ts_code, datetime, open, high, low, close, volume, amount,
    amplitude, pct_change, change, turnover, frequency
    """
    parsed = []
    for line in klines:
        parts = line.split(",")
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
        except (ValueError, TypeError):
            continue

        def _safe_float(i: int):
            try:
                return float(parts[i])
            except (ValueError, TypeError, IndexError):
                return None

        amplitude = _safe_float(7)
        pct_change = _safe_float(8)
        change = _safe_float(9)
        turnover = _safe_float(10)

        parsed.append(
            (
                ts_code,
                dt_text,
                open_px,
                high_px,
                low_px,
                close_px,
                volume,
                amount,
                amplitude,
                pct_change,
                change,
                turnover,
                frequency,
            )
        )

    if not parsed:
        return pd.DataFrame()

    df = pd.DataFrame(
        parsed,
        columns=[
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
        ],
    )
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    if df.empty:
        return pd.DataFrame()
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df
