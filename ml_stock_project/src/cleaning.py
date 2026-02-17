from __future__ import annotations

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply required cleaning steps without introducing future leakage.
    """
    out = df.copy()

    # Ensure correct dtypes and ordering
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "stock"])
    out["stock"] = out["stock"].astype(str)

    # 1) Remove duplicates
    out = out.drop_duplicates(subset=["stock", "date"], keep="last")

    # 2) Filter invalid rows
    out = out[(out["close"] > 0) & (out["high"] >= out["low"]) & (out["volume"] > 0)]

    # Sort for grouped rolling/shift operations
    out = out.sort_values(["stock", "date"]).reset_index(drop=True)

    # 3) Remove first 60 trading days of each stock
    out = out[out.groupby("stock").cumcount() >= 60].copy()

    # Previous close used by backtest limit-up filter
    out["prev_close"] = out.groupby("stock", group_keys=False)["close"].shift(1)

    # 4) 1-day return with clipping (winsorize)
    out["ret_1"] = out.groupby("stock", group_keys=False)["close"].pct_change(1)
    out["ret_1"] = out["ret_1"].clip(-0.2, 0.2)

    return out.reset_index(drop=True)
