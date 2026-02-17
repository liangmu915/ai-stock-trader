from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI implemented with EWM smoothing."""
    diff = series.diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build leakage-safe stock-level features, clip extremes, and create
    cross-sectional percentile rank features by date.
    """
    # Sort by stock/date first so all rolling and shift operations are leakage-safe.
    out = df.copy().sort_values(["stock", "date"]).reset_index(drop=True)

    g = out.groupby("stock", group_keys=False)
    if "industry" in out.columns:
        out["industry"] = out["industry"].fillna("UNKNOWN").astype(str)

    # Momentum
    out["ret_5"] = g["close"].pct_change(5)
    out["ret_10"] = g["close"].pct_change(10)
    out["ret_20"] = g["close"].pct_change(20)
    out["ret_60"] = g["close"].pct_change(60)

    # Moving-average deviation
    for n in [5, 10, 20, 60]:
        ma = g["close"].transform(lambda s: s.rolling(n, min_periods=n).mean())
        out[f"dev_ma_{n}"] = out["close"] / ma

    # Volatility (std of daily returns)
    daily_ret = g["close"].pct_change(1)
    out["std_5"] = daily_ret.groupby(out["stock"]).transform(lambda s: s.rolling(5, min_periods=5).std())
    out["std_20"] = daily_ret.groupby(out["stock"]).transform(lambda s: s.rolling(20, min_periods=20).std())

    # Volume features
    vol_ma_5 = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    vol_ma_20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    out["vol_ratio_5"] = out["volume"] / vol_ma_5
    out["vol_ratio_20"] = out["volume"] / vol_ma_20

    # RSI(14)
    out["rsi_14"] = g["close"].transform(lambda s: _rsi(s, 14))

    # MACD(12,26,9)
    out["ema_12"] = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    out["ema_26"] = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out.groupby("stock", group_keys=False)["macd"].transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # ATR(14)
    prev_close = g["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.groupby(out["stock"]).transform(lambda s: s.rolling(14, min_periods=14).mean())

    # Sector strength (from equal-weight industry return; no external index required)
    if "industry" in out.columns:
        out["sector_eq_ret"] = out.groupby(["date", "industry"], group_keys=False)["ret_1"].transform("mean")
        market_eq_ret = out.groupby("date", group_keys=False)["ret_1"].transform("mean")
        out["sector_rel_strength"] = out["sector_eq_ret"] - market_eq_ret
    else:
        out["sector_eq_ret"] = np.nan
        out["sector_rel_strength"] = np.nan

    raw_feature_cols = [
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_20",
        "ret_60",
        "dev_ma_5",
        "dev_ma_10",
        "dev_ma_20",
        "dev_ma_60",
        "std_5",
        "std_20",
        "vol_ratio_5",
        "vol_ratio_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "sector_eq_ret",
        "sector_rel_strength",
    ]

    # Conservative clipping ranges
    clip_map = {
        "ret_1": (-0.2, 0.2),
        "ret_5": (-0.5, 0.5),
        "ret_10": (-0.8, 0.8),
        "ret_20": (-1.2, 1.2),
        "ret_60": (-2.0, 2.0),
        "dev_ma_5": (0.5, 1.5),
        "dev_ma_10": (0.5, 1.5),
        "dev_ma_20": (0.5, 1.5),
        "dev_ma_60": (0.4, 1.8),
        "std_5": (0.0, 0.25),
        "std_20": (0.0, 0.25),
        "vol_ratio_5": (0.0, 10.0),
        "vol_ratio_20": (0.0, 10.0),
        "rsi_14": (0.0, 100.0),
        "macd": (-2.0, 2.0),
        "macd_signal": (-2.0, 2.0),
        "macd_hist": (-2.0, 2.0),
        "atr_14": (0.0, np.inf),
        "sector_eq_ret": (-0.2, 0.2),
        "sector_rel_strength": (-0.2, 0.2),
    }
    for col in raw_feature_cols:
        low, high = clip_map[col]
        out[col] = out[col].clip(lower=low, upper=high)

    # Cross-sectional rank features by date. The model uses these ranked features.
    rank_feature_cols: List[str] = []
    for col in raw_feature_cols:
        rank_col = f"rk_{col}"
        out[rank_col] = out.groupby("date", group_keys=False)[col].rank(pct=True)
        out[rank_col] = out[rank_col].astype("float32")
        rank_feature_cols.append(rank_col)

    # Explicit sector strength rank feature requested by strategy.
    if "industry" in out.columns:
        out["sector_strength_rank"] = out.groupby("date", group_keys=False)["sector_rel_strength"].rank(pct=True)
    else:
        out["sector_strength_rank"] = np.nan
    out["sector_strength_rank"] = out["sector_strength_rank"].fillna(0.5).astype("float32")

    # Low-memory industry encoding (replace one-hot on full universe).
    if "industry" in out.columns:
        ind_cat = out["industry"].fillna("UNKNOWN").astype("category")
        out["industry_code"] = ind_cat.cat.codes.astype("int16")
    else:
        out["industry_code"] = -1
    final_feature_cols = rank_feature_cols + ["sector_strength_rank", "industry_code"]
    return out, final_feature_cols


def add_labels(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Add 5-day future return and binary label, then drop rows with NaNs from
    rolling/shift operations.
    """
    out = df.copy().sort_values(["stock", "date"]).reset_index(drop=True)
    # Forward 5-day label target, clipped but not removed (except NaN boundary rows).
    out["future_return_5d"] = out.groupby("stock", group_keys=False)["close"].shift(-5) / out["close"] - 1.0
    out["future_return_5d"] = out["future_return_5d"].clip(-0.3, 0.3)
    out["label"] = (out["future_return_5d"] > 0).astype("int8")

    # Drop rows not usable for training/backtest
    required = feature_cols + ["future_return_5d", "label", "prev_close", "open", "close"]
    out = out.dropna(subset=required)
    return out.reset_index(drop=True)
