from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .metrics import calc_performance_metrics


@dataclass
class TranchePosition:
    """
    One tranche opened on a trade day and held for fixed hold_days.
    weights: equal-weight stock allocation inside the tranche.
    age: number of days elapsed since opening (0 on open day).
    """

    weights: Dict[str, float]
    age: int = 0


def backtest_topk(
    test_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    k: int = 10,
    hold_days: int = 5,
    max_per_sector: int | None = None,
):
    """
    Event-driven daily backtest (no leakage, overlap handled by capital tranches).

    Execution logic (trade date = t):
    1) Use signal from t-1 features to predict.
    2) Pick TopK by predicted probability.
    3) Open new tranche at t open (skip limit-up at open).
    4) For all active tranches, compute daily PnL on date t:
       - new tranche (age=0): close/open - 1
       - existing tranche (age>0): close/prev_close - 1
    5) Portfolio daily return = sum(tranche_return * 1/hold_days), cash return is 0.
    """
    # test_df must include at least: date, stock, open, close, prev_close + feature_cols.
    df = test_df.copy()
    df = df.sort_values(["date", "stock"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Needed for trade filter and daily holding return.
    if "prev_close" not in df.columns:
        df["prev_close"] = df.groupby("stock", group_keys=False)["close"].shift(1)

    # Group rows by date for fast daily simulation.
    daily_map: Dict[pd.Timestamp, pd.DataFrame] = {
        d: ddf.set_index("stock", drop=False)
        for d, ddf in df.groupby("date", sort=True)
    }
    dates = sorted(daily_map.keys())
    if len(dates) <= 1:
        returns = pd.Series(dtype=float)
        cumulative = pd.Series(dtype=float)
        perf = calc_performance_metrics(returns, annual_days=252)
        return {
            "daily_returns": returns,
            "cumulative_returns": cumulative,
            "annualized_return": perf["annualized_return"],
            "max_drawdown": perf["max_drawdown"],
            "sharpe": perf["sharpe"],
        }

    active_tranches: List[TranchePosition] = []
    daily_records = []
    tranche_capital = 1.0 / float(hold_days)

    # Start from second date: need t-1 signal date for no-leakage trading.
    for i in range(1, len(dates)):
        signal_date = dates[i - 1]
        trade_date = dates[i]

        signal_day = daily_map[signal_date]
        trade_day = daily_map[trade_date]

        # 1) Predict from t-1 features (no future leakage)
        signal_x = signal_day[feature_cols]
        prob = model.predict_proba(signal_x)[:, 1]
        signal_candidates = signal_day.copy()
        signal_candidates["score"] = prob

        # 2) Select tradable TopK by probability with optional sector cap.
        ordered = signal_candidates.sort_values("score", ascending=False)
        use_sector_cap = (max_per_sector is not None) and ("industry" in ordered.columns)
        sector_count: Dict[str, int] = {}

        # 3) Build tradable list on t (limit-up at open filter + sector cap)
        tradable = []
        for _, sig_row in ordered.iterrows():
            if len(tradable) >= k:
                break
            stk = sig_row["stock"]
            if stk not in trade_day.index:
                continue
            row = trade_day.loc[stk]
            if pd.isna(row["prev_close"]) or row["prev_close"] <= 0:
                continue
            if row["open"] >= row["prev_close"] * 1.095:
                continue

            if use_sector_cap:
                sector = str(sig_row.get("industry", "UNKNOWN"))
                c = sector_count.get(sector, 0)
                if c >= int(max_per_sector):
                    continue
                sector_count[sector] = c + 1
            tradable.append(stk)

        if tradable:
            w = 1.0 / float(len(tradable))
            active_tranches.append(TranchePosition(weights={stk: w for stk in tradable}, age=0))

        # 4) Compute today's return for all active tranches
        port_ret = 0.0
        still_active: List[TranchePosition] = []

        for tr in active_tranches:
            tranche_ret = 0.0
            used_weight = 0.0

            for stk, w in tr.weights.items():
                if stk not in trade_day.index:
                    # Missing quote: treat as 0 return for that stock-day.
                    continue
                row = trade_day.loc[stk]

                if tr.age == 0:
                    # New position: open -> close
                    if row["open"] > 0:
                        r = row["close"] / row["open"] - 1.0
                    else:
                        r = 0.0
                else:
                    # Existing position: prev_close -> close
                    if pd.notna(row["prev_close"]) and row["prev_close"] > 0:
                        r = row["close"] / row["prev_close"] - 1.0
                    else:
                        r = 0.0

                tranche_ret += w * float(r)
                used_weight += w

            # If all holdings missing, tranche_ret remains 0
            if used_weight > 0:
                tranche_ret = tranche_ret / used_weight

            port_ret += tranche_capital * tranche_ret

            # Age update and expiration
            tr.age += 1
            if tr.age < hold_days:
                still_active.append(tr)

        active_tranches = still_active
        daily_records.append((trade_date, float(port_ret), len(active_tranches)))

    bt = pd.DataFrame(daily_records, columns=["date", "ret", "active_tranches"]).sort_values("date")
    returns = bt.set_index("date")["ret"]
    cumulative = (1.0 + returns.fillna(0.0)).cumprod()
    perf = calc_performance_metrics(returns.fillna(0.0), annual_days=252)

    return {
        "daily_returns": returns,
        "cumulative_returns": cumulative,
        "annualized_return": perf["annualized_return"],
        "max_drawdown": perf["max_drawdown"],
        "sharpe": perf["sharpe"],
    }
