from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def calc_performance_metrics(daily_returns: pd.Series, annual_days: int = 252) -> dict:
    """
    Compute annualized return, max drawdown and Sharpe from daily return series.
    """
    if daily_returns.empty:
        return {
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }

    daily_returns = daily_returns.fillna(0.0)
    cum = (1.0 + daily_returns).cumprod()
    years = max(len(daily_returns) / annual_days, 1e-9)
    annualized_return = float(cum.iloc[-1] ** (1.0 / years) - 1.0)

    rolling_peak = cum.cummax()
    drawdown = cum / rolling_peak - 1.0
    max_drawdown = float(drawdown.min())

    vol = float(daily_returns.std(ddof=1))
    mean_ret = float(daily_returns.mean())
    sharpe = float((mean_ret / vol) * np.sqrt(annual_days)) if vol > 0 else 0.0

    return {
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def calc_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    y_true = pd.Series(y_true).astype(int)
    y_prob = pd.Series(y_prob).astype(float)
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))
