from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_stock_info(db_path: Path) -> pd.DataFrame:
    """Load stock_info table if present; otherwise return empty dataframe."""
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stock_info'"
        ).fetchone()
        if not exists:
            return pd.DataFrame()
        return pd.read_sql_query("SELECT * FROM stock_info", conn)


def _compute_quintile_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional quintile test by date.
    Q1 = lowest score, Q5 = highest score.
    """
    tmp = df.copy()
    # Stable cross-sectional rank percentile in each date.
    tmp["pct_rank"] = tmp.groupby("date", group_keys=False)["score"].rank(method="first", pct=True)
    date_size = tmp.groupby("date")["stock"].transform("size")
    # Map percentile to quintiles 1..5; keep NaN for dates with fewer than 5 stocks.
    tmp["quintile"] = np.where(date_size >= 5, np.ceil(tmp["pct_rank"] * 5.0), np.nan)
    tmp["quintile"] = tmp["quintile"].clip(1, 5)
    tmp = tmp.dropna(subset=["quintile"]).copy()
    tmp["quintile"] = tmp["quintile"].astype(int)

    daily_q = (
        tmp.groupby(["date", "quintile"], as_index=False)["future_return_5d"]
        .mean()
        .pivot(index="date", columns="quintile", values="future_return_5d")
        .sort_index()
    )
    daily_q.columns = [f"Q{int(c)}" for c in daily_q.columns]
    return daily_q


def _plot_quintile_cumulative(quintile_daily: pd.DataFrame, output_path: Path) -> None:
    """Plot cumulative curves for Q1..Q5 daily group returns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cum = (1.0 + quintile_daily.fillna(0.0)).cumprod()
    plt.figure(figsize=(11, 5))
    for col in sorted(cum.columns):
        plt.plot(cum.index, cum[col], label=col)
    plt.title("Quintile Cumulative Returns (Future 5D Return)")
    plt.xlabel("Date")
    plt.ylabel("Net Value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_yearly_returns(yearly_ret: pd.Series, output_path: Path) -> None:
    """Plot yearly return bars."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    yearly_ret.plot(kind="bar")
    plt.title("TopK Strategy Yearly Returns")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_predictions(
    test_df: pd.DataFrame,
    y_score: np.ndarray,
    topk: int,
    output_dir: Path,
    db_path: Optional[Path] = None,
    strategy_daily_returns: Optional[pd.Series] = None,
    max_per_sector: Optional[int] = None,
) -> Dict[str, object]:
    """
    Produce diagnostics:
    1) Quintile cumulative returns (Q1..Q5)
    2) Yearly TopK returns (prefer real strategy daily returns)
    3) TopK hit rate
    4) Holding concentration by industry/market/(optional) market-cap bucket
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = test_df.copy()
    eval_df["score"] = y_score
    eval_df["date"] = pd.to_datetime(eval_df["date"])
    eval_df = eval_df.sort_values(["date", "stock"]).reset_index(drop=True)

    # 1) Quintile performance
    quintile_daily = _compute_quintile_daily_returns(eval_df[["date", "stock", "score", "future_return_5d"]])
    quintile_daily.to_csv(output_dir / "quintile_daily_returns.csv")
    _plot_quintile_cumulative(quintile_daily, output_dir / "quintile_cumulative.png")

    # TopK daily selections based on score (for yearly stats, hit rate, concentration)
    def _pick_day(day_df: pd.DataFrame) -> pd.DataFrame:
        ordered = day_df.sort_values("score", ascending=False)
        if (max_per_sector is None) or ("industry" not in ordered.columns):
            return ordered.head(topk)

        picks = []
        sec_cnt: Dict[str, int] = {}
        for _, row in ordered.iterrows():
            sector = str(row.get("industry", "UNKNOWN"))
            c = sec_cnt.get(sector, 0)
            if c >= int(max_per_sector):
                continue
            picks.append(row)
            sec_cnt[sector] = c + 1
            if len(picks) >= topk:
                break
        if not picks:
            return ordered.head(0)
        return pd.DataFrame(picks)

    picked_frames = []
    for _, day_df in eval_df.groupby("date", sort=True):
        one = _pick_day(day_df)
        if not one.empty:
            picked_frames.append(one[["date", "stock", "score", "future_return_5d", "label"]])
    if picked_frames:
        topk_df = pd.concat(picked_frames, ignore_index=True)
    else:
        topk_df = eval_df.head(0)[["date", "stock", "score", "future_return_5d", "label"]]
    topk_df.to_csv(output_dir / "topk_daily_picks.csv", index=False)

    # 2) Yearly returns (use real backtest daily returns if provided)
    if strategy_daily_returns is not None and len(strategy_daily_returns) > 0:
        daily_ret = strategy_daily_returns.copy()
        daily_ret.index = pd.to_datetime(daily_ret.index)
        daily_ret = daily_ret.sort_index()
    else:
        # Fallback: this is only a proxy, because future_5d overlaps across days.
        daily_ret = topk_df.groupby("date", as_index=True)["future_return_5d"].mean().sort_index()
    yearly_ret = (1.0 + daily_ret).groupby(daily_ret.index.year).prod() - 1.0
    yearly_ret.index = yearly_ret.index.astype(str)
    yearly_ret.to_csv(output_dir / "yearly_returns.csv", header=["return"])
    _plot_yearly_returns(yearly_ret, output_dir / "yearly_returns.png")

    # 4) TopK hit rate
    daily_hit = topk_df.groupby("date", as_index=True)["label"].mean().sort_index()
    overall_hit_rate = float(daily_hit.mean()) if len(daily_hit) else np.nan
    daily_hit.to_csv(output_dir / "daily_topk_hit_rate.csv", header=["hit_rate"])

    # 3) Concentration checks (industry/market/market-cap bucket if available)
    stock_info = _load_stock_info(db_path) if db_path is not None else pd.DataFrame()
    concentration_summary: Dict[str, object] = {}
    if not stock_info.empty:
        info = stock_info.copy()
        info["ts_code"] = info["ts_code"].astype(str)
        merged = topk_df.merge(info, left_on="stock", right_on="ts_code", how="left")

        # Industry and market distribution (count share)
        industry_share = merged["industry"].fillna("UNKNOWN").value_counts(normalize=True)
        market_share = merged["market"].fillna("UNKNOWN").value_counts(normalize=True)
        industry_share.to_csv(output_dir / "industry_distribution.csv", header=["weight_share"])
        market_share.to_csv(output_dir / "market_distribution.csv", header=["weight_share"])

        # Optional market cap bucket if table includes market cap like columns.
        cap_col = None
        for candidate in ["market_cap", "total_mv", "total_market_cap", "mkt_cap"]:
            if candidate in merged.columns:
                cap_col = candidate
                break

        if cap_col is not None:
            cap = pd.to_numeric(merged[cap_col], errors="coerce")
            valid = merged.loc[cap.notna()].copy()
            valid["_cap"] = cap[cap.notna()]
            if not valid.empty:
                valid["cap_bucket"] = pd.qcut(valid["_cap"], q=3, labels=["Small", "Mid", "Large"], duplicates="drop")
                cap_share = valid["cap_bucket"].value_counts(normalize=True)
                cap_share.to_csv(output_dir / "market_cap_distribution.csv", header=["weight_share"])
                concentration_summary["market_cap_distribution"] = cap_share.to_dict()
            else:
                concentration_summary["market_cap_distribution"] = "No valid market-cap values."
        else:
            concentration_summary["market_cap_distribution"] = "No market-cap column in stock_info."

        # Daily industry HHI to gauge concentration.
        hhi_daily = (
            merged.assign(industry=merged["industry"].fillna("UNKNOWN"))
            .groupby(["date", "industry"])
            .size()
            .groupby(level=0)
            .apply(lambda s: ((s / s.sum()) ** 2).sum())
        )
        hhi_daily.to_csv(output_dir / "daily_industry_hhi.csv", header=["hhi"])

        concentration_summary["industry_distribution"] = industry_share.to_dict()
        concentration_summary["market_distribution"] = market_share.to_dict()
        concentration_summary["industry_hhi_mean"] = float(hhi_daily.mean()) if len(hhi_daily) else np.nan
    else:
        concentration_summary["note"] = "stock_info table not found; concentration by industry/market/cap skipped."

    return {
        "quintile_daily_returns": quintile_daily,
        "yearly_returns": yearly_ret,
        "daily_hit_rate": daily_hit,
        "topk_hit_rate_mean": overall_hit_rate,
        "concentration": concentration_summary,
    }
