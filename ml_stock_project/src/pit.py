from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import lightgbm as lgb

from .features import add_labels
from .train import train_model


@dataclass
class PitRunResult:
    target_date: pd.Timestamp
    topk: pd.DataFrame
    all_scores: pd.DataFrame
    train_rows: int
    valid_rows: int


def _predict_score(model, x: pd.DataFrame):
    """Predict positive class probability for either sklearn-LGBM or raw Booster."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    # lightgbm.Booster
    return model.predict(x)


def _apply_sector_cap(out: pd.DataFrame, topk: int, max_per_sector: int | None) -> pd.DataFrame:
    """Apply TopK with optional per-sector hard cap."""
    if (max_per_sector is not None) and ("industry" in out.columns):
        chosen = []
        sec_cnt: dict[str, int] = {}
        for _, row in out.iterrows():
            sector = str(row.get("industry", "UNKNOWN"))
            c = sec_cnt.get(sector, 0)
            if c >= int(max_per_sector):
                continue
            chosen.append(row)
            sec_cnt[sector] = c + 1
            if len(chosen) >= topk:
                break
        return pd.DataFrame(chosen) if chosen else out.head(0).copy()
    return out.head(topk).copy()


def _tail_valid_split(df: pd.DataFrame, valid_days: int = 252) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time split inside one training sample:
    - train: earlier dates
    - valid: last `valid_days` unique trading dates
    """
    d = df.copy()
    dates = sorted(d["date"].dropna().unique())
    if len(dates) < 2:
        return pd.DataFrame(), pd.DataFrame()

    n_valid = min(valid_days, max(1, len(dates) // 5))
    valid_date_set = set(dates[-n_valid:])
    valid_df = d[d["date"].isin(valid_date_set)].copy()
    train_df = d[~d["date"].isin(valid_date_set)].copy()
    return train_df, valid_df


def _fit_for_target(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_date: pd.Timestamp,
    mode: str,
    static_train_start: str,
    static_train_end: str,
    rolling_years: int,
    valid_days: int,
) -> tuple[object, int, int]:
    """
    Train one model for one target date with either:
    - static range
    - rolling N-year range ending at target_date - 1 day
    """
    t = pd.Timestamp(target_date).normalize()
    if mode == "static":
        start = pd.Timestamp(static_train_start)
        end = min(pd.Timestamp(static_train_end), t - pd.Timedelta(days=1))
    elif mode == "rolling":
        start = t - pd.DateOffset(years=rolling_years)
        end = t - pd.Timedelta(days=1)
    else:
        raise ValueError("mode must be 'static' or 'rolling'")

    train_unlabeled = feature_df[(feature_df["date"] >= start) & (feature_df["date"] <= end)].copy()
    labeled = add_labels(train_unlabeled, feature_cols)
    train_df, valid_df = _tail_valid_split(labeled, valid_days=valid_days)

    if train_df.empty or valid_df.empty:
        raise ValueError(
            f"Not enough train/valid data for target_date={t.date()} in mode={mode}. "
            f"window={start.date()}~{end.date()}"
        )

    model, _imp, _info = train_model(train_df, valid_df, feature_cols)
    return model, len(train_df), len(valid_df)


def predict_topk_for_date(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_date: str,
    topk: int = 10,
    mode: str = "static",
    static_train_start: str = "2008-01-01",
    static_train_end: str = "2024-12-31",
    rolling_years: int = 5,
    valid_days: int = 252,
    max_per_sector: int | None = None,
) -> PitRunResult:
    """
    Point-in-time prediction for one target date.

    Rules:
    - Training always uses rows where date < target_date.
    - Prediction uses the target_date cross-section only.
    """
    t = pd.Timestamp(target_date).normalize()
    model, train_rows, valid_rows = _fit_for_target(
        feature_df=feature_df,
        feature_cols=feature_cols,
        target_date=t,
        mode=mode,
        static_train_start=static_train_start,
        static_train_end=static_train_end,
        rolling_years=rolling_years,
        valid_days=valid_days,
    )

    cross = feature_df[feature_df["date"] == t].copy()
    if cross.empty:
        raise ValueError(f"No rows found for target_date={t.date()}")
    cross = cross.dropna(subset=feature_cols)
    if cross.empty:
        raise ValueError(f"No valid feature rows for target_date={t.date()}")

    score = _predict_score(model, cross[feature_cols])
    base_cols = ["date", "stock", "open", "high", "low", "close", "volume"]
    enrich_cols = [c for c in ["name", "symbol", "market", "industry", "list_date"] if c in cross.columns]
    out = cross[base_cols + enrich_cols].copy()
    out["score"] = score
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    picks = _apply_sector_cap(out, topk=topk, max_per_sector=max_per_sector)

    return PitRunResult(
        target_date=t,
        topk=picks,
        all_scores=out,
        train_rows=train_rows,
        valid_rows=valid_rows,
    )


def predict_topk_for_date_with_saved_model(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_date: str,
    model_path: str,
    topk: int = 10,
    max_per_sector: int | None = None,
) -> PitRunResult:
    """
    Point-in-time prediction using an already-trained LightGBM model file.
    """
    model = lgb.Booster(model_file=model_path)
    t = pd.Timestamp(target_date).normalize()

    cross = feature_df[feature_df["date"] == t].copy()
    if cross.empty:
        raise ValueError(f"No rows found for target_date={t.date()}")
    missing_cols = [c for c in feature_cols if c not in cross.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns for inference: {missing_cols[:8]}")
    cross = cross.dropna(subset=feature_cols)
    if cross.empty:
        raise ValueError(f"No valid feature rows for target_date={t.date()}")

    score = _predict_score(model, cross[feature_cols])
    base_cols = ["date", "stock", "open", "high", "low", "close", "volume"]
    enrich_cols = [c for c in ["name", "symbol", "market", "industry", "list_date"] if c in cross.columns]
    out = cross[base_cols + enrich_cols].copy()
    out["score"] = score
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    picks = _apply_sector_cap(out, topk=topk, max_per_sector=max_per_sector)

    return PitRunResult(
        target_date=t,
        topk=picks,
        all_scores=out,
        train_rows=0,
        valid_rows=0,
    )


def available_target_dates(feature_df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> list[pd.Timestamp]:
    """Return tradable date list in [start_date, end_date]."""
    d = pd.to_datetime(feature_df["date"]).dropna().sort_values().unique()
    dates = pd.to_datetime(d)
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date)]
    return [pd.Timestamp(x).normalize() for x in dates]
