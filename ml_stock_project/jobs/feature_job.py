from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.cleaning import clean_data  # noqa: E402
from src.data_io import load_data, load_stock_info_snapshot  # noqa: E402
from src.features import build_features  # noqa: E402


BASE_COLS = ["date", "stock", "open", "high", "low", "close", "volume"]
INFO_COLS = ["name", "symbol", "market", "industry", "list_date"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="后台特征工程任务")
    p.add_argument("--raw-db", required=True)
    p.add_argument("--out-db", required=True)
    p.add_argument("--feature-csv", required=True)
    p.add_argument("--eng-table", default="engineered_features")
    p.add_argument("--sample-stocks", type=int, default=0)
    return p.parse_args()


def _read_feature_list(path: str) -> list[str]:
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError("特征文件缺少 feature 列")
    return df["feature"].dropna().astype(str).tolist()


def _validate_feature_requirements(df: pd.DataFrame, required_cols: list[str]) -> tuple[bool, list[str]]:
    miss = [c for c in required_cols if c not in df.columns]
    return len(miss) == 0, miss


def main() -> int:
    args = parse_args()
    t0 = time.time()
    step = 0
    total_steps = 6

    def mark(msg: str) -> None:
        nonlocal step
        step += 1
        elapsed = time.time() - t0
        eta = (elapsed / step) * (total_steps - step) if step > 0 else 0.0
        print(f"[{step}/{total_steps}] {msg} | elapsed={elapsed:.1f}s | eta={eta:.1f}s", flush=True)

    try:
        mark("读取模型特征列表")
        required_features = _read_feature_list(args.feature_csv)
        print(f"required features={len(required_features)}", flush=True)

        mark("加载 raw 数据")
        df = load_data(
            data_source="sqlite",
            data_path=Path(args.raw_db),
            sqlite_table=config.SQLITE_TABLE,
            sample_stocks=None if args.sample_stocks <= 0 else int(args.sample_stocks),
        )
        print(f"raw rows={len(df):,}, stocks={df['stock'].nunique():,}", flush=True)
        info = load_stock_info_snapshot(
            data_source="sqlite",
            data_path=Path(args.raw_db),
            snapshot_at=config.INDUSTRY_SNAPSHOT_AT,
        )
        if not info.empty:
            df = df.merge(info, on="stock", how="left")

        mark("清洗与构建特征")
        df = clean_data(df)
        print(f"after clean rows={len(df):,}", flush=True)
        feat_df, _ = build_features(df)
        print(f"feature rows={len(feat_df):,}", flush=True)

        mark("校验模型所需特征")
        ok, miss = _validate_feature_requirements(feat_df, required_features)
        if not ok:
            raise ValueError(f"特征缺失: {miss[:20]}")

        mark("写入 engineered 数据库")
        keep = [c for c in (BASE_COLS + INFO_COLS + ["prev_close", "ret_1"] + required_features) if c in feat_df.columns]
        out_df = feat_df[keep].copy()
        out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        out_db = Path(args.out_db)
        out_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(out_db)) as conn:
            out_df.to_sql(args.eng_table, conn, if_exists="replace", index=False)
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{args.eng_table}_stock_date ON {args.eng_table}(stock, date)")
            meta = pd.DataFrame(
                [
                    {
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source_db": str(Path(args.raw_db).resolve()),
                        "feature_path": str(Path(args.feature_csv).resolve()),
                        "feature_count": len(required_features),
                        "row_count": len(out_df),
                    }
                ]
            )
            meta.to_sql("engineered_meta", conn, if_exists="replace", index=False)

        mark("完成")
        print(f"Output DB: {out_db.resolve()}", flush=True)
        print(f"Rows: {len(out_df):,}", flush=True)
        print(f"Elapsed: {time.time() - t0:.1f}s", flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
