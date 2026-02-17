from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.backtest import backtest_topk  # noqa: E402
from src.cleaning import clean_data  # noqa: E402
from src.data_io import load_data, load_stock_info_snapshot  # noqa: E402
from src.evaluation import evaluate_predictions  # noqa: E402
from src.features import add_labels, build_features  # noqa: E402
from src.metrics import calc_auc  # noqa: E402
from src.plots import plot_cumulative_return, plot_feature_importance, plot_score_distribution  # noqa: E402
from src.train import train_model  # noqa: E402


def _split_by_custom_date(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
):
    d = pd.to_datetime(df["date"])
    train = df[(d >= train_start) & (d <= train_end)].copy()
    valid = df[(d >= valid_start) & (d <= valid_end)].copy()
    test = df[(d >= test_start) & (d <= test_end)].copy()
    return train, valid, test


def _load_engineered_df(db_path: str, table_name: str) -> pd.DataFrame:
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date", "stock"]).sort_values(["date", "stock"]).reset_index(drop=True)


def _infer_feature_cols_from_engineered(df: pd.DataFrame) -> list[str]:
    base_cols = ["date", "stock", "open", "high", "low", "close", "volume"]
    info_cols = ["name", "symbol", "market", "industry", "list_date"]
    exclude = set(base_cols + info_cols + ["prev_close", "ret_1", "future_return_5d", "label"])
    return [c for c in df.columns if c not in exclude]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="后台训练任务")
    p.add_argument("--db-path", required=True)
    p.add_argument("--mode", choices=["raw", "engineered"], default="raw")
    p.add_argument("--eng-table", default="engineered_features")
    p.add_argument("--sample-stocks", type=int, default=0)
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--valid-start", required=True)
    p.add_argument("--valid-end", required=True)
    p.add_argument("--test-start", required=True)
    p.add_argument("--test-end", required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--max-per-sector", type=int, default=2)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=64)
    p.add_argument("--max-depth", type=int, default=-1)
    p.add_argument("--min-child-samples", type=int, default=50)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--reg-alpha", type=float, default=1.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    step = 0
    total_steps = 9

    def mark(msg: str) -> None:
        nonlocal step
        step += 1
        elapsed = time.time() - t0
        eta = (elapsed / step) * (total_steps - step) if step > 0 else 0.0
        print(f"[{step}/{total_steps}] {msg} | elapsed={elapsed:.1f}s | eta={eta:.1f}s", flush=True)

    try:
        if args.mode == "raw":
            mark("加载 raw 数据并构建特征")
            df = load_data(
                data_source="sqlite",
                data_path=Path(args.db_path),
                sqlite_table=config.SQLITE_TABLE,
                sample_stocks=None if args.sample_stocks <= 0 else int(args.sample_stocks),
            )
            info = load_stock_info_snapshot(
                data_source="sqlite",
                data_path=Path(args.db_path),
                snapshot_at=config.INDUSTRY_SNAPSHOT_AT,
            )
            if not info.empty:
                df = df.merge(info, on="stock", how="left")
            print(f"raw rows={len(df):,}, stocks={df['stock'].nunique():,}", flush=True)
            df = clean_data(df)
            print(f"after clean rows={len(df):,}", flush=True)
            feat_df, feature_cols = build_features(df)
            print(f"feature rows={len(feat_df):,}, feature cols={len(feature_cols):,}", flush=True)
        else:
            mark("加载 engineered 数据")
            feat_df = _load_engineered_df(args.db_path, args.eng_table)
            feature_cols = _infer_feature_cols_from_engineered(feat_df)

        mark("标签生成")
        model_df = add_labels(feat_df, feature_cols)

        mark("按日期切分")
        train_df, valid_df, test_df = _split_by_custom_date(
            model_df,
            args.train_start,
            args.train_end,
            args.valid_start,
            args.valid_end,
            args.test_start,
            args.test_end,
        )
        print(f"Train={len(train_df):,}, Valid={len(valid_df):,}, Test={len(test_df):,}", flush=True)
        if train_df.empty or valid_df.empty or test_df.empty:
            raise RuntimeError("Train/Valid/Test 有空集，请检查日期范围。")

        mark("训练 LightGBM")
        model_params = dict(
            objective="binary",
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            num_leaves=int(args.num_leaves),
            max_depth=int(args.max_depth),
            min_child_samples=int(args.min_child_samples),
            subsample=float(args.subsample),
            colsample_bytree=float(args.colsample_bytree),
            reg_alpha=float(args.reg_alpha),
            reg_lambda=float(args.reg_lambda),
            random_state=int(config.RANDOM_SEED),
            n_jobs=-1,
        )
        model, importance_df, train_info = train_model(train_df, valid_df, feature_cols, model_params=model_params, log_period=100)
        print(f"Validation AUC={train_info['valid_auc']:.6f}", flush=True)

        mark("测试和回测")
        test_pred = model.predict_proba(test_df[feature_cols])[:, 1]
        test_auc = calc_auc(test_df["label"], pd.Series(test_pred))
        bt = backtest_topk(
            test_df=test_df,
            model=model,
            feature_cols=feature_cols,
            k=int(args.topk),
            hold_days=config.HOLD_DAYS,
            max_per_sector=int(args.max_per_sector),
        )
        print(f"Test AUC={test_auc:.6f}", flush=True)
        print(f"Annualized={bt['annualized_return']:.2%}, MDD={bt['max_drawdown']:.2%}, Sharpe={bt['sharpe']:.4f}", flush=True)

        mark("保存模型与图表")
        out_dir = Path(config.OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(config.MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"lgbm_daily_model_{ts}.txt"
        feat_path = model_dir / f"feature_columns_{ts}.csv"
        model.booster_.save_model(str(model_path))
        pd.DataFrame({"feature": feature_cols}).to_csv(feat_path, index=False)
        model.booster_.save_model(str(model_dir / "lgbm_daily_model_latest.txt"))
        pd.DataFrame({"feature": feature_cols}).to_csv(model_dir / "feature_columns_latest.csv", index=False)

        importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
        bt["daily_returns"].to_csv(out_dir / "daily_returns.csv", header=["return"])
        bt["cumulative_returns"].to_csv(out_dir / "cumulative_returns.csv", header=["cum_return"])
        plot_cumulative_return(bt["cumulative_returns"], out_dir / "cumulative_return.png")
        plot_feature_importance(importance_df, out_dir / "feature_importance_top30.png", top_n=30)
        plot_score_distribution(pd.Series(test_pred), out_dir / "score_distribution.png")

        mark("诊断分析")
        eval_res = evaluate_predictions(
            test_df=test_df,
            y_score=test_pred,
            topk=int(args.topk),
            output_dir=out_dir,
            db_path=Path(args.db_path),
            strategy_daily_returns=bt["daily_returns"],
            max_per_sector=int(args.max_per_sector),
        )
        print(f"Top{args.topk} hit rate={eval_res['topk_hit_rate_mean']:.2%}", flush=True)
        mark("完成")
        print(f"Model saved: {model_path.resolve()}", flush=True)
        print(f"Elapsed: {time.time() - t0:.1f}s", flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
