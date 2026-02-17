from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from src.cleaning import clean_data
from src.data_io import load_data, load_stock_info_snapshot
from src.features import build_features
from src.pit import available_target_dates, predict_topk_for_date, predict_topk_for_date_with_saved_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Point-in-time stock prediction (TopK) for A-share daily model.")
    p.add_argument("--mode", choices=["static", "rolling"], default="static", help="Training mode.")
    p.add_argument(
        "--use-saved-model",
        action="store_true",
        help="Use existing LightGBM model file for inference only (no retraining).",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=str(config.MODEL_DIR / "lgbm_daily_model_latest.txt"),
        help="Path to saved LightGBM model (.txt).",
    )
    p.add_argument(
        "--feature-path",
        type=str,
        default=str(config.MODEL_DIR / "feature_columns_latest.csv"),
        help="Path to saved feature column list csv.",
    )
    p.add_argument("--target-date", type=str, default=None, help="Single target date, e.g. 2025-03-10")
    p.add_argument("--start-date", type=str, default=None, help="Start date for rolling batch mode.")
    p.add_argument("--end-date", type=str, default=None, help="End date for rolling batch mode.")
    p.add_argument("--topk", type=int, default=config.TOPK, help="Top K picks.")
    p.add_argument("--rolling-years", type=int, default=5, help="Rolling training window length.")
    p.add_argument("--max-per-sector", type=int, default=config.MAX_PER_SECTOR, help="Hard cap per sector in TopK.")
    p.add_argument("--static-train-start", type=str, default="2008-01-01")
    p.add_argument("--static-train-end", type=str, default="2024-12-31")
    p.add_argument("--valid-days", type=int, default=252, help="Validation tail length in trading days.")
    p.add_argument("--sample-stocks", type=int, default=config.SAMPLE_STOCKS, help="Optional stock subset for quick tests.")
    p.add_argument("--output-dir", type=str, default=str(config.OUTPUT_DIR / "pit"), help="Output directory.")
    return p.parse_args()


def _normalize_date_str(s: str) -> str:
    t = pd.Timestamp(s)
    return t.strftime("%Y-%m-%d")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing features...")
    raw = load_data(
        data_source=config.DATA_SOURCE,
        data_path=Path(config.DATA_PATH),
        sqlite_table=config.SQLITE_TABLE,
        sample_stocks=args.sample_stocks,
    )
    info = load_stock_info_snapshot(
        data_source=config.DATA_SOURCE,
        data_path=Path(config.DATA_PATH),
        snapshot_at=config.INDUSTRY_SNAPSHOT_AT,
    )
    if not info.empty:
        raw = raw.merge(info, on="stock", how="left")
        snap_text = config.INDUSTRY_SNAPSHOT_AT or "latest available"
        print(f"Using static industry snapshot: {snap_text} (not as-of history).")
    clean = clean_data(raw)
    feat_df, current_feature_cols = build_features(clean)
    feat_df = feat_df.sort_values(["date", "stock"]).reset_index(drop=True)

    if args.use_saved_model:
        feature_file = Path(args.feature_path)
        if not feature_file.exists():
            raise ValueError(f"Feature column file not found: {feature_file}")
        feature_cols = pd.read_csv(feature_file)["feature"].astype(str).tolist()
    else:
        feature_cols = current_feature_cols

    if args.target_date:
        targets = [_normalize_date_str(args.target_date)]
    else:
        if args.mode != "rolling":
            raise ValueError("In static mode, --target-date is required.")
        if not args.start_date or not args.end_date:
            raise ValueError("In rolling mode without --target-date, both --start-date and --end-date are required.")
        targets = [
            d.strftime("%Y-%m-%d")
            for d in available_target_dates(feat_df, args.start_date, args.end_date)
        ]
        if not targets:
            raise ValueError("No target dates found in the specified range.")

    all_topk = []
    for i, td in enumerate(targets, start=1):
        if args.use_saved_model:
            print(f"[{i}/{len(targets)}] Predicting {td} (saved-model) ...")
            res = predict_topk_for_date_with_saved_model(
                feature_df=feat_df,
                feature_cols=feature_cols,
                target_date=td,
                model_path=args.model_path,
                topk=args.topk,
                max_per_sector=args.max_per_sector,
            )
        else:
            print(f"[{i}/{len(targets)}] Predicting {td} ({args.mode}) ...")
            res = predict_topk_for_date(
                feature_df=feat_df,
                feature_cols=feature_cols,
                target_date=td,
                topk=args.topk,
                mode=args.mode,
                static_train_start=args.static_train_start,
                static_train_end=args.static_train_end,
                rolling_years=args.rolling_years,
                valid_days=args.valid_days,
                max_per_sector=args.max_per_sector,
            )

        one_topk = res.topk.copy()
        one_topk["rank"] = range(1, len(one_topk) + 1)
        one_topk["train_rows"] = res.train_rows
        one_topk["valid_rows"] = res.valid_rows
        all_topk.append(one_topk)

        one_topk.to_csv(out_dir / f"topk_{res.target_date.strftime('%Y%m%d')}.csv", index=False, encoding="utf-8-sig")

    final_topk = pd.concat(all_topk, ignore_index=True)
    final_topk.to_csv(out_dir / "topk_all.csv", index=False, encoding="utf-8-sig")
    print(f"Saved PIT recommendations to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
