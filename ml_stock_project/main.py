from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from src.backtest import backtest_topk
from src.cleaning import clean_data
from src.data_io import load_data, load_stock_info_snapshot
from src.evaluation import evaluate_predictions
from src.features import add_labels, build_features
from src.metrics import calc_auc
from src.plots import plot_cumulative_return, plot_feature_importance, plot_score_distribution
from src.split import split_by_date
from src.train import train_model


def main() -> None:
    """
    Run the full daily ML + backtest pipeline.

    Pipeline order:
    load -> clean -> feature -> label -> split -> train -> backtest -> diagnostics.
    """
    print("Loading data...")
    df = load_data(
        data_source=config.DATA_SOURCE,
        data_path=Path(config.DATA_PATH),
        sqlite_table=config.SQLITE_TABLE,
        sample_stocks=config.SAMPLE_STOCKS,
    )
    info = load_stock_info_snapshot(
        data_source=config.DATA_SOURCE,
        data_path=Path(config.DATA_PATH),
        snapshot_at=config.INDUSTRY_SNAPSHOT_AT,
    )
    if not info.empty:
        df = df.merge(info, on="stock", how="left")
        snap_text = config.INDUSTRY_SNAPSHOT_AT or "latest available"
        print(f"Using static industry snapshot: {snap_text} (not as-of history).")
    print(f"Raw rows: {len(df):,}, stocks: {df['stock'].nunique():,}")

    print("Cleaning data...")
    df = clean_data(df)
    print(f"After clean rows: {len(df):,}")

    print("Building features...")
    df, feature_cols = build_features(df)

    print("Generating labels...")
    model_df = add_labels(df, feature_cols)
    print(f"Model rows: {len(model_df):,}")

    print("Splitting by date...")
    train_df, valid_df, test_df = split_by_date(model_df)
    print(f"Train rows: {len(train_df):,} | Valid rows: {len(valid_df):,} | Test rows: {len(test_df):,}")

    if train_df.empty or valid_df.empty or test_df.empty:
        raise RuntimeError("One of train/valid/test splits is empty. Check date coverage in your data.")

    print("Training LightGBM...")
    model, importance_df, train_info = train_model(train_df, valid_df, feature_cols)
    print(f"Validation AUC: {train_info['valid_auc']:.6f}")

    # Test AUC (classification quality, separate from strategy metrics)
    test_pred = model.predict_proba(test_df[feature_cols])[:, 1]
    test_auc = calc_auc(test_df["label"], pd.Series(test_pred))
    print(f"Test AUC: {test_auc:.6f}")

    print("Running TopK backtest...")
    bt = backtest_topk(
        test_df=test_df,
        model=model,
        feature_cols=feature_cols,
        k=config.TOPK,
        hold_days=config.HOLD_DAYS,
        max_per_sector=config.MAX_PER_SECTOR,
    )

    print("===== Backtest Summary =====")
    print(f"Annualized Return: {bt['annualized_return']:.4%}")
    print(f"Max Drawdown:      {bt['max_drawdown']:.4%}")
    print(f"Sharpe Ratio:      {bt['sharpe']:.4f}")

    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"lgbm_daily_model_{ts}.txt"
    feature_path = model_dir / f"feature_columns_{ts}.csv"
    model.booster_.save_model(str(model_path))
    pd.DataFrame({"feature": feature_cols}).to_csv(feature_path, index=False)

    # Keep latest snapshot for downstream scripts that use fixed file names.
    latest_model_path = model_dir / "lgbm_daily_model_latest.txt"
    latest_feature_path = model_dir / "feature_columns_latest.csv"
    model.booster_.save_model(str(latest_model_path))
    pd.DataFrame({"feature": feature_cols}).to_csv(latest_feature_path, index=False)

    # Save artifacts
    importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
    bt["daily_returns"].to_csv(out_dir / "daily_returns.csv", header=["return"])
    bt["cumulative_returns"].to_csv(out_dir / "cumulative_returns.csv", header=["cum_return"])

    plot_cumulative_return(bt["cumulative_returns"], out_dir / "cumulative_return.png")
    plot_feature_importance(importance_df, out_dir / "feature_importance_top30.png", top_n=30)
    plot_score_distribution(pd.Series(test_pred), out_dir / "score_distribution.png")

    print("Running diagnostics: quintile/yearly/concentration/topk-hit...")
    eval_res = evaluate_predictions(
        test_df=test_df,
        y_score=test_pred,
        topk=config.TOPK,
        output_dir=out_dir,
        db_path=Path(config.DATA_PATH) if str(config.DATA_SOURCE).lower() == "sqlite" else None,
        strategy_daily_returns=bt["daily_returns"],
        max_per_sector=config.MAX_PER_SECTOR,
    )

    print("===== Diagnostic Summary =====")
    print(f"Top{config.TOPK} mean hit rate: {eval_res['topk_hit_rate_mean']:.4%}")
    if len(eval_res["yearly_returns"]) > 0:
        print("Yearly returns:")
        for y, r in eval_res["yearly_returns"].items():
            print(f"  {y}: {r:.2%}")
    conc = eval_res["concentration"]
    if "industry_hhi_mean" in conc:
        print(f"Industry concentration (HHI mean): {conc['industry_hhi_mean']:.4f}")
    else:
        print(f"Concentration note: {conc.get('note', 'N/A')}")

    print(f"Saved model to: {model_path.resolve()}")
    print(f"Saved latest model to: {latest_model_path.resolve()}")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
