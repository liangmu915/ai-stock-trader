from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier

from .metrics import calc_auc


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict | None = None,
    log_period: int = 100,
):
    """
    Train LGBMClassifier with early stopping on validation AUC.

    Returns:
    - trained model
    - feature importance dataframe
    - training info dict with validation AUC
    """
    x_train = train_df[feature_cols]
    y_train = train_df["label"]
    x_valid = valid_df[feature_cols]
    y_valid = valid_df["label"]

    categorical_cols = [c for c in feature_cols if c == "industry_code"]

    default_params = dict(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    if model_params:
        default_params.update(model_params)

    model = LGBMClassifier(**default_params)

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=categorical_cols if categorical_cols else "auto",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=log_period),
        ],
    )

    valid_pred = model.predict_proba(x_valid)[:, 1]
    valid_auc = calc_auc(y_valid, valid_pred)

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return model, importance_df, {"valid_auc": valid_auc}
