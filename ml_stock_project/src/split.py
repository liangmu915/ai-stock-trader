from __future__ import annotations

import pandas as pd
import config


def split_by_date(df: pd.DataFrame):
    """
    Fixed chronological split from config.py.
    """
    d = pd.to_datetime(df["date"])

    train = df[(d >= config.TRAIN_START) & (d <= config.TRAIN_END)].copy()
    valid = df[(d >= config.VALID_START) & (d <= config.VALID_END)].copy()
    test = df[(d >= config.TEST_START) & (d <= config.TEST_END)].copy()

    return train, valid, test
