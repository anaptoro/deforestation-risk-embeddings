from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.train_logit import FEATURE_COLS, train_from_csv


def make_df(n=200, years=(2018, 2019, 2020), pos_rate=0.3, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_COLS))).astype(np.float32)
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df["tYear"] = rng.choice(list(years), size=n)
    df["label"] = (rng.random(n) < pos_rate).astype(int)
    return df


def test_train_from_csv_smoke(tmp_path: Path):
    df = make_df(n=300, years=(2018, 2019, 2020), pos_rate=0.35)
    train_csv = tmp_path / "train.csv"
    df.to_csv(train_csv, index=False)

    res, info = train_from_csv(
        train_csv=train_csv,
        train_years=[2018, 2019],
        test_year=2020,
        C=1.0,
    )


    assert res.w_raw.shape == (66,)
    assert isinstance(res.b_raw, float)


    assert info["train_n"] > 0
    assert 0.0 <= info["train_pos_rate"] <= 1.0
    assert info["test_year"] == 2020
    assert "test_n" in info


def test_train_from_csv_raises_on_missing_features(tmp_path: Path):
    df = make_df(n=50, years=(2018,), pos_rate=0.2)

    # drop one required feature
    df = df.drop(columns=["A00"])

    train_csv = tmp_path / "train_missing.csv"
    df.to_csv(train_csv, index=False)

    with pytest.raises(ValueError, match="Missing feature cols"):
        train_from_csv(train_csv=train_csv, train_years=[2018])


def test_train_from_csv_raises_on_empty_train_split(tmp_path: Path):
    df = make_df(n=80, years=(2018, 2019), pos_rate=0.2)
    train_csv = tmp_path / "train.csv"
    df.to_csv(train_csv, index=False)

    with pytest.raises(ValueError, match="No rows found for train_years"):
        train_from_csv(train_csv=train_csv, train_years=[2050])
