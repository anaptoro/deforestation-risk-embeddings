from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


FEATURE_COLS = [f"A{i:02d}" for i in range(64)] + ["dist_to_nonforest_m", "dist_to_road_m"]


@dataclass
class TrainResult:
    model: Pipeline
    w_raw: np.ndarray
    b_raw: float


def load_xy(df: pd.DataFrame, feature_cols=FEATURE_COLS):
    X = df[feature_cols].to_numpy(np.float32)
    y = df["label"].to_numpy(np.int32)
    return X, y


def fit_logit(Xtr, ytr, C: float = 1.0, max_iter: int = 5000) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=max_iter, C=C)),
        ]
    )
    model.fit(Xtr, ytr)
    return model


def raw_space_weights(model: Pipeline) -> tuple[np.ndarray, float]:
    """Convert (scaled-space) weights to raw feature space for Earth Engine."""
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    w_scaled = clf.coef_.ravel()
    b_scaled = float(clf.intercept_[0])

    w_raw = w_scaled / scaler.scale_
    b_raw = b_scaled - float(np.sum(w_scaled * (scaler.mean_ / scaler.scale_)))
    return w_raw, b_raw


def train_from_csv(
    train_csv: str | Path,
    train_years: list[int],
    test_year: int | None = None,
    C: float = 1.0,
) -> tuple[TrainResult, dict]:
    df = pd.read_csv(train_csv)

    # sanity checks
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature cols in train CSV: {missing[:5]} ... ({len(missing)} missing)")
    if "tYear" not in df.columns or "label" not in df.columns:
        raise ValueError("Train CSV must have columns: tYear, label")

    train_df = df[df["tYear"].isin(train_years)].copy()
    if len(train_df) == 0:
        raise ValueError(f"No rows found for train_years={train_years}. Available years: {sorted(df['tYear'].unique())}")

    Xtr, ytr = load_xy(train_df)
    model = fit_logit(Xtr, ytr, C=C)

    w_raw, b_raw = raw_space_weights(model)
    res = TrainResult(model=model, w_raw=w_raw, b_raw=b_raw)

    info = {"train_n": int(len(ytr)), "train_pos_rate": float(ytr.mean())}
    if test_year is not None:
        test_df = df[df["tYear"].eq(test_year)].copy()
        info["test_n"] = int(len(test_df))
        info["test_year"] = int(test_year)
    return res, info
