from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.modeling.train_logit import FEATURE_COLS, train_from_csv
from src.modeling.metrics import eval_probs, topk_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--unbiased_csv", default=None)
    ap.add_argument("--train_years", required=True, help="Comma list, e.g. 2018,2019")
    ap.add_argument("--test_year", type=int, default=None)
    ap.add_argument("--out_json", default="models/logit_weights.json")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    train_years = [int(x.strip()) for x in args.train_years.split(",") if x.strip()]

    res, info = train_from_csv(args.train_csv, train_years, test_year=args.test_year, C=args.C)
    print("Train info:", info)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"w": [float(x) for x in res.w_raw], "b": float(res.b_raw), "feature_cols": FEATURE_COLS}
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved weights to: {out_path}")

    # optional unbiased evaluation
    if args.unbiased_csv:
        df_u = pd.read_csv(args.unbiased_csv)
        Xu = df_u[FEATURE_COLS].to_numpy(np.float32)
        yu = df_u["label"].to_numpy(np.int32)
        p = res.model.predict_proba(Xu)[:, 1]
        print("Unbiased metrics:", eval_probs(yu, p, name="unbiased"))
        print("TopK:", topk_report(yu, p))


if __name__ == "__main__":
    main()
