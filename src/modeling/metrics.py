from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)


def eval_probs(y, p, name: str = "") -> dict:
    """Return common probabilistic metrics."""
    y = np.asarray(y).astype(float)
    p = np.clip(np.asarray(p).astype(float), 1e-6, 1 - 1e-6)

    out = {
        "name": name,
        "n": int(len(y)),
        "pos_rate": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    return out


def summarize_at_threshold(y, p, thr: float, name: str = "") -> dict:
    """Return confusion matrix + accuracy metrics for a threshold."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p).astype(float), 1e-6, 1 - 1e-6)
    pred = (p >= thr).astype(int)

    cm = confusion_matrix(y, pred)
    acc = accuracy_score(y, pred)
    bacc = balanced_accuracy_score(y, pred)

    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "name": name,
        "thr": float(thr),
        "cm": cm.tolist(),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "precision": float(precision),
        "recall": float(recall),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def topk_report(y, p, top_k_list=(1, 2, 5, 10)) -> list[dict]:
    """Map-style evaluation: capture + precision in top K% alerts."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    total_pos = max(int(y.sum()), 1)
    out = []
    for k in top_k_list:
        thr = float(np.quantile(p, 1 - k / 100.0))
        sel = p >= thr
        capture = float(y[sel].sum()) / total_pos
        prec = float(y[sel].mean()) if sel.sum() else 0.0
        out.append({"top_k_pct": int(k), "thr": thr, "capture": capture, "precision": prec})
    return out
