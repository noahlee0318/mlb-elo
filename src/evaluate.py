"""Chunk 5, Deliverable 3: metric helpers for scoring a probability column
against binary outcomes. No RNG, no model fitting.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def evaluate(y_true, y_prob, name, n_bins=10):
    """Metric dict for one probability column.

    accuracy uses a STRICT p > 0.5 threshold: a prediction of exactly 0.5 is
    counted as NOT a home pick (predicts the away team). log loss clips
    probabilities to [1e-6, 1-1e-6]. auc is None when y_prob has zero
    variance (roc_auc_score is undefined there and is not called). n_bins is
    accepted for interface stability but does not affect the returned metrics.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    pc = np.clip(p, EPS, 1 - EPS)

    pred = (p > 0.5).astype(int)
    acc = float((pred == y.astype(int)).mean())
    ll = float(log_loss(y, pc, labels=[0, 1]))
    brier = float(brier_score_loss(y, pc))
    if p.std() == 0 or len(np.unique(y)) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(y, p))
    return {"name": name, "n": int(len(y)), "accuracy": acc, "log_loss": ll,
            "brier": brier, "auc": auc, "mean_pred": float(p.mean()),
            "base_rate": float(y.mean())}


def compare(results):
    """DataFrame of metric dicts, sorted ascending by log loss (best first)."""
    return (pd.DataFrame(results)
            .sort_values("log_loss", kind="stable")
            .reset_index(drop=True))
