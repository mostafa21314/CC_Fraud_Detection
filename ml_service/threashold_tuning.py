# ml_service/threshold_tuning.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

from config import OOF_PROBS_PATH, OOF_LABELS_PATH, ARTIFACTS_DIR


THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"


def load_oof_arrays():
    if not OOF_PROBS_PATH.exists() or not OOF_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Missing OOF files. Expected:\n"
            f"  {OOF_PROBS_PATH}\n"
            f"  {OOF_LABELS_PATH}\n"
            f"Run hyperparam_search / recompute_best_oof first."
        )

    probs = np.load(OOF_PROBS_PATH)
    labels = np.load(OOF_LABELS_PATH)

    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Shape mismatch: probs.shape={probs.shape}, labels.shape={labels.shape}"
        )

    return probs, labels


def sweep_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    t_min: float = 0.01,
    t_max: float = 0.99,
    t_step: float = 0.01,
):
    """
    Sweep thresholds in [t_min, t_max] and compute metrics for the fraud class.
    Returns a list of dicts.
    """
    results = []

    thresholds = np.arange(t_min, t_max + 1e-9, t_step)

    # Global ROC-AUC / PR-AUC (threshold-independent) for reporting
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = float("nan")

    print(f"[THRESH] Global ROC_AUC={roc_auc:.4f}, PR_AUC={pr_auc:.4f}")

    for t in thresholds:
        y_pred = (probs >= t).astype(int)

        # precision_recall_fscore_support returns arrays for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            y_pred,
            pos_label=1,
            average=None,
            labels=[0, 1],
            zero_division=0,
        )
        # We requested labels=[0,1], so index 1 is fraud
        prec_fraud = float(precision[1])
        rec_fraud = float(recall[1])
        f1_fraud = float(f1[1])
        sup_fraud = int(support[1])

        results.append(
            {
                "threshold": float(t),
                "precision_fraud": prec_fraud,
                "recall_fraud": rec_fraud,
                "f1_fraud": f1_fraud,
                "support_fraud": sup_fraud,
            }
        )

    return results, roc_auc, pr_auc


def pick_best_threshold(results, min_recall: float | None = None):
    """
    Pick the threshold that maximizes F1 on the fraud class.
    Optionally enforce a minimum recall constraint.
    """
    best = None
    best_score = -1.0

    for r in results:
        f1 = r["f1_fraud"]
        rec = r["recall_fraud"]

        if min_recall is not None and rec < min_recall:
            continue

        if f1 > best_score:
            best_score = f1
            best = r

    if best is None:
        raise RuntimeError(
            f"No threshold satisfied the minimum recall constraint (min_recall={min_recall})."
        )

    return best


def main():
    # 1) Load OOF arrays
    probs, labels = load_oof_arrays()
    print(f"[THRESH] Loaded OOF arrays: probs.shape={probs.shape}, labels.shape={labels.shape}")
    print(f"[THRESH] Positive (fraud) rate in OOF: {labels.mean():.6f}")

    # 2) Sweep thresholds
    results, roc_auc, pr_auc = sweep_thresholds(probs, labels)

    # 3) Pick best threshold by F1_fraud
    best = pick_best_threshold(results, min_recall=None)

    print(
        "[THRESH] Best threshold found:\n"
        f"        t={best['threshold']:.4f}, "
        f"F1_fraud={best['f1_fraud']:.4f}, "
        f"Recall_fraud={best['recall_fraud']:.4f}, "
        f"Precision_fraud={best['precision_fraud']:.4f}"
    )

    # 4) Save to JSON
    payload = {
        "threshold": best["threshold"],
        "metrics_at_threshold": {
            "precision_fraud": best["precision_fraud"],
            "recall_fraud": best["recall_fraud"],
            "f1_fraud": best["f1_fraud"],
            "support_fraud": best["support_fraud"],
        },
        "global_metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        },
    }

    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[THRESH] Saved best threshold and metrics to {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()