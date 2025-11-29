# ml_service/online_sgd_update.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

from data_pipeline import prepare_train_test_splits
from model import FraudMLP, MLPConfig
from losses import compute_loss, FocalParams

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

BEST_CONFIG_PATH = ARTIFACTS_DIR / "best_config.json"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "final_model.pt"
FINAL_METRICS_PATH = ARTIFACTS_DIR / "final_metrics_test.json"
FEEDBACK_LOG_PATH = ARTIFACTS_DIR / "feedback_log.jsonl"


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype("float32"))
        self.y = torch.from_numpy(y.astype("int64"))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_best_config() -> Dict[str, Any]:
    if not BEST_CONFIG_PATH.exists():
        raise FileNotFoundError(f"best_config.json not found at {BEST_CONFIG_PATH}")
    with open(BEST_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_package() -> Dict[str, Any]:
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"final_model.pt not found at {FINAL_MODEL_PATH}")
    print(f"[ONLINE-SGD] Loading existing model from {FINAL_MODEL_PATH}...")
    pkg = torch.load(FINAL_MODEL_PATH, map_location="cpu")
    return pkg


def load_feedback(feature_names: List[str]) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """
    Read feedback_log.jsonl and convert to X_feedback, y_feedback.

    Each line is a JSON record with:
      - "features": { feature_name: value, ... }
      - "true_label": 0 or 1
    """
    if not FEEDBACK_LOG_PATH.exists():
        print(f"[ONLINE-SGD] No feedback log found at {FEEDBACK_LOG_PATH}.")
        return None, None

    X_rows = []
    y_rows = []

    with FEEDBACK_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print("[ONLINE-SGD] Skipping malformed feedback line.")
                continue

            if "features" not in rec or "true_label" not in rec:
                print("[ONLINE-SGD] Skipping feedback without features/true_label.")
                continue

            feat_dict = rec["features"]
            true_label = rec["true_label"]

            row = []
            missing = False
            for name in feature_names:
                if name not in feat_dict:
                    print(f"[ONLINE-SGD] Feedback missing feature '{name}', skipping record.")
                    missing = True
                    break
                row.append(float(feat_dict[name]))
            if missing:
                continue

            X_rows.append(row)
            y_rows.append(int(true_label))

    if not X_rows:
        print("[ONLINE-SGD] Feedback log present but no valid samples found.")
        return None, None

    X_fb = np.array(X_rows, dtype=np.float32)
    y_fb = np.array(y_rows, dtype=np.int64)

    print(f"[ONLINE-SGD] Loaded {X_fb.shape[0]} feedback samples.")
    return X_fb, y_fb


def online_sgd_update(
    model: FraudMLP,
    X_fb: np.ndarray,
    y_fb: np.ndarray,
    hp: Dict[str, Any],
    device: torch.device,
    lr_factor: float = 0.1,
    max_grad_norm: float = 1.0,
    max_loss: float = 10.0,
) -> FraudMLP:
    """
    Online-style SGD: one update step per feedback sample.

    - lr_factor: new_lr = original_lr * lr_factor
    - max_grad_norm: clip gradient norm per step
    - max_loss: clamp loss to this max value to reduce impact of outliers
    """
    model.train()

    base_lr = hp["learning_rate"]
    lr = base_lr * lr_factor

    print(f"[ONLINE-SGD] Using online learning rate lr={lr:.6f}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=hp["weight_decay"],
    )

    focal_params = FocalParams(alpha=hp["alpha"], gamma=hp["gamma"])

    n_samples = X_fb.shape[0]
    print(f"[ONLINE-SGD] Starting online updates for {n_samples} samples...")

    for idx in range(n_samples):
        x = torch.from_numpy(X_fb[idx:idx+1]).to(device)  # shape (1, D)
        y = torch.from_numpy(y_fb[idx:idx+1]).to(device)  # shape (1,)

        optimizer.zero_grad()
        logits = model(x)
        loss = compute_loss(
            logits,
            y,
            loss_type=hp["loss_type"],
            pos_weight=hp["pos_weight"],
            focal_params=focal_params,
            lam=hp["lam"],
        )

        # Clamp loss to avoid insane outliers
        loss = torch.clamp(loss, max=max_loss)

        loss.backward()

        # Gradient clipping to guard against outlier updates
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        if (idx + 1) % 50 == 0 or idx == n_samples - 1:
            print(
                f"[ONLINE-SGD] Processed {idx+1}/{n_samples} samples "
                f"(last loss={loss.item():.4f})"
            )

    print("[ONLINE-SGD] Finished online SGD updates on feedback.")
    return model


def evaluate(
    model: FraudMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate the (updated) model on the held-out 20% test set.
    """
    ds = NumpyDataset(X_test, y_test)
    loader = DataLoader(ds, batch_size=4096, shuffle=False, drop_last=False)

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1],
        zero_division=0,
        average=None,
    )

    prec_fraud = float(precision[1])
    rec_fraud = float(recall[1])
    f1_fraud = float(f1[1])

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = float("nan")

    cls_report = classification_report(
        labels,
        preds,
        labels=[0, 1],
        target_names=["Not Fraud", "Fraud"],
        zero_division=0,
    )

    print("\n[ONLINE-SGD] Test set evaluation after online update:")
    print(f"        Threshold: {threshold:.4f}")
    print(f"        Fraud Precision: {prec_fraud:.4f}")
    print(f"        Fraud Recall:    {rec_fraud:.4f}")
    print(f"        Fraud F1:        {f1_fraud:.4f}")
    print(f"        ROC-AUC:         {roc_auc:.4f}")
    print(f"        PR-AUC:          {pr_auc:.4f}")
    print("\n[ONLINE-SGD] Confusion matrix (rows=true, cols=pred, order=[0,1]):")
    print(cm)
    print("\n[ONLINE-SGD] Classification report:")
    print(cls_report)

    metrics = {
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
        "fraud_metrics": {
            "precision_fraud": prec_fraud,
            "recall_fraud": rec_fraud,
            "f1_fraud": f1_fraud,
        },
        "macro_metrics": {
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
        },
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "classification_report": cls_report,
        "n_test_samples": int(labels.shape[0]),
    }

    return metrics


def main():
    # 1) Load existing model package
    pkg = load_model_package()
    mlp_config_dict = pkg["mlp_config"]
    feature_names = list(pkg["feature_names"])
    threshold = float(pkg["threshold"])

    mlp_config = MLPConfig(**mlp_config_dict)
    model = FraudMLP(mlp_config)
    model.load_state_dict(pkg["state_dict"])

    # 2) Load test data (for evaluation only)
    splits = prepare_train_test_splits()
    X_test = splits.X_test
    y_test = splits.y_test

    print(f"[ONLINE-SGD] Test shape: X={X_test.shape}, y={y_test.shape}")
    print(f"[ONLINE-SGD] Num features: {len(feature_names)}")

    # 3) Load best hyperparameters (for optimizer / loss settings)
    hp = load_best_config()
    print("[ONLINE-SGD] Loaded best hyperparameters:")
    print(hp)
    print(f"[ONLINE-SGD] Using fixed threshold: {threshold:.4f}")

    # 4) Load feedback samples
    X_fb, y_fb = load_feedback(feature_names)
    if X_fb is None or y_fb is None:
        print("[ONLINE-SGD] No valid feedback samples, aborting online update.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ONLINE-SGD] Using device: {device}")
    model = model.to(device)

    # 5) Online SGD: one step per feedback sample
    model = online_sgd_update(
        model,
        X_fb,
        y_fb,
        hp,
        device,
        lr_factor=0.1,       # 10x smaller LR than offline
        max_grad_norm=1.0,   # gradient clipping
        max_loss=10.0,       # clamp very large losses
    )

    # 6) Evaluate on held-out test set
    metrics = evaluate(model, X_test, y_test, threshold, device)

    # 7) Save updated metrics
    with open(FINAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[ONLINE-SGD] Saved updated test metrics to {FINAL_METRICS_PATH}")

    # 8) Overwrite final model package (same threshold + hyperparams)
    new_pkg = {
        "state_dict": model.state_dict(),
        "mlp_config": mlp_config_dict,
        "threshold": threshold,
        "feature_names": feature_names,
        "best_hyperparams": hp,
    }
    torch.save(new_pkg, FINAL_MODEL_PATH)
    print(f"[ONLINE-SGD] Overwrote final model at {FINAL_MODEL_PATH}")
    print("[ONLINE-SGD] Done.")


if __name__ == "__main__":
    main()