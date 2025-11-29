# ml_service/train_final.py

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

import torch
from torch.utils.data import Dataset, DataLoader

from data_pipeline import prepare_train_test_splits
from model import FraudMLP, MLPConfig
from losses import compute_loss, FocalParams
from config import RANDOM_STATE


# Paths
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_CONFIG_PATH = ARTIFACTS_DIR / "best_config.json"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "final_model.pt"
FINAL_METRICS_PATH = ARTIFACTS_DIR / "final_metrics_test.json"
TEST_PROBS_PATH = ARTIFACTS_DIR / "test_probs.npy"
TEST_LABELS_PATH = ARTIFACTS_DIR / "test_labels.npy"


# ------------------------
# Helper dataset
# ------------------------

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype("float32"))
        self.y = torch.from_numpy(y.astype("int64"))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ------------------------
# Load artifacts
# ------------------------

def load_best_config() -> Dict[str, Any]:
    if not BEST_CONFIG_PATH.exists():
        raise FileNotFoundError(f"best_config.json not found at {BEST_CONFIG_PATH}")
    with open(BEST_CONFIG_PATH, "r", encoding="utf-8") as f:
        hp = json.load(f)
    return hp


def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"threshold.json not found at {THRESHOLD_PATH}")
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj["threshold"])


# ------------------------
# Training on full 80% split
# ------------------------

def train_on_full_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hp: Dict[str, Any],
    device: torch.device,
):
    """
    Train the model on the full 80% training split using the best hyperparameters.
    Returns the trained model instance and its MLPConfig.
    """
    train_ds = NumpyDataset(X_train, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=hp["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    mlp_config = MLPConfig(
        input_dim=X_train.shape[1],
        num_hidden_layers=hp["num_hidden_layers"],
        hidden_size=hp["hidden_size"],
        activation=hp["activation"],
        dropout=hp["dropout"],
        weight_decay=hp["weight_decay"],
    )
    model = FraudMLP(mlp_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )

    focal_params = FocalParams(alpha=hp["alpha"], gamma=hp["gamma"])

    print("[FINAL] Starting training on full 80% training split...")
    model.train()
    for epoch in range(hp["num_epochs"]):
        running_loss = 0.0
        batch_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = compute_loss(
                logits,
                yb,
                loss_type=hp["loss_type"],
                pos_weight=hp["pos_weight"],
                focal_params=focal_params,
                lam=hp["lam"],
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / max(1, batch_count)
        print(
            f"[FINAL] Epoch {epoch+1}/{hp['num_epochs']} - "
            f"Avg train loss: {avg_loss:.4f}"
        )

    print("[FINAL] Finished training on full train split.")
    return model, mlp_config


# ------------------------
# Evaluation on 20% test split
# ------------------------

def evaluate_on_test(
    model: FraudMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate the trained model on the held-out 20% test split.
    Returns a metrics dictionary.
    """
    test_ds = NumpyDataset(X_test, y_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=4096,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Save raw probs/labels for any later analysis
    np.save(TEST_PROBS_PATH, probs)
    np.save(TEST_LABELS_PATH, labels)

    # Apply threshold
    preds = (probs >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    # Precision/recall/F1 per class and macro/weighted if needed
    # average=None → per-class arrays
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1],
        zero_division=0,
        average=None,
    )

    # fraud is class 1
    prec_fraud = float(precision[1])
    rec_fraud = float(recall[1])
    f1_fraud = float(f1[1])

    # Macro statistics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    # Threshold-independent metrics
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = float("nan")

    print("\n[FINAL] Test set evaluation:")
    print(f"        Threshold: {threshold:.4f}")
    print(f"        Fraud Precision: {prec_fraud:.4f}")
    print(f"        Fraud Recall:    {rec_fraud:.4f}")
    print(f"        Fraud F1:        {f1_fraud:.4f}")
    print(f"        ROC-AUC:         {roc_auc:.4f}")
    print(f"        PR-AUC:          {pr_auc:.4f}")
    print("\n[FINAL] Confusion matrix (rows=true, cols=pred, order=[0,1]):")
    print(cm)

    # For completeness, also keep a text classification report
    cls_report = classification_report(
        labels,
        preds,
        labels=[0, 1],
        target_names=["Not Fraud", "Fraud"],
        zero_division=0,
    )
    print("\n[FINAL] Classification report:")
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


# ------------------------
# Main
# ------------------------

def main():
    # 1) Load data
    splits = prepare_train_test_splits()
    X_train = splits.X_train
    y_train = splits.y_train
    X_test = splits.X_test
    y_test = splits.y_test
    feature_names = splits.feature_names

    print(f"[FINAL] Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"[FINAL] Test  shape: X={X_test.shape}, y={y_test.shape}")
    print(f"[FINAL] Number of features: {len(feature_names)}")

    # 2) Load best hyperparameters & threshold
    hp = load_best_config()
    threshold = load_threshold()

    print("[FINAL] Loaded best hyperparameters:")
    print(hp)
    print(f"[FINAL] Loaded decision threshold: {threshold:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FINAL] Using device: {device}")

    # 3) Train on full training split
    model, mlp_config = train_on_full_train(X_train, y_train, hp, device)

    # 4) Evaluate on held-out test split
    metrics = evaluate_on_test(model, X_test, y_test, threshold, device)

    # 5) Save metrics
    with open(FINAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[FINAL] Saved test metrics to {FINAL_METRICS_PATH}")

    # 6) Save model package (state_dict + config + threshold + feature names)
    package = {
        "state_dict": model.state_dict(),
        "mlp_config": asdict(mlp_config),
        "threshold": float(threshold),
        "feature_names": feature_names,
        "best_hyperparams": hp,
    }
    torch.save(package, FINAL_MODEL_PATH)
    print(f"[FINAL] Saved final model package to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()