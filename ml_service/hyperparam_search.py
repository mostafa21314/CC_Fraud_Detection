# ml_service/hyperparam_search.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from data_pipeline import prepare_train_test_splits
from model import FraudMLP, MLPConfig
from losses import compute_loss, FocalParams
from config import RANDOM_STATE, BEST_CONFIG_PATH, OOF_PROBS_PATH, OOF_LABELS_PATH

# ------------------------
# Helper dataset
# ------------------------

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X is already float32 from data_pipeline, but we cast again to be safe
        self.X = torch.from_numpy(X.astype("float32"))
        self.y = torch.from_numpy(y.astype("int64"))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ------------------------
# Hyperparameter sampling
# ------------------------

def sample_hyperparams(rng: np.random.Generator) -> Dict[str, Any]:
    """
    Sample one random hyperparameter configuration according to the Phase IV design.
    """
    num_hidden_layers = rng.choice([1, 2])
    hidden_size = int(rng.choice([16, 32, 64, 128]))
    activation = rng.choice(["relu", "leaky_relu"])
    dropout = float(rng.choice([0.0, 0.05, 0.10, 0.20]))
    weight_decay = float(rng.choice([0.0, 1e-5, 1e-4, 1e-3]))

    # Optimization
    learning_rate = float(rng.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]))
    batch_size = int(rng.choice([512, 1024, 2048]))
    num_epochs = int(rng.choice([5, 8, 10]))  # you can tune this

    # Loss-related (we'll use "mixed" loss so focal is always included)
    loss_type = "mixed"
    pos_weight = float(rng.choice([2.0, 3.0, 5.0, 8.0, 10.0]))
    alpha = float(rng.choice([0.5, 0.8, 0.9]))
    gamma = float(rng.choice([1.0, 2.0, 3.0]))
    lam = float(rng.choice([0.3, 0.5, 0.7]))

    return {
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "activation": activation,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "loss_type": loss_type,
        "pos_weight": pos_weight,
        "alpha": alpha,
        "gamma": gamma,
        "lam": lam,
    }


# ------------------------
# Training for one fold
# ------------------------

def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hp: Dict[str, Any],
    device: torch.device,
    fold_id: int,
) -> np.ndarray:
    """
    Train on a single fold and return predicted probabilities for the validation indices.
    """

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    print(
        f"    [FOLD {fold_id}] Train size = {X_tr.shape[0]}, "
        f"Val size = {X_val.shape[0]}"
    )

    train_ds = NumpyDataset(X_tr, y_tr)
    val_ds = NumpyDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=hp["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hp["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # Build model config
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

    # Training loop
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
            f"        [FOLD {fold_id}] Epoch {epoch+1}/{hp['num_epochs']} "
            f"- Avg train loss: {avg_loss:.4f}"
        )

    # Validation: collect probabilities
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)  # (len(val_idx),)
    print(f"    [FOLD {fold_id}] Finished validation, collected {all_probs.shape[0]} probs.")
    return all_probs


# ------------------------
# Evaluate one hyperparameter configuration with 5-fold CV
# ------------------------

def evaluate_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hp: Dict[str, Any],
    device: torch.device,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Perform stratified K-fold CV for a given hyperparameter config.
    Returns average metrics across folds and the full OOF probs for this config.
    (We will only keep OOF for the *best* config later.)
    """
    print("[CFG] Evaluating hyperparameters:")
    print(
        f"      layers={hp['num_hidden_layers']}, hidden={hp['hidden_size']}, "
        f"act={hp['activation']}, dropout={hp['dropout']}, wd={hp['weight_decay']}, "
        f"lr={hp['learning_rate']}, batch_size={hp['batch_size']}, "
        f"epochs={hp['num_epochs']}, loss={hp['loss_type']}, "
        f"pos_w={hp['pos_weight']}, alpha={hp['alpha']}, "
        f"gamma={hp['gamma']}, lam={hp['lam']}"
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probs = np.zeros_like(y_train, dtype=np.float32)

    fold_metrics = {
        "f1_fraud": [],
        "recall_fraud": [],
        "roc_auc": [],
        "pr_auc": [],
    }

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  [CFG] Starting fold {fold_idx+1}/{n_splits}...")
        probs_val = train_one_fold(
            X_train, y_train, tr_idx, val_idx, hp, device, fold_id=fold_idx+1,
        )
        oof_probs[val_idx] = probs_val

        y_val = y_train[val_idx]

        # Use a default threshold 0.5 during hyperparameter selection
        y_pred = (probs_val >= 0.5).astype(int)

        f1 = f1_score(y_val, y_pred, pos_label=1)
        recall = recall_score(y_val, y_pred, pos_label=1)
        try:
            roc = roc_auc_score(y_val, probs_val)
        except ValueError:
            roc = np.nan
        try:
            pr = average_precision_score(y_val, probs_val)
        except ValueError:
            pr = np.nan

        fold_metrics["f1_fraud"].append(f1)
        fold_metrics["recall_fraud"].append(recall)
        fold_metrics["roc_auc"].append(roc)
        fold_metrics["pr_auc"].append(pr)

        print(
            f"  [CFG] Fold {fold_idx+1}/{n_splits} done: "
            f"F1_fraud={f1:.4f}, Recall_fraud={recall:.4f}, "
            f"ROC_AUC={roc:.4f}, PR_AUC={pr:.4f}"
        )

    avg_metrics = {k: float(np.nanmean(v)) for k, v in fold_metrics.items()}
    avg_metrics["oof_probs"] = oof_probs  # attach for convenience

    print(
        f"[CFG] Completed 5-fold CV: "
        f"mean F1_fraud={avg_metrics['f1_fraud']:.4f}, "
        f"mean Recall_fraud={avg_metrics['recall_fraud']:.4f}, "
        f"mean ROC_AUC={avg_metrics['roc_auc']:.4f}, "
        f"mean PR_AUC={avg_metrics['pr_auc']:.4f}"
    )

    return avg_metrics


# ------------------------
# Main search routine
# ------------------------

def run_random_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
) -> None:
    """
    Main random search loop:

    - Sample n_iter hyperparameter configs.
    - Evaluate each with stratified 5-fold CV.
    - Select the best by F1 on the fraud class.
    - Re-run CV with the best config to get final OOF probabilities and save them.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    rng = np.random.default_rng(RANDOM_STATE)

    best_hp: Dict[str, Any] = {}
    best_score = -np.inf

    # First pass: search configurations
    print(f"[INFO] Starting random search with n_iter={n_iter}...")
    for i in tqdm(range(n_iter), desc="Random search"):
        print(f"\n[INFO] === Hyperparameter config {i+1}/{n_iter} ===")
        hp = sample_hyperparams(rng)
        metrics = evaluate_hyperparams(X_train, y_train, hp, device)

        score = metrics["f1_fraud"]  # primary selection criterion

        if score > best_score:
            best_score = score
            best_hp = hp
            print(
                f"[INFO] New best config found at iter {i+1} with "
                f"F1_fraud={best_score:.4f}"
            )
        else:
            print(
                f"[INFO] Config {i+1} F1_fraud={score:.4f} "
                f"(current best={best_score:.4f})"
            )

    print(f"\n[INFO] Best F1_fraud={best_score:.4f} with hyperparams:")
    print(best_hp)

    # Second pass: re-run CV with best config to generate OOF probabilities
    print("[INFO] Recomputing OOF probabilities for the best configuration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_probs_final = np.zeros_like(y_train, dtype=np.float32)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  [BEST CFG] Fold {fold_idx+1}/5 for final OOF...")
        probs_val = train_one_fold(
            X_train, y_train, tr_idx, val_idx, best_hp, device, fold_id=fold_idx+1,
        )
        oof_probs_final[val_idx] = probs_val

    # Save best hyperparams and OOF arrays
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(best_hp, f, indent=2)

    np.save(OOF_PROBS_PATH, oof_probs_final)
    np.save(OOF_LABELS_PATH, y_train.astype("int8"))

    print(f"[INFO] Saved best hyperparameters to {BEST_CONFIG_PATH}")
    print(f"[INFO] Saved OOF probabilities to {OOF_PROBS_PATH}")
    print(f"[INFO] Saved OOF labels to {OOF_LABELS_PATH}")


# ------------------------
# Entry point
# ------------------------

def main():
    # 1) Load data and get 80% training split
    splits = prepare_train_test_splits()
    X_train = splits.X_train
    y_train = splits.y_train

    print(f"[INFO] Training set shape: X={X_train.shape}, y={y_train.shape}")

    # 2) Run random search + OOF generation
    run_random_search(X_train, y_train, n_iter=20)  # adjust n_iter as needed


if __name__ == "__main__":
    main()