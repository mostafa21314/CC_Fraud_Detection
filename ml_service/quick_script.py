# recompute_best_oof.py

from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch

from data_pipeline import prepare_train_test_splits
from hyperparam_search import (
    NumpyDataset,
    train_one_fold,
    BEST_CONFIG_PATH,
    OOF_PROBS_PATH,
    OOF_LABELS_PATH,
)
from config import RANDOM_STATE

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

def main():
    # 1) Load 80% train split
    splits = prepare_train_test_splits()
    X_train = splits.X_train
    y_train = splits.y_train

    print(f"[REBUILD] Training set shape: X={X_train.shape}, y={y_train.shape}")

    # 2) Hard-code best hyperparams (convert to plain Python types)
    best_hp = {
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "activation": "leaky_relu",
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "learning_rate": 0.01,
        "batch_size": 2048,
        "num_epochs": 8,
        "loss_type": "mixed",
        "pos_weight": 5.0,
        "alpha": 0.8,
        "gamma": 1.0,
        "lam": 0.7,
    }

    print("[REBUILD] Using best hyperparameters:")
    print(best_hp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[REBUILD] Using device: {device}")

    # 3) 5-fold CV with this best config to regenerate OOF probs
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_probs_final = np.zeros_like(y_train, dtype=np.float32)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"[REBUILD] Fold {fold_idx+1}/5...")
        probs_val = train_one_fold(
            X_train, y_train, tr_idx, val_idx, best_hp, device, fold_id=fold_idx+1,
        )
        oof_probs_final[val_idx] = probs_val

    # 4) Save best config + OOF arrays
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(best_hp, f, indent=2)

    np.save(OOF_PROBS_PATH, oof_probs_final)
    np.save(OOF_LABELS_PATH, y_train.astype("int8"))

    print(f"[REBUILD] Saved best hyperparameters to {BEST_CONFIG_PATH}")
    print(f"[REBUILD] Saved OOF probabilities to {OOF_PROBS_PATH}")
    print(f"[REBUILD] Saved OOF labels to {OOF_LABELS_PATH}")

if __name__ == "__main__":
    main()
