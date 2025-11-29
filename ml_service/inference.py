# ml_service/inference.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

from model import FraudMLP, MLPConfig

ARTIFACTS_DIR = Path("artifacts")
FINAL_MODEL_PATH = ARTIFACTS_DIR / "final_model.pt"


@dataclass
class LoadedModelPackage:
    model: FraudMLP
    threshold: float
    feature_names: List[str]


class FraudModelService:
    """
    Wrapper around the trained FraudMLP model for inference.

    - Loads the trained weights, config, threshold, and feature names.
    - Provides prediction methods that accept numpy arrays or dicts of features.
    """

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.package = self._load_model_package()
        self.model = self.package.model.to(self.device)
        self.model.eval()
        self.threshold = self.package.threshold
        self.feature_names = self.package.feature_names

        print(f"[SERVICE] Loaded model on device={self.device}")
        print(f"[SERVICE] Threshold = {self.threshold:.4f}")
        print(f"[SERVICE] Num features = {len(self.feature_names)}")

    def _load_model_package(self) -> LoadedModelPackage:
        if not FINAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Final model not found at {FINAL_MODEL_PATH}")

        print(f"[SERVICE] Loading model package from {FINAL_MODEL_PATH}...")
        raw = torch.load(FINAL_MODEL_PATH, map_location="cpu")

        mlp_config_dict = raw["mlp_config"]
        mlp_config = MLPConfig(**mlp_config_dict)

        model = FraudMLP(mlp_config)
        model.load_state_dict(raw["state_dict"])

        threshold = float(raw["threshold"])
        feature_names = list(raw["feature_names"])

        return LoadedModelPackage(
            model=model,
            threshold=threshold,
            feature_names=feature_names,
        )

    # ---------- internal helpers ----------

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float32)
        else:
            X = X.astype(np.float32, copy=False)

        return torch.from_numpy(X)

    # ---------- array-based API ----------

    def predict_proba_from_array(self, X: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray of shape (n_samples, n_features) in the same feature order
           as self.feature_names.

        Returns:
            probs_fraud: np.ndarray of shape (n_samples,)
                         probabilities P(y=1 | x) for each sample.
        """
        self.model.eval()
        with torch.no_grad():
            xb = self._to_tensor(X).to(self.device)
            logits = self.model(xb)
            probs = torch.sigmoid(logits)
            return probs.cpu().numpy()

    def predict_label_from_array(self, X: np.ndarray) -> np.ndarray:
        """
        Uses the stored threshold to convert probabilities to binary labels.

        Returns:
            labels: np.ndarray of shape (n_samples,), with 0/1 predictions.
        """
        probs = self.predict_proba_from_array(X)
        labels = (probs >= self.threshold).astype(int)
        return labels

    # ---------- dict-based API (for REST) ----------

    def features_dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert a single transaction feature dict into a 2D array (1, n_features)
        in the correct column order.

        The keys of `features` must include ALL names in self.feature_names.
        """
        row = []
        for name in self.feature_names:
            if name not in features:
                raise KeyError(f"Missing feature '{name}' in input payload.")
            row.append(float(features[name]))
        return np.array([row], dtype=np.float32)

    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict for a single transaction represented as {feature_name: value}.

        Returns a dict with:
          - prob_not_fraud: P(y=0 | x)
          - prob_fraud:     P(y=1 | x)
          - is_fraud:       0/1 decision based on threshold
          - threshold:      decision threshold used
        """
        X = self.features_dict_to_array(features)
        probs_fraud = self.predict_proba_from_array(X)  # P(y=1 | x)
        labels = (probs_fraud >= self.threshold).astype(int)

        prob_fraud = float(probs_fraud[0])
        prob_not_fraud = float(1.0 - prob_fraud)
        label = int(labels[0])

        return {
            "prob_not_fraud": prob_not_fraud,
            "prob_fraud": prob_fraud,
            "is_fraud": label,
            "threshold": self.threshold,
        }