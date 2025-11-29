# ml_service/model.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


ActivationType = Literal["relu", "leaky_relu"]


@dataclass
class MLPConfig:
    """
    Configuration for the fraud-detection MLP, matching the Phase IV design:

    - num_hidden_layers: 1 or 2
    - hidden_size: size of the first hidden layer ∈ {16, 32, 64, 128}
        * if num_hidden_layers == 2, the second hidden layer has hidden_size // 2 units
    - activation: "relu" or "leaky_relu(0.01)" as per the report
    - dropout: dropout rate ∈ {0.0, 0.05, 0.10, 0.20}
    - weight_decay: L2 penalty (to be passed to the optimizer, not used inside the net)
    """
    input_dim: int
    num_hidden_layers: int          # 1 or 2
    hidden_size: int                # {16, 32, 64, 128}
    activation: ActivationType = "relu"
    dropout: float = 0.0            # {0.0, 0.05, 0.10, 0.20}
    weight_decay: float = 0.0       # {0, 1e-5, 1e-4, 1e-3}


class FraudMLP(nn.Module):
    """
    Compact, expressive MLP for binary fraud detection.

    Architecture (Phase IV spec):
      - 1 or 2 hidden layers
      - First hidden layer: hidden_size units
      - If 2 layers: second hidden layer: hidden_size // 2 units
      - Activation: ReLU or LeakyReLU(0.01)
      - Dropout after each hidden layer (rate from config)
      - Output: single neuron (logit); apply sigmoid externally for probabilities.
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        # Basic sanity checks to enforce the design
        assert config.num_hidden_layers in {1, 2}, "num_hidden_layers must be 1 or 2"
        assert config.hidden_size in {16, 32, 64, 128}, "hidden_size must be one of {16, 32, 64, 128}"
        assert 0.0 <= config.dropout <= 0.5, "dropout should be between 0.0 and 0.5"

        if config.activation == "relu":
            act_cls = nn.ReLU
        elif config.activation == "leaky_relu":
            act_cls = lambda: nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")

        layers = []
        in_dim = config.input_dim

        # First hidden layer
        layers.append(nn.Linear(in_dim, config.hidden_size))
        layers.append(act_cls())
        if config.dropout > 0.0:
            layers.append(nn.Dropout(p=config.dropout))
        in_dim = config.hidden_size

        # Optional second hidden layer (half the size of the first)
        if config.num_hidden_layers == 2:
            second_size = max(1, config.hidden_size // 2)
            layers.append(nn.Linear(in_dim, second_size))
            layers.append(act_cls())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(p=config.dropout))
            in_dim = second_size

        # Output layer: single neuron (logit)
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x: (batch_size, input_dim)
        Returns: logits of shape (batch_size,)
        """
        logits = self.net(x).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: returns fraud probabilities in [0, 1],
        i.e., sigmoid(logits).
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs