# ml_service/losses.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


LossType = Literal["bce", "weighted_bce", "focal", "mixed"]


@dataclass
class FocalParams:
    alpha: float = 0.8   # weight for positive class
    gamma: float = 2.0   # focusing parameter


def bce_with_logits_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Standard BCE-with-logits.
    logits: (B,)
    targets: (B,) with values in {0, 1}
    """
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def weighted_bce_with_logits_loss(
    logits: Tensor,
    targets: Tensor,
    pos_weight: float,
) -> Tensor:
    """
    Weighted BCE-with-logits using PyTorch's pos_weight.
    pos_weight > 1.0 increases penalty on positive (fraud) errors.
    """
    targets = targets.float()
    pos_w = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_w,
    )


def focal_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.8,
    gamma: float = 2.0,
) -> Tensor:
    """
    Binary focal loss on logits.

    Implementation:
      CE = BCE-with-logits(logits, targets) [per-sample]
      p_t = p if y=1 else (1-p)
      loss = alpha_t * (1 - p_t)^gamma * CE
    """
    targets = targets.float()

    # BCE per sample (no reduction)
    ce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    # Probabilities
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

    # Alpha balancing
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

    # Modulating factor
    modulating_factor = (1.0 - p_t) ** gamma

    loss = alpha_t * modulating_factor * ce_loss
    return loss.mean()


def mixed_wbce_focal_loss(
    logits: Tensor,
    targets: Tensor,
    pos_weight: float,
    focal_params: FocalParams,
    lam: float = 0.5,
) -> Tensor:
    """
    Mix of Weighted BCE and Focal Loss:
      L = (1 - λ) * WBCE + λ * Focal
    """
    wbce = weighted_bce_with_logits_loss(logits, targets, pos_weight=pos_weight)
    fl = focal_loss_with_logits(
        logits,
        targets,
        alpha=focal_params.alpha,
        gamma=focal_params.gamma,
    )
    return (1.0 - lam) * wbce + lam * fl


def compute_loss(
    logits: Tensor,
    targets: Tensor,
    loss_type: LossType,
    pos_weight: float = 1.0,
    focal_params: Optional[FocalParams] = None,
    lam: float = 0.5,
) -> Tensor:
    """
    Unified entry point for training scripts.

    loss_type:
      - "bce"          → plain BCE-with-logits
      - "weighted_bce" → cost-sensitive BCE
      - "focal"        → focal loss only
      - "mixed"        → (1-λ)*WBCE + λ*Focal
    """
    if focal_params is None:
        focal_params = FocalParams()

    if loss_type == "bce":
        return bce_with_logits_loss(logits, targets)
    elif loss_type == "weighted_bce":
        return weighted_bce_with_logits_loss(logits, targets, pos_weight=pos_weight)
    elif loss_type == "focal":
        return focal_loss_with_logits(
            logits,
            targets,
            alpha=focal_params.alpha,
            gamma=focal_params.gamma,
        )
    elif loss_type == "mixed":
        return mixed_wbce_focal_loss(
            logits,
            targets,
            pos_weight=pos_weight,
            focal_params=focal_params,
            lam=lam,
        )
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
