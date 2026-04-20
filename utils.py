"""
utils/utils.py  —  INF8225 Équipe 5
Utilitaires d'entraînement (loss, métriques, helpers).
"""

import torch
import torch.nn.functional as F
from thop import profile, clever_format


# ── Loss ──────────────────────────────────────────────────────────────────────

def structure_loss(pred_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    BCE pondérée + IoU pondéré (structure loss de PraNet).
    Les pixels proches des bords du masque reçoivent un poids plus élevé.

    pred_logits : (B, 1, H, W) logits bruts
    mask        : (B, 1, H, W) masque binaire float [0,1]
    """
    weight_map = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    bce = F.binary_cross_entropy_with_logits(pred_logits, mask, reduction='none')
    w_bce = (weight_map * bce).sum(dim=(2, 3)) / weight_map.sum(dim=(2, 3))

    pred_prob = torch.sigmoid(pred_logits)
    inter = ((pred_prob * mask) * weight_map).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weight_map).sum(dim=(2, 3))
    w_iou = 1 - (inter + 1) / (union - inter + 1)

    return (w_bce + w_iou).mean()


# ── Métriques ─────────────────────────────────────────────────────────────────

def dice_score(pred_binary: torch.Tensor, gt_binary: torch.Tensor,
               smooth: float = 1e-6) -> float:
    """mDice sur tenseurs binaires (valeurs 0 ou 1)."""
    p = pred_binary.contiguous().view(-1).float()
    g = gt_binary.contiguous().view(-1).float().to(p.device)
    return ((2 * (p * g).sum() + smooth) / (p.sum() + g.sum() + smooth)).item()


def iou_score(pred_binary: torch.Tensor, gt_binary: torch.Tensor,
              smooth: float = 1e-6) -> float:
    """mIoU sur tenseurs binaires."""
    p = pred_binary.contiguous().view(-1).float()
    g = gt_binary.contiguous().view(-1).float().to(p.device)
    inter = (p * g).sum()
    return ((inter + smooth) / (p.sum() + g.sum() - inter + smooth)).item()


# ── Helpers d'entraînement ────────────────────────────────────────────────────

def clip_gradient(optimizer, max_norm: float = 0.5):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-max_norm, max_norm)


class RunningAverage:
    """Moyenne courante simple pour logger la loss."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum, self._count = 0.0, 0

    def update(self, val: float, n: int = 1):
        self._sum += val * n
        self._count += n

    @property
    def avg(self) -> float:
        return self._sum / max(self._count, 1)


def count_params_flops(model: torch.nn.Module, img_size: int = 352) -> tuple:
    """Retourne (FLOPs en G, params en M) via thop."""
    dummy = torch.randn(1, 3, img_size, img_size).cuda()
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    return flops / 1e9, params / 1e6
