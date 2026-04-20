"""
mkunet_network.py  —  INF8225 Équipe 5
Implémentation personnelle de MK-UNet.

Référence : Rahman & Marculescu, ICCV 2025 CVAMD.
L'architecture suit la description du papier ; le code et les noms
de variables sont entièrement les nôtres.

Classes exportées
-----------------
MKDC    Multi-Kernel Depthwise Convolution block
MKIR    Multi-Kernel Inverted Residual block
CA      Channel Attention module
SA      Spatial Attention module
GAG     Grouped Attention Gate
MKUNet  Réseau complet encodeur-décodeur
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fonctions utilitaires privées ─────────────────────────────────────────────

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _make_activation(name: str) -> nn.Module:
    lookup = {"relu": nn.ReLU(inplace=True),
              "relu6": nn.ReLU6(inplace=True),
              "gelu": nn.GELU()}
    if name not in lookup:
        raise ValueError(f"Activation inconnue : '{name}'")
    return lookup[name]


def _channel_shuffle(tensor: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Mélange de canaux entre num_groups groupes (ShuffleNet)."""
    B, C, H, W = tensor.shape
    return (tensor.view(B, num_groups, C // num_groups, H, W)
                  .transpose(1, 2).contiguous()
                  .view(B, C, H, W))


def _init_weights(module: nn.Module):
    """Initialisation He pour Conv2d, constantes pour BN."""
    if isinstance(module, nn.Conv2d):
        fan_out = (module.kernel_size[0] * module.kernel_size[1]
                   * module.out_channels // module.groups)
        nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ── MKDC — Multi-Kernel Depthwise Convolution ────────────────────────────────

class MKDC(nn.Module):
    """
    Applique en parallèle une dépthwise conv par noyau dans kernel_sizes.
    Chaque branche : DWConv(k) → BN → Activation.
    Retourne une liste de tenseurs (un par branche).

    Paramètres
    ----------
    num_channels  : C_in = C_out (opération dépthwise)
    kernel_sizes  : liste de noyaux, ex. [1, 3, 5]
    stride        : stride commun à toutes les branches
    activation    : nom de l'activation ('relu6' par défaut)
    """
    def __init__(self, num_channels: int, kernel_sizes: list,
                 stride: int, activation: str = "relu6"):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=k,
                          stride=stride, padding=k // 2,
                          groups=num_channels, bias=False),
                nn.BatchNorm2d(num_channels),
                _make_activation(activation),
            )
            for k in kernel_sizes
        ])
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> list:
        return [branch(x) for branch in self.branches]


# ── MKIR — Multi-Kernel Inverted Residual ────────────────────────────────────

class MKIR(nn.Module):
    """
    Bloc résiduel inversé multi-noyaux (extension de MobileNetV2).

    Schéma :
      x ──► PW-expand ──► MKDC ──► agrégation ──► PW-project ──► (+) ──► sortie
      │                                                                ▲
      └────────────────────────── skip (si stride=1) ─────────────────┘

    Agrégation : somme des sorties MKDC (conserve C_expanded canaux).

    Paramètres
    ----------
    in_channels     : C d'entrée
    out_channels    : C de sortie
    expansion_ratio : facteur t de PW-expand (C_expanded = in_channels * t)
    stride          : 1 → skip connection possible, 2 → pas de skip
    kernel_sizes    : noyaux passés au MKDC
    activation      : activation utilisée partout
    """
    def __init__(self, in_channels: int, out_channels: int,
                 expansion_ratio: int = 2, stride: int = 1,
                 kernel_sizes: list = [1, 3, 5],
                 activation: str = "relu6"):
        super().__init__()
        assert stride in (1, 2)
        self.use_skip = (stride == 1)
        C_exp = in_channels * expansion_ratio

        # Pointwise expand
        self.pw_expand = nn.Sequential(
            nn.Conv2d(in_channels, C_exp, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_exp),
            _make_activation(activation),
        )
        # Multi-kernel depthwise (branches en parallèle, agrégation par somme)
        self.mkdc = MKDC(C_exp, kernel_sizes, stride, activation)

        # Pointwise project
        self.pw_project = nn.Sequential(
            nn.Conv2d(C_exp, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Ajustement dimensionnel du skip si nécessaire
        self.skip_adjust = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if self.use_skip and in_channels != out_channels else None
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.pw_expand(x)
        # Somme des sorties de chaque branche MKDC
        aggregated = sum(self.mkdc(expanded))
        projected = self.pw_project(aggregated)

        if self.use_skip:
            skip = self.skip_adjust(x) if self.skip_adjust else x
            return skip + projected
        return projected


def _build_mkir_stage(in_channels: int, out_channels: int,
                      num_blocks: int, stride: int, **kwargs) -> nn.Sequential:
    """Empile num_blocks blocs MKIR (1er avec stride, suivants avec stride=1)."""
    blocks = [MKIR(in_channels, out_channels, stride=stride, **kwargs)]
    for _ in range(1, num_blocks):
        blocks.append(MKIR(out_channels, out_channels, stride=1, **kwargs))
    return nn.Sequential(*blocks)


# ── CA — Channel Attention ────────────────────────────────────────────────────

class CA(nn.Module):
    """
    Attention canal CBAM.
    avg-pool + max-pool → MLP partagé → addition → sigmoid.

    Paramètres
    ----------
    num_channels    : C d'entrée/sortie
    reduction_ratio : taux de compression du MLP (défaut 16)
    """
    def __init__(self, num_channels: int, reduction_ratio: int = 16):
        super().__init__()
        C_bottleneck = max(1, num_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(num_channels, C_bottleneck, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_bottleneck, num_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne le masque d'attention canal ∈ (0,1)^{B×C×1×1}."""
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


# ── SA — Spatial Attention ────────────────────────────────────────────────────

class SA(nn.Module):
    """
    Attention spatiale CBAM.
    cat(avg_channel, max_channel) → conv → sigmoid.

    Paramètres
    ----------
    kernel_size : taille du noyau de la conv spatiale (3, 7 ou 11)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7, 11)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne le masque d'attention spatiale ∈ (0,1)^{B×1×H×W}."""
        avg_c = x.mean(dim=1, keepdim=True)
        max_c = x.max(dim=1, keepdim=True).values
        return self.sigmoid(self.conv(torch.cat([avg_c, max_c], dim=1)))


# ── GAG — Grouped Attention Gate ─────────────────────────────────────────────

class GAG(nn.Module):
    """
    Porte d'attention groupée pour les skip connections.

    Calcule ψ = σ(W_ψ · ReLU(W_g·g + W_x·x)) puis renvoie x · ψ.
    g est le signal décodeur, x la feature encodeur correspondante.

    Paramètres
    ----------
    gate_channels   : C de g (signal décodeur)
    feat_channels   : C de x (feature encodeur)
    inter_channels  : C intermédiaires du bottleneck
    kernel_size     : noyau des projections W_g et W_x (1 ou 3)
    """
    def __init__(self, gate_channels: int, feat_channels: int,
                 inter_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.Wg = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels,
                      kernel_size=kernel_size, padding=pad, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(feat_channels, inter_channels,
                      kernel_size=kernel_size, padding=pad, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
        self.apply(_init_weights)

    def forward(self, gate: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """gate : signal décodeur | feat : skip encodeur → feat pondéré."""
        return feat * self.psi(self.relu(self.Wg(gate) + self.Wx(feat)))


# ── MKUNet — Réseau complet ───────────────────────────────────────────────────

# Largeurs de canaux de la variante de base (Table 1 du papier)
_BASE_CHANNELS = [16, 32, 64, 96, 160]

class MKUNet(nn.Module):
    """
    MK-UNet : encodeur-décodeur à noyaux multiples (~0.8 M paramètres).

    Encodeur  : 5 étages MKIR + MaxPool 2×2
    Décodeur  : CA → SA → upsample → GAG(skip) → add, ×5 étages
    Sortie    : logits à résolution pleine (avant sigmoid)

    Paramètres
    ----------
    num_classes    : sorties (1 pour segmentation binaire)
    in_channels    : canaux d'entrée (3 = RGB, géré automatiquement si 1)
    channels       : largeurs des 5 étages [C1…C5]
    blocks_per_stage : répétitions MKIR par étage
    kernel_sizes   : noyaux MKDC, ex. [1, 3, 5]
    expansion_ratio: facteur t du MKIR
    gag_kernel     : noyau du GAG
    """
    def __init__(self, num_classes: int = 1, in_channels: int = 3,
                 channels: list = _BASE_CHANNELS,
                 blocks_per_stage: list = [1, 1, 1, 1, 1],
                 kernel_sizes: list = [1, 3, 5],
                 expansion_ratio: int = 2,
                 gag_kernel: int = 3):
        super().__init__()
        C = channels
        N = blocks_per_stage
        mkir_kw = dict(kernel_sizes=kernel_sizes, expansion_ratio=expansion_ratio)

        # Encodeur
        self.enc1 = _build_mkir_stage(in_channels, C[0], N[0], stride=1, **mkir_kw)
        self.enc2 = _build_mkir_stage(C[0], C[1], N[1], stride=1, **mkir_kw)
        self.enc3 = _build_mkir_stage(C[1], C[2], N[2], stride=1, **mkir_kw)
        self.enc4 = _build_mkir_stage(C[2], C[3], N[3], stride=1, **mkir_kw)
        self.enc5 = _build_mkir_stage(C[3], C[4], N[4], stride=1, **mkir_kw)

        # GAG pour chaque skip connection
        self.gag = nn.ModuleList([
            GAG(C[3], C[3], C[3] // 2, kernel_size=gag_kernel),
            GAG(C[2], C[2], C[2] // 2, kernel_size=gag_kernel),
            GAG(C[1], C[1], C[1] // 2, kernel_size=gag_kernel),
            GAG(C[0], C[0], C[0] // 2, kernel_size=gag_kernel),
        ])

        # Décodeur
        self.dec1 = _build_mkir_stage(C[4], C[3], 1, stride=1, **mkir_kw)
        self.dec2 = _build_mkir_stage(C[3], C[2], 1, stride=1, **mkir_kw)
        self.dec3 = _build_mkir_stage(C[2], C[1], 1, stride=1, **mkir_kw)
        self.dec4 = _build_mkir_stage(C[1], C[0], 1, stride=1, **mkir_kw)
        self.dec5 = _build_mkir_stage(C[0], C[0], 1, stride=1, **mkir_kw)

        # Attention par étage décodeur (CA partagé par étage, SA global partagé)
        ratios = [16, 16, 16, 8, 4]
        self.ca_list = nn.ModuleList([CA(C[4-i], reduction_ratio=ratios[i]) for i in range(5)])
        self.sa = SA(kernel_size=7)

        # Tête de segmentation finale
        self.seg_head = nn.Conv2d(C[0], num_classes, kernel_size=1)

    @staticmethod
    def _upsample2x(x: torch.Tensor) -> torch.Tensor:
        return F.relu(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Encodeur (skip features s1…s4, bottleneck bt)
        s1 = F.max_pool2d(self.enc1(x),  2, 2)
        s2 = F.max_pool2d(self.enc2(s1), 2, 2)
        s3 = F.max_pool2d(self.enc3(s2), 2, 2)
        s4 = F.max_pool2d(self.enc4(s3), 2, 2)
        bt = F.max_pool2d(self.enc5(s4), 2, 2)

        # Décodeur avec MKIRA (CA → SA → MKIR) + GAG
        d = self.ca_list[0](bt) * bt
        d = self.sa(d) * d
        d = self._upsample2x(self.dec1(d))
        d = d + self.gag[0](d, s4)

        d = self.ca_list[1](d) * d
        d = self.sa(d) * d
        d = self._upsample2x(self.dec2(d))
        d = d + self.gag[1](d, s3)

        d = self.ca_list[2](d) * d
        d = self.sa(d) * d
        d = self._upsample2x(self.dec3(d))
        d = d + self.gag[2](d, s2)

        d = self.ca_list[3](d) * d
        d = self.sa(d) * d
        d = self._upsample2x(self.dec4(d))
        d = d + self.gag[3](d, s1)

        d = self.ca_list[4](d) * d
        d = self.sa(d) * d
        d = self._upsample2x(self.dec5(d))

        return self.seg_head(d)   # logits, shape (B, num_classes, H, W)
