"""The learned grader: predict the *distribution* of human ratings.

Why a distribution and not a single number: on AVA, "6.0 because everyone said
6" and "6.0 because half said 3 and half said 9" are very different photos. The
second is divisive, and a coach should say so rather than average it away. This
is the NIMA formulation (Talebi & Milanfar) — softmax over rating bins, trained
with earth-mover distance, which respects that bin 3 is closer to bin 4 than to
bin 9 (plain cross-entropy does not).

The head is deliberately small. CLIP stays frozen, so this trains in seconds on
precomputed embeddings and can be honestly evaluated many times over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .embed import EMBED_DIM
from .features import N_FEATURES

DEFAULT_BINS = 10


def emd_loss(pred: torch.Tensor, target: torch.Tensor, r: int = 2) -> torch.Tensor:
    """Squared earth-mover distance between two rating distributions."""
    cdf_pred = torch.cumsum(pred, dim=1)
    cdf_true = torch.cumsum(target, dim=1)
    diff = torch.abs(cdf_pred - cdf_true) ** r
    return torch.mean(torch.mean(diff, dim=1) ** (1.0 / r))


class Grader(nn.Module):
    """CLIP embedding (+ geometric features) -> rating distribution.

    `channels` exists for the ablation the write-up needs: "both" is the real
    model, "clip" and "geo" isolate how much each half actually contributes.
    """

    def __init__(
        self,
        n_bins: int = DEFAULT_BINS,
        channels: str = "both",
        hidden: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if channels not in ("both", "clip", "geo"):
            raise ValueError(f"channels must be both|clip|geo, got {channels}")
        self.channels = channels
        self.n_bins = n_bins

        in_dim = 0
        if channels in ("both", "clip"):
            in_dim += EMBED_DIM
        if channels in ("both", "geo"):
            in_dim += N_FEATURES
        self.in_dim = in_dim

        # Standardization for the geometric block, filled from the training set.
        self.register_buffer("geo_mean", torch.zeros(N_FEATURES))
        self.register_buffer("geo_std", torch.ones(N_FEATURES))

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(64, n_bins),
        )

    def set_geo_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.geo_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.geo_std.copy_(torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32))

    def forward(self, clip: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.channels in ("both", "clip"):
            parts.append(clip)
        if self.channels in ("both", "geo"):
            parts.append((geo - self.geo_mean) / self.geo_std)
        return torch.softmax(self.net(torch.cat(parts, dim=1)), dim=1)

    def bin_values(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.arange(1, self.n_bins + 1, dtype=torch.float32, device=device)

    def score(self, dist: torch.Tensor) -> torch.Tensor:
        """Expected rating, on the dataset's own 1..n_bins scale."""
        return (dist * self.bin_values(dist.device)).sum(dim=1)


@dataclass
class Metrics:
    srcc: float  # rank correlation with human mean — the headline number
    plcc: float
    mse: float
    acc: float  # binary good/bad agreement at the AVA convention threshold
    emd: float
    n: int

    def __str__(self) -> str:
        return (
            f"SRCC {self.srcc:.4f}  PLCC {self.plcc:.4f}  "
            f"MSE {self.mse:.4f}  acc {self.acc * 100:.2f}%  EMD {self.emd:.4f}  (n={self.n})"
        )


def evaluate(
    pred_scores: np.ndarray,
    true_scores: np.ndarray,
    pred_dist: np.ndarray | None = None,
    true_dist: np.ndarray | None = None,
    threshold: float = 5.0,
) -> Metrics:
    from scipy.stats import pearsonr, spearmanr

    srcc = float(spearmanr(pred_scores, true_scores).statistic)
    plcc = float(pearsonr(pred_scores, true_scores).statistic)
    mse = float(np.mean((pred_scores - true_scores) ** 2))
    acc = float(np.mean((pred_scores >= threshold) == (true_scores >= threshold)))

    emd = float("nan")
    if pred_dist is not None and true_dist is not None:
        emd = float(
            np.mean(
                np.sqrt(
                    np.mean(
                        (np.cumsum(pred_dist, 1) - np.cumsum(true_dist, 1)) ** 2, axis=1
                    )
                )
            )
        )
    return Metrics(srcc, plcc, mse, acc, emd, len(true_scores))
