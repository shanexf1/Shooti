"""Grade a photo with the trained model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .embed import embed_bgr, pick_device
from .features import geometric_features
from .model import Grader

CKPT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


@dataclass
class Grade:
    score: float  # expected human rating, 1-10
    distribution: np.ndarray  # predicted vote distribution over 10 bins
    spread: float  # std dev of the distribution — how divisive the photo is
    percentile: float | None = None  # where it sits among AVA photos, if known

    @property
    def divisive(self) -> bool:
        return self.spread > 1.7


@lru_cache(maxsize=4)
def load_grader(channels: str = "both", device: str | None = None) -> tuple[Grader, str]:
    device = device or pick_device()
    path = CKPT_DIR / f"grader_{channels}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train one with:\n"
            f"  python -m shooti.grader.prepare --split train\n"
            f"  python -m shooti.grader.prepare --split validation\n"
            f"  python -m shooti.grader.train"
        )
    blob = torch.load(path, map_location=device, weights_only=False)
    model = Grader(n_bins=blob["n_bins"], channels=blob["channels"])
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, device


@torch.no_grad()
def grade_batch(
    bgr_list: list[np.ndarray],
    channels: str = "both",
    geo: np.ndarray | None = None,
) -> list[Grade]:
    """Grade several frames at once — used by the counterfactual crop search."""
    if not bgr_list:
        return []
    model, device = load_grader(channels)

    clip = torch.from_numpy(embed_bgr(bgr_list, device=device)).float().to(device)
    if geo is None:
        geo = np.stack([geometric_features(b) for b in bgr_list])
    geo_t = torch.from_numpy(np.asarray(geo, dtype=np.float32)).to(device)

    dist = model(clip, geo_t).cpu().numpy()
    bins = np.arange(1, dist.shape[1] + 1, dtype=np.float64)

    grades = []
    for row in dist:
        score = float(row @ bins)
        var = float(row @ (bins - score) ** 2)
        grades.append(Grade(score=score, distribution=row, spread=float(np.sqrt(var))))
    return grades


def grade(bgr: np.ndarray, channels: str = "both") -> Grade:
    return grade_batch([bgr], channels=channels)[0]


CHANNEL_LABELS = {
    "both": "CLIP + geometry",
    "clip": "CLIP only",
    "geo": "Geometry only",
}


def available_channels() -> list[str]:
    return [c for c in ("both", "clip", "geo") if (CKPT_DIR / f"grader_{c}.pt").exists()]


@torch.no_grad()
def grade_all_channels(bgr: np.ndarray) -> dict[str, Grade]:
    """Score one photo with every trained grader.

    Embeds the image and measures its geometry once, then runs each head over the
    same inputs — each head slices the channels it was trained on, so three
    verdicts cost one CLIP pass rather than three.
    """
    channels = available_channels()
    if not channels:
        return {}

    _, device = load_grader(channels[0])
    clip = torch.from_numpy(embed_bgr([bgr], device=device)).float().to(device)
    geo = torch.from_numpy(geometric_features(bgr)[None, :]).float().to(device)

    out: dict[str, Grade] = {}
    for name in channels:
        model, _ = load_grader(name)
        dist = model(clip, geo).cpu().numpy()[0]
        bins = np.arange(1, len(dist) + 1, dtype=np.float64)
        score = float(dist @ bins)
        spread = float(np.sqrt(dist @ (bins - score) ** 2))
        out[name] = Grade(score=score, distribution=dist, spread=spread)
    return out
