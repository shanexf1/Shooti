"""Frozen CLIP image embeddings.

CLIP does the semantic heavy lifting — what the photo is *of*, its style, its
light — which is exactly the part a geometry-only model can't see. We never
fine-tune it: only a small head on top is trained, so the whole thing fits on a
laptop and 20k images embed in minutes.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

MODEL_ID = "openai/clip-vit-base-patch32"
EMBED_DIM = 512


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def _load(device: str):
    processor = CLIPImageProcessor.from_pretrained(MODEL_ID)
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


@torch.no_grad()
def embed_pils(images: list[Image.Image], device: str | None = None) -> np.ndarray:
    """Embed a batch of PIL images. Returns (N, EMBED_DIM) float32, L2-normalized."""
    if not images:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    device = device or pick_device()
    processor, model = _load(device)
    batch = processor(images=images, return_tensors="pt").to(device)
    out = model(**batch).image_embeds
    out = out / out.norm(dim=-1, keepdim=True)
    return out.float().cpu().numpy().astype(np.float32)


def embed_bgr(bgr_list: list[np.ndarray], device: str | None = None) -> np.ndarray:
    """Embed OpenCV BGR arrays."""
    pils = [Image.fromarray(b[:, :, ::-1]) for b in bgr_list]
    return embed_pils(pils, device=device)
