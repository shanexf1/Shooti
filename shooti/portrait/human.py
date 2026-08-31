"""Is this a photograph of a real human being?

Face detection answers a different question. On the test set YuNet found the
cartoon face painted on a Thomas the Tank Engine toy at 0.83 confidence with all
five landmarks correctly placed. It was not wrong — that IS a face. It is not a
human, and a portrait tool coaching you on a toy train's headroom is broken.

CLIP zero-shot, deliberately with the simplest possible prompt pair. Elaborate
prompt sets did worse: adding "a doll or mannequin face" made it reject real
children, presumably on smooth skin.

MEASURED PERFORMANCE, on 24 face crops mined from a different AVA shard and
labelled by hand, with no tuning against them: 22/24 correct (92%).
  - 19/20 real humans kept; 1 rejected (a child, margin -0.003)
  - 3/4 non-humans caught; 1 let through (a scarecrow, ambiguous to me too)
Both errors sat within 0.003 of the decision boundary, hence the uncertain band
below. n=24 is small; treat 92% as indicative, not established.

Because it rejects real humans a few percent of the time, this NEVER blocks —
refusing to analyse a genuine portrait is worse than analysing a cat.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

HUMAN_PROMPT = ("a photo of a person",)
OTHER_PROMPT = ("a photo of a toy",)
UNCERTAIN_BAND = 0.005  # both held-out errors fell inside this


@dataclass
class HumanCheck:
    verdict: str  # "human" | "not-human" | "uncertain" | "unavailable"
    margin: float
    note: str

    @property
    def looks_human(self) -> bool:
        return self.verdict in ("human", "uncertain", "unavailable")


@lru_cache(maxsize=1)
def _prompts(device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

    from ..grader.embed import MODEL_ID

    tok = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    model = CLIPTextModelWithProjection.from_pretrained(MODEL_ID).to(device).eval()

    def embed(prompts):
        with torch.no_grad():
            batch = tok(list(prompts), padding=True, return_tensors="pt").to(device)
            emb = model(**batch).text_embeds
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu().numpy()

    return embed(HUMAN_PROMPT), embed(OTHER_PROMPT)


def check(bgr: np.ndarray, face_box: tuple[int, int, int, int]) -> HumanCheck:
    from ..grader.embed import embed_bgr, pick_device

    h, w = bgr.shape[:2]
    x, y, bw, bh = face_box
    pad = 0.6
    crop = bgr[
        max(0, int(y - pad * bh)) : min(h, int(y + bh + pad * bh)),
        max(0, int(x - pad * bw)) : min(w, int(x + bw + pad * bw)),
    ]
    if crop.size == 0:
        return HumanCheck("unavailable", 0.0, "Face crop was empty.")

    try:
        device = pick_device()
        human_t, other_t = _prompts(device)
        emb = embed_bgr([crop], device=device)[0]
    except Exception as exc:  # never block the app on this
        return HumanCheck("unavailable", 0.0, f"Human check unavailable ({type(exc).__name__}).")

    margin = float((emb @ human_t.T).max() - (emb @ other_t.T).max())

    if abs(margin) < UNCERTAIN_BAND:
        return HumanCheck(
            "uncertain", margin,
            "Can't tell whether this is a real person — it sits right on the "
            "decision boundary, where this check's known errors live. Proceeding anyway.",
        )
    if margin > 0:
        return HumanCheck("human", margin, "Reads as a real person.")
    return HumanCheck(
        "not-human", margin,
        "This looks like a drawing, toy, statue or animal rather than a real "
        "person. Portrait rules assume a human subject, so treat everything below "
        "with suspicion. (This check is right about 92% of the time on 24 hand-"
        "labelled crops — it can be wrong.)",
    )
