"""Does intent-conditioning actually beat one universal rule set?

v2 established that v1's single rule set correlates with human ratings at
SRCC -0.006 — noise. The v3 hypothesis is that rules work when the RIGHT rules
are applied. This measures it instead of assuming it.

Method, on the 5,110 held-out AVA photos:

  1. Assign each photo a profile with CLIP zero-shot text-image similarity.
  2. Recompute the rule score under that profile, from the cached 18-d features.
  3. Compare its correlation with human ratings against the universal score.

Honest caveat, stated up front: AVA carries no stated intent, so CLIP inferring
what the photo IS stands in for what the photographer WANTED. In the app the
user states their intent directly, which is strictly better information. This
experiment therefore measures a lower bound on what conditioning can buy.

    python -m shooti.intent.experiment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..grader.train import CACHE_DIR, load_split
from .profiles import PROFILES

CKPT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

# Several prompts per profile, averaged — single prompts are noisy in CLIP space.
PROMPTS: dict[str, tuple[str, ...]] = {
    "portrait": (
        "a portrait photograph of one person",
        "a close-up photo of a person's face",
        "a posed headshot of a single person",
    ),
    "group": (
        "a group photo of several people together",
        "a family photograph with multiple people",
        "a photo of a crowd of people posing",
    ),
    "symmetry": (
        "a symmetrical photograph of architecture",
        "a photo of a building facade shot straight on",
        "a symmetric interior with a central axis, or a mirrored reflection",
    ),
    "landscape": (
        "a landscape photograph with a wide horizon",
        "a scenic photo of mountains, sea or open countryside",
        "a sunset over an open natural scene",
    ),
    "street": (
        "a candid street photograph of everyday life",
        "documentary photojournalism in a city",
        "an unposed photo of strangers in public",
    ),
    "action": (
        "an action photograph of sport or fast movement",
        "a photo of an athlete or animal in motion",
        "a decisive moment of someone running or jumping",
    ),
    "minimal": (
        "a minimalist photograph with a lot of empty negative space",
        "a small lone subject in a vast empty field of fog or sky",
        "a very sparse simple composition",
    ),
    "product": (
        "a product photograph of an object on a plain background",
        "a flat lay of food or objects seen from above",
        "a still life studio photo of a single object",
    ),
}


def profile_text_embeddings(device: str) -> tuple[list[str], np.ndarray]:
    """Mean CLIP text embedding per profile, in the joint image-text space."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

    from ..grader.embed import MODEL_ID

    tok = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    model = CLIPTextModelWithProjection.from_pretrained(MODEL_ID).to(device).eval()

    keys, vecs = [], []
    with torch.no_grad():
        for key, prompts in PROMPTS.items():
            batch = tok(list(prompts), padding=True, return_tensors="pt").to(device)
            emb = model(**batch).text_embeds
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(0)
            mean = mean / mean.norm()
            keys.append(key)
            vecs.append(mean.float().cpu().numpy())
    return keys, np.stack(vecs)


def assign_profiles(clip: np.ndarray, device: str) -> tuple[np.ndarray, list[str]]:
    keys, text = profile_text_embeddings(device)
    sims = clip @ text.T  # both L2-normalized -> cosine
    return np.asarray(keys)[sims.argmax(axis=1)], keys


def conditioned_scores(geo: np.ndarray, assigned: np.ndarray) -> np.ndarray:
    from .engine import score_only

    out = np.empty(len(geo), dtype=np.float64)
    for i, (row, key) in enumerate(zip(geo, assigned)):
        out[i] = score_only(row, PROFILES[str(key)])
    return out


def main() -> None:
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = load_split(args.split)
    if args.limit:
        data = {k: v[: args.limit] for k, v in data.items()}

    human = data["mean_score"].astype(np.float64)
    universal = data["rule_score"].astype(np.float64)
    n = len(human)
    print(f"split={args.split}  n={n}  device={device}\n")

    assigned, keys = assign_profiles(data["clip"], device)
    print("CLIP zero-shot profile assignment:")
    for key in keys:
        c = int((assigned == key).sum())
        print(f"  {key:10s} {c:5d}  ({c / n * 100:5.1f}%)")

    cond = conditioned_scores(data["geo"], assigned)

    r_uni = spearmanr(universal, human)
    r_cond = spearmanr(cond, human)

    print("\nCorrelation with human mean rating (higher is better, 0 = no signal):")
    print(f"  universal rule set (v1):        SRCC {r_uni.statistic:+.4f}  (p={r_uni.pvalue:.3g})")
    print(f"  intent-conditioned (v3):        SRCC {r_cond.statistic:+.4f}  (p={r_cond.pvalue:.3g})")
    print(f"  change:                         {r_cond.statistic - r_uni.statistic:+.4f}")

    # Per-profile, to see whether any single rule set works even if the mix does not.
    print("\nPer-profile correlation (within the photos CLIP assigned to it):")
    per = {}
    for key in keys:
        mask = assigned == key
        if mask.sum() < 100:
            print(f"  {key:10s} n={int(mask.sum()):5d}  (too few to report)")
            continue
        r = spearmanr(cond[mask], human[mask])
        per[key] = {"n": int(mask.sum()), "srcc": float(r.statistic), "p": float(r.pvalue)}
        print(f"  {key:10s} n={int(mask.sum()):5d}  SRCC {r.statistic:+.4f}  (p={r.pvalue:.3g})")

    learned = json.loads((CKPT_DIR / "report_test.json").read_text())
    best = learned["learned"]["clip"]["srcc"]
    print(f"\nFor scale, the learned grader on the same photos: SRCC {best:+.4f}")
    print(
        "\nCONFOUND, checked by eye (out/assignment_check.png): the CLIP zero-shot\n"
        "assignment is unreliable. 'portrait' collected a pickup truck and a cat;\n"
        "'action' collected a stationary butterfly and a goose. Only 'landscape'\n"
        "looked consistently right. So this result does NOT cleanly test the\n"
        "hypothesis -- it cannot separate 'conditioning does not help' from\n"
        "'the intent was inferred wrongly'. What it does show is that AUTOMATIC\n"
        "intent inference plus these rule sets does not rescue rule-based scoring."
    )

    out = CKPT_DIR / f"report_intent_{args.split}.json"
    out.write_text(json.dumps({
        "n": int(n),
        "universal_srcc": float(r_uni.statistic),
        "universal_p": float(r_uni.pvalue),
        "conditioned_srcc": float(r_cond.statistic),
        "conditioned_p": float(r_cond.pvalue),
        "per_profile": per,
        "assignment_counts": {k: int((assigned == k).sum()) for k in keys},
        "learned_grader_srcc": best,
        "caveat": (
            "AVA has no stated intent; CLIP zero-shot infers what the photo IS as a "
            "proxy for what the photographer WANTED. The app receives stated intent "
            "directly, so this is a lower bound."
        ),
        "confound": (
            "Inspected by eye (out/assignment_check.png): the CLIP zero-shot profile "
            "assignment is unreliable -- 'portrait' collected a pickup truck and a cat, "
            "'action' collected a stationary butterfly and a goose; only 'landscape' "
            "looked consistently correct. This result therefore cannot separate "
            "'conditioning does not help' from 'intent was inferred wrongly'. It is "
            "evidence that AUTOMATIC intent inference plus these hand-tuned rule sets "
            "does not rescue rule-based scoring; it is NOT a clean test of whether "
            "user-stated intent would."
        ),
        "multiple_comparisons_note": (
            "Eight per-profile tests were run. 'action' at p=0.036 is what multiple "
            "comparisons produce by chance and should not be read as a real effect."
        ),
    }, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
