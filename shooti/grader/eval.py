"""Re-score saved checkpoints against any cached split.

`train.py` already reports held-out numbers (it selects epochs on a slice carved
out of train and reports on the untouched validation split). This exists to
re-check a checkpoint later without retraining, or to score a new cache.

Caveat worth keeping in mind: whether the result is "held out" depends on the
split you pass, not on this script. The dataset's own `test` split is a
byte-identical copy of `validation`, so it carries no extra information.

    python -m shooti.grader.eval --split validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .grade import CKPT_DIR, load_grader
from .train import load_split, rule_baseline, thirds_correlation
from .model import evaluate


@torch.no_grad()
def eval_checkpoint(channels: str, data: dict[str, np.ndarray]):
    model, device = load_grader(channels)
    clip = torch.from_numpy(data["clip"]).float()
    geo = torch.from_numpy(data["geo"]).float()

    outs = []
    for i in range(0, len(clip), 1024):
        outs.append(model(clip[i : i + 1024].to(device), geo[i : i + 1024].to(device)).cpu().numpy())
    dist = np.concatenate(outs, axis=0)

    bins = np.arange(1, dist.shape[1] + 1, dtype=np.float64)
    scores = dist @ bins
    return evaluate(scores, data["mean_score"].astype(np.float64), dist, data["dist"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    data = load_split(args.split)
    n = len(data["mean_score"])
    print(f"split={args.split}  n={n}  (never used for training or model selection)\n")

    base = rule_baseline(data)
    corr = thirds_correlation(data)

    print(f"  {'model':<24} {'SRCC':>8} {'PLCC':>8} {'MSE':>8} {'acc':>8}")
    print(f"  {'v1 rules (baseline)':<24} {base.srcc:>8.4f} {base.plcc:>8.4f} "
          f"{base.mse:>8.4f} {base.acc * 100:>7.2f}%")

    results = {}
    for channels in ("clip", "geo", "both"):
        if not (CKPT_DIR / f"grader_{channels}.pt").exists():
            continue
        m = eval_checkpoint(channels, data)
        results[channels] = m
        print(f"  {'learned: ' + channels:<24} {m.srcc:>8.4f} {m.plcc:>8.4f} "
              f"{m.mse:>8.4f} {m.acc * 100:>7.2f}%")

    print(f"\n  thirds_dist vs human rating: SRCC {corr['thirds_dist_srcc']:+.4f} "
          f"(p={corr['thirds_dist_p']:.3g})")
    print(f"  center_dist vs human rating: SRCC {corr['center_dist_srcc']:+.4f} "
          f"(p={corr['center_dist_p']:.3g})")

    out = CKPT_DIR / f"report_{args.split}.json"
    out.write_text(json.dumps({
        "split": args.split,
        "n": int(n),
        "baseline_rules": base.__dict__,
        "premise_check": corr,
        "learned": {k: v.__dict__ for k, v in results.items()},
    }, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
