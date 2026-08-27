"""Train the grader head and report honest numbers.

Split discipline, which this dataset makes easy to get wrong: its `test` split is
a byte-identical copy of its `validation` split (verified — all 5,110 image ids
match), so using both would silently report selection numbers as held-out ones.
Instead we carve a selection set out of `train` for early stopping and never
touch `validation` until the final report. train and validation are genuinely
disjoint (overlap: 0).

Every run prints four things the write-up needs:

1. The learned grader's agreement with human ratings (SRCC/PLCC/acc), on data
   used for neither fitting nor epoch selection.
2. The v1 rule score's agreement on the *same* images — the baseline v2 has to
   beat to justify existing.
3. An ablation: CLIP only, geometry only, both. If geometry adds nothing, that
   is worth knowing and saying.
4. The correlation between thirds-distance and human rating — a direct test of
   v1's core assumption that off-thirds means worse.

    python -m shooti.grader.train --ablate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .features import FEATURE_NAMES
from .model import DEFAULT_BINS, Grader, Metrics, emd_loss, evaluate

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"
CKPT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def load_split(name: str) -> dict[str, np.ndarray]:
    path = CACHE_DIR / f"ava_{name}.npz"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun: python -m shooti.grader.prepare --split {name}"
        )
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def split_off_selection(
    data: dict[str, np.ndarray], frac: float, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Carve a selection set out of train, so `validation` stays truly held out."""
    n = len(data["mean_score"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_sel = int(round(frac * n))
    sel_idx, fit_idx = perm[:n_sel], perm[n_sel:]
    fit = {k: v[fit_idx] for k, v in data.items()}
    sel = {k: v[sel_idx] for k, v in data.items()}
    return fit, sel


def rule_baseline(data: dict[str, np.ndarray]) -> Metrics:
    """How well does v1's hand-tuned score track human ratings?

    v1 scores 0-100, humans 1-10, so it is rescaled linearly onto the human
    range before MSE/accuracy. SRCC is unaffected by that rescaling.
    """
    rule = data["rule_score"].astype(np.float64)
    human = data["mean_score"].astype(np.float64)
    scaled = 1.0 + 9.0 * (rule / 100.0)
    return evaluate(scaled, human)


def thirds_correlation(data: dict[str, np.ndarray]) -> dict[str, float]:
    """Direct test of v1's premise: does off-thirds predict lower human ratings?"""
    from scipy.stats import spearmanr

    idx = FEATURE_NAMES.index("thirds_dist")
    dist = data["geo"][:, idx].astype(np.float64)
    human = data["mean_score"].astype(np.float64)
    res = spearmanr(dist, human)

    centre_idx = FEATURE_NAMES.index("center_dist")
    centre = data["geo"][:, centre_idx].astype(np.float64)
    res_c = spearmanr(centre, human)
    return {
        "thirds_dist_srcc": float(res.statistic),
        "thirds_dist_p": float(res.pvalue),
        "center_dist_srcc": float(res_c.statistic),
        "center_dist_p": float(res_c.pvalue),
    }


def _tensors(data: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(data["clip"]).float(),
        torch.from_numpy(data["geo"]).float(),
        torch.from_numpy(data["dist"]).float(),
        torch.from_numpy(data["mean_score"]).float(),
    )


@torch.no_grad()
def predict(model: Grader, clip: torch.Tensor, geo: torch.Tensor, device: str) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(clip), 1024):
        d = model(clip[i : i + 1024].to(device), geo[i : i + 1024].to(device))
        outs.append(d.cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_one(
    channels: str,
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    device: str,
    verbose: bool = True,
) -> tuple[Grader, Metrics, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_clip, tr_geo, tr_dist, tr_mean = _tensors(train_data)
    va_clip, va_geo, va_dist, va_mean = _tensors(val_data)

    model = Grader(n_bins=DEFAULT_BINS, channels=channels).to(device)
    model.set_geo_stats(train_data["geo"].mean(0), train_data["geo"].std(0))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(
        TensorDataset(tr_clip, tr_geo, tr_dist), batch_size=batch_size, shuffle=True
    )

    bin_vals = np.arange(1, DEFAULT_BINS + 1, dtype=np.float64)
    best_srcc = -np.inf
    best_state = None
    best_metrics: Metrics | None = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for clip_b, geo_b, dist_b in loader:
            opt.zero_grad()
            pred = model(clip_b.to(device), geo_b.to(device))
            loss = emd_loss(pred, dist_b.to(device))
            loss.backward()
            opt.step()
            total += float(loss) * len(clip_b)
        sched.step()

        pred_dist = predict(model, va_clip, va_geo, device)
        pred_scores = pred_dist @ bin_vals
        m = evaluate(
            pred_scores, va_mean.numpy().astype(np.float64), pred_dist, val_data["dist"]
        )
        history.append({"epoch": epoch, "train_emd": total / len(tr_clip), "val_srcc": m.srcc})
        if verbose:
            print(f"    epoch {epoch:3d}  train_emd {total / len(tr_clip):.4f}  val {m}")

        if m.srcc > best_srcc:
            best_srcc = m.srcc
            best_metrics = m
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    return model, best_metrics, history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", default="both", choices=["both", "clip", "geo"])
    ap.add_argument("--ablate", action="store_true", help="train all three channel configs")
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    full_train = load_split("train")
    test_data = load_split("validation")  # never used for fitting or selection
    fit_data, sel_data = split_off_selection(full_train, args.holdout_frac, args.seed)
    print(
        f"fit n={len(fit_data['mean_score'])}  "
        f"selection n={len(sel_data['mean_score'])} (carved from train)  "
        f"held-out n={len(test_data['mean_score'])}  device={device}"
    )

    print("\n=== Baseline: v1 hand-tuned rule score, held-out images ===")
    base = rule_baseline(test_data)
    print(f"  rule score: {base}")

    print("\n=== Is v1's premise even true? (rank correlation with human rating) ===")
    corr = thirds_correlation(test_data)
    print(
        f"  distance from nearest thirds intersection: SRCC {corr['thirds_dist_srcc']:+.4f} "
        f"(p={corr['thirds_dist_p']:.3g})"
    )
    print(
        f"  distance from dead center:                 SRCC {corr['center_dist_srcc']:+.4f} "
        f"(p={corr['center_dist_p']:.3g})"
    )
    print("  (negative SRCC would mean farther-from-thirds -> lower human rating)")

    configs = ["clip", "geo", "both"] if args.ablate else [args.channels]
    results: dict[str, Metrics] = {}
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for channels in configs:
        print(f"\n=== Training channels={channels} ===")
        model, sel_metrics, history = train_one(
            channels,
            fit_data,
            sel_data,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=device,
            verbose=True,
        )
        # The number that counts: the selected checkpoint on data it never saw,
        # not the selection set it was chosen on.
        va_clip, va_geo, _, va_mean = _tensors(test_data)
        pred_dist = predict(model, va_clip, va_geo, device)
        bins = np.arange(1, DEFAULT_BINS + 1, dtype=np.float64)
        metrics = evaluate(
            pred_dist @ bins, va_mean.numpy().astype(np.float64), pred_dist, test_data["dist"]
        )
        results[channels] = metrics

        ckpt = CKPT_DIR / f"grader_{channels}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "channels": channels,
                "n_bins": DEFAULT_BINS,
                "feature_names": list(FEATURE_NAMES),
                "metrics": metrics.__dict__,
                "selection_metrics": sel_metrics.__dict__,
                "history": history,
            },
            ckpt,
        )
        print(f"  selection set: {sel_metrics}")
        print(f"  HELD OUT:      {metrics}")
        print(f"  saved {ckpt} ({ckpt.stat().st_size / 1e6:.2f} MB)")

    print("\n=== Summary (held-out, never used for fitting or selection) ===")
    print(f"  {'model':<22} {'SRCC':>8} {'PLCC':>8} {'MSE':>8} {'acc':>8}")
    print(f"  {'v1 rules (baseline)':<22} {base.srcc:>8.4f} {base.plcc:>8.4f} "
          f"{base.mse:>8.4f} {base.acc * 100:>7.2f}%")
    for name, m in results.items():
        print(f"  {'learned: ' + name:<22} {m.srcc:>8.4f} {m.plcc:>8.4f} "
              f"{m.mse:>8.4f} {m.acc * 100:>7.2f}%")

    report = {
        "baseline_rules": base.__dict__,
        "premise_check": corr,
        "learned": {k: v.__dict__ for k, v in results.items()},
        "n_fit": int(len(fit_data["mean_score"])),
        "n_selection": int(len(sel_data["mean_score"])),
        "n": int(len(test_data["mean_score"])),
        "split_note": (
            "Selection set carved from the train split; metrics above are on the "
            "dataset's validation split, which was used for neither fitting nor "
            "epoch selection. The dataset's own 'test' split is a duplicate of "
            "'validation' and is therefore unused."
        ),
    }
    (CKPT_DIR / "report_test.json").write_text(json.dumps(report, indent=2))
    print(f"\n  wrote {CKPT_DIR / 'report_test.json'}")


if __name__ == "__main__":
    main()
