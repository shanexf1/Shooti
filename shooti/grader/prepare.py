"""Build the training cache: CLIP embeddings + geometric features + labels.

Reads the AVA subset's parquet shards directly (HF streaming stalls on this
repo), decodes images in memory, and writes one compact .npz that training
reads. Shards can be deleted afterwards with --cleanup, so peak disk is one
shard rather than the full 4 GB.

Also records the v1 rule score per image, so the learned grader can be compared
against the hand-tuned rules on identical data — the whole point of v2.

    python -m shooti.grader.prepare --split validation
    python -m shooti.grader.prepare --split train --limit 12000
"""

from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from ..rules import analyze
from .embed import embed_pils, pick_device
from .features import N_FEATURES, features_from_analysis

DATASET_ID = "trojblue/AVA-aesthetics-10pct-min50-10bins"
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"
CV_MAX_EDGE = 512  # detection runs on a downscale; every feature is normalized
N_BINS = 10

SHARDS = {
    "train": [f"data/train-0000{i}-of-00006.parquet" for i in range(6)],
    "validation": [f"data/validation-0000{i}-of-00002.parquet" for i in range(2)],
    "test": [f"data/test-0000{i}-of-00002.parquet" for i in range(2)],
}


def _decode(raw: bytes, max_edge: int = CV_MAX_EDGE) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = image.size
    scale = max_edge / max(w, h)
    small = (
        image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        if scale < 1.0
        else image
    )
    return image, cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR)


def _cv_worker(raw: bytes) -> tuple[np.ndarray, float] | None:
    """CV pass for one image. Runs in a worker process — the CPU bottleneck."""
    try:
        _, bgr = _decode(raw)
        analysis = analyze(bgr, download_model=False)
        return features_from_analysis(analysis, bgr), float(analysis.score)
    except Exception:
        return None


def prepare(
    split: str,
    limit: int | None,
    batch_size: int,
    workers: int,
    out: Path,
    cleanup: bool,
) -> None:
    from huggingface_hub import hf_hub_download

    from .subject_warmup import warm_model

    warm_model()  # fetch the YuNet weights once, before forking workers

    device = pick_device()
    print(f"device={device} workers={workers} split={split} limit={limit} -> {out}")

    clip_parts: list[np.ndarray] = []
    geo_rows: list[np.ndarray] = []
    rule_rows: list[float] = []
    mean_rows: list[float] = []
    dist_rows: list[np.ndarray] = []
    vote_rows: list[int] = []
    id_rows: list[str] = []

    started = time.time()
    skipped = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for shard in SHARDS[split]:
            if limit is not None and len(geo_rows) >= limit:
                break
            path = hf_hub_download(DATASET_ID, shard, repo_type="dataset")
            print(f"  shard {shard}", flush=True)

            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=batch_size):
                if limit is not None and len(geo_rows) >= limit:
                    break
                rows = batch.to_pylist()

                raws = [r["image"]["bytes"] for r in rows]
                cv_out = list(pool.map(_cv_worker, raws, chunksize=4))

                keep_pils: list[Image.Image] = []
                for r, res in zip(rows, cv_out):
                    counts = np.asarray(r["rating_counts"], dtype=np.float64)
                    if res is None or counts.sum() <= 0 or len(counts) != N_BINS:
                        skipped += 1
                        continue
                    try:
                        pil, _ = _decode(r["image"]["bytes"], max_edge=336)
                    except Exception:
                        skipped += 1
                        continue
                    feat, rule = res
                    geo_rows.append(feat)
                    rule_rows.append(rule)
                    mean_rows.append(float(r["mean_score"]))
                    dist_rows.append((counts / counts.sum()).astype(np.float32))
                    vote_rows.append(int(r["total_votes"]))
                    id_rows.append(str(r["image_id"]))
                    keep_pils.append(pil)

                if keep_pils:
                    clip_parts.append(embed_pils(keep_pils, device=device))

                done = len(geo_rows)
                rate = done / max(time.time() - started, 1e-6)
                print(f"    {done} imgs  {rate:.1f}/s  skipped={skipped}", flush=True)

            if cleanup:
                Path(path).unlink(missing_ok=True)
                print(f"  removed {shard}", flush=True)

    if not geo_rows:
        raise SystemExit("no rows collected")

    clip = np.concatenate(clip_parts, axis=0)
    geo = np.stack(geo_rows)
    assert clip.shape[0] == geo.shape[0] == len(mean_rows), (clip.shape, geo.shape)
    assert geo.shape[1] == N_FEATURES

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        clip=clip,
        geo=geo,
        rule_score=np.asarray(rule_rows, dtype=np.float32),
        mean_score=np.asarray(mean_rows, dtype=np.float32),
        dist=np.stack(dist_rows),
        votes=np.asarray(vote_rows, dtype=np.int32),
        image_id=np.asarray(id_rows),
    )
    elapsed = time.time() - started
    size_mb = out.stat().st_size / 1e6
    print(
        f"wrote {out} ({size_mb:.1f} MB)  n={geo.shape[0]}  skipped={skipped}  "
        f"{elapsed / 60:.1f} min ({geo.shape[0] / elapsed:.1f}/s)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation", choices=list(SHARDS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--cleanup", action="store_true", help="delete each parquet shard after processing"
    )
    args = ap.parse_args()

    out = args.out or CACHE_DIR / f"ava_{args.split}.npz"
    prepare(args.split, args.limit, args.batch_size, args.workers, out, args.cleanup)


if __name__ == "__main__":
    main()
