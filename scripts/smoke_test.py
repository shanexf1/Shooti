"""Run the CV pipeline over synthetic frames and print what it measured.

No API key needed — this exercises everything except the Claude call. Writes
annotated PNGs to out/ so the overlay can be eyeballed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shooti import analyze  # noqa: E402
from shooti.coach import measurements_text  # noqa: E402
from shooti.overlay import render  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


def tilted_horizon_scene(w=1200, h=800, tilt_deg=6.0) -> np.ndarray:
    """Sky over ground, horizon rolled by tilt_deg, with a dead-center subject."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (200, 150, 90)  # sky (BGR)
    slope = -np.tan(np.radians(tilt_deg))
    y_mid = h * 0.42
    for x in range(w):
        y = int(y_mid + slope * (x - w / 2.0))
        img[max(0, y) :, x] = (70, 110, 60)  # ground
    # A "subject": textured rectangle sitting dead center.
    cv2.rectangle(img, (w // 2 - 70, h // 2 - 110), (w // 2 + 70, h // 2 + 110), (40, 40, 40), -1)
    for i in range(0, 220, 12):
        cv2.line(img, (w // 2 - 70, h // 2 - 110 + i), (w // 2 + 70, h // 2 - 110 + i), (150, 150, 150), 2)
    return img


def offcenter_blob_scene(w=1000, h=1000) -> np.ndarray:
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.circle(img, (int(w * 0.12), int(h * 0.86)), 90, (30, 60, 200), -1)
    for r in range(20, 90, 8):
        cv2.circle(img, (int(w * 0.12), int(h * 0.86)), r, (255, 255, 255), 2)
    return img


def empty_scene(w=800, h=600) -> np.ndarray:
    return np.full((h, w, 3), 128, np.uint8)


def report(name: str, img: np.ndarray) -> None:
    analysis = analyze(img, download_model=True)
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(measurements_text(analysis))
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / f"{name}.png"), render(img, analysis))
    print(f"-> wrote out/{name}.png")


def main() -> int:
    scenes = {
        "tilted_horizon": tilted_horizon_scene(),
        "offcenter_blob": offcenter_blob_scene(),
        "flat_gray": empty_scene(),
    }
    for path in sorted((Path(__file__).resolve().parent.parent / "samples").glob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            loaded = cv2.imread(str(path))
            if loaded is not None:
                scenes[f"sample_{path.stem}"] = loaded

    for name, img in scenes.items():
        report(name, img)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
