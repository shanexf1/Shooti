"""Fetch the YuNet weights once in the parent process.

Worker processes run detection with download_model=False, so if the model file
were missing every worker would silently fall back to the saliency path and the
cached features would be wrong. Warming here makes that impossible.
"""

from __future__ import annotations

from ..subject import ensure_model


def warm_model() -> None:
    path = ensure_model(download=True)
    if path is None:
        raise SystemExit(
            "Could not obtain the YuNet model. Face features would be silently "
            "absent from the training cache, so refusing to continue."
        )
    print(f"  yunet: {path}")
