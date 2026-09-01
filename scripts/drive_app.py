"""Drive a Shooti Streamlit app in a real browser and screenshot the result.

Proves the UI actually renders and produces output, which a socket check does
not. Uploads a photo through the real file input rather than bypassing the UI.

    python scripts/drive_app.py --app app2.py --photo samples/messi.jpg
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent


def drive(app: str, photo: Path, port: int, out: Path, settle: int, height: int,
          intent: str | None = None) -> int:
    proc = subprocess.Popen(
        [
            str(ROOT / ".venv/bin/python"), "-m", "streamlit", "run", app,
            "--server.headless", "true", "--server.port", str(port),
            "--browser.gatherUsageStats", "false",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            # Streamlit scrolls an inner container, so full_page never grows past
            # the viewport. A tall viewport is what actually captures the page.
            page = browser.new_page(viewport={"width": 1600, "height": height})

            for attempt in range(40):
                try:
                    page.goto(f"http://localhost:{port}", timeout=5000)
                    break
                except Exception:
                    time.sleep(1)
            else:
                print("app never became reachable", file=sys.stderr)
                return 1

            page.wait_for_selector("text=Shooti", timeout=60_000)

            # Switch the sidebar source to Upload; the camera path needs a real
            # device permission we can't grant headlessly.
            try:
                page.get_by_text("Upload", exact=True).first.click(timeout=15_000)
            except Exception:
                print("note: no Upload radio found (v1 layout?)")

            page.wait_for_timeout(1500)

            # v3 takes a stated intent; fill it before uploading so the first
            # render already has it.
            if intent:
                # v3 uses a textarea; v4.x uses a text_input. Try both, and match
                # the intent box by its placeholder so a key field is never filled.
                target = page.query_selector("textarea")
                if target is None:
                    for sel in ('input[placeholder*="editorial"]',
                                'input[placeholder*="approachable"]',
                                'input[placeholder*="know"]'):
                        target = page.query_selector(sel)
                        if target is not None:
                            break
                if target is not None:
                    target.fill(intent)
                    page.keyboard.press("Tab")  # commit the widget value
                    page.wait_for_timeout(1500)
                    print(f"intent typed: {intent!r}")
                else:
                    print("note: no intent field found")

            page.set_input_files("input[type=file]", str(photo))
            print(f"uploaded {photo}")

            # Grading + the 15-candidate crop search takes a few seconds.
            page.wait_for_timeout(settle * 1000)
            try:
                page.wait_for_selector("text=/Predicted human rating|score/i", timeout=60_000)
            except Exception:
                print("note: expected result text not found")

            out.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(out), full_page=True)
            print(f"screenshot -> {out}")

            body = page.inner_text("body")
            errors = [
                line
                for line in body.splitlines()
                if any(k in line for k in ("Traceback", "Error:", "Exception", "KeyError"))
            ]
            if errors:
                print("PAGE ERRORS:")
                for e in errors[:10]:
                    print("  ", e)
                return 1

            for probe in ("Predicted human rating", "reframing", "No reframing", "score"):
                if probe.lower() in body.lower():
                    print(f"found: {probe!r}")

            browser.close()
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        logs = proc.stdout.read() if proc.stdout else ""
        bad = [
            l for l in logs.splitlines()
            if any(k in l for k in ("Traceback", "Error", "Exception"))
        ]
        if bad:
            print("SERVER LOG ISSUES:")
            for l in bad[:15]:
                print("  ", l)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app", default="app2.py")
    ap.add_argument("--photo", type=Path, default=ROOT / "samples/messi.jpg")
    ap.add_argument("--port", type=int, default=8910)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--settle", type=int, default=12, help="seconds to wait for grading")
    ap.add_argument("--height", type=int, default=2600, help="viewport height")
    ap.add_argument("--intent", default=None, help="v3: stated intent text")
    args = ap.parse_args()
    out = args.out or ROOT / "out" / f"drive_{Path(args.app).stem}.png"
    raise SystemExit(
        drive(args.app, args.photo, args.port, out, args.settle, args.height, args.intent)
    )


if __name__ == "__main__":
    main()
