"""Validate every ```mermaid block in a markdown file.

GitHub renders these server-side, so a syntax error shows up as a broken box for
every reader and never as a local failure. This runs Mermaid's own parser in
headless Chromium (already installed for scripts/drive_app.py) so bad syntax
fails here instead.

    python scripts/check_mermaid.py docs/architecture.md
    python scripts/check_mermaid.py docs/architecture.md --render out/
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs"

PAGE = """<!doctype html><html><head><style>
body{background:#fff;margin:0;padding:16px;font-family:system-ui}
</style></head><body><div id="out"></div><script type="module">
import mermaid from "%s";
mermaid.initialize({ startOnLoad: false, theme: 'default' });
window.checkDiagram = async (src) => {
  try { await mermaid.parse(src); return { ok: true }; }
  catch (e) { return { ok: false, error: String(e && e.message ? e.message : e) }; }
};
window.renderDiagram = async (src) => {
  const { svg } = await mermaid.render('g' + Math.floor(Math.random() * 1e9), src);
  document.getElementById('out').innerHTML = svg;
  return true;
};
window.__ready = true;
</script></body></html>""" % MERMAID_CDN


def extract(md: str) -> list[tuple[int, str]]:
    """Return (line_number, source) for each mermaid fence."""
    blocks, cur, start = [], None, 0
    for i, line in enumerate(md.splitlines(), 1):
        if cur is None and re.match(r"^\s*```mermaid\s*$", line):
            cur, start = [], i
        elif cur is not None and re.match(r"^\s*```\s*$", line):
            blocks.append((start, "\n".join(cur)))
            cur = None
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        raise SystemExit(f"unclosed mermaid fence starting at line {start}")
    return blocks


def main() -> int:
    args = sys.argv[1:]
    render_dir: Path | None = None
    if "--render" in args:
        i = args.index("--render")
        render_dir = Path(args[i + 1]) if len(args) > i + 1 else Path("out")
        del args[i : i + 2]

    targets = [Path(a) for a in args] or [Path("docs/architecture.md")]
    failures = 0

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1700, "height": 1400})
        page.set_content(PAGE)
        try:
            page.wait_for_function("window.__ready === true", timeout=30_000)
        except Exception:
            print("could not load mermaid from the CDN (offline?)", file=sys.stderr)
            browser.close()
            return 2

        for path in targets:
            blocks = extract(path.read_text())
            print(f"{path}: {len(blocks)} mermaid block(s)")
            for n, (line, src) in enumerate(blocks, 1):
                result = page.evaluate("src => window.checkDiagram(src)", src)
                if not result["ok"]:
                    failures += 1
                    print(f"  FAIL  line {line}: {result['error']}")
                    continue

                first = src.strip().splitlines()[0]
                note = ""
                if render_dir is not None:
                    # Parsing is not looking right — render so a human can check.
                    render_dir.mkdir(parents=True, exist_ok=True)
                    page.evaluate("src => window.renderDiagram(src)", src)
                    page.wait_for_timeout(1200)
                    el = page.query_selector("#out svg")
                    out = render_dir / f"{path.stem}-{n}.png"
                    el.screenshot(path=str(out))
                    box = el.bounding_box()
                    note = f"  -> {out} ({int(box['width'])}x{int(box['height'])})"
                print(f"  OK    line {line}: {first}{note}")
        browser.close()

    print("\nall diagrams parse" if not failures else f"\n{failures} diagram(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
