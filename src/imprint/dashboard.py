"""Read-only dashboard: render any Imprint profile JSON as a static HTML page.

    python -m imprint.dashboard my.profile.json -o dashboard.html

Shows each trait's value and band label, the standing rules, candidate rules
awaiting corroboration, and observation counts. It never writes the profile.
"""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from .traits import BANDS, TRAIT_LABELS, TRAIT_META, TRAIT_NAMES, band_index

_CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--bar:#4f46e5;--edge:#a8a29e}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;--bar:#818cf8;--edge:#78716c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:880px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.row{display:grid;grid-template-columns:130px 1fr 56px;gap:12px;align-items:center}
.track{position:relative;height:10px;border-radius:5px;background:var(--line)}
.fill{position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:var(--bar)}
.edge{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--edge)}
.band{grid-column:2/4;font-size:13px;color:var(--muted)}ul{margin:6px 0;padding-left:20px}
.stats{display:flex;gap:24px;flex-wrap:wrap}.stat b{display:block;font-size:20px}
@media (max-width:560px){.row{grid-template-columns:96px 1fr 44px}}
"""


def render_html(profile: dict, title: str = "Imprint profile") -> str:
    e = html.escape
    traits = profile.get("traits") or {}
    rows = []
    for t in TRAIT_NAMES:
        if t not in traits:
            continue
        v = float(traits[t].get("value", 0.5))
        c = float(traits[t].get("confidence", 0))
        edges, sentences = BANDS[t]
        marks = "".join(f'<span class="edge" style="left:{x * 100:.1f}%"></span>' for x in edges)
        rows.append(
            f'<div class="card"><div class="row"><b>{e(TRAIT_LABELS[t])}</b>'
            f'<div class="track" title="{e(TRAIT_META[t])}"><span class="fill" style="width:{v * 100:.1f}%"></span>{marks}</div>'
            f'<span>{v:.2f}</span><div class="band">{e(sentences[band_index(t, v)])} '
            f'<span title="confidence">· conf {c:.2f}</span></div></div></div>')
    obs = profile.get("observations") or []
    by_source = Counter(o.get("source", "?") for o in obs)
    rules = profile.get("rules") or []
    cands = profile.get("candidateRules") or []
    rule_list = "".join(f"<li>{e(r)}</li>" for r in rules) or '<li class="muted">None yet.</li>'
    cand_list = "".join(
        f"<li>{e(c.get('text', ''))} <span class='muted'>({'likes' if c.get('polarity', 1) > 0 else 'dislikes'},"
        f" seen {len(c.get('evidence') or [])}×)</span></li>" for c in cands) or '<li class="muted">None.</li>'
    stats = "".join(f'<div class="stat"><b>{n}</b><span class="muted">{e(k)}</span></div>' for k, n in [
        ("interactions", profile.get("interactions", 0)), ("observations", len(obs)),
        *[(f"{s} observations", n) for s, n in sorted(by_source.items())]])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title>
<style>{_CSS}</style></head><body><main>
<h1>{e(title)}</h1><p class="muted">Last updated {e(str(profile.get('lastUpdated', 'unknown')))}. Read-only view; bar ticks mark band edges.</p>
<div class="card stats">{stats}</div>
<h2>Traits</h2>{''.join(rows)}
<h2>Standing rules</h2><div class="card"><ul>{rule_list}</ul></div>
<h2>Candidate rules <span class="muted">(not injected until corroborated)</span></h2><div class="card"><ul>{cand_list}</ul></div>
</main></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render an Imprint profile as static HTML.")
    ap.add_argument("profile", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("imprint-dashboard.html"))
    ap.add_argument("--title", default="Imprint profile")
    a = ap.parse_args(argv)
    a.out.write_text(render_html(json.loads(a.profile.read_text(encoding="utf-8")), a.title),
                     encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
