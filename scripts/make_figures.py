"""Write simple SVG training figures for the README (no extra deps)."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "figures"
LOG = ROOT / "artifacts" / "logs"


def _rows(name: str) -> list[dict[str, str]]:
    path = LOG / name
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _f(row: dict[str, str], key: str) -> float | None:
    try:
        v = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return None
    if v != v:  # nan
        return None
    return v


def _svg(xs: list[float], series: dict[str, list[float]], out: Path, title: str) -> None:
    w, h, pad = 720, 280, 48
    if not xs:
        out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><text x="20" y="40">{title}: no data</text></svg>\n', encoding="utf-8")
        return
    ymin = min(v for ys in series.values() for v in ys if v is not None)
    ymax = max(v for ys in series.values() for v in ys if v is not None)
    if ymin == ymax:
        ymax = ymin + 1
    xmin, xmax = min(xs), max(xs)
    if xmin == xmax:
        xmax = xmin + 1

    def xy(x: float, y: float) -> tuple[float, float]:
        px = pad + (x - xmin) / (xmax - xmin) * (w - 2 * pad)
        py = h - pad - (y - ymin) / (ymax - ymin) * (h - 2 * pad)
        return px, py

    colors = ["#2563eb", "#dc2626", "#059669"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="ui-sans-serif,system-ui">',
        f'<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{pad}" y="24" font-size="16" fill="#0f172a">{title}</text>',
    ]
    for i, (name, ys) in enumerate(series.items()):
        pts = []
        for x, y in zip(xs, ys):
            if y is None:
                continue
            px, py = xy(x, y)
            pts.append(f"{px:.1f},{py:.1f}")
        if pts:
            parts.append(f'<polyline fill="none" stroke="{colors[i % 3]}" stroke-width="2" points="{" ".join(pts)}"/>')
            parts.append(f'<text x="{w - pad - 8}" y="{36 + 16 * i}" text-anchor="end" font-size="12" fill="{colors[i % 3]}">{name}</text>')
    parts.append(f'<text x="{pad}" y="{h - 12}" font-size="11" fill="#64748b">step {xmin:.0f} – {xmax:.0f}</text>')
    parts.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pre = _rows("pretrain.csv")
    xs, tr, va = [], [], []
    seen = set()
    for i, row in enumerate(pre):
        step = _f(row, "step")
        if step is None:
            continue
        key = int(step)
        if key in seen and _f(row, "val") is None:
            continue
        # Keep val rows and a thinned train curve so the SVG stays small.
        if _f(row, "val") is None and i % 40 != 0:
            continue
        seen.add(key)
        xs.append(step)
        tr.append(_f(row, "loss"))
        va.append(_f(row, "val"))
    if xs:
        _svg(xs, {"train": tr, "val": va}, FIG / "pretrain_loss.svg", "Pretrain loss")

    stages = []
    for name, file in (("mid", "mid.csv"), ("sft", "sft.csv"), ("dpo", "dpo.csv")):
        rows = _rows(file)
        if rows:
            last = _f(rows[-1], "loss")
            stages.append((name, last if last is not None else 0.0))
    if stages:
        _svg(
            list(range(len(stages))),
            {"loss": [v for _, v in stages]},
            FIG / "stage_losses.svg",
            "Later-stage train loss",
        )
    _stage_ladder()
    print(f"wrote figures under {FIG}", flush=True)


def _stage_ladder() -> None:
    import json

    path = ROOT / "docs" / "stage_eval.json"
    if not path.exists():
        return
    blob = json.loads(path.read_text(encoding="utf-8"))
    table = blob.get("table") or {}
    order = [("gold", "Gold"), ("pretrain", "Pretrain"), ("mid", "Mid"), ("sft", "SFT")]
    rows = []
    for key, label in order:
        cell = table.get(key)
        if not cell:
            continue
        hand = cell.get("hand", "0/1")
        frozen = cell.get("frozen", "0/1")
        hp, hn = [int(x) for x in hand.split("/")]
        fp, fn = [int(x) for x in frozen.split("/")]
        rows.append((label, hp / hn, fp / fn, hand, frozen))
    if not rows:
        return
    w, h, pad = 720, 300, 56
    n = len(rows)
    group = (w - 2 * pad) / n
    bar_w = group * 0.28
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="ui-sans-serif,system-ui">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{pad}" y="28" font-size="16" fill="#0f172a">Hidden-test pass rate by stage</text>',
        f'<text x="{w - pad}" y="28" text-anchor="end" font-size="12" fill="#2563eb">Hand (30)</text>',
        f'<text x="{w - pad}" y="44" text-anchor="end" font-size="12" fill="#059669">Frozen synth (40)</text>',
    ]
    for i, (label, hr, fr, hs, fs) in enumerate(rows):
        x0 = pad + i * group + group * 0.18
        y0 = 64
        bh = h - pad - y0

        def bar(x, rate, color, caption):
            hh = max(rate * bh, 2)
            y = y0 + bh - hh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{hh:.1f}" fill="{color}" rx="3"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="11" fill="#0f172a">{caption}</text>')

        bar(x0, hr, "#2563eb", hs)
        bar(x0 + bar_w + 8, fr, "#059669", fs)
        parts.append(f'<text x="{x0 + bar_w + 4:.1f}" y="{h - 18}" text-anchor="middle" font-size="13" fill="#334155">{label}</text>')
    parts.append("</svg>")
    (FIG / "stage_ladder.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
