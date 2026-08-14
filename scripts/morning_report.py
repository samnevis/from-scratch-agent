"""Write a morning summary of the overnight 10h run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "logs" / "MORNING.md"


def _tail_csv(path: Path, n: int = 8) -> list[dict]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return rows[-n:]


def _ckpt_info(path: Path) -> str:
    if not path.exists():
        return "missing"
    mb = path.stat().st_size / 1e6
    return f"{path} ({mb:.1f} MB)"


def main() -> None:
    lines = ["# Overnight KataLM run", ""]
    log = ROOT / "artifacts" / "logs" / "run_10h.jsonl"
    if log.exists():
        events = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        lines.append("## Pipeline events")
        for e in events[-20:]:
            lines.append(f"- `{e.get('event')}` { {k: v for k, v in e.items() if k not in ('t', 'event')} }")
        lines.append("")
    bin_path = ROOT / "data" / "tokenized" / "pretrain.bin"
    if bin_path.exists():
        lines.append(f"- train tokens: **{bin_path.stat().st_size // 2:,}**")
    val_path = ROOT / "data" / "tokenized" / "pretrain_val.bin"
    if val_path.exists():
        lines.append(f"- val tokens: **{val_path.stat().st_size // 2:,}**")
    tok = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
    if tok.exists():
        lines.append(f"- tokenizer: {tok} ({tok.stat().st_size / 1e6:.2f} MB)")
    lines.append("")
    lines.append("## Checkpoints")
    for p in [
        ROOT / "artifacts/checkpoints/pretrain/best.pt",
        ROOT / "artifacts/checkpoints/pretrain/last.pt",
        ROOT / "artifacts/checkpoints/mid/best.pt",
        ROOT / "artifacts/checkpoints/sft/best.pt",
        ROOT / "artifacts/checkpoints/dpo/best.pt",
    ]:
        lines.append(f"- {_ckpt_info(p)}")
    lines.append("")
    pre = _tail_csv(ROOT / "artifacts/logs/pretrain.csv")
    if pre:
        lines.append("## Pretrain log (tail)")
        lines.append("| " + " | ".join(pre[0].keys()) + " |")
        lines.append("| " + " | ".join("---" for _ in pre[0]) + " |")
        for row in pre:
            lines.append("| " + " | ".join(str(row.get(k, ""))[:16] for k in pre[0]) + " |")
        lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
