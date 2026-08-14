from __future__ import annotations

import json
from pathlib import Path

from katas.hand import HAND_KATAS
from katas.schema import Kata

ROOT = Path(__file__).resolve().parent
HAND_JSONL = ROOT / "hand.jsonl"
SYNTH_JSONL = ROOT / "synth.jsonl"
EVAL_IDS_PATH = ROOT / "agent_eval_ids.txt"


def all_hand() -> list[Kata]:
    return [Kata.from_dict(k) for k in HAND_KATAS]


def write_hand_jsonl(path: Path = HAND_JSONL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for kata in all_hand():
            f.write(json.dumps(kata.to_dict()) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[Kata]:
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(Kata.from_dict(json.loads(line)))
    return items


def load_bank(include_synth: bool = True) -> list[Kata]:
    items = all_hand()
    if include_synth and SYNTH_JSONL.exists():
        items.extend(load_jsonl(SYNTH_JSONL))
    return items


def by_id(kid: str, include_synth: bool = True) -> Kata:
    for k in load_bank(include_synth=include_synth):
        if k.id == kid:
            return k
    raise KeyError(kid)


def freeze_eval_ids(ids: list[str], path: Path = EVAL_IDS_PATH) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def load_eval_ids(path: Path = EVAL_IDS_PATH) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
