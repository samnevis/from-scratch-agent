"""Eval gold + pretrain/mid/SFT on hand (30) and frozen synth (40)."""

from __future__ import annotations

import json
from pathlib import Path

from agent.loop import run_episode
from agent.policies import gold_policy
from katas.bank import all_hand, load_bank, load_eval_ids

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "stage_eval.json"
TOK = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
CKPTS = {
    "pretrain": ROOT / "artifacts/checkpoints/pretrain/best.pt",
    "mid": ROOT / "artifacts/checkpoints/mid/best.pt",
    "sft": ROOT / "artifacts/checkpoints/sft/best.pt",
}
FROZEN_N = 40


def _splits() -> dict[str, list]:
    ids = load_eval_ids()[:FROZEN_N]
    want = set(ids)
    frozen = [k for k in load_bank() if k.id in want]
    frozen.sort(key=lambda k: ids.index(k.id))
    return {"hand": all_hand(), "frozen": frozen}


def _score(name: str, policy, splits: dict[str, list], blob: dict) -> None:
    blob.setdefault("runs", {})
    for split, katas in splits.items():
        key = f"{name}/{split}"
        if key in blob["runs"] and blob["runs"][key].get("done"):
            print(f"skip {key} {blob['runs'][key]['pass']}/{blob['runs'][key]['n']}", flush=True)
            continue
        ok = 0
        rows = []
        for i, k in enumerate(katas, 1):
            st = run_episode(k, policy)
            hit = st.hidden_ok()
            ok += int(hit)
            rows.append({"id": k.id, "pass": hit, "steps": st.steps})
            print(f"{key} {i}/{len(katas)} {k.id} {'PASS' if hit else 'FAIL'}", flush=True)
        blob["runs"][key] = {"n": len(katas), "pass": ok, "done": True, "rows": rows}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"done {key} {ok}/{len(katas)}", flush=True)


def main() -> None:
    splits = _splits()
    blob = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    blob["splits"] = {k: [x.id for x in v] for k, v in splits.items()}
    _score("gold", gold_policy, splits, blob)
    from agent.model_policy import ModelPolicy

    for name, ckpt in CKPTS.items():
        policy = ModelPolicy(ckpt, TOK, max_new=160)
        _score(name, policy, splits, blob)
        del policy
    table = {}
    for name in ("gold", "pretrain", "mid", "sft"):
        table[name] = {
            split: f"{blob['runs'][f'{name}/{split}']['pass']}/{blob['runs'][f'{name}/{split}']['n']}"
            for split in ("hand", "frozen")
        }
    blob["table"] = table
    OUT.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
