from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.loop import run_episode
from agent.policies import gold_policy, recovery_policy
from katas.bank import all_hand, load_bank, load_eval_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="gold", choices=["gold", "recovery", "model"])
    p.add_argument("--split", default="hand")
    p.add_argument("--ckpt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--out", default="artifacts/logs/eval.json")
    args = p.parse_args()
    if args.policy == "gold":
        policy = gold_policy
    elif args.policy == "recovery":
        policy = recovery_policy
    else:
        from agent.model_policy import ModelPolicy

        policy = ModelPolicy(args.ckpt, args.tokenizer)
    if args.split == "hand":
        katas = all_hand()
    elif args.split == "agent_eval":
        ids = set(load_eval_ids())
        katas = [k for k in load_bank() if k.id in ids]
    else:
        katas = [k for k in load_bank() if k.split == args.split]
    rows = []
    ok = 0
    for k in katas:
        st = run_episode(k, policy)
        hit = st.hidden_ok()
        ok += int(hit)
        rows.append({"id": k.id, "pass": hit, "steps": st.steps})
    summary = {"n": len(katas), "pass": ok, "rate": (ok / len(katas) if katas else 0.0), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"success {ok}/{len(katas)}")


if __name__ == "__main__":
    main()
