from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.loop import run_episode
from agent.policies import gold_policy, recovery_policy
from katas.bank import all_hand, by_id, load_bank, write_hand_jsonl


def _policy(name: str, ckpt: str | None, tokenizer: str | None):
    if name == "gold":
        return gold_policy
    if name == "recovery":
        return recovery_policy
    if name == "model":
        if not ckpt or not tokenizer:
            raise SystemExit("--ckpt and --tokenizer required for --policy model")
        from agent.model_policy import ModelPolicy

        return ModelPolicy(ckpt, tokenizer)
    raise SystemExit(f"unknown policy {name}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="kata-agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run one kata episode")
    run_p.add_argument("--kata", required=True)
    run_p.add_argument("--policy", default="gold", choices=["gold", "recovery", "model"])
    run_p.add_argument("--ckpt")
    run_p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    run_p.add_argument("--max-steps", type=int, default=8)

    ev = sub.add_parser("eval", help="eval a policy on a split")
    ev.add_argument("--policy", default="gold", choices=["gold", "recovery", "model"])
    ev.add_argument("--split", default="hand", choices=["hand", "agent_eval", "train"])
    ev.add_argument("--ckpt")
    ev.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ev.add_argument("--limit", type=int, default=0)

    sub.add_parser("list", help="list hand katas")
    dump = sub.add_parser("dump-hand", help="write katas/hand.jsonl")
    dump.add_argument("--out", default="katas/hand.jsonl")

    args = p.parse_args(argv)
    if args.cmd == "list":
        for k in all_hand():
            print(f"{k.id}\t{k.entry_point}\t{k.prompt[:60]}")
        return
    if args.cmd == "dump-hand":
        path = write_hand_jsonl(Path(args.out))
        print(path)
        return
    policy = _policy(args.policy, getattr(args, "ckpt", None), getattr(args, "tokenizer", None))
    if args.cmd == "run":
        kata = by_id(args.kata)
        st = run_episode(kata, policy, max_steps=args.max_steps)
        print(json.dumps({"id": kata.id, "hidden_pass": st.hidden_ok(), "steps": st.steps, "transcript": st.transcript}, indent=2))
        return
    if args.cmd == "eval":
        if args.split == "hand":
            katas = all_hand()
        elif args.split == "agent_eval":
            from katas.bank import load_eval_ids

            ids = load_eval_ids()
            want = set(ids)
            katas = [k for k in load_bank() if k.id in want]
            katas.sort(key=lambda k: ids.index(k.id))
        else:
            katas = [k for k in load_bank() if k.split == args.split]
        if args.limit:
            katas = katas[: args.limit]
        ok = 0
        for k in katas:
            st = run_episode(k, policy)
            hit = st.hidden_ok()
            ok += int(hit)
            print(f"{k.id}\t{'PASS' if hit else 'FAIL'}\tsteps={st.steps}")
        n = max(len(katas), 1)
        print(f"success {ok}/{len(katas)} ({ok / n:.1%})")


if __name__ == "__main__":
    main()
