"""DPO pairs: chosen = gold pass trace, rejected = bad JSON / wrong tool / fail finish."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.loop import run_episode
from agent.policies import format_prompt, gold_policy
from katas.bank import all_hand


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/post/dpo.jsonl"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for k in all_hand():
            st = run_episode(k, gold_policy)
            chosen = format_prompt(k, st.transcript)
            rejected_json = f"<|user|>{k.prompt}<|end|><|assistant|>not json at all<|end|>"
            rejected_tool = (
                f"<|user|>{k.prompt}<|end|><|assistant|>"
                + json.dumps({"tool": "shell", "args": {"cmd": "rm -rf /"}})
                + "<|end|>"
            )
            rejected_fail = (
                f"<|user|>{k.prompt}<|end|><|assistant|>"
                + json.dumps({"tool": "finish", "args": {"status": "fail"}})
                + "<|end|>"
            )
            for rej in (rejected_json, rejected_tool, rejected_fail):
                f.write(json.dumps({"id": k.id, "chosen": chosen, "rejected": rej}) + "\n")
                n += 1
    print(f"wrote {args.out} pairs={n}")


if __name__ == "__main__":
    main()
