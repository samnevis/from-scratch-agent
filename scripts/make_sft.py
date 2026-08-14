"""Next-action SFT: prefix ends at <|assistant|>, completion is the next tool JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.loop import run_episode
from agent.policies import format_prompt, gold_policy, recovery_policy
from katas.bank import all_hand, load_bank


def _actions(kata, policy) -> list[tuple[str, str]]:
    st = run_episode(kata, policy)
    rows = []
    for i in range(len(st.transcript)):
        prefix = format_prompt(kata, st.transcript[:i])
        call = {"tool": st.transcript[i].get("tool"), "args": st.transcript[i].get("args") or {}}
        completion = json.dumps(call, separators=(",", ":")) + "<|end|>"
        rows.append((prefix, completion, kata.id, i))
    rows.append(
        (
            f"<|user|>{kata.prompt}<|end|><|assistant|>",
            kata.solution.rstrip() + "\n<|end|>",
            kata.id + "_code",
            -1,
        )
    )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/post/sft.jsonl"))
    p.add_argument("--include-synth", action="store_true")
    args = p.parse_args()
    katas = list(all_hand())
    if args.include_synth:
        katas.extend(k for k in load_bank(include_synth=True) if k.split != "agent_eval")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for k in katas:
            for policy in (gold_policy, recovery_policy):
                for prefix, completion, kid, turn in _actions(k, policy):
                    f.write(
                        json.dumps({"prefix": prefix, "completion": completion, "id": kid, "turn": turn})
                        + "\n"
                    )
                    n += 1
    print(f"wrote {args.out} rows={n}")


if __name__ == "__main__":
    main()
