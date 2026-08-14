"""Canonical + recovery tool traces from verified katas (no Qwen teacher)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.loop import run_episode
from agent.policies import format_prompt, gold_policy, recovery_policy
from katas.bank import all_hand, load_bank


def _trace_text(kata, policy) -> str:
    st = run_episode(kata, policy)
    # full conversation including final assistant finish
    return format_prompt(kata, st.transcript)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/mid/mix.jsonl"))
    p.add_argument("--include-synth", action="store_true")
    args = p.parse_args()
    katas = load_bank(include_synth=args.include_synth) if args.include_synth else all_hand()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for k in katas:
            if k.split == "agent_eval":
                continue
            f.write(json.dumps({"bucket": "traces", "text": _trace_text(k, gold_policy), "id": k.id}) + "\n")
            f.write(json.dumps({"bucket": "traces", "text": _trace_text(k, recovery_policy), "id": k.id + "_rec"}) + "\n")
            f.write(
                json.dumps(
                    {
                        "bucket": "single_turn",
                        "text": f"<|user|>{k.prompt}<|end|><|assistant|>{k.solution}<|end|>",
                        "id": k.id + "_st",
                    }
                )
                + "\n"
            )
            f.write(json.dumps({"bucket": "python", "text": k.solution, "id": k.id + "_py"}) + "\n")
            n += 4
    print(f"wrote {args.out} lines={n}")


if __name__ == "__main__":
    main()
