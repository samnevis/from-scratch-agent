from __future__ import annotations

import json

from agent.loop import EpisodeState
from katas.schema import Kata


def gold_policy(state: EpisodeState) -> str:
    """Canonical success trace (PLAN.md §4.3). No teacher model."""
    n = len(state.transcript)
    if n == 0:
        return json.dumps({"tool": "read_task", "args": {}})
    if n == 1:
        return json.dumps({"tool": "write_solution", "args": {"code": state.kata.solution}})
    if n == 2:
        return json.dumps({"tool": "run_tests", "args": {}})
    return json.dumps({"tool": "finish", "args": {"status": "pass", "note": "gold"}})


def recovery_policy(state: EpisodeState) -> str:
    """Buggy first write, then gold fix."""
    n = len(state.transcript)
    if n == 0:
        return json.dumps({"tool": "read_task", "args": {}})
    if n == 1:
        return json.dumps({"tool": "write_solution", "args": {"code": "def broken():\n    return None\n"}})
    if n == 2:
        return json.dumps({"tool": "run_tests", "args": {}})
    if n == 3:
        return json.dumps({"tool": "write_solution", "args": {"code": state.kata.solution}})
    if n == 4:
        return json.dumps({"tool": "run_tests", "args": {}})
    return json.dumps({"tool": "finish", "args": {"status": "pass", "note": "recovered"}})


def format_prompt(kata: Kata, transcript: list[dict]) -> str:
    parts = [f"<|user|>{kata.prompt}<|end|>"]
    for turn in transcript:
        call = {"tool": turn.get("tool"), "args": turn.get("args", {})}
        parts.append(f"<|assistant|>{json.dumps(call)}<|end|>")
        parts.append(f"<|obs|>{turn.get('obs', '')}<|end|>")
    parts.append("<|assistant|>")
    return "".join(parts)
