from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from katas.schema import Kata
from sandbox.runner import run_tests

TOOL_RE = re.compile(r"\{[^{}]*\"tool\"[^{}]*\}", re.DOTALL)


@dataclass
class EpisodeState:
    kata: Kata
    solution: str = ""
    steps: int = 0
    finished: bool = False
    finish_status: str | None = None
    last_obs: str = ""
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def hidden_ok(self) -> bool:
        tests = self.kata.hidden_tests or self.kata.tests
        return run_tests(self.solution, tests).ok


def parse_tool_call(text: str) -> dict[str, Any] | None:
    text = text.strip()
    for piece in ("<|end|>", "<|assistant|>", "<|user|>", "<|obs|>", "<|tool|>"):
        text = text.replace(piece, "")
    text = text.strip()
    # Recover JSON if a code-only model dumped a function instead of a tool call.
    if text.startswith("def ") or text.startswith("class "):
        return {"tool": "write_solution", "args": {"code": text if text.endswith("\n") else text + "\n"}}
    candidates = []
    try:
        obj = json.loads(text)
        candidates.append(obj)
    except json.JSONDecodeError:
        for m in TOOL_RE.finditer(text):
            try:
                candidates.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                continue
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                candidates.append(json.loads(text[start : end + 1]))
            except json.JSONDecodeError:
                pass
    for obj in candidates:
        if isinstance(obj, dict) and "tool" in obj:
            obj.setdefault("args", {})
            return obj
    return None


class ToolEnv:
    def __init__(self, kata: Kata, max_steps: int = 8, timeout_s: float = 5.0) -> None:
        self.state = EpisodeState(kata=kata)
        self.max_steps = max_steps
        self.timeout_s = timeout_s

    def step(self, call: dict[str, Any]) -> str:
        st = self.state
        if st.finished:
            return "already finished"
        st.steps += 1
        if st.steps > self.max_steps:
            st.finished = True
            st.finish_status = "fail"
            return "step cap"
        name = call.get("tool")
        args = call.get("args") or {}
        if name == "read_task":
            obs = (
                f"prompt: {st.kata.prompt}\n"
                f"entry_point: {st.kata.entry_point}\n"
                f"visible_tests:\n" + "\n".join(st.kata.visible_tests)
            )
        elif name == "write_solution":
            st.solution = str(args.get("code", ""))
            obs = "ok"
        elif name == "run_tests":
            if not st.solution.strip():
                obs = "no solution written"
            else:
                result = run_tests(st.solution, st.kata.visible_tests, timeout_s=self.timeout_s)
                obs = result.summary()
        elif name == "finish":
            st.finished = True
            st.finish_status = str(args.get("status", "fail"))
            hidden = st.hidden_ok()
            obs = f"hidden_pass={hidden}"
        else:
            obs = f"invalid tool: {name}"
        st.last_obs = obs
        st.transcript.append({"tool": name, "args": args, "obs": obs})
        return obs


Policy = Callable[[EpisodeState], str]


def run_episode(kata: Kata, policy: Policy, max_steps: int = 8) -> EpisodeState:
    env = ToolEnv(kata, max_steps=max_steps)
    while not env.state.finished and env.state.steps < max_steps:
        raw = policy(env.state)
        call = parse_tool_call(raw)
        if call is None:
            env.state.transcript.append({"tool": None, "args": {}, "obs": "invalid json", "raw": raw})
            env.state.steps += 1
            if env.state.steps >= max_steps:
                break
            continue
        env.step(call)
    if not env.state.finished:
        env.state.finish_status = "fail"
        env.state.finished = True
    return env.state
