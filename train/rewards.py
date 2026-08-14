"""Episode rewards for optional RL (PLAN.md §4.4)."""

from __future__ import annotations

from agent.loop import EpisodeState


def episode_reward(state: EpisodeState) -> float:
    reward = 0.0
    if state.hidden_ok():
        reward += 1.0
    valid = sum(1 for t in state.transcript if t.get("tool") in {"read_task", "write_solution", "run_tests", "finish"})
    if valid:
        reward += 0.2
    reward -= 0.05 * state.steps
    return reward
