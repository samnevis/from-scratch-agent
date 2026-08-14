from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Kata:
    id: str
    prompt: str
    entry_point: str
    tests: list[str]
    hidden_tests: list[str]
    solution: str
    split: str = "train"
    difficulty: str = "easy"
    tags: list[str] = field(default_factory=list)

    @property
    def visible_tests(self) -> list[str]:
        return self.tests

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "entry_point": self.entry_point,
            "tests": self.tests,
            "hidden_tests": self.hidden_tests,
            "solution": self.solution,
            "split": self.split,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Kata":
        return cls(
            id=raw["id"],
            prompt=raw["prompt"],
            entry_point=raw["entry_point"],
            tests=list(raw.get("tests") or raw.get("visible_tests") or []),
            hidden_tests=list(raw.get("hidden_tests") or []),
            solution=raw.get("solution", ""),
            split=raw.get("split", "train"),
            difficulty=raw.get("difficulty", "easy"),
            tags=list(raw.get("tags") or []),
        )
