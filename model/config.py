from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class KataLMConfig:
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    block_size: int = 512
    vocab_size: int = 32000
    tie_embeddings: bool = True
    special_tokens: list[str] = field(
        default_factory=lambda: [
            "<|user|>",
            "<|assistant|>",
            "<|tool|>",
            "<|obs|>",
            "<|end|>",
            "<|pad|>",
        ]
    )
    name: str = "katalm-88m"

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @classmethod
    def from_yaml(cls, path: str | Path) -> "KataLMConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**known)


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
