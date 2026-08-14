from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from model.config import KataLMConfig
from model.device import require_cuda
from model.transformer import KataLM


def load_cfg(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cosine_lr(step: int, warmup: int, total: int, peak: float, min_lr: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if step >= total:
        return min_lr
    t = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * t)) * (peak - min_lr)


def build_model(cfg: dict[str, Any], device: torch.device | None = None) -> tuple[KataLM, torch.device]:
    device = device or require_cuda("model init")
    model_cfg = KataLMConfig.from_yaml(cfg["model"])
    model = KataLM(model_cfg).to(device)
    if cfg.get("init_ckpt"):
        ckpt_path = Path(cfg["init_ckpt"])
        if ckpt_path.exists():
            blob = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(blob["model"])
    return model, device


def save_ckpt(path: Path, model: KataLM, optimizer, step: int, loss: float, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "loss": loss,
        "config": model.cfg.__dict__,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
