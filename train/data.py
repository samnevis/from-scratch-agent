from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenBinDataset(Dataset):
    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.mem = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        if self.mem.size < seq_len + 1:
            raise ValueError(f"{path} too short for seq_len={seq_len}")

    def __len__(self) -> int:
        return (self.mem.size - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.seq_len
        sl = np.array(self.mem[i : i + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(sl[:-1])
        y = torch.from_numpy(sl[1:])
        return x, y


class TextJsonlDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, seq_len: int) -> None:
        import json

        self.seq_len = seq_len
        ids: list[int] = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("completion") or obj.get("prefix") or ""
                if obj.get("completion") and obj.get("prefix") is not None:
                    text = obj["prefix"] + obj["completion"]
                ids.extend(tokenizer.encode(text))
                ids.append(tokenizer.token_id("<|end|>"))
        if len(ids) < seq_len + 1:
            ids = (ids * ((seq_len + 2) // max(len(ids), 1) + 1))[: seq_len + 2]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return max((len(self.data) - 1) // self.seq_len, 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = (idx * self.seq_len) % max(len(self.data) - self.seq_len - 1, 1)
        x = self.data[i : i + self.seq_len]
        y = self.data[i + 1 : i + self.seq_len + 1]
        return x, y


class PrefixedSFTDataset(Dataset):
    """Teacher-force completion tokens only (prefix labels = -100)."""

    def __init__(self, path: str | Path, tokenizer, seq_len: int) -> None:
        import json

        pad = tokenizer.token_id("<|pad|>")
        rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "prefix" in obj and "completion" in obj:
                    pre = tokenizer.encode(obj["prefix"])
                    comp = tokenizer.encode(obj["completion"])
                else:
                    text = obj.get("text") or ""
                    ids = tokenizer.encode(text)
                    cut = max(len(ids) // 4, 1)
                    pre, comp = ids[:cut], ids[cut:]
                ids = (pre + comp)[:seq_len]
                labels = ([-100] * min(len(pre), seq_len) + comp)[:seq_len]
                if len(ids) < 2:
                    continue
                x = torch.tensor(ids[:-1], dtype=torch.long)
                y = torch.tensor(labels[1:], dtype=torch.long)
                if x.numel() < seq_len:
                    n_pad = seq_len - x.numel()
                    x = torch.cat([x, torch.full((n_pad,), pad, dtype=torch.long)])
                    y = torch.cat([y, torch.full((n_pad,), -100, dtype=torch.long)])
                rows.append((x, y))
        if not rows:
            raise ValueError(f"no SFT rows in {path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rows[idx]
