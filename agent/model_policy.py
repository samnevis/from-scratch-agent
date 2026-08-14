from __future__ import annotations

from pathlib import Path

import torch

from model.config import KataLMConfig
from model.device import require_cuda
from model.transformer import KataLM
from tokenizer.tokenizer import KataTokenizer


class ModelPolicy:
    def __init__(self, ckpt: str | Path, tokenizer_path: str | Path, max_new: int = 384) -> None:
        device = require_cuda("model policy generation")
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = KataLMConfig(**blob["config"]) if isinstance(blob.get("config"), dict) else blob["config"]
        self.model = KataLM(cfg).to(device)
        self.model.load_state_dict(blob["model"])
        self.model.eval()
        self.tok = KataTokenizer.load(tokenizer_path)
        self.device = device
        self.max_new = max_new
        self.eos = self.tok.token_id("<|end|>")

    def __call__(self, state) -> str:
        from agent.policies import format_prompt

        prompt = format_prompt(state.kata, state.transcript)
        ids = torch.tensor([self.tok.encode(prompt)], dtype=torch.long, device=self.device)
        out = self.model.generate(
            ids,
            max_new_tokens=self.max_new,
            temperature=0.0,
            top_k=None,
            eos_id=self.eos,
        )
        new_ids = out[0, ids.size(1) :].tolist()
        text = self.tok.decode(new_ids)
        for piece in ("<|end|>", "<|assistant|>", "<|user|>", "<|obs|>"):
            text = text.replace(piece, "")
        return text.strip()
