"""Week 1 GPU gate: one train step of KataLM on the RTX 4060. Never CPU."""

from __future__ import annotations

import argparse
import time

import torch

from model.config import KataLMConfig
from model.device import device_summary, require_cuda
from model.transformer import KataLM


def one_step(cfg: KataLMConfig, seq: int, microbatch: int, device: torch.device) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = KataLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, cfg.vocab_size, (microbatch, seq), device=device)
    y = torch.randint(0, cfg.vocab_size, (microbatch, seq), device=device)
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda"):
        _, loss = model(x, y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    out = {
        "params_m": round(model.n_params() / 1e6, 2),
        "seq": seq,
        "microbatch": microbatch,
        "loss": float(loss.item()),
        "step_s": round(time.perf_counter() - t0, 3),
        "peak_gb": round(peak, 2),
    }
    del model, opt, x, y, loss
    torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wiki-params", action="store_true", help="untied vocab 100277 ≈ wiki 88M")
    args = p.parse_args()
    device = require_cuda("gpu smoke")
    print(device_summary())
    prod = KataLMConfig()
    print("production", one_step(prod, 512, 4, device))
    if args.wiki_params:
        wiki = KataLMConfig(vocab_size=100277, tie_embeddings=False)
        print("wiki88m", one_step(wiki, 512, 2, device))
    print("GPU_SMOKE_OK")


if __name__ == "__main__":
    main()
