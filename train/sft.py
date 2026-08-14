from __future__ import annotations

import argparse
import time
from itertools import cycle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.device import device_summary, require_cuda
from tokenizer.tokenizer import KataTokenizer
from train.common import append_csv, build_model, cosine_lr, load_cfg, save_ckpt
from train.data import PrefixedSFTDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sft.yaml")
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    if args.max_iters is not None:
        cfg["max_iters"] = args.max_iters
    max_seconds = args.max_seconds if args.max_seconds is not None else cfg.get("max_seconds")
    device = require_cuda("sft")
    print(device_summary(), flush=True)
    tok = KataTokenizer.load(cfg["tokenizer"])
    model, device = build_model(cfg, device)
    ds = PrefixedSFTDataset(cfg["data_jsonl"], tok, cfg["seq_len"])
    print(f"sft_rows={len(ds)}", flush=True)
    mb = min(int(cfg["microbatch"]), max(len(ds), 1))
    loader = DataLoader(ds, batch_size=max(mb, 1), shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("use_amp", True))
    it = cycle(loader)
    model.train()
    opt.zero_grad(set_to_none=True)
    best = float("inf")
    t0 = time.perf_counter()
    last_step = 0
    pbar = tqdm(range(cfg["max_iters"]), desc="sft")
    for step in pbar:
        last_step = step
        lr = cosine_lr(step, cfg["warmup_iters"], cfg["max_iters"], cfg["lr"], cfg.get("min_lr", 0.0))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=cfg.get("use_amp", True)):
            logits, _ = model(x, None)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )
            loss = loss / cfg["grad_accum"]
        scaler.scale(loss).backward()
        if (step + 1) % cfg["grad_accum"] == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        loss_v = float(loss.item() * cfg["grad_accum"])
        pbar.set_postfix(loss=f"{loss_v:.4f}")
        if step % cfg["log_interval"] == 0:
            append_csv(Path(cfg["log_csv"]), {"step": step, "loss": loss_v, "lr": lr})
            if loss_v < best:
                best = loss_v
                save_ckpt(Path(cfg["out_dir"]) / "best.pt", model, opt, step, loss_v)
        if max_seconds is not None and (time.perf_counter() - t0) >= float(max_seconds):
            print(f"hit max_seconds={max_seconds} at step={step}", flush=True)
            break
    save_ckpt(Path(cfg["out_dir"]) / "last.pt", model, opt, last_step, best)
    print(f"done best_loss={best:.4f} steps={last_step + 1}", flush=True)


if __name__ == "__main__":
    main()
