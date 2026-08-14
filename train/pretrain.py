from __future__ import annotations

import argparse
import time
from itertools import cycle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.device import device_summary, require_cuda
from train.common import append_csv, build_model, cosine_lr, load_cfg, save_ckpt
from train.data import TokenBinDataset


@torch.no_grad()
def _val_loss(model, loader, device: torch.device, n_batches: int, use_amp: bool) -> float:
    model.eval()
    it = cycle(loader)
    total = 0.0
    n = 0
    for _ in range(max(n_batches, 1)):
        x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pretrain.yaml")
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--data-bin", default=None)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    if args.max_iters is not None:
        cfg["max_iters"] = args.max_iters
    max_seconds = args.max_seconds if args.max_seconds is not None else cfg.get("max_seconds")
    if args.data_bin:
        cfg["data_bin"] = args.data_bin
    device = require_cuda("pretrain")
    print(device_summary(), flush=True)
    model, device = build_model(cfg, device)
    print(f"params={model.n_params()/1e6:.2f}M", flush=True)
    ds = TokenBinDataset(cfg["data_bin"], cfg["seq_len"])
    print(f"dataset_seqs={len(ds)} tokens~={len(ds)*cfg['seq_len']}", flush=True)
    mb = min(int(cfg["microbatch"]), max(len(ds), 1))
    loader = DataLoader(ds, batch_size=max(mb, 1), shuffle=True, drop_last=True, pin_memory=False)
    val_loader = None
    val_path = cfg.get("val_bin")
    if val_path and Path(val_path).exists():
        val_ds = TokenBinDataset(val_path, cfg["seq_len"])
        val_mb = min(mb, max(len(val_ds), 1))
        val_loader = DataLoader(val_ds, batch_size=max(val_mb, 1), shuffle=False, drop_last=False, pin_memory=False)
        print(f"val_seqs={len(val_ds)}", flush=True)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("use_amp", True))
    it = cycle(loader)
    model.train()
    opt.zero_grad(set_to_none=True)
    best = float("inf")
    t0 = time.perf_counter()
    last_step = 0
    last_val = float("nan")
    start_step = 0
    out_dir = Path(cfg["out_dir"])
    resume_cands = list(out_dir.glob("step_*.pt"))
    for name in ("best.pt", "last.pt"):
        p = out_dir / name
        if p.exists() and p.stat().st_size > 1_000_000:
            resume_cands.append(p)
    if resume_cands:
        newest = max(resume_cands, key=lambda p: p.stat().st_mtime)
        blob = torch.load(newest, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        if blob.get("optimizer"):
            try:
                opt.load_state_dict(blob["optimizer"])
            except Exception as e:
                print(f"optimizer resume skipped: {e!r}", flush=True)
        start_step = int(blob.get("step", 0)) + 1
        best = float(blob.get("loss", best))
        print(f"resumed {newest} step={start_step} loss={best}", flush=True)
    log_path = Path(cfg["log_csv"])
    if start_step == 0 and log_path.exists():
        log_path.unlink()
    val_every = int(cfg.get("val_interval", 500))
    val_batches = int(cfg.get("val_batches", 20))
    pbar = tqdm(range(start_step, cfg["max_iters"]), desc="pretrain")
    for step in pbar:
        last_step = step
        lr = cosine_lr(step, cfg["warmup_iters"], cfg["max_iters"], cfg["lr"], cfg.get("min_lr", 0.0))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=cfg.get("use_amp", True)):
            _, loss = model(x, y)
            loss = loss / cfg["grad_accum"]
        scaler.scale(loss).backward()
        if (step + 1) % cfg["grad_accum"] == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        loss_v = float(loss.item() * cfg["grad_accum"])
        sleep_ms = float(cfg.get("step_sleep_ms") or 0)
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
        elapsed = time.perf_counter() - t0
        tok_s = ((step + 1) * x.numel()) / max(elapsed, 1e-6)
        pbar.set_postfix(loss=f"{loss_v:.4f}", val=f"{last_val:.4f}", lr=f"{lr:.2e}", tok_s=f"{tok_s:.0f}")
        if step % cfg["log_interval"] == 0:
            append_csv(
                log_path,
                {
                    "step": step,
                    "loss": loss_v,
                    "val": last_val,
                    "lr": lr,
                    "tok_s": tok_s,
                    "elapsed_s": elapsed,
                },
            )
        if val_loader is not None and step > 0 and step % val_every == 0:
            last_val = _val_loss(model, val_loader, device, val_batches, cfg.get("use_amp", True))
            print(f"val step={step} val_loss={last_val:.4f} train_loss={loss_v:.4f}", flush=True)
            append_csv(
                log_path,
                {
                    "step": step,
                    "loss": loss_v,
                    "val": last_val,
                    "lr": lr,
                    "tok_s": tok_s,
                    "elapsed_s": elapsed,
                },
            )
            if last_val < best:
                best = last_val
                save_ckpt(Path(cfg["out_dir"]) / "best.pt", model, opt, step, last_val)
        if step > 0 and step % cfg["ckpt_interval"] == 0:
            save_ckpt(out_dir / f"step_{step}.pt", model, opt, step, loss_v)
            save_ckpt(out_dir / "last.pt", model, opt, step, loss_v)
            ckpts = sorted(out_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
            for old in ckpts[:-2]:
                old.unlink(missing_ok=True)
        if max_seconds is not None and elapsed >= float(max_seconds):
            print(f"hit max_seconds={max_seconds} at step={step}", flush=True)
            break
    if val_loader is not None:
        last_val = _val_loss(model, val_loader, device, val_batches, cfg.get("use_amp", True))
        print(f"final_val={last_val:.4f}", flush=True)
        if last_val < best:
            best = last_val
            save_ckpt(Path(cfg["out_dir"]) / "best.pt", model, opt, last_step, last_val)
    save_ckpt(Path(cfg["out_dir"]) / "last.pt", model, opt, last_step, loss_v if last_step else best)
    if not (Path(cfg["out_dir"]) / "best.pt").exists():
        save_ckpt(Path(cfg["out_dir"]) / "best.pt", model, opt, last_step, best)
    done_path = Path("artifacts/logs/pretrain_complete")
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(f"steps={last_step+1} best={best} elapsed={time.perf_counter()-t0:.1f}\n", encoding="utf-8")
    print(f"done best_loss={best:.4f} steps={last_step+1} elapsed={time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
