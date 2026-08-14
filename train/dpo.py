from __future__ import annotations

import argparse
import json
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.device import device_summary, require_cuda
from tokenizer.tokenizer import KataTokenizer
from train.common import append_csv, build_model, load_cfg, save_ckpt


class DPOPairs(Dataset):
    def __init__(self, path: str | Path, tokenizer: KataTokenizer, seq_len: int) -> None:
        self.seq_len = seq_len
        self.tok = tokenizer
        self.rows = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def _ids(self, text: str) -> torch.Tensor:
        ids = self.tok.encode(text)[: self.seq_len]
        if len(ids) < 2:
            ids = ids + [self.tok.token_id("<|end|>")]
        t = torch.tensor(ids, dtype=torch.long)
        if t.numel() < self.seq_len:
            pad = torch.full((self.seq_len - t.numel(),), self.tok.token_id("<|pad|>"), dtype=torch.long)
            t = torch.cat([t, pad])
        return t

    def __len__(self) -> int:
        return max(len(self.rows), 1)

    def __getitem__(self, idx: int):
        row = self.rows[idx % len(self.rows)]
        return self._ids(row["chosen"]), self._ids(row["rejected"])


def sequence_logprobs(model, idx: torch.Tensor, pad_id: int) -> torch.Tensor:
    logits, _ = model(idx[:, :-1], None)
    logp = F.log_softmax(logits, dim=-1)
    tgt = idx[:, 1:]
    token_lp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    mask = tgt.ne(pad_id).float()
    return (token_lp * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dpo.yaml")
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    if args.max_iters is not None:
        cfg["max_iters"] = args.max_iters
    max_seconds = args.max_seconds if args.max_seconds is not None else cfg.get("max_seconds")
    device = require_cuda("dpo")
    print(device_summary())
    tok = KataTokenizer.load(cfg["tokenizer"])
    model, device = build_model(cfg, device)
    ref, _ = build_model(cfg, device)
    ref.load_state_dict(model.state_dict())
    ref.eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ds = DPOPairs(cfg["pairs_jsonl"], tok, cfg["seq_len"])
    loader = DataLoader(ds, batch_size=cfg["microbatch"], shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    it = cycle(loader)
    beta = cfg.get("beta", 0.1)
    pad_id = tok.token_id("<|pad|>")
    model.train()
    best = float("inf")
    t0 = time.perf_counter()
    last_step = 0
    pbar = tqdm(range(cfg["max_iters"]), desc="dpo")
    for step in pbar:
        last_step = step
        chosen, rejected = next(it)
        chosen, rejected = chosen.to(device), rejected.to(device)
        with torch.amp.autocast("cuda", enabled=cfg.get("use_amp", True)):
            pol_c = sequence_logprobs(model, chosen, pad_id)
            pol_r = sequence_logprobs(model, rejected, pad_id)
            with torch.no_grad():
                ref_c = sequence_logprobs(ref, chosen, pad_id)
                ref_r = sequence_logprobs(ref, rejected, pad_id)
            pi = pol_c - pol_r
            pj = ref_c - ref_r
            loss = -F.logsigmoid(beta * (pi - pj)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        loss_v = float(loss.item())
        pbar.set_postfix(loss=f"{loss_v:.4f}")
        if step % cfg["log_interval"] == 0:
            append_csv(Path(cfg["log_csv"]), {"step": step, "loss": loss_v})
            if loss_v < best:
                best = loss_v
                save_ckpt(Path(cfg["out_dir"]) / "best.pt", model, opt, step, loss_v)
        if max_seconds is not None and (time.perf_counter() - t0) >= float(max_seconds):
            print(f"hit max_seconds={max_seconds} at step={step}", flush=True)
            break
    save_ckpt(Path(cfg["out_dir"]) / "last.pt", model, opt, last_step, best)


if __name__ == "__main__":
    main()
