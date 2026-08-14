"""Full laptop-safe pipeline: pretrain -> mid -> SFT -> DPO -> eval -> figures.

Keeps the eased GPU settings. CUDA only. Resume-friendly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts" / "logs" / "run_complete.jsonl"
PRETRAIN_S = 10 * 3600
MID_S = 90 * 60
SFT_S = 50 * 60
DPO_S = 40 * 60
EVAL_S = 25 * 60
MIN_PRETRAIN_TOKENS = 1_000_000


def log(event: str, **kw) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"t": time.time(), "event": event, **kw}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[{event}] {extra}", flush=True)


def run(label: str, argv: list[str], timeout: float) -> int:
    log("start", label=label, timeout=round(timeout, 1), cmd=" ".join(argv))
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TQDM_DISABLE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-u", *argv],
            cwd=ROOT,
            timeout=max(timeout, 30),
            env=env,
        )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        log("timeout", label=label, elapsed=time.perf_counter() - t0)
        return 124
    log("done", label=label, rc=rc, elapsed=round(time.perf_counter() - t0, 1))
    return rc


def _ready() -> bool:
    bin_path = ROOT / "data" / "tokenized" / "pretrain.bin"
    tok = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
    return bin_path.exists() and tok.exists() and (bin_path.stat().st_size // 2) >= MIN_PRETRAIN_TOKENS


def main() -> None:
    deadline = time.time() + PRETRAIN_S + MID_S + SFT_S + DPO_S + EVAL_S + 600
    log("pipeline_start", deadline=deadline, kind="complete")
    if not _ready():
        log("missing_data")
        run(
            "prepare_pretrain",
            ["-m", "scripts.prepare_pretrain", "--max-fw-docs", "50000", "--max-py-files", "30000", "--vocab-size", "32000"],
            timeout=4500,
        )
    mix = ROOT / "data" / "mid" / "mix.jsonl"
    if not (mix.exists() and mix.stat().st_size > 1000):
        run("traces", ["-m", "scripts.make_canonical_traces", "--include-synth"], timeout=600)
    if not (ROOT / "data" / "post" / "sft.jsonl").exists():
        run("sft_jsonl", ["-m", "scripts.make_sft"], timeout=120)
    if not (ROOT / "data" / "post" / "dpo.jsonl").exists():
        run("dpo_jsonl", ["-m", "scripts.make_dpo_pairs"], timeout=120)

    complete = ROOT / "artifacts" / "logs" / "pretrain_complete"
    if complete.exists():
        complete.unlink()
    rc = run(
        "pretrain",
        ["-m", "train.pretrain", "--config", "configs/pretrain.yaml", "--max-iters", "2000000", "--max-seconds", str(PRETRAIN_S - 30)],
        timeout=PRETRAIN_S,
    )
    if rc != 0:
        log("pretrain_nonzero", rc=rc)

    for label, extra, t in (
        ("mid", ["-m", "train.midtrain", "--config", "configs/mid.yaml", "--max-iters", "20000"], MID_S),
        ("sft", ["-m", "train.sft", "--config", "configs/sft.yaml", "--max-iters", "12000"], SFT_S),
        ("dpo", ["-m", "train.dpo", "--config", "configs/dpo.yaml", "--max-iters", "8000"], DPO_S),
    ):
        run(label, extra + ["--max-seconds", str(int(t - 20))], timeout=t)

    run("gold_eval", ["-m", "agent.cli", "eval", "--policy", "gold", "--split", "hand"], timeout=90)
    ckpt = ROOT / "artifacts/checkpoints/dpo/best.pt"
    if not ckpt.exists():
        ckpt = ROOT / "artifacts/checkpoints/sft/best.pt"
    if not ckpt.exists():
        ckpt = ROOT / "artifacts/checkpoints/mid/best.pt"
    if not ckpt.exists():
        ckpt = ROOT / "artifacts/checkpoints/pretrain/best.pt"
    tok = ROOT / "artifacts/tokenizer/tokenizer.json"
    if ckpt.exists() and tok.exists():
        run(
            "model_eval_hand",
            ["-m", "agent.cli", "eval", "--policy", "model", "--split", "hand", "--ckpt", str(ckpt), "--tokenizer", str(tok), "--limit", "30"],
            timeout=1200,
        )
        run(
            "model_eval_frozen",
            ["-m", "agent.cli", "eval", "--policy", "model", "--split", "agent_eval", "--ckpt", str(ckpt), "--tokenizer", str(tok), "--limit", "40"],
            timeout=1500,
        )
    run("figures", ["-m", "scripts.make_figures"], timeout=120)
    run("morning_report", ["-m", "scripts.morning_report"], timeout=30)
    (ROOT / "artifacts" / "logs" / "SHIP_READY").write_text("ok\n", encoding="utf-8")
    log("pipeline_end", remaining_s=round(max(deadline - time.time(), 0.0), 1))


if __name__ == "__main__":
    main()
