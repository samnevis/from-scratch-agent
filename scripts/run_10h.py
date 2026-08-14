"""10-hour 4060 job: real data -> 88M-arch pretrain -> mid -> SFT -> DPO -> eval.

Leaves a small buffer before the deadline. CUDA only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts" / "logs" / "run_10h.jsonl"
# Original 10h window from first launch ~11:56pm ET; stop before ~9:45am.
WAKE_UNIX = 1786628700  # 2026-08-13 13:45:00 UTC
DATA_CAP_S = 75 * 60
LATER_RESERVE_S = 45 * 60  # mid+sft+dpo+eval
MIN_PRETRAIN_TOKENS = 1_000_000


def log(event: str, **kw) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"t": time.time(), "event": event, **kw}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[{event}] {extra}", flush=True)


def remaining(deadline: float) -> float:
    return max(deadline - time.time(), 1.0)


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


def _pretrain_ready() -> bool:
    bin_path = ROOT / "data" / "tokenized" / "pretrain.bin"
    tok = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
    if not bin_path.exists() or not tok.exists():
        return False
    tokens = bin_path.stat().st_size // 2
    return tokens >= MIN_PRETRAIN_TOKENS


def main() -> None:
    deadline = time.time() + 9.25 * 3600
    log("pipeline_start", deadline=deadline, budget_h=remaining(deadline) / 3600, resume=_pretrain_ready())
    if _pretrain_ready():
        log("skip_prepare", tokens= (ROOT / "data" / "tokenized" / "pretrain.bin").stat().st_size // 2)
    else:
        data_t = min(DATA_CAP_S, remaining(deadline) * 0.25)
        rc = run(
            "prepare_pretrain",
            [
                "-m",
                "scripts.prepare_pretrain",
                "--max-fw-docs",
                "50000",
                "--max-py-files",
                "30000",
                "--max-seconds",
                str(int(data_t - 30)),
                "--vocab-size",
                "32000",
            ],
            timeout=data_t,
        )
        if rc != 0:
            log("prepare_failed_fallback_toy", rc=rc)
            run(
                "prepare_toy",
                ["-m", "scripts.prepare_pretrain", "--toy-only", "--vocab-size", "32000"],
                timeout=120,
            )

    mix = ROOT / "data" / "mid" / "mix.jsonl"
    if mix.exists() and mix.stat().st_size > 1000:
        log("skip_traces")
    else:
        run("traces", ["-m", "scripts.make_canonical_traces", "--include-synth"], timeout=min(600, remaining(deadline)))
    if not (ROOT / "data" / "post" / "sft.jsonl").exists():
        run("sft_jsonl", ["-m", "scripts.make_sft"], timeout=120)
    if not (ROOT / "data" / "post" / "dpo.jsonl").exists():
        run("dpo_jsonl", ["-m", "scripts.make_dpo_pairs"], timeout=120)

    complete = ROOT / "artifacts" / "logs" / "pretrain_complete"
    if complete.exists():
        log("skip_pretrain", marker=str(complete))
        rc = 0
    else:
        left = remaining(deadline)
        pretrain_t = max(left - LATER_RESERVE_S, left * 0.7)
        rc = run(
            "pretrain",
            [
                "-m",
                "train.pretrain",
                "--config",
                "configs/pretrain.yaml",
                "--max-iters",
                "2000000",
                "--max-seconds",
                str(int(pretrain_t - 20)),
            ],
            timeout=pretrain_t,
        )
        if rc != 0:
            log("pretrain_nonzero", rc=rc)
            run("morning_report", ["-m", "scripts.morning_report"], timeout=30)
            log("pipeline_end", remaining_s=round(remaining(deadline), 1), aborted="pretrain_failed")
            return

    # Split leftover across mid / sft / dpo.
    for label, argv_extra, share in (
        ("mid", ["-m", "train.midtrain", "--config", "configs/mid.yaml", "--max-iters", "20000"], 0.40),
        ("sft", ["-m", "train.sft", "--config", "configs/sft.yaml", "--max-iters", "12000"], 0.30),
        ("dpo", ["-m", "train.dpo", "--config", "configs/dpo.yaml", "--max-iters", "8000"], 0.25),
    ):
        slice_t = remaining(deadline) * share
        if slice_t < 60:
            log("skip", label=label, remaining=remaining(deadline))
            continue
        run(
            label,
            argv_extra + ["--max-seconds", str(int(slice_t - 15))],
            timeout=slice_t,
        )

    if remaining(deadline) > 30:
        run("gold_eval", ["-m", "agent.cli", "eval", "--policy", "gold", "--split", "hand"], timeout=60)
        ckpt = ROOT / "artifacts/checkpoints/dpo/best.pt"
        if not ckpt.exists():
            ckpt = ROOT / "artifacts/checkpoints/sft/best.pt"
        if not ckpt.exists():
            ckpt = ROOT / "artifacts/checkpoints/mid/best.pt"
        if not ckpt.exists():
            ckpt = ROOT / "artifacts/checkpoints/pretrain/best.pt"
        tok = ROOT / "artifacts/tokenizer/tokenizer.json"
        if ckpt.exists() and tok.exists() and remaining(deadline) > 90:
            run(
                "model_eval_hand",
                [
                    "-m",
                    "agent.cli",
                    "eval",
                    "--policy",
                    "model",
                    "--split",
                    "hand",
                    "--ckpt",
                    str(ckpt),
                    "--tokenizer",
                    str(tok),
                    "--limit",
                    "10",
                ],
                timeout=min(900, remaining(deadline) - 20),
            )
    run("morning_report", ["-m", "scripts.morning_report"], timeout=30)
    log("pipeline_end", remaining_s=round(remaining(deadline), 1))


if __name__ == "__main__":
    main()
