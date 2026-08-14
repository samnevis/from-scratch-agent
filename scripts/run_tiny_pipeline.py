"""Run pretrain → mid → SFT → DPO with tiny configs. Each stage is capped at 55 minutes."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

STAGE_TIMEOUT_S = 55 * 60
ROOT = Path(__file__).resolve().parents[1]


def run(label: str, argv: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, *argv],
        cwd=ROOT,
        timeout=STAGE_TIMEOUT_S,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed rc={proc.returncode} after {elapsed:.1f}s")
    print(f"=== {label} done in {elapsed:.1f}s ===", flush=True)


def main() -> None:
    run("prepare-toy-shards", ["-m", "scripts.prepare_pretrain", "--toy-only", "--vocab-size", "256"])
    run("canonical-traces", ["-m", "scripts.make_canonical_traces"])
    run("sft-jsonl", ["-m", "scripts.make_sft"])
    run("dpo-pairs", ["-m", "scripts.make_dpo_pairs"])
    run("pretrain", ["-m", "train.pretrain", "--config", "configs/pretrain_tiny.yaml"])
    run("midtrain", ["-m", "train.midtrain", "--config", "configs/mid_tiny.yaml"])
    run("sft", ["-m", "train.sft", "--config", "configs/sft_tiny.yaml"])
    run("dpo", ["-m", "train.dpo", "--config", "configs/dpo_tiny.yaml"])
    run("gold-eval", ["-m", "agent.cli", "eval", "--policy", "gold", "--split", "hand"])
    print("\nTINY PIPELINE COMPLETE (CUDA). Checkpoints under artifacts/checkpoints/tiny/", flush=True)


if __name__ == "__main__":
    main()
