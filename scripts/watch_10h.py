"""Keep the complete CUDA pipeline alive."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = time.time() + 16 * 3600
LOG = ROOT / "artifacts" / "logs" / "watch_10h.log"
JSONL = ROOT / "artifacts" / "logs" / "run_complete.jsonl"


def _log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _python_cmds() -> list[str]:
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Select-Object -ExpandProperty CommandLine"
    )
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def _pipeline_alive(cmds: list[str]) -> bool:
    keys = (
        "scripts.run_complete",
        "scripts.run_10h",
        "train.pretrain",
        "train.midtrain",
        "train.sft",
        "train.dpo",
        "agent.cli",
    )
    return any(any(k in c for k in keys) for c in cmds)


def _pipeline_ended() -> bool:
    if not JSONL.exists():
        return False
    lines = JSONL.read_text(encoding="utf-8").splitlines()
    return bool(lines) and "pipeline_end" in lines[-1]


def main() -> None:
    _log("watchdog start complete")
    while time.time() < DEADLINE - 90:
        time.sleep(180)
        if _pipeline_ended():
            _log("pipeline_end seen; exiting")
            return
        if _pipeline_alive(_python_cmds()):
            continue
        left = DEADLINE - time.time()
        if left < 180:
            _log("too close to deadline; not restarting")
            return
        _log(f"pipeline dead; restarting run_complete left_s={left:.0f}")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["TQDM_DISABLE"] = "1"
        subprocess.Popen(
            [sys.executable, "-u", "-m", "scripts.run_complete"],
            cwd=ROOT,
            env=env,
        )
    _log("watchdog deadline reached")


if __name__ == "__main__":
    main()
