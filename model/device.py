from __future__ import annotations

import os
import subprocess

import torch


def ease_host_load() -> None:
    """Lower process priority so the desktop/GPU driver is less likely to TDR."""
    try:
        import ctypes

        BELOW_NORMAL = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(),
            BELOW_NORMAL,
        )
        print("process_priority=below_normal", flush=True)
    except Exception as e:
        print(f"priority skip: {e!r}", flush=True)


def try_cap_gpu(watts: int = 55, max_clock_mhz: int = 1600) -> None:
    """Best-effort power/clock cap. Needs admin; ignored if it fails."""
    smi = "nvidia-smi"
    for args in ([smi, "-pl", str(watts)], [smi, "-lgc", f"300,{max_clock_mhz}"]):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                print(f"gpu_cap ok: {' '.join(args[1:])}", flush=True)
            else:
                print(f"gpu_cap skip: {' '.join(args[1:])} ({(r.stderr or r.stdout).strip()[:120]})", flush=True)
        except Exception as e:
            print(f"gpu_cap skip: {e!r}", flush=True)


def require_cuda(reason: str = "training") -> torch.device:
    """Refuse CPU. PLAN.md: laptop 4060 only; never silently fall back."""
    if os.environ.get("KATA_ALLOW_CPU") == "1":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is required for {reason}. Refusing to run on CPU. "
            "This machine should expose the RTX 4060 via PyTorch."
        )
    ease_host_load()
    if os.environ.get("KATA_CAP_GPU", "1") == "1":
        try_cap_gpu()
    return torch.device("cuda")


def device_summary() -> str:
    if not torch.cuda.is_available():
        return "cuda_available=False"
    props = torch.cuda.get_device_properties(0)
    return (
        f"name={torch.cuda.get_device_name(0)} "
        f"vram_gb={props.total_memory / 1024**3:.2f} "
        f"cap={props.major}.{props.minor} "
        f"torch={torch.__version__} cuda_built={torch.version.cuda}"
    )
