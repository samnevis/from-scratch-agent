"""Grow the pretrain shards with more FineWeb using the existing 32k BPE.

Does not retrain the tokenizer. CUDA is not used.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tokenizer.tokenizer import KataTokenizer


def _stream_append(path: Path, rows, text_key: str, limit: int, skip: int, deadline: float, label: str) -> int:
    n = 0
    skipped = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            if time.time() >= deadline:
                break
            if skipped < skip:
                skipped += 1
                if skipped % 5000 == 0:
                    print(f"{label} skipped={skipped}", flush=True)
                continue
            if n >= limit:
                break
            text = row.get(text_key) or ""
            if not text.strip():
                continue
            f.write(text.replace("\x00", " ") + "\n\n")
            n += 1
            if n % 500 == 0:
                print(f"{label} extra_docs={n}", flush=True)
    print(f"{label} extra done docs={n} skipped={skipped} path={path}", flush=True)
    return n


def _tokenize_files(tok: KataTokenizer, paths: list[Path], deadline: float) -> list[int]:
    ids: list[int] = []
    for path in paths:
        if not path.exists():
            continue
        print(f"tokenizing {path} size={path.stat().st_size}", flush=True)
        buf: list[str] = []
        buf_n = 0
        with path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                buf.append(line)
                buf_n += len(line)
                if buf_n >= 400_000:
                    ids.extend(tok.encode("".join(buf)))
                    buf, buf_n = [], 0
            if buf:
                ids.extend(tok.encode("".join(buf)))
        print(f"running tokens={len(ids)}", flush=True)
        if time.time() >= deadline:
            print("tokenize time cap; stopping", flush=True)
            break
    return ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extra-fw-docs", type=int, default=100000)
    p.add_argument("--skip-fw-docs", type=int, default=50000)
    p.add_argument("--max-seconds", type=float, default=1500)
    args = p.parse_args()
    deadline = time.time() + args.max_seconds
    raw = Path("data/raw")
    lang = raw / "lang.txt"
    py = raw / "python.txt"
    extra = raw / "katas_and_wiki.txt"
    tok = KataTokenizer.load("artifacts/tokenizer/tokenizer.json")
    print(f"tokenizer vocab={tok.vocab_size}", flush=True)

    from datasets import load_dataset

    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        _stream_append(lang, ds, "text", args.extra_fw_docs, args.skip_fw_docs, deadline, "fineweb_extra")
    except Exception as e:
        print(f"fineweb_extra failed: {e!r}", flush=True)

    texts = [lang]
    if py.exists() and py.stat().st_size > 1000:
        texts.extend([py, py])
    if extra.exists():
        texts.append(extra)
    ids = _tokenize_files(tok, texts, deadline + 300)
    if len(ids) < 50_000:
        raise SystemExit(f"too few tokens after expand: {len(ids)}")
    n = len(ids)
    split = max(int(n * 0.97), 1)
    out = Path("data/tokenized")
    out.mkdir(parents=True, exist_ok=True)
    np.asarray(ids[:split], dtype=np.uint16).tofile(out / "pretrain.bin")
    np.asarray(ids[split:] or ids[:2048], dtype=np.uint16).tofile(out / "pretrain_val.bin")
    print(f"expanded train_tokens={split} val_tokens={n - split}", flush=True)


if __name__ == "__main__":
    main()
