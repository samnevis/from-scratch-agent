"""Download a bounded pretrain mix, train BPE, write uint16 shards.

PLAN.md §4.1: FineWeb-Edu sample + the-stack-smol Python + synth katas.
Never pulls full FineWeb / Stack v2. Time-capped for overnight runs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tokenizer.tokenizer import train_bpe


def _write_bin(ids: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(ids, dtype=np.uint16)
    arr.tofile(path)
    print(f"wrote {path} tokens={len(arr)}", flush=True)


def _stream_write(path: Path, rows, text_key: str, limit: int, deadline: float, label: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if time.time() >= deadline or n >= limit:
                break
            text = row.get(text_key) or ""
            if not text.strip():
                continue
            f.write(text.replace("\x00", " ") + "\n\n")
            n += 1
            if n % 500 == 0:
                print(f"{label} docs={n}", flush=True)
    print(f"{label} done docs={n} path={path}", flush=True)
    return n


def _try_write(path: Path, loader, text_key: str, limit: int, deadline: float, label: str) -> bool:
    try:
        ds = loader()
        n = _stream_write(path, ds, text_key, limit, deadline, label)
        ok = path.exists() and path.stat().st_size > 1000 and n > 0
        if not ok:
            print(f"{label} produced too little text", flush=True)
        return ok
    except Exception as e:
        print(f"{label} failed: {e!r}", flush=True)
        return False


def _load(*args, streaming: bool = True, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, split="train", streaming=streaming, **kwargs)


def _append_katas(path: Path) -> None:
    from katas.bank import all_hand, load_bank

    with path.open("a", encoding="utf-8") as f:
        for k in list(all_hand()) + load_bank(include_synth=True):
            f.write(k.prompt + "\n" + k.solution + "\n\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-fw-docs", type=int, default=40000)
    p.add_argument("--max-py-files", type=int, default=25000)
    p.add_argument("--max-seconds", type=float, default=5400)
    p.add_argument("--toy-only", action="store_true")
    args = p.parse_args()
    raw = args.out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_seconds
    texts: list[Path] = []

    if args.toy_only:
        texts = [Path("tests/fixtures/tiny.txt")]
    else:
        fw_path = raw / "lang.txt"
        py_path = raw / "python.txt"
        extra_path = raw / "katas_and_wiki.txt"
        got_fw = False
        for label, loader, key, limit in (
            ("fineweb", lambda: _load("HuggingFaceFW/fineweb-edu", "sample-10BT"), "text", args.max_fw_docs),
            ("wikitext", lambda: _load("wikitext", "wikitext-103-raw-v1", streaming=False), "text", 10**9),
            ("wikipedia", lambda: _load("wikipedia", "20220301.en", streaming=True, trust_remote_code=True), "text", args.max_fw_docs),
        ):
            if _try_write(fw_path, loader, key, limit, deadline, label):
                got_fw = True
                break
        got_py = False
        for label, loader, key, limit in (
            ("stack-smol", lambda: _load("bigcode/the-stack-smol", data_dir="data/python"), "content", args.max_py_files),
            ("code_search_net", lambda: _load("code_search_net", "python"), "whole_func_string", args.max_py_files),
            ("mbpp", lambda: _load("mbpp", streaming=False), "code", 10**9),
        ):
            if _try_write(py_path, loader, key, limit, deadline, label):
                got_py = True
                break
        extra_path.write_text("", encoding="utf-8")
        _append_katas(extra_path)
        extra_path.write_text(
            extra_path.read_text(encoding="utf-8", errors="ignore")
            + Path("tests/fixtures/tiny.txt").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        if got_fw:
            texts.append(fw_path)
        if got_py:
            texts.append(py_path)
            texts.append(py_path)  # upsample Python
        texts.append(extra_path)
        if not texts:
            raise SystemExit("no pretrain text sources available")

    tok_path = Path("artifacts/tokenizer/tokenizer.json")
    bpe_files = texts
    # Train BPE on a prefix if files are huge (faster, still representative).
    bpe_slice = raw / "bpe_slice.txt"
    if not args.toy_only:
        with bpe_slice.open("w", encoding="utf-8") as out:
            nchars = 0
            for path in texts:
                chunk = path.read_text(encoding="utf-8", errors="ignore")[: 8_000_000]
                out.write(chunk)
                nchars += len(chunk)
                if nchars >= 12_000_000:
                    break
        bpe_files = [bpe_slice]
    vocab = min(args.vocab_size, 8000 if args.toy_only else args.vocab_size)
    tok = train_bpe(bpe_files, vocab_size=vocab, out_path=tok_path, min_frequency=2)
    print(f"tokenizer vocab={tok.vocab_size}", flush=True)

    ids: list[int] = []
    for path in texts:
        print(f"tokenizing {path} size={path.stat().st_size}", flush=True)
        with path.open(encoding="utf-8", errors="ignore") as f:
            buf = []
            buf_n = 0
            for line in f:
                buf.append(line)
                buf_n += len(line)
                if buf_n >= 400_000:
                    ids.extend(tok.encode("".join(buf)))
                    buf, buf_n = [], 0
            if buf:
                ids.extend(tok.encode("".join(buf)))
        print(f"running tokens={len(ids)}", flush=True)
        if time.time() >= deadline + 600:
            print("tokenize time cap; stopping", flush=True)
            break
    min_tokens = 4096 if args.toy_only else 50_000
    if len(ids) < min_tokens and ids:
        ids = ids * ((min_tokens // len(ids)) + 1)
    n = len(ids)
    split = max(int(n * 0.97), 1)
    _write_bin(ids[:split], args.out_dir / "tokenized" / "pretrain.bin")
    _write_bin(ids[split:] or ids[:2048], args.out_dir / "tokenized" / "pretrain_val.bin")
    print(f"tokenizer={tok_path} vocab={tok.vocab_size} train_tokens={split}", flush=True)


if __name__ == "__main__":
    main()
