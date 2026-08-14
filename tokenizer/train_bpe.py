from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer.tokenizer import train_bpe


def main() -> None:
    p = argparse.ArgumentParser(description="Train a BPE tokenizer on text files")
    p.add_argument("files", nargs="+", type=Path, help="UTF-8 text files")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--out", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    args = p.parse_args()
    missing = [f for f in args.files if not f.exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    tok = train_bpe(args.files, vocab_size=args.vocab_size, out_path=args.out)
    print(f"saved {args.out} vocab_size={tok.vocab_size}")


if __name__ == "__main__":
    main()
