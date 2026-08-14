from __future__ import annotations

from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL = ["<|user|>", "<|assistant|>", "<|tool|>", "<|obs|>", "<|end|>", "<|pad|>", "<|unk|>"]


class KataTokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_id(self, piece: str) -> int:
        tid = self._tok.token_to_id(piece)
        if tid is None:
            raise KeyError(piece)
        return tid

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        if add_special:
            text = f"<|user|>{text}<|end|>"
        # Disable padding so a single prompt is not stretched to a pad length.
        self._tok.no_padding()
        return self._tok.encode(text).ids

    def decode(self, ids: Sequence[int]) -> str:
        self._tok.no_padding()
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_bytelevel_decoder()
        self._tok.save(str(path))

    def _ensure_bytelevel_decoder(self) -> None:
        # Without this, HF BPE joins pieces with spaces and JSON tool calls
        # become '{ " tool ": ... }' which fails to parse.
        self._tok.decoder = ByteLevelDecoder()

    @classmethod
    def load(cls, path: str | Path) -> "KataTokenizer":
        wrapped = cls(Tokenizer.from_file(str(path)))
        wrapped._ensure_bytelevel_decoder()
        return wrapped


def train_bpe(
    files: Sequence[str | Path],
    vocab_size: int = 32000,
    out_path: str | Path | None = None,
    min_frequency: int = 2,
) -> KataTokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL,
        min_frequency=min_frequency,
        show_progress=False,
    )
    tokenizer.train([str(p) for p in files], trainer)
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is not None:
        tokenizer.enable_padding(pad_id=pad_id, pad_token="<|pad|>")
    wrapped = KataTokenizer(tokenizer)
    if out_path is not None:
        wrapped.save(out_path)
    return wrapped
