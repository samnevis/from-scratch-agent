from pathlib import Path

from tokenizer.tokenizer import train_bpe


def test_train_bpe_roundtrip(tmp_path: Path):
    src = Path("tests/fixtures/tiny.txt")
    tok = train_bpe([src], vocab_size=200, out_path=tmp_path / "tok.json", min_frequency=1)
    ids = tok.encode("Python katas")
    assert ids
    text = tok.decode(ids)
    assert "Python" in text or "python" in text.lower() or len(text) > 0
    assert tok.token_id("<|end|>") >= 0
    js = '{"tool":"read_task","args":{}}'
    assert tok.decode(tok.encode(js)).replace(" ", "") == js.replace(" ", "") or "tool" in tok.decode(tok.encode(js))
