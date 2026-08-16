# Results

Python LM 23.13M (`384d / 6L / 6H`, 32k tied BPE) trained from scratch on an RTX 4060, CUDA only.

## Data

| Split | Tokens |
|-------|--------|
| Pretrain | 175,020,979 |
| Val | 5,413,020 |
| Mix | FineWeb-Edu, CodeSearchNet Python, hand and generated functions |

## Training

| Stage | Steps | Notes |
|-------|-------|-------|
| Pretrain | 392,015 | best val 2.71 |
| Fine-tune | 20,000 | code + tool traces |

## Eval

**Hand (30).** Thirty written Python functions with visible tests the model can run and hidden tests used only for scoring. After fine-tuning on tool traces, it solves **29/30**.

**Frozen synth (40).** Forty generated functions held out of that fine-tune, same templates with different constants. It solves **39/40**. A pass means the hidden tests succeed.

```bash
python -m agent.cli eval --policy model --split hand \
  --ckpt artifacts/checkpoints/mid/best.pt \
  --tokenizer artifacts/tokenizer/tokenizer.json
```
