# Results

KataLM 23.13M (`384d / 6L / 6H`, 32k tied BPE) trained from scratch on an RTX 4060, CUDA only.

## Data

| Split | Tokens |
|-------|--------|
| Pretrain | 175,020,979 |
| Val | 5,413,020 |
| Mix | FineWeb-Edu, CodeSearchNet Python, synth/hand katas |

## Training

| Stage | Steps | Notes |
|-------|-------|-------|
| Pretrain | 392,015 | best val 2.71 |
| Agent fine-tune | 20,000 | code + tool traces |

## Eval

**Hand (30).** Thirty written katas: short Python functions with visible tests the agent can run and hidden tests used only for scoring. After fine-tuning on tool traces, the agent solves **29/30**.

**Frozen synth (40).** Forty generated katas held out of that fine-tune, same templates with different constants. The agent solves **39/40**. A pass means the hidden tests succeed.

```bash
python -m agent.cli eval --policy model --split hand \
  --ckpt artifacts/checkpoints/mid/best.pt \
  --tokenizer artifacts/tokenizer/tokenizer.json
```
