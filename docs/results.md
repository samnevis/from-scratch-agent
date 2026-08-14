# Results

KataLM 23.13M (`384d / 6L / 6H`, 32k tied BPE) trained from scratch on an RTX 4060, CUDA only.

## Data

| Split | Tokens |
|-------|--------|
| Pretrain | 175,020,979 |
| Val | 5,413,020 |
| Mix | FineWeb-Edu, CodeSearchNet Python, synth/hand katas |

## Training

| Stage | Steps | Val / notes |
|-------|-------|-------------|
| Pretrain | 392,015 | best val 2.71 |
| Mid | 20,000 | traces + code |
| SFT | 12,000 | agent checkpoint |

## Eval

| Policy | Split | Success |
|--------|-------|---------|
| Gold | 30 hand katas | 30/30 |
| SFT | 30 hand katas | 29/30 |
| SFT | 40 frozen synth | 35/40 |

Checkpoint: `artifacts/checkpoints/sft/best.pt`

```bash
python -m agent.cli eval --policy gold --split hand
python -m agent.cli eval --policy model --split hand \
  --ckpt artifacts/checkpoints/sft/best.pt \
  --tokenizer artifacts/tokenizer/tokenizer.json
python -m agent.cli eval --policy model --split agent_eval \
  --ckpt artifacts/checkpoints/sft/best.pt \
  --tokenizer artifacts/tokenizer/tokenizer.json --limit 40
```
