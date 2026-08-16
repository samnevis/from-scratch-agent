# Results

KataLM 23.13M (`384d / 6L / 6H`, 32k tied BPE) trained from scratch on an RTX 4060, CUDA only.

**Gold** is a scripted solver that already knows the answer (it checks the tests). **Pretrain / Mid / SFT** are the trained model after that stage.

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
| SFT | 12,000 | post-train |

## Eval

Hand (30) = written katas. Frozen synth (40) = held-out generated katas. Pass = hidden tests succeed.

| Stage | Hand (30) | Frozen synth (40) |
|-------|-----------|-------------------|
| Gold | 30/30 | 40/40 |
| Pretrain | 0/30 | 0/40 |
| Mid | 29/30 | 39/40 |
| SFT | 29/30 | 35/40 |

![Stage ladder](../artifacts/figures/stage_ladder.svg)

```bash
python -m scripts.eval_stages
```
