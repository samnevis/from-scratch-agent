# KataAgent

From-scratch 23M language model trained into a Python kata agent: write → test → fix.

![Fail then fix](artifacts/figures/fail_fix.gif)

**Gold** is a scripted solver that already knows the answer (it checks the tests). **Pretrain / Mid / SFT** are the same trained model after each stage.

| Stage | Hand (30) | Frozen synth (40) |
|-------|-----------|-------------------|
| Gold | 30/30 | 40/40 |
| Pretrain | 0/30 | 0/40 |
| Mid | 29/30 | 39/40 |
| SFT | 29/30 | 35/40 |

Hand = 30 written katas. Frozen synth = 40 held-out generated katas. A pass means hidden tests succeed.

![Stage ladder](artifacts/figures/stage_ladder.svg)

## Method
- Own 32k BPE + KataLM from random init (23.13M, `384d / 6L / 6H`, tied embeddings)
- Pretrain on FineWeb-Edu, CodeSearchNet Python, and katas (175M tokens)
- Mid-train on code and tool traces, then SFT
- Tools: `read_task`, `write_solution`, `run_tests`, `finish`
- CUDA only (RTX 4060)

## Training

Pretrain val **2.71** after 392k steps.

![Pretrain loss](artifacts/figures/pretrain_loss.svg)

## Run
```bash
pip install -e ".[dev]"
pytest -q
python -m agent.cli eval --policy gold --split hand
python -m agent.cli eval --policy model --split hand \
  --ckpt artifacts/checkpoints/sft/best.pt \
  --tokenizer artifacts/tokenizer/tokenizer.json
```

Weights live in `artifacts/checkpoints/` (not in git). Tokenizer and figures are in the repo.

## Links
- [PLAN.md](./PLAN.md)
- [docs/results.md](./docs/results.md)
