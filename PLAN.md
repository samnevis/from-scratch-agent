# Python LM — Pretrain → Mid → Post Python Function Agent

**Domain (LOCKED):** Python katas — short functions, unit-test rewards, multi-turn write → test → fix agent.  
**Base (LOCKED):** **Train the LM from scratch** (random init → your checkpoint). No Qwen/HF code model as the agent brain for v1.  
**Deferred:** After v1 ships, optionally mid+post-train a Qwen2.5-Coder and compare — see §12. Not in critical path.

| Constraint | Value |
|------------|--------|
| Hardware | ASUS Zephyrus, RTX 4060 Laptop (**8 GB VRAM**) |
| Cloud budget | **$100 available, not obligated** — laptop-first; spend only if needed (§5 budget) |
| Prior work | Wiki transformer → reuse/adapt architecture + training loop |
| Goal | Full stack you own: tokenizer → pretrain → mid → post → agent + evals |

**Honesty bar:** a ~100–200M from-scratch model will **not** match Qwen-1.5B on HumanEval. Success = working agent loop, rising easy-kata metrics across stages, clean ablations, and a story you can defend.

---

## 0. Locked decisions

| Decision | Lock | Why |
|----------|------|-----|
| Domain | Python katas | Verifiable reward; narrow enough for a tiny LM |
| Base model | **Your** decoder, **match Wiki-Transformer ~88M** | Same machine class; proven trainable in your prior project |
| Exact arch target | `d_model=384`, `n_layer=6`, `n_head=6`, `d_ff=1536` | Copied from `Wiki-Transformer/wikipedia_transformer.py` |
| Stretch size | Only if 88M underfits *and* VRAM/budget allow; do not start at 120M | Prefer more tokens over more params |
| Init | Random (normal / GPT-2-style) | True from-scratch |
| Tokenizer | **Train your own BPE** (vocab 16k–32k) on pretrain mix | Full ownership |
| Pretrain data | FineWeb-Edu **sample** + Python (`the-stack-smol`) + synth kata text/solutions | Language + code bias without multi-TB |
| Pretrain tokens | Target **~2–4B** if $ allows; minimum useful **~0.5–1B** | Prefer fewer params well-trained over huge undertrained |
| Mid-train | Required | Code + single-turn katas + **agent tool traces** |
| Post-train | SFT on wins → DPO → optional light RL | Test-verified rewards |
| Trace data | Mostly **canonical/programmatic** gold traces; self-play later | No Qwen teacher in v1 |
| Qwen | **Deferred comparison only** (§12) | After your stack works |
| Primary kata data | Own synth generator | Controllable, clean, no eval leak |
| Public eval | EvalPlus HE+ / MBPP+ (report honestly; expect low absolutes) | Comparability |
| Primary eval | 100 held-out **easy** synth katas, multi-turn success | Where gains show up |
| Sandbox | Subprocess + tempdir + timeout + import deny-list | Fast on macOS |
| Tools | `read_task`, `write_solution`, `run_tests`, `finish` | Minimal ReAct |

---

## 1. What you will ship

### Product
A local kata coding agent driven by **your** pretrained + mid/post-trained model.

### Deliverables
1. Custom tokenizer + from-scratch base checkpoint
2. Checkpoints: `pretrain` → `mid` → `sft` → `dpo` (+ optional `rl`)
3. Agent runtime (CLI) + sandbox + transcripts
4. Synth kata bank + frozen agent eval (100)
5. Stage ablations + curves + **skimmable README with figures** (§1.1)
6. *(Deferred)* Qwen mid/post comparison report

### 1.1 Final README showcase (remember at ship time)

Interviewers skim. **Keep the README short** (rough target: one screen of text + figures — ~80–120 lines max). Put detail in `PLAN.md` / `docs/` / notebooks, not a novel README.

**Must include (as images under `artifacts/figures/`, embedded in README):**
1. **Pretrain loss** vs steps/tokens (train + val if available)
2. **Stage ladder chart** — primary metric (agent success on frozen 100-katas) for `pretrain→mid→sft→dpo` (+ optional RL)
3. **Comparison bar chart** — your final agent vs prompted baselines and 1–2 references (e.g. same-size ablations; optional frozen open model as *upper bound*, clearly labeled; frontier only as published numbers or API baseline if you run it — don’t imply you trained them)
4. **Example generations / transcripts** — 2–3 compact figures or fenced blocks: one pass, one recover-from-fail, one failure (honest)
5. **Fail → fix replay GIF(s)** — screen-capture or animated frames of the agent writing bad code, seeing test failure, then fixing and passing (highest wow-per-hour demo asset)

**Optional extras (link, don’t dump):** more curves in `artifacts/figures/`, EvalPlus table in a collapsible section or `docs/results.md`.

**README outline (keep tight):**
1. One-liner + **fail→fix GIF** + short cli snippet  
2. Method in 5 bullets  
3. Figures (loss, stage ladder, comparisons)  
4. How to run  
5. Links to plan / checkpoints  

Log metrics to CSV/W&B **during** training so week-10 plotting is mechanical, not a scavenger hunt. Record the GIF from a real transcript in week 10 (scripted replay OK if timing is stable).

### Non-goals (v1)
- Matching Qwen/ChatGPT code quality
- LeetCode Medium/Hard train set
- SWE-bench / repo agents
- Spending the $100 on a 1B+ from-scratch model

---

## 2. Model architecture (from scratch)

Reuse the wiki transformer as the starting point; modernize only if it doesn’t explode scope.

### Default config — **KataLM-88M** (match Wiki-Transformer)

Source of truth: `/Users/samkhoshnevis/Documents/importants/code/Wiki-Transformer/wikipedia_transformer.py`

| Hyperparam | Wiki value | KataLM lock |
|------------|------------|-------------|
| `d_model` / `n_embd` | 384 | **384** |
| `num_layers` / `n_layer` | 6 | **6** |
| `num_heads` / `n_head` | 6 | **6** |
| `dim_feedforward` / `d_ff` | 1536 | **1536** |
| Dropout | 0.1 | 0.1 |
| `max_seq_length` (wiki) | 128 | **512** pretrain target (start 256 if needed; 128 is tight for agent traces) |
| Vocab (wiki) | tiktoken ~**100277** (untied emb + lm_head) | Prefer **own BPE 16k–32k** + **tied** emb → fewer embedding params, more capacity in blocks for same VRAM |
| Measured params (wiki config, untied, vocab 100277) | **~87.8M** | Expect **~40–90M** depending on vocab/tying; keep block dims fixed |
| README claim | “~150M” | **Incorrect** — ignore; use measured ~88M |

**Note:** Wiki README says training was on an **RTX 4090**; your Zephyrus is a **4060 8GB**. Matching the **88M arch** is still right (comfortable on 8GB). Spend extra budget on **tokens**, not width/depth.

**4060 rule:** if OOM, shrink **seq/batch** first, not layers. Only then fall back to wiki’s seq 128 for smoke tests.

### Tokenizer
1. Train BPE on a slice of the pretrain corpus (FineWeb-Edu sample + Python)
2. Vocab **32k** default (16k if you want faster embedding matmuls)
3. Special tokens: `<|user|>`, `<|assistant|>`, `<|tool|>`, `<|obs|>`, `<|end|>` (or equivalent) — add **before** mid-train
4. Save `tokenizer.json` + train script in-repo

### Implementation layout
```text
model/
  transformer.py    # blocks, attention, LM head
  config.py
tokenizer/
  train_bpe.py
  tokenizer.py
train/
  pretrain.py
  ...
```

Port patterns from the wiki project (attention, residual, training loop); don’t rewrite theory from zero unless you want to.

---

## 3. Agent design (unchanged product)

### Episode
1. Load kata: `prompt`, `entry_point`, `visible_tests`, `hidden_tests`
2. Agent ≤ **`max_steps = 8`**
3. Success = all **hidden** tests pass
4. Fail = step cap / invalid tools / finish failing

### Tools (`configs/schema.yaml`)

```text
{"tool":"read_task","args":{}}
{"tool":"write_solution","args":{"code":"def foo(x):\n    ..."}}
{"tool":"run_tests","args":{}}
{"tool":"finish","args":{"status":"pass"|"fail","note":"..."}}
```

| Tool | Behavior |
|------|----------|
| `read_task` | Problem + entry point + **visible** asserts |
| `write_solution` | Full replace of solution buffer |
| `run_tests` | Sandbox run; pass counts + truncated stderr |
| `finish` | End episode; env re-checks hidden tests |

No raw shell / network / pip in v1.

### Sandbox v1
- Tempdir + `subprocess` + **5s** timeout
- Static deny-list: `os`, `sys`, `subprocess`, `socket`, `ctypes`, …
- Truncate I/O to ~2k chars
- Optional Docker later for cloud rollouts

---

## 4. Data plan by stage

### 4.1 Pretrain corpus

| Source | Role | How much |
|--------|------|----------|
| **FineWeb-Edu sample** (HF sample-10BT or smaller sample) | General language | Majority of tokens |
| **`bigcode/the-stack-smol` Python** | Code prior | Upsample vs natural % (e.g. 20–40% of mix) |
| **Synth kata prompts + gold solutions** | Domain hint | Small but useful (1–5%) |
| Your old **wiki dump** (optional) | Extra text; reuse pipeline | If easy |

**Prep pipeline:** download → clean → train tokenizer → tokenize to memory-mapped `.bin` shards → shuffle.

**Token budget:**
- Ideal: ~20× params (Chinchilla-ish) → **~1.5–2B tokens** for ~88M
- With $100 + laptop: aim **≥1B**; celebrate **2B+** if cloud efficiency is good
- Log tokens/sec, loss curves, wall-clock $ spent

**Do not** download all of The Stack v2 / full FineWeb.

### 4.2 Kata problem bank (mid/post/eval)

| Source | Role | Notes |
|--------|------|-------|
| **Synth generator (primary)** | 15k–25k verified `(prompt, tests, solution)` | Your templates |
| MBPP sanitized **train** only | Extra easy seeds | Never MBPP/HE test or EvalPlus tasks |
| KodCode Prefill / easy (optional) | Volume | **CC BY-NC** — personal OK |

Template families: strings, lists, dicts, easy math, filter/map, fizzbuzz-like, trivial scans.

JSONL schema:
```json
{"id":"synth_00142","prompt":"...","entry_point":"is_palindrome","tests":["assert ..."],"solution":"def ...","split":"train","difficulty":"easy"}
```

Freeze **100** `agent_eval` ids before any mid/post train. Hash-check train ∩ eval = ∅.

### 4.3 Mid-train mixture

| Bucket | % | Content |
|--------|--:|---------|
| A. Short Python | 25% | stack-smol / synth solutions as raw LM text |
| B. Single-turn kata | 40% | problem → code (verified) |
| C. Agent traces | 35% | Multi-turn tool format, **mostly canonical gold** |

**Canonical trace (v1 workhorse — no Qwen needed):**
```text
User: <kata>
Assistant: {"tool":"read_task","args":{}}
User/obs: <problem + visible tests>
Assistant: {"tool":"write_solution","args":{"code":"<gold>"}}
User/obs: ok
Assistant: {"tool":"run_tests","args":{}}
User/obs: passed n/n
Assistant: {"tool":"finish","args":{"status":"pass"}}
```

Also generate **recovery** traces: inject a buggy first `write_solution`, show failing `run_tests`, then gold fix — teaches the fix loop without a strong teacher model.

**Self-play traces:** only after single-turn SFT exists and pass rate on easy train katas is non-trivial; filter to successes only.

### 4.4 Post-train

**SFT:** successful trajectories + strong single-turn examples (2k–6k).  
**DPO:** chosen = pass / valid schema; rejected = fail / bad JSON / wrong tool (2k–8k pairs).  
**Optional RL:** episode reward `+1` pass, `+0.2` valid schema, `−0.05`/step; REINFORCE/GRPO only if DPO works and $ remains.

### 4.5 Contamination
Never train on HumanEval, HumanEval+, MBPP test, or MBPP+ items.

---

## 5. Training stages (compute map)

### Stage 0 — Agent + kata env (before serious pretrain finishes)
Build sandbox, tools, loop, synth generator, 30 hand katas. Can parallelize with tokenizer/data prep.

### Stage 1 — Pretrain (main $ sink)

| Item | Lock |
|------|------|
| Objective | Next-token prediction |
| Optim | AdamW, β=(0.9, 0.95), wd=0.1, grad clip=1.0 |
| LR | ~3e-4 peak (tune), cosine decay, warmup ~1–2% steps |
| Batch | Max microbatch that fits; accumulate to ~0.25–0.5M tokens/step if possible |
| Seq | 512 |
| Eval | Held-out FineWeb-Edu + Python shard perplexity; sample generations weekly |
| Checkpoints | Every N steps + best val loss; save optimizer state for resume |

**Where:** **laptop-first** for the full pretrain if loss/time are acceptable. Rent cloud only when a gate in §5 budget fires.

### Stage 2 — Mid-train
LoRA **or** full FT (~88M full FT should be comfortable on 4060). Mixture A/B/C, seq 512–1024, shorter than pretrain. Default: laptop.

### Stage 3 — Post-train
SFT → DPO on laptop; optional RL only if useful and still local-first.

### Budget policy (LOCKED) — don’t spend if you don’t need to

**Default: $0 cloud.** The 4060 should handle ~88M pretrain/mid/post; overnight laptop runs are fine.

Spend from the $100 **only** if one of these is true:
1. Pretrain smoke works but full run would take an unreasonable wall-clock (e.g. multi-week) and a spot GPU clearly finishes it
2. Laptop thermally throttles / unstable for long jobs
3. You need a short burst to finish a stage before a deadline

If you do spend:

| Bucket | Cap | Notes |
|--------|----:|-------|
| Pretrain burst | ≤70 | Only after laptop dry-run |
| Mid/post overflow | ≤20 | Prefer not to |
| Buffer | rest | Failed pods |

**Rules:** no cloud until smoke loss trends down locally; 50–100 step dry-run on the pod; auto-shutdown; stop once metrics are “good enough” for the project — don’t burn leftover $ for vanity tokens.

---

## 6. Evaluation

### Primary (care about this)
- **Agent success rate** on 100 easy held-out synth katas
- Schema validity, avg steps, recovery rate
- Ablations: `pretrain-only` (prompted agent) → `+mid` → `+sft` → `+dpo`

### Secondary (report, don’t obsess)
- EvalPlus HumanEval+ / MBPP+ pass@1 — expect **low** absolute numbers; track relative movement
- Pretrain val loss / samples (“can it write a plausible `def`?”)

### Success criteria (v1 done)
- [ ] From-scratch checkpoint exists and generates coherent-ish Python pieces
- [ ] Agent runs end-to-end with your weights
- [ ] Mid and/or post clearly improves agent success vs pretrain-only agent
- [ ] Curves + ablation table + demo

Not required for v1: beating any Qwen.

---

## 7. Repo structure

```text
from-scratch-agent/
├── PLAN.md
├── README.md
├── configs/
│   ├── model_88m.yaml
│   ├── pretrain.yaml
│   ├── mid.yaml
│   ├── sft.yaml
│   ├── dpo.yaml
│   └── schema.yaml
├── model/
├── tokenizer/
├── data/                    # gitignore large artifacts
├── katas/                   # templates, generate, verify, split
├── sandbox/
├── agent/
├── train/
│   ├── pretrain.py
│   ├── midtrain.py
│   ├── sft.py
│   ├── dpo.py
│   └── rewards.py
├── eval/
├── scripts/
│   ├── prepare_pretrain.py
│   ├── make_canonical_traces.py
│   ├── make_dpo_pairs.py
│   └── cloud_launch.sh
├── artifacts/
│   └── figures/             # README plots (loss, stage ladder, comparisons)
└── docs/                    # optional long results; keep root README short
```

---

## 8. Timeline (~12 weeks part-time)

| Week | Focus | Machine |
|-----:|-------|---------|
| 1 | Model+tokenizer scaffold; sandbox+agent loop; 30 hand katas | Laptop |
| 2 | Synth generator; freeze agent_eval-100; data download scripts | Laptop |
| 3 | Tokenize shards; pretrain smoke (hours, watch loss↓) | Laptop |
| 4–5 | Pretrain long run on **laptop**; cloud only if §5 gate fires | Laptop (cloud optional) |
| 6 | Canonical traces + mid mix; mid-train | Laptop / $ |
| 7 | SFT on wins / single-turn | Laptop |
| 8 | DPO + agent eval ablations | Laptop |
| 9 | Fix failure modes; more easy data if needed | Laptop |
| 10 | CLI, **plot figures**, skimmable README (§1.1), demo | Laptop |
| 11–12 | Buffer / optional RL / writeup | Remaining $ |

**Deferred after v1:** Qwen mid+post comparison (§12).

---

## 9. Week 1 checklist

1. [ ] Port/adapt decoder to wiki dims (384 / 6 / 6 / 1536); unit-test forward + loss
2. [ ] BPE train script on a tiny text sample
3. [ ] Sandbox + 4 tools + agent loop
4. [ ] 30 hand-written katas with hidden tests
5. [ ] Confirm 4060 trains ~88M one step without OOM (seq 256–512, microbatch 1–8)
6. [ ] **No cloud** until smoke loss trends down on a toy shard

---

## 10. Risks

| Risk | Mitigation |
|------|------------|
| From-scratch model too weak for any kata | Keep templates trivial; measure train solve-rate; shrink task space |
| $100 undershoots token count | Keep **88M** wiki-matched size; buy more tokens, don’t upscale params |
| Pretrain loss falls but code is garbage | Upsample Python + synth solutions in mix |
| No strong teacher for traces | Canonical + recovery traces first |
| EvalPlus looks “failed” | Emphasize agent_eval ablations in writeup |
| Scope creep into Qwen | Leave §12 closed until v1 ships |

---

## 11. Portfolio framing (v1)

> I pretrained an **~88M** decoder **from scratch** (same scale as my wiki transformer), mid-trained it on code and tool-use kata traces, then post-trained with SFT/DPO using **unit-test rewards**, and wrapped it in a write→test→fix agent. Ablations show what each stage buys on a frozen easy-kata suite.

---

## 12. Deferred — Qwen comparison (not now)

After KataLM v1 is done, optional follow-up:

1. Take `Qwen/Qwen2.5-Coder-0.5B` or `1.5B-Instruct`
2. Run the **same** mid/post recipes + same eval suites
3. Compare: absolute agent success, $$, and “what from-scratch taught me vs adapting a strong base”

Until then: **do not** spend budget or weeks on Qwen mid/post. Optional: use a quantized Qwen **only** as a non-training baseline demo (“upper reference”) if you want a video contrast — still not training it in v1.

---

## 13. North star

**Own the stack:** tokenizer → from-scratch pretrain → mid → post → kata agent, measured on easy verifiable tasks, mostly on a 4060 — **$100 cloud only if necessary** — with Qwen comparison explicitly deferred.
