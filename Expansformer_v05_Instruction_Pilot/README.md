# Expansformer v0.5 — Instruction Pilot

A **separate, reversible research experiment**, written for the Expansformer v0.4 code supplied in the conversation. This does **not** alter original v0.4 checkpoints, training scripts, source corpus, or tokenizer. It initializes every model parameter from `runs/v04_transfer_431/best.pt` (expected transfer step 800) and preserves MoE, parametric slot memory, intermediate exits and next-token confidence heads.

## Scope and limitations

- Supervised learning uses **answer/completion tokens only**. The question is given as input context and is masked from target loss. End-of-answer EOS is supervised.
- Existing `NeuralTransformerLM.training_loss` retains the same weighted auxiliary early-exit, MoE balance, and correctness-head objectives, also using the masked answer targets.
- The starter data is a *synthetic deterministic teaching/functional smoke corpus*: 244 training, 61 development, 61 reserved test examples. Numeric ranges are disjoint across splits; **the four task templates overlap**, so held-out accuracy would not establish broad reasoning. There are also a few explicitly labeled definition examples. No claim of factual review or independent legal clearance is made.
- This kit is not a production instruction corpus. Before any scientific claims, replace it with a reviewed, provenance-tracked dataset and genuinely untouched external tasks.
- The final test split is locked behind `--final-test`. Use it **only after** all decisions about training are made. Once used for decisions, it is no longer an untouched test.
- The same 4,096-ID BPE tokenizer is copied from your original checkpoint directory, verified by SHA256, and never retrained.
- Checkpoints can be resumed only using the same optimizer/learning schedule, dataset file hashes, tokenizer, architecture, and output directory. Changing `--stop-after` is allowed.
- This is a proof-of-training pilot, not a guarantee that a 28M-parameter pretrained model will achieve instruction competence with this small dataset.

## Layout 2 data (`data_v2/`, opt-in)

The original `data/` (layout 1) is unchanged and remains the default in every command below. It has three measurement problems: `compare` always answers with the second number (copying scores 100%), numeric ranges are disjoint across splits so "unseen numbers" and "larger than ever trained" are mixed together, and `definition` has 4/1/1 items. `data_v2/` addresses these:

- compare operand order is balanced (7 or 8 of 15 per group), so copying either position scores about 53%;
- in-range dev/test numbers are interleaved with train numbers inside the same range (everything below 100);
- extrapolation (every operand and answer 100 or more, none in train) is its own labeled category: `extrapolation_arithmetic`, `extrapolation_comparison`;
- definitions are 24/10/10. Exact match on unseen definitions is not informative; use masked answer CE and token accuracy for that category.

Build and test: `python3 build_starter_data.py --layout v2` (writes `data_v2/`); `python -m unittest discover -s Expansformer_v05_Instruction_Pilot/tests`. To use it, point `--data` at `data_v2`. Checkpoints record dataset hashes, so a layout 1 checkpoint cannot be evaluated or resumed against layout 2 data (the trainer refuses). Counts are 204/115/115 (train has fewer numeric rows than layout 1). Still synthetic, template-based and unreviewed; the guarantees above are about measurement design, not language ability. Not yet checked with the real 4,096-ID tokenizer.

## 1. Install in WSL

Place ZIP in `~/ACTIVE_PROJECTS/Expansformer_v03_Research`, then:

```bash
cd ~/ACTIVE_PROJECTS/Expansformer_v03_Research
unzip -q Expansformer_v05_Instruction_Pilot.zip
cp -n Expansformer_v05_Instruction_Pilot/train_expansformer_v05_instruction.py .
```

The existing project must provide `train_expansformer_v01.py`, `train_expansformer_v04.py`, `expansformer_v03_model.py`, `expansformer_v04_model.py`, `corpus_builder.py`, and its `expansformer_model.zip` or original source directory.

## 2. First bounded pilot: 25 updates

```bash
python3 train_expansformer_v05_instruction.py train \
  --init-from ./runs/v04_transfer_431/best.pt \
  --data ./Expansformer_v05_Instruction_Pilot/data \
  --out ./runs/v05_instruction_pilot \
  --device cuda --amp bf16 \
  --steps 200 --stop-after 25 \
  --batch-size 4 --grad-accum 4 \
  --lr 0.00002 --min-lr 0.000002 \
  --warmup-steps 15 \
  --eval-every 25 --save-every 25
```

**No other v0.4 flags required:** the v0.5 script loads the exact architecture and research configuration from the parent checkpoint. It starts a fresh optimizer, NOT the parent optimizer state. It writes `latest.pt`, `best.pt`, `initial_dev.json`, `metrics.jsonl`, a verified tokenizer copy, and provenance into its own output directory.

## 3. Check development-set generation (NOT final test)

```bash
python3 train_expansformer_v05_instruction.py evaluate \
  --checkpoint ./runs/v05_instruction_pilot/best.pt \
  --data ./Expansformer_v05_Instruction_Pilot/data \
  --split dev --max-examples 12 --max-new-tokens 40 \
  --out ./runs/v05_instruction_pilot/dev_quality.json \
  --device cuda
```

This compares teacher-forced answer loss and token accuracy **and** greedy exact answers. Greedy generation is a narrow behavioral probe, not a comprehensive quality metric. Inspect `dev_quality.json` for individual outputs. Results can be poor even when loss declines.

## 4. Resume to 100 then 200 if warranted

Only change `--init-from` to `--resume` and `--stop-after` to the new target; keep other settings identical:

```bash
python3 train_expansformer_v05_instruction.py train \
  --resume ./runs/v05_instruction_pilot/latest.pt \
  --data ./Expansformer_v05_Instruction_Pilot/data \
  --out ./runs/v05_instruction_pilot \
  --device cuda --amp bf16 \
  --steps 200 --stop-after 100 \
  --batch-size 4 --grad-accum 4 \
  --lr 0.00002 --min-lr 0.000002 \
  --warmup-steps 15 \
  --eval-every 25 --save-every 25
```

For the final run change `--stop-after 100` to `--stop-after 200`. Do not assume more updates help: compare development errors and stop if generation is not improving.

## 5. Check regression on old research text

Uses the original transfer validation documents as a **development regression signal**, not as an untouched benchmark:

```bash
python3 train_expansformer_v05_instruction.py lm-check \
  --parent ./runs/v04_transfer_431/best.pt \
  --checkpoint ./runs/v05_instruction_pilot/best.pt \
  --corpus ./runs/v04_staged_research_data_499 \
  --windows 64 \
  --out ./runs/v05_instruction_pilot/lm_regression.json \
  --device cuda
```

A large loss increase on the original research documents suggests forgetting. The script compares both checkpoints against exactly the same token windows and tokenizer.

## 6. Prompt it directly

```bash
python3 train_expansformer_v05_instruction.py generate \
  --checkpoint ./runs/v05_instruction_pilot/best.pt \
  --prompt $'Question: What is 11 plus 4?\nAnswer: ' \
  --max-new-tokens 40 --device cuda
```

## 7. One-time final synthetic test (after model selection)

```bash
python3 train_expansformer_v05_instruction.py evaluate \
  --checkpoint ./runs/v05_instruction_pilot/best.pt \
  --data ./Expansformer_v05_Instruction_Pilot/data \
  --split test --final-test --max-examples 0 \
  --max-new-tokens 40 \
  --out ./runs/v05_instruction_pilot/final_synthetic_test.json \
  --device cuda
```

Do not repeatedly tune after inspecting the test results. Even a good result on this synthetic set would only demonstrate generalization to **unseen numbers in familiar templates**, not real open-ended QA.

## Verification performed before delivery

- `py_compile`: passed.
- Four unit tests: passed (answer-only masking, masked padding, duplicate-prompt rejection across splits, oversize example rejection).
- Complete CPU smoke test with a synthetic, tiny v0.4 checkpoint: v0.5 init, 2 updates, resume to 4, development evaluation, and generation all ran.
- Supplied 4096-ID BPE tokenizer: all 366 examples encode without context truncation; maximum input length was 37 tokens for this starter dataset.
- **Not tested with your final 28.44M-parameter v0.4/RTX 5050 checkpoint**, which remains on your machine. GPU performance and learning quality must be validated locally.

## Suggested expansion after pilot

Create genuinely reviewed examples across human–AI collaboration, cognitive mechanisms, causal reasoning, explanation, uncertainty, domain knowledge, arithmetic, and instruction compliance. Include questions that cannot be answered by copying publication metadata. Record sources, permissions, source-to-question mapping, reviewer signoff, and split identity. For architecture evidence, separately train compute-matched no-MoE/no-memory/no-exit baselines; changing modules at inference does not prove causal contribution.
