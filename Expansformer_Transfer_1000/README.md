# Expansformer v0.4 — Transfer from completed 1,000-step checkpoint

Additive warm-start trainer for the prepared, 431-document staged research snapshot.
Do **not** overwrite `train_expansformer_v04.py` or any original run files.

## Verified offline against uploaded files

- Expanded corpus: **431 selected**, 68 excluded; snapshot SHA256
  `e07b57fa83c2ac567c26d4d3ff39038bd4a85fe59666857c58fa154579c6d526`.
- All **221** old training and **39** old validation documents match identical text SHA256 values in the expanded corpus.
- Inherited split + new additions: **366 training** (221 old + 145 new), **65 validation** (39 old + 26 new), seed 42.
- The uploaded **750-step** checkpoint's state dictionary loaded strictly into the 28,437,129-parameter neural model and its tokenizer SHA256 matched. **Your local WSL 1,000-step checkpoint was not uploaded/verified in this test.** Verify its `step` before proceeding.
- No CUDA training was performed in this environment.

## Installation

Copy the zip into `~/ACTIVE_PROJECTS/Expansformer_v03_Research`, then:

```bash
cd ~/ACTIVE_PROJECTS/Expansformer_v03_Research
unzip -q Expansformer_Transfer_1000.zip
cp -n Expansformer_Transfer_1000/train_expansformer_v04_transfer.py ./train_expansformer_v04_transfer.py
```

The `cp -n` option preserves an existing transfer trainer; if yours differs, compare files before replacing.

## Verify local checkpoint step

```bash
python3 - <<'PY'
from pathlib import Path
import torch
p = Path('runs/v04_rtx5050_enhanced/best.pt')
ckpt = torch.load(p, map_location='cpu', weights_only=True)
print('Parent checkpoint:', p, 'step:', ckpt['step'])
assert ckpt['step'] == 1000, 'Expected best.pt from completed 1000-step run'
PY
```

## Transfer: first 25 updates

```bash
python3 train_expansformer_v04_transfer.py train \
  --data ./runs/v04_staged_research_data_499 \
  --strict-provenance \
  --out ./runs/v04_transfer_431 \
  --init-from ./runs/v04_rtx5050_enhanced/best.pt \
  --tokenizer bpe \
  --vocab-size 4096 \
  --tokenizer-sample-bytes 786432 \
  --n-layers 5 \
  --n-heads 6 \
  --embed-dim 384 \
  --context-length 512 \
  --batch-size 4 \
  --grad-accum 12 \
  --steps 800 \
  --stop-after 25 \
  --lr 0.00005 \
  --min-lr 0.000005 \
  --warmup-steps 50 \
  --dropout 0.1 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --eval-every 25 \
  --eval-windows 64 \
  --save-every 25 \
  --moe-experts 4 \
  --moe-top-k 2 \
  --moe-balance-weight 0.01 \
  --auxiliary-exits \
  --exit-loss-weight 0.15 \
  --memory-slots 16 \
  --memory-every 2 \
  --next-token-heads \
  --next-token-head-weight 0.1 \
  --attention sdpa \
  --amp bf16
```

## Continue to new-run update 100

Repeat the same command, replacing `--init-from ...` with
`--resume ./runs/v04_transfer_431/latest.pt` and setting `--stop-after 100`.
Do NOT use both `--init-from` and `--resume` together.

## Safeguards

- Keeps original BPE merges/token IDs and full neural model weights; refuses mismatched architecture/tokenizer.
- Inherits original train/validation assignments, avoids putting old training docs into validation.
- Creates a separate output directory with fresh optimizer and new LR schedule.
- Records parent checkpoint identity, original step, and transfer corpus provenance.
- On later resume validates that the transfer corpus/splits are unchanged.
- The 431-document set is an isolated experimental candidate corpus, not independently rights-cleared or editor-reviewed.

The old validation examples were previously used in checkpoint selection. Compare both old and new checkpoints on identical *separately held-out* evaluation examples before making transfer-quality claims.
