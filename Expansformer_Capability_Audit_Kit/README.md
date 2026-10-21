# Expansformer v0.4 — Automated Capability and Architecture Audit

This is an **evaluation-only** audit for the v0.4 original 1,000-update checkpoint and the subsequent 800-update expanded-corpus transfer checkpoint. It **does not** retrain, fine-tune, or overwrite model weights.

## Installation in WSL

Place the script in the root of your existing `Expansformer_v03_Research` project (beside `expansformer_v04_model.py`, `train_expansformer_v04_transfer.py`, and `corpus_builder.py`). If you downloaded the ZIP:

```bash
cd ~/ACTIVE_PROJECTS/Expansformer_v03_Research
unzip -oq Expansformer_Capability_Audit_Kit.zip
cp Expansformer_Capability_Audit_Kit/audit_expansformer_v04.py ./audit_expansformer_v04.py
```

Requirements: the existing project Python environment, with PyTorch and its native model-loading files already working. Your WSL folder must contain the expanded snapshot `runs/v04_staged_research_data_499`, the **1,000-update** `best.pt`, and the **800-update** transfer `best.pt`.

## One-command audit

```bash
cd ~/ACTIVE_PROJECTS/Expansformer_v03_Research
python3 audit_expansformer_v04.py \
  --original ./runs/v04_rtx5050_enhanced/best.pt \
  --transfer ./runs/v04_transfer_431/best.pt \
  --data ./runs/v04_staged_research_data_499 \
  --out ./runs/v04_capability_audit \
  --device cuda \
  --windows 128 \
  --depth-windows 12 \
  --memory-windows 12 \
  --batch-size 4 \
  --new-tokens 64
```

The default behavior creates SHA-256-verified frozen copies of both checkpoints in `runs/v04_capability_audit/frozen/`, which requires about 600 MB of additional disk space. Pass `--no-freeze` if you already have verified, safely stored checkpoint copies.

For a short preliminary run:

```bash
python3 audit_expansformer_v04.py --device cuda --windows 16 --depth-windows 4 --memory-windows 4 --new-tokens 24 --no-freeze
```

## Outputs

- `runs/v04_capability_audit/REPORT.md`: readable audit and interpretation limits.
- `runs/v04_capability_audit/audit.json`: machine-readable evaluation metrics, checkpoint hashes, per-layer exit metrics, router frequencies, confidence-head calibration, memory-gate perturbation, timing, and generation.
- `runs/v04_capability_audit/manual_quality_review.csv`: generated answers from both models; leave reviewer scores blank until human review.
- `runs/v04_capability_audit/frozen/`: immutable-by-convention SHA-256-verified checkpoint copies (unless `--no-freeze`). These are ordinary files; restrict permissions and storage access as needed.

## Interpretation boundaries

- Matched cohorts are development-validation examples, not genuinely untouched test sets. Recorded coverage specifies the fraction evaluated.
- Memory-gate zeroing is **inference-time sensitivity testing**, *not* a properly trained no-memory control. To prove the architectural value of memory, retrain a matched no-memory model and compare on untouched tasks.
- MoE routing frequencies cannot establish beneficial expertise; early-exit confidence predicts whether token argmax is correct, not whether an answer is factual or safe.
- Early-exit latency is an isolated single-window microbenchmark; system-level throughput requires further benchmarking.
- Greedy generated outputs are **qualitative**; no automatic reasoning accuracy claims are produced. Have a reviewer score the CSV and create a new genuinely held-out task set.
- The research corpus remains an experimentally staged dataset with source-declared licenses rather than independently cleared rights.

## Verification performed

- Python syntax compilation.
- End-to-end CPU synthetic-checkpoint smoke test for matched cohort evaluation, depth analysis, memory perturbation, routing, calibration, generation, JSON/Markdown/CSV output.
- Independent smoke test for checkpoint freezing and timing.
- Negative test confirms that wrong checkpoint step fails closed.
- The **actual 1,000-update and 800-update checkpoints have not been run through this script in this environment**; execute in WSL to obtain real model evidence.
