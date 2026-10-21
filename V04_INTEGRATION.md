# Expansformer v0.3 → v0.4 neural upgrade

This is an **additive upgrade**, using the user-provided v0.3 project as the source of truth. Existing v0.3 scripts, expanded dataset pipeline, Wikipedia acquisition, evaluation examples, model archive and files are preserved. The upgrade adds **new entry points** (`train_expansformer_v04.py`, `expansformer_v04_model.py`, `evaluate_adaptive_v04.py`, `compare_runs_v04.py`) and `tests/test_v04.py`. It does **not** convert existing checkpoints or configure new components on old models.

## Run as a standalone patch against your WSL project

Extract `Expansformer_v04_Upgrade.zip` into `~/ACTIVE_PROJECTS/Expansformer_v04_Upgrade` then:

```bash
cd ~/ACTIVE_PROJECTS/Expansformer_v04_Upgrade
python apply_v04_upgrade.py --target ~/ACTIVE_PROJECTS/Expansformer_v03_Research --dry-run
python apply_v04_upgrade.py --target ~/ACTIVE_PROJECTS/Expansformer_v03_Research
cd ~/ACTIVE_PROJECTS/Expansformer_v03_Research
source ../Expansformer_v02_Research/.venv/bin/activate
python -m unittest discover -s tests -q
python train_expansformer_v04.py smoke
```

If a destination file already exists with the same checksum, the patch skips it. If different, the patch **refuses to overwrite** unless you explicitly supply `--replace`, which creates a backup under `.expansformer_upgrade_backups/` first. The patch verifies SHA-256 hashes of its payload before writing and supports `--verify` after installation. It **does not run training automatically**.

## Training (new weights; separate directory)

```bash
python train_expansformer_v04.py train \
  --data ../Expansformer_v02_Research/corpus_wiki \
  --strict-provenance --out ./runs/v04_trial \
  --tokenizer bpe --vocab-size 512 \
  --n-layers 4 --n-heads 4 --embed-dim 256 --context-length 128 \
  --batch-size 8 --grad-accum 2 --steps 150 \
  --eval-every 25 --save-every 25 \
  --moe-experts 4 --moe-top-k 2 --auxiliary-exits \
  --memory-slots 16 --memory-every 2 --next-token-heads
```

`--data` accepts text documents via `corpus_builder.py`; **the expanded v0.3 `dataset_pipeline.py` does not automatically become the v0.4 trainer's loader**. The larger dataset infrastructure remains available for curation, but verify the output layout and manifests before pointing the trainer to curated documents.

## Scientific status

- v0.4 neural memory, MoE and early-exit predictors are **experimental, opt-in** features.
- The `next-token-heads` signal measures prediction of next tokens, **not factual validity**.
- MoE and memory are not yet proven superior on large held-out corpora. CPU toy results in `docs/ablations_v04_toy.json` are historical and not a GPU benchmark.
- Existing v0.2/v0.3 checkpoints remain separate. v0.4 needs training from scratch.
- No Ollama export, pretrained weights, or validated large corpus are included.
- For the original v0.4 design and verification provenance, see `docs/README_v04.md`, `docs/ARCHITECTURE_DECISIONS_v04.md` and `docs/verification_v04_source.json`.
