# Expansformer v0.2.1 — independent from-scratch language model lab

**Working code for original random-initialized model weights, curated corpus acquisition, training, held-out evaluation, and strict traceability.** The included `expansformer_model/` is copied unchanged from the supplied v0.1 ZIP. The old trainer is also preserved as a rollback baseline.

> **Status:** CPU functionality verified on PyTorch 2.10.0. This is not an already-trained, high-quality assistant. No pretrained model, pretrained tokenizer, or cloud inference API is required. **Ollama/GGUF conversion is not implemented** and may require inference-backend support for the architecture.

## Installation in WSL

```bash
unzip Expansformer_v02_1_Corpus.zip
cd Expansformer_v02_1_Corpus
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.3'
python hardware_report.py
python train_expansformer_v02.py smoke
```

For an NVIDIA GPU, install the correct CUDA-enabled PyTorch wheel from <https://pytorch.org/get-started/locally/> instead of relying on the generic pip command. Verify `nvidia-smi` and `torch.cuda.is_available()`. GPU training was **not** tested in the delivery environment.

## You don't have a corpus yet? Opt-in bootstrap

You can fetch a small **educational example corpus** from Wikipedia with provenance, article URLs, revision timestamps, history links, license metadata and SHA-256 hashes.

```bash
python corpus_builder.py wiki \
  --out ./corpus_wiki \
  --user-agent 'ExpansformerResearch/0.2 contact: you@example.com' \
  --max-pages 32
python corpus_builder.py audit --data ./corpus_wiki --require-license
```

Replace the example contact address with one you actually control. This downloads text **only when explicitly invoked**. It requires internet access and Wikimedia API availability; the live network acquisition path was not exercised by the automated offline tests. Wikipedia text is generally CC BY-SA 4.0; the manifest is a **license claim**, not a legal clearance. Preserve attribution, license links, article histories and applicable share-alike obligations. See <https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse>. No photos or other separately licensed media are acquired.

**This small bootstrap dataset is only for validating learning and infrastructure.** Do not mistake a few dozen encyclopedia pages for broad knowledge. For serious pretraining, add far more diverse properly licensed documents (technical manuals, science, math, history, economics, philosophy, multilingual material) and evaluate leakage, source reliability and domain balance independently.

You can instead create a directory of UTF-8 `.txt` / `.md` files. At least two **distinct** documents are required for a document-level holdout. For a single document, provide a separate `--val-data` folder rather than splitting adjacent text into supposedly independent validation.

## Train from scratch

Quick small experiment:

```bash
python train_expansformer_v02.py train \
  --data ./corpus_wiki --strict-provenance \
  --out ./runs/expansformer_v02 \
  --tokenizer byte \
  --n-layers 2 --n-heads 4 --embed-dim 64 \
  --context-length 64 --batch-size 4 --grad-accum 2 \
  --steps 200 --warmup-steps 10 \
  --eval-every 20 --eval-windows 16 --save-every 20
```

For a larger run with your own BPE tokenizer:

```bash
python train_expansformer_v02.py train \
  --data ./corpus_wiki --strict-provenance \
  --out ./runs/expansformer_bpe \
  --tokenizer bpe --vocab-size 512 --tokenizer-sample-bytes 131072 \
  --n-layers 4 --n-heads 4 --embed-dim 256 \
  --context-length 128 --batch-size 8 --grad-accum 2 \
  --steps 1000 --warmup-steps 30 \
  --eval-every 100 --eval-windows 64 --save-every 100
```

BPE training uses an intentionally transparent pure-Python implementation: correct and testable on small inputs, not a production-scale tokenizer trainer. The example 512-entry vocabulary is intentionally tiny; it is **not** a recommendation for a high-quality, full-scale LLM. Corpus files are currently read and tokenized in RAM, so very large pretraining requires a streaming/token-sharded data loader before scaling.

The trainer writes `config.json`, `tokenizer.json`, `provenance.json`, `metrics.jsonl`, `latest.pt`, and `best.pt`. You can choose `--attention classic` for the baseline attention or `--attention sdpa` (default) for PyTorch scaled dot-product attention. The latter keeps the **same trainable weights and mathematical attention**, but GPU performance gains are not promised without measurement. Configure `--early-stop-patience 5 --min-delta 0.001` to stop when validation ceases improving.

## Resume safely

```bash
python train_expansformer_v02.py train \
  --data ./corpus_wiki --strict-provenance \
  --out ./runs/expansformer_v02 \
  --tokenizer byte \
  --n-layers 2 --n-heads 4 --embed-dim 64 \
  --context-length 64 --batch-size 4 --grad-accum 2 \
  --steps 200 --warmup-steps 10 \
  --eval-every 20 --eval-windows 16 --save-every 20 \
  --resume ./runs/expansformer_v02/latest.pt
```

Keep all original training arguments **exactly the same**, including corpus contents, seed, tokenizer, schedule, attention implementation and split settings. The trainer rejects mismatched datasets and configurations and rewinds metric lines beyond the checkpoint if a crash happened after metrics were appended. Reproducible CPU resume was tested; a final missing metrics row is restored with a null measurement rather than an invented loss; CUDA/kernel determinism has not been established. **Do not load unknown third-party checkpoints**, even with safe loading conventions.

If you want to interrupt before the full schedule, set `--stop-after 40` on the **first** command, while preserving `--steps 200`; resume without `--stop-after`.

## Generate with the actual trained weights

```bash
python train_expansformer_v02.py generate \
  --checkpoint ./runs/expansformer_v02/best.pt \
  --prompt 'What can we conclude from these observations?' \
  --max-new-tokens 80 --temperature 0
```

**Generation is NOT automatically fact-checked.** An untrained or tiny model may produce meaningless or false output regardless of its prompting or documentation.

## Separate evidence retrieval and claim handling

```bash
python evidence_index.py index --data ./corpus_wiki --db ./evidence.sqlite3
python evidence_index.py search --db ./evidence.sqlite3 --query 'scientific evidence'
python claim_gate.py --corpus ./corpus_wiki --claims ./examples/claims.json
```

The index returns document paths, exact provenance hashes, titles, source URLs, licenses and matching excerpts. The claim gate accepts only **exact source quotations** as source-matched (not verified true), reports inferences as unverified and rejects unsupported assertions by default. This is a **standalone inspection tool**, not yet wired into model decoding. It cannot prove correctness or determine whether a source is authoritative. Model outputs must not be advertised as grounded unless actually passed through the validation procedure.

## Validation and observed scope

```bash
python train_expansformer_v02.py smoke
```

Eleven automated tests verify: tokenizer Unicode/BPE roundtrip, deterministic document-level split and disjoint hashes, manifest hash tampering, SDPA/classic logit and gradient parity, checkpoint/resume weight identity, changed-dataset resume refusal, SQLite retrieval, strict claim gate behavior, source acquisition through a **mocked** network response, and parameter-accounting consistency. A three-step toy run reduced validation cross-entropy; toy data is not evidence of broad competence.

## Why this version is stronger than v0.1

- Deterministic **document-level** train/validation split instead of splitting one document into adjacent chunks; source duplicates rejected.
- A corpus acquisition path that logs title, domain, revision, license metadata and content hash, with manifest tamper detection.
- Validation samples non-overlapping windows from throughout the held-out token stream; metrics include loss, perplexity, token accuracy, bits/byte, and validation coverage.
- Strict checkpoint fingerprint, tokenizer checksum, full optimizer and sampling/RNG state; atomic checkpoint writes, metric-tail recovery, early stopping.
- Opt-in drop-in PyTorch scaled dot-product attention with regression-tested logits and gradients.
- Evidence source index and bounded claim gate **do not equate retrieval, agreement or model confidence with truth**.
- A hardware resource estimator and safe micro configurations before investing in a longer run.

## Current limitations / development voids

1. No substantial licensed multidisciplinary corpus has been downloaded or trained in this package; live Wikimedia acquisition still needs environment verification.
2. No trustworthiness metric or calibrated abstention for the neural network itself. The external claim gate does **not** inspect free-form generations automatically.
3. No native Ollama inference or general GGUF export. Converting this original weight format alone is not sufficient if the architecture or tokenizer is unsupported. See [Ollama importing models](https://github.com/ollama/ollama/blob/main/docs/import.mdx).
4. No large-scale data streaming/sharding, distributed GPU training, packed-sequence masks, model KV cache, or throughput benchmark for deployment. These must be validated before a serious-scale release.
5. Exact hashes prevent accidental tampering but do **not** prove that facts, annotations, licensing or source quality are correct. Document overlap beyond exact duplication requires stronger controls.
6. No independently held-out multidisciplinary evaluation benchmark. Novelty, quality and strategic precision have **not** been demonstrated.

Original code is preserved for recovery; evaluate improvements against that version under matched conditions before accepting any measurable performance claims.

## v0.2.1 corpus collector maintenance update (October 17, 2026)

This preserves the v0.2 neural architecture, trainer, checkpoints, and public Python import/function contracts. Only `corpus_builder.py` and its offline tests are extended.

- Default `--max-pages 32` now draws topics in *round-robin domain order*, rather than silently excluding the last two domains (medicine and arts). The topic pool still contains only 40 manually selected English Wikipedia articles: it is not a substantial pretraining dataset.
- The collector handles API maxlag warnings and HTTP 429/503 throttling with bounded retries, and honors `Retry-After` (capped to 120 seconds). It keeps one request at a time and does not run background acquisition.
- Corpus outputs are staged in a sibling temporary directory, hashes are recomputed from saved UTF-8 text, and the corpus is published only after validation. This reduces partial-output failures; it does not make downloads transactional against third-party changes.
- Saved manifests now include `quality_diagnostics` (word counts, unique word ratio, repeated lines) and `license_status=SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED`. The `audit` command summarizes domain counts and warnings. **No script establishes source truthfulness or grants reuse rights.**
- Existing `read_documents`, `split_documents`, `corpus_fingerprint`, and trainer checkpoint interfaces are unchanged. Previously trained v0.2 checkpoints remain compatible if the underlying dataset and training arguments remain unchanged.

Run the no-network suite:

```bash
python -m unittest discover -s tests -v
```

Dry-run topic balancing without downloading:

```bash
python -c "from corpus_builder import TOPICS,balanced_topics; from collections import Counter; print(Counter(d for d,t in balanced_topics(list(TOPICS),32)))"
```

When ready to acquire a new dataset explicitly:

```bash
python corpus_builder.py wiki \
  --out ./corpus_wiki_balanced \
  --user-agent 'ExpansformerResearch/0.2.1 (your-real-email@example.org)' \
  --max-pages 32 --delay 1
python corpus_builder.py audit --data ./corpus_wiki_balanced --require-license
```

Use a real contact in the User-Agent. Follow [Wikimedia API etiquette](https://www.mediawiki.org/wiki/API:Etiquette) and [2026 Wikimedia API rate limits](https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits). These commands have not been executed against live Wikipedia in this release.
