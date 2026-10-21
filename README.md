# Expansformer v0.3 — Controlled neural architecture experiments

**Stand-alone from-scratch language-model training. No Ava integration. No pretrained weights. No automatic web calls.**

This release preserves `expansformer_model/`, `train_expansformer_v01.py`, `train_expansformer_v02.py`, `corpus_builder.py`, the evidence index, and the original v0.2 tests unchanged. It adds a new `train_expansformer_v03.py` and `expansformer_v03_model.py`. Your previously trained **v0.2 checkpoints remain on the v0.2 execution path**; the v0.3 checkpoint schema is intentionally separate.

## What actually changed

| Mode | CLI switches | Trainable mechanism | Trade-offs |
| --- | --- | --- | --- |
| Dense control | none | Original TransformerLM, numerically identical at initialization | Strongest baseline, fewest parameters |
| Sparse expert | `--moe-experts 4 --moe-top-k 2` | Each dense SwiGLU FFN is replaced by four real SwiGLU experts with top-2 token routing | More parameters; dynamic routing may be slower; no performance gain assumed |
| Intermediate exits | `--auxiliary-exits --exit-loss-weight 0.15` | Intermediate decoder norms receive supervised next-token CE in addition to final output | Extra optimization objective; early depth requires validation |
| Both | both sets of switches | Expert layers and intermediate exits train jointly | Experimental; potentially more costly and harder to optimize |

**Important:** The source `expansformer_model/MoE.py` is a separate classification research prototype. This release implements a genuine token-level LM mixture-of-experts mechanism instead of pretending that classifier plugs into the LLM. Ideas about adaptive depth in the uploaded `enhanced_agi_core.py` and `cognitive_orchestrator_v2.py` informed the intermediate-exit experiment, but their classes are NOT imported or assumed to work. `autonomous_ai_development_engine.py`, `enhanced_innovation_engine.py`, and `ApexModal(1).py` remain external-system candidates, not hidden dependencies or trained language modules. See `ARCHITECTURE_DECISIONS.md`.

## Install / smoke test (Ubuntu WSL)

```bash
cd ~/ACTIVE_PROJECTS/Expansformer_v03_Research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.3'  # use appropriate official CUDA PyTorch wheel for your GPU
python train_expansformer_v03.py smoke
```

A CPU PyTorch installation works for tests. To train on GPU verify `python -c 'import torch; print(torch.cuda.is_available())'`. If false, install the CUDA-enabled PyTorch wheel appropriate for your driver/GPU. The environment used for packaging was CPU-only; no GPU benchmarking occurred in this session.

## **Corpus path**

If your original 32-document corpus is available, point v0.3 at it without
duplicating or overwriting it. In the current workspace, the referenced v0.2
directory is empty, so these commands require that you restore or provide the
corpus first:

```bash
DATA=../Expansformer_v02_Research/corpus_wiki
test -f "$DATA/manifest.jsonl" || { echo "Corpus manifest not found: $DATA/manifest.jsonl" >&2; exit 1; }
python corpus_builder.py audit --data "$DATA" --require-license
```

`--require-license` checks supplied metadata, not legal or factual truth. This corpus is useful for controlled experiments only, **not** for a high-quality strategic assistant.

## Dataset pipeline (experimental)

The local-first knowledge and reasoning dataset scaffold, provenance importer,
quality audit, document-level split builder, and opt-in Wikipedia category
collector are documented in [expansformer_dataset/README.md](expansformer_dataset/README.md).
The scaffold contains no source corpus by default. Wikipedia acquisition is
explicit, bounded, and still requires article-specific license and quality
review before training use. A separate, unadmitted 240-document topic-gap
candidate batch is available at [topic_gap_batch_240_revised/](topic_gap_batch_240_revised/).
An additional human-focused research batch is staged separately in
[topic_gap_human_expansion/](topic_gap_human_expansion/); the category counts,
source screens, duplicate findings, and unmet target gaps are recorded in
[the corpus candidate quality review](topic_gap_batch_240_revised_QUALITY_REVIEW.md).
Neither batch has been admitted to the canonical dataset or accepted for training.

## The first matched 100-step comparison

Set the data path once, keep all controls identical. These are **new random-init training runs**, not resumes of your prior 1,000-step checkpoint.

```bash
DATA=../Expansformer_v02_Research/corpus_wiki
test -f "$DATA/manifest.jsonl" || { echo "Set DATA to a prepared corpus; manifest not found: $DATA/manifest.jsonl" >&2; exit 1; }
COMMON="--data $DATA --strict-provenance --tokenizer bpe --vocab-size 512 --n-layers 4 --n-heads 4 --embed-dim 256 --context-length 128 --batch-size 8 --grad-accum 2 --steps 100 --warmup-steps 10 --eval-every 25 --eval-windows 64 --save-every 25 --seed 42"

# Control: original transformer architecture
python train_expansformer_v03.py train $COMMON --out ./runs/v03_dense_100

# Candidate: MoE and trainable intermediate exits
python train_expansformer_v03.py train $COMMON --moe-experts 4 --moe-top-k 2 --auxiliary-exits --out ./runs/v03_moe_exits_100

# Report measured, compatible held-out comparison
python compare_runs.py ./runs/v03_dense_100 ./runs/v03_moe_exits_100
```

This is shell syntax for bash. `COMMON` is only simple space-separated flags; if your corpus path contains spaces, pass the arguments directly or use a shell array.

When scaling to a larger corpus, increase vocabulary, context, and parameter count in separate *new* runs with corresponding controls. This is a tiny benchmark, not meaningful domain competence evidence.

## Study intermediate depth

```bash
DATA=../Expansformer_v02_Research/corpus_wiki
test -f "$DATA/manifest.jsonl" || { echo "Set DATA to a prepared corpus; manifest not found: $DATA/manifest.jsonl" >&2; exit 1; }
python evaluate_exits_v03.py \
  --checkpoint ./runs/v03_moe_exits_100/best.pt \
  --data "$DATA" \
  --max-loss-increase 0.03
```

The tool evaluates each depth on the same held-out text and returns the shallowest one whose held-out loss is within the **user-specified loss increase** relative to full depth. This does NOT establish calibrated confidence, response factuality, or inference-speed benefit. Test the chosen depth on untouched tasks before accepting it.

```bash
python train_expansformer_v03.py generate \
  --checkpoint ./runs/v03_moe_exits_100/best.pt \
  --inference-depth 2 \
  --prompt 'Scientific reasoning requires' \
  --temperature 0 --max-new-tokens 80
```

Omit `--inference-depth` for full computation. Asking for an untrained intermediate exit is an error, not silent fallback.

## Resume without modifying the experiment contract

V0.3 retains the v0.2 strict fingerprints, optimizer/RNG resume and atomic checkpoints. An intentional early stop can use `--steps 1000 --stop-after 100`; then use the *same* configuration with `--resume <out>/latest.pt --stop-after 250`. Only `--stop-after` and `--resume` should change. For ablation comparison, fully complete both experiments at the same planned step count.

**No cross-version checkpoint migration:** to continue your trained `wiki_pilot_01`, run the original `train_expansformer_v02.py`, with your matching old arguments. Switching to MoE changes trainable weight structure and requires new training; it cannot preserve the original v0.2 optimizer state and model equivalence.

## Evidence limits and known voids

- `train_loss` records **total weighted training objective**; `train_lm_loss` separately reports the main next-token CE. `train_exit_loss` and `train_router_balance_loss` show auxiliary terms. **Validation `val_loss` reports primary next-token CE**, not the training objective.
- Full-depth v0.3 generation is not connected to `claim_gate.py`: generated text is neither fact-checked nor calibrated. Retrieval and evidence inspection tools remain separate.
- Architecture enhancements are opt-in and **not demonstrated to beat the dense baseline**. MoE adds parameters and routing complexity; trainable exits add additional objectives.
- Ollama/GGUF import, GPU throughput, a large licensed corpus, reproducible multi-seed held-out gains, multimodal pretraining, and reliable abstention remain unverified.
- Original v0.2 documentation and limitations are retained under `docs/README_v021_ARCHIVE.md`.

Run `python -m unittest discover -s tests -v` for all old and new offline tests. The local verification summary is `verification_v03.json`; a tiny synthetic ablation is not a claim about language intelligence.
