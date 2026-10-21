# Expansformer v0.4 — experimentally enhanced from-scratch language model

This is a **standalone PyTorch research model**. It does not use or wrap other cognitive systems. It includes the original Expansformer architecture, v0.1/v0.2/v0.3 programs and tests, plus two **trainable opt-in neural enhancements** derived from concepts in five original user-provided Python files.

The original five scripts **are not runtime imports**. Their specific opportunities, incorrect proxy signals and deferred functionality are in [`ARCHITECTURE_DECISIONS.md`](ARCHITECTURE_DECISIONS.md). See [`RESEARCH_CONTRACT.md`](RESEARCH_CONTRACT.md) for acceptance criteria and [`ablations_v04_toy.json`](ablations_v04_toy.json) for actual CPU ablation output. No pretrained weights, corpus, or Ollama export included.

## Features

- **Preserved:** causal attention, RoPE, RMSNorm, SwiGLU; v0.3 MoE experts, trained intermediate prediction exits; native from-scratch training, checkpoint resume, source fingerprints and tokenizer checks.
- **New:** `--memory-slots N` enables trainable internal key/value slots that are accessed at every `--memory-every` blocks. They contain neural parameters, **not** externally verified facts.
- **New:** `--next-token-heads` with `--auxiliary-exits` adds heads trained against true next-token labels. Their probability estimates are **not evidence of factual correctness**. The extra training objective is weighted by `--next-token-head-weight`.
- **New:** `--experimental-adaptive-exit-threshold` for research-only shallow inference and `evaluate_adaptive_v04.py` to measure depth decisions against held-out next tokens.
- **New:** strict v0.4 checkpoint separation, v0.4 controlled experiment comparison, explicit documented ablations.

## Install and test in WSL

```bash
cd ~/ACTIVE_PROJECTS/Expansformer_v04_Neural
# Reusing your already-functioning v0.2 CUDA environment is fine:
source ../Expansformer_v02_Research/.venv/bin/activate
python train_expansformer_v04.py smoke
```

Alternatively create a clean venv: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Confirm CUDA with `python -c 'import torch; print(torch.cuda.is_available())'`. Package testing was on CPU, not on your CUDA host.

## Use the existing 32-document Wikipedia pilot corpus for a small controlled comparison

**Start with small, equal-budget experiments**, not a large training commitment. The following runs use the same 32-document corpus, seed, tokenizer and training schedule. Change the `--out` and last switch only. All three start from new random initializations; v0.2 weights cannot be resumed as v0.4.

```bash
COMMON="--data ../Expansformer_v02_Research/corpus_wiki --strict-provenance --tokenizer byte --n-layers 4 --n-heads 4 --embed-dim 256 --context-length 128 --batch-size 8 --grad-accum 2 --steps 150 --eval-every 25 --save-every 25 --seed 42 --auxiliary-exits"

# Control: dense transformer with v0.3 auxiliary exits
python train_expansformer_v04.py train $COMMON --out ./runs/dense_control

# Memory-only neural alteration
python train_expansformer_v04.py train $COMMON --out ./runs/latent_memory --memory-slots 16 --memory-every 2

# Memory and trainable next-token accuracy predictor
python train_expansformer_v04.py train $COMMON --out ./runs/memory_predictor --memory-slots 16 --memory-every 2 --next-token-heads

python compare_runs_v04.py ./runs/dense_control ./runs/latent_memory
python compare_runs_v04.py ./runs/latent_memory ./runs/memory_predictor
```

Use different seeds and much larger held-out data before deciding if any enhancement deserves permanent adoption. `compare_runs_v04.py` requires both runs to have finished their planned steps, and refuses mismatched settings, corpus fingerprints or tokenizer hashes. Note: adding neural memory adds parameters and compute; the comparison is **not parameter-matched**.

## Generate from a trained v0.4 checkpoint

```bash
python train_expansformer_v04.py generate --checkpoint ./runs/memory_predictor/best.pt --prompt "Scientific reasoning requires" --max-new-tokens 60 --temperature 0.7
```

**Adaptive exit stays off by default.** To investigate, first audit on the exact held-out split:

```bash
python evaluate_adaptive_v04.py --checkpoint ./runs/memory_predictor/best.pt --data ../Expansformer_v02_Research/corpus_wiki --windows 32 --threshold 0.95
```

Then, strictly for experimentation, use `--experimental-adaptive-exit-threshold 0.95` when generating; it emits a warning. The neural head estimates correct *next tokens*, not whether a statement is trustworthy or well supported. No automated action is authorized on its output.

## Which variant is best?

**Not determined.** In the committed 12-step CPU toy run, memory slightly worsened held-out loss versus the dense model, and adding the accuracy head recovered some but not all of that difference. This is recorded as a failed/undetermined performance hypothesis, **not a claimed improvement**. All extensions remain opt-in.

## What remains missing

A substantial multilingual/multidisciplinary legally reviewed training corpus, verified source evidence wired into response generation, calibrated refusals, domain reasoning benchmarks, multi-seed comparisons, profiler-derived training cost, GPU confirmation for v0.4, and Ollama-compatible GGUF/runtime architecture support. None is silently simulated or asserted complete.
