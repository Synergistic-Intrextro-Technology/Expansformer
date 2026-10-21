# Expansformer v0.4 research contract

**Outcome:** Keep intact Expansformer v0.3 behavior and add isolated, trainable memory-slot and next-token exit-head mechanisms that can be trained, checkpointed, restored, and compared against the unchanged baseline.

**Success conditions established before experiment:**
1. V0.3 model with enhancements disabled matches v0.4 state tensors and logits bit-for-bit on CPU.
2. Gradient reaches every new trainable component when enabled, including joint MoE + exits + memory + next-token heads.
3. No future-token leakage in causal outputs.
4. An interrupted run resumed from checkpoint exactly matches uninterrupted CPU model weights.
5. Factual confidence is never inferred from token statistics or model descriptions; evaluation labels remain explicitly next-token only.
6. Training settings, source provenance and tokenizer hashes are verified before resumption.
7. Baseline and candidates evaluated on identical split and tokenizer, with changed parameter counts declared. Improvement requires separate trials on larger datasets and multiple seeds, not only tests or toy loss.

**Rollback:** v0.3 source and tests unchanged; v0.4 all switches default disabled. Checkpoint formats separated.

**Validation:** Run `python train_expansformer_v04.py smoke`. Run `python evaluate_adaptive_v04.py` on a separate held-out corpus before attempting predictive early exits in generation. Its result is evidence of *next-token* selection behavior only.

**Authority:** Human operator must approve choosing a more costly/worse-validation variant for expensive training. No automatic selection, implicit model modifications or retraining.
