# Research contract and evidence boundaries

**Objective:** Train a standalone, independently initialized Expansformer model with reproducible loss measurements, auditable corpus metadata, and an honest separation between text generation, source matching and validated facts.

**Baseline:** The unchanged `expansformer_model/` and preserved `train_expansformer_v01.py`; classic causal attention remains available as a comparable control.

**Acceptance for this release:** 100% offline regression tests pass; tokenizer roundtrip including UTF-8; source hash change blocks continuation; document-level validation has no exact duplicate content across train/validation; interrupted-run model weights match uninterrupted CPU run for the tested settings; attention patch matches classic forward/backward within specified tolerances; train/eval/checkpoint/generation execute end to end.

**Guards:** No pretrained weights; no silently sourced training data; no conversion of generated confidence into factual verification; no labeling retrieved text as truthful without external corroboration. Do not overwrite existing training output. Keep original source recoverable.

**Test results:** See `verification_results.json` and `smoke_stderr.txt`. Small synthetic fixtures establish correctness of code paths, not useful language competence. Apparent speed improvements from a single short run are not a defensible benchmark due to warmup and I/O asymmetry.

**Measurement plan for genuine quality improvement:** Predeclare a licensed, multi-domain train corpus and frozen, deduplicated evaluation split; report held-out NLL, byte-normalized cross-entropy, token accuracy, domain-level task suites, abstention precision/recall, citation support rate, memory, latency and token throughput. Compare equal parameter budgets and training compute, seeds where affordable. Adopt changes only when benefits replicate without degrading reliability.

**Revalidation triggers:** Corpus changes; tokenizer changes; model architecture, attention kernel, PyTorch/CUDA version, optimizer settings, seed or evaluation procedures; checkpoint format changes; any new claim-verification logic. CUDA tests and live network fetch remain open until run on deployment hardware.

**Authority:** Local scripts perform only their explicitly invoked operation. Wikimedia fetch requires an explicit `wiki` command and user-agent/contact. This package does not deploy to Ollama or execute model-suggested actions.

**Unresolved:** Sustained, quality-filtered multi-domain data; scalable token shards; independent factual QA scoring; GPU training validation; native Ollama backend and tokenizer conversion.

## v0.3 addendum

The original v0.2 and v0.2.1 features remain unchanged. Neural extensions live in `expansformer_v03_model.py` and `train_expansformer_v03.py`, with independent checkpoint format. Optional sparse experts and intermediate exits must never be described as established improvements without controlled multi-seed comparisons. The evaluator always selects on primary next-token CE, not weighted auxiliary loss. Corpus provenance is not factual verification. No Ava wiring or use of experimental Python modules as unverified runtime dependencies is present.
